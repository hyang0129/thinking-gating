#!/usr/bin/env python3
"""
capture_readout.py — D0 decision readout (System 1, training-free) on a capture.

For every item of an existing paired thinking-off/on capture this runs ONE
forward pass and reads the model's answer as a distribution over the item's
valid options (LLM2Jev-style; proposal v3 §4, design D0). No token is
generated. The result is the "S1" tier of the S1 / no-think / think table
(proposal §6, Phase 0 step 1), on exactly the items the capture labelled.

Prompt, per item:

  1. The thinking-OFF prompt the capture used. The raw prompt is rebuilt from
     the stored question and matched to the row's `prompt_hash` with the same
     candidates as scripts/rescore_confidence.py (BBH v3 used the task's
     PROMPT_TEMPLATE_V1), so the question text, options and answer-format
     instruction are byte-identical to what the generations saw.
  2. READOUT_INSTRUCTION appended to the user turn: answer only, no working.
  3. The chat template with thinking OFF (the capture's own toggle), then an
     assistant prefix ending right before the option token:
         mmlu_pro                  "Answer:"    -> " A" ... " J"
         bbh, lettered options     "Answer: ("  -> "A", "B", ...
         bbh, yes/no, true/false,  "Answer:"    -> " Yes"/" No", " True"/...,
             valid/invalid                          " valid"/" invalid"
     These match how the thinking-off generations actually wrote their final
     line (mmlu_pro mostly "Answer: B", BBH mostly "Answer: (B)").

For "Answer:" items the model may open a parenthesis first ("Answer: (Yes)";
the BBH v3 prompt asks for "Answer: (A)"), and on BBH yes/no-style items
readout v1 found almost all the next-token mass on " (". So (v2) each such
option is read along both formats and marginalised exactly:

    P(option) = P(" X" | "Answer:") + P(" (" | "Answer:") * P("X" | "Answer: (")

which costs a second forward pass over the same prompt plus " (".
`mass_direct`, `mass_paren` and `p_bridge` record the split.

The next-token distribution at the last prompt position is restricted to the
option tokens and renormalised. Each option's probability is the sum over its
surface forms (leading space or not; for words also lower/capitalised). A form
counts only if, appended to the actual prompt, it tokenizes as exactly the
prompt's tokens plus ONE new token -- the check is made per row, in context.
For "Answer:" + "A" it fails (Qwen merges ":A"), so letters there are read
only as " A". A secondary form is kept only if it passes for EVERY option of
the item, so no option gets an extra form the others lack; the primary form
(leading space after ":", none after "(") must pass for every option or the
row is flagged and left unscored. `option_mass` records how much of the full
next-token distribution the options held before renormalising.

BBH free-form subtasks (object_counting, multistep_arithmetic_two,
word_sorting, dyck_languages) have no closed option set and are excluded;
they are still written, with `covered: false`, so coverage is countable.

Output (a capture-style dir; summary.json is written last -- use it as the
cell's output_check):

    <out-dir>/
      config.json        args, model, instruction, prefixes, git commit, versions
      readout.jsonl      one row per capture row, same order
      decide_hidden.npz  hidden state at the decide position (the last prompt
                         token, the one whose logits are read) for the covered
                         rows: `hidden` (n, n_layers, d) fp16, `layers` (HF
                         hidden_states indices, 0 = embeddings; the last one is
                         after the final norm), `sample_id`
      summary.json       counts, coverage, accuracy, flags

Usage (GPU node; Qwen3-8B needs ~17 GB):
    .venv/bin/python scripts/capture_readout.py \\
        --capture-dir shared/icr_capture/mmlu_pro_thinking_qwen3v3 \\
        --out-dir shared/icr_capture/mmlu_pro_readout_qwen3v3

    # CPU, tokenizer only: build every prompt and run the option-token checks
    .venv/bin/python scripts/capture_readout.py --check-only \\
        --capture-dir shared/icr_capture/bbh_thinking_qwen3v3 --out-dir /tmp/x
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402

import scripts.capture_inference_thinking as cap  # noqa: E402
from scripts.rescore_confidence import match_prompt, prompt_candidates  # noqa: E402
from utils.capture_io import load_config, load_meta  # noqa: E402

logger = logging.getLogger("capture_readout")

READOUT_VERSION = 2

READOUT_INSTRUCTION = (
    "\n\nDo not explain and do not show any working. "
    "Reply with the final answer line only.")

# Answer kinds this readout can score, with their assistant prefix and option
# labels. Labels are the canonical spelling of each option (what the gold uses).
PREFIX_COLON = "Answer:"
PREFIX_PAREN = "Answer: ("
WORD_OPTIONS = {
    "yes_no": ["Yes", "No"],
    "true_false": ["True", "False"],
    "valid_invalid": ["valid", "invalid"],
}
EXCLUDED_KINDS = ("number", "dyck", "word_list")

_MMLU_OPTION = re.compile(r"^\(([A-J])\) ", re.MULTILINE)
_BBH_OPTION = re.compile(r"^\s*\(([A-Z])\)", re.MULTILINE)


# ---------------------------------------------------------------------------
# Answer specification per item (pure; tested without a model)
# ---------------------------------------------------------------------------

def bbh_subtask(row: dict) -> str | None:
    """Subtask from the row ('subtask' field, else the 'bbh-<name>-<idx>' id)."""
    if row.get("subtask"):
        return row["subtask"]
    sid = str(row.get("sample_id", ""))
    if sid.startswith("bbh-") and "-" in sid[4:]:
        return sid[4:].rsplit("-", 1)[0]
    return None


def answer_spec(task: str, row: dict) -> dict:
    """What to read for one item.

    Returns {covered, kind, family, prefix, options, gold, exclude_reason}.
    `options` are canonical labels in display order; `gold` is one of them.
    """
    q = row["question"]
    if task == "mmlu_pro":
        letters = _MMLU_OPTION.findall(q)
        spec = {"kind": "letter", "family": row.get("category"),
                "prefix": PREFIX_COLON, "options": letters,
                "gold": str(row["answer"]).strip().upper()}
    elif task == "bbh":
        from tasks.bbh import SUBTASK_KIND, infer_answer_kind

        sub = bbh_subtask(row)
        kind = SUBTASK_KIND.get(sub) if sub else infer_answer_kind(q)
        if kind in EXCLUDED_KINDS or kind is None:
            return {"covered": False, "kind": kind, "family": sub,
                    "exclude_reason": f"free-form answer ({kind})"}
        gold_raw = str(row["answer"]).strip()
        if kind == "letter":
            m = re.fullmatch(r"\(([A-Z])\)", gold_raw)
            spec = {"kind": kind, "family": sub, "prefix": PREFIX_PAREN,
                    "options": _BBH_OPTION.findall(q),
                    "gold": m.group(1) if m else gold_raw}
        else:
            opts = WORD_OPTIONS[kind]
            gold = next((o for o in opts if o.lower() == gold_raw.lower()), gold_raw)
            spec = {"kind": kind, "family": sub, "prefix": PREFIX_COLON,
                    "options": list(opts), "gold": gold}
    else:
        raise ValueError(f"no readout spec for task {task!r} "
                         "(mmlu_pro and bbh only; gsm8k/math500 need a verify format)")

    # "Answer:" items are also read through "Answer: (": the model may open a
    # parenthesis first ("(Yes)", "(C)"), and the BBH v3 prompt asks for one.
    spec["alt_prefix"] = PREFIX_PAREN if spec["prefix"] == PREFIX_COLON else None
    opts = spec["options"]
    if len(opts) < 2 or len(set(opts)) != len(opts):
        return {**spec, "covered": False,
                "exclude_reason": f"could not parse a clean option list: {opts}"}
    if spec["gold"] not in opts:
        return {**spec, "covered": False,
                "exclude_reason": f"gold {spec['gold']!r} not among options {opts}"}
    return {**spec, "covered": True, "exclude_reason": None}


def surface_forms(label: str, prefix: str) -> list[str]:
    """Surface forms of one option, primary first.

    Primary: " X" after a colon, "X" after an open paren. Secondary: the other
    spacing and, for words, the lower-case / capitalised spellings.
    """
    spellings = [label]
    if len(label) > 1:
        for s in (label.lower(), label.capitalize()):
            if s not in spellings:
                spellings.append(s)
    lead = " " if prefix.endswith(":") else ""
    other = "" if lead else " "
    forms = [lead + label]
    for s in spellings:
        for sp in (lead, other):
            f = sp + s
            if f not in forms:
                forms.append(f)
    return forms


def single_token_in_context(tokenizer: Any, context_ids: list[int], context: str,
                            form: str) -> int | None:
    """Token id if `context + form` tokenizes as context_ids + [one token]."""
    ids = tokenizer(context + form, add_special_tokens=False).input_ids
    if len(ids) == len(context_ids) + 1 and ids[:len(context_ids)] == context_ids:
        return int(ids[-1])
    return None


def option_token_map(tokenizer: Any, context: str, options: list[str],
                     prefix: str, context_ids: list[int] | None = None) -> dict:
    """Map each option to the token ids of its surface forms, checked in context.

    Returns {"token_ids": {label: [ids]}, "forms": {label: [surface forms]},
    "ok": bool, "failed": [labels whose primary form is not one token]}.
    A secondary form (by position in surface_forms) is kept only if it passes
    for every option and its tokens are unclaimed by any other option.
    """
    if context_ids is None:
        context_ids = tokenizer(context, add_special_tokens=False).input_ids
    per_opt = {o: surface_forms(o, prefix) for o in options}
    checked = {o: [single_token_in_context(tokenizer, context_ids, context, f)
                   for f in forms] for o, forms in per_opt.items()}
    failed = [o for o in options if checked[o][0] is None]
    if failed:
        return {"token_ids": {}, "forms": {}, "ok": False, "failed": failed}

    token_ids = {o: [checked[o][0]] for o in options}
    primaries = [checked[o][0] for o in options]
    if len(set(primaries)) != len(primaries):
        return {"token_ids": {}, "forms": {}, "ok": False,
                "failed": ["primary tokens collide"]}
    forms = {o: [per_opt[o][0]] for o in options}
    n_forms = min(len(v) for v in per_opt.values())
    for k in range(1, n_forms):
        ids = [checked[o][k] for o in options]
        if any(i is None for i in ids):
            continue
        claimed = {t for v in token_ids.values() for t in v}
        new = {o: i for o, i in zip(options, ids) if i not in token_ids[o]}
        if any(i in claimed for i in new.values()) or len(set(new.values())) != len(new):
            continue
        for o, i in new.items():
            token_ids[o].append(i)
            forms[o].append(per_opt[o][k])
    return {"token_ids": token_ids, "forms": forms, "ok": True, "failed": []}


def build_readout_prompt(raw_prompt: str, render_off, prefix: str) -> str:
    """Thinking-off chat prompt + answer-only instruction + assistant prefix."""
    return render_off(raw_prompt + READOUT_INSTRUCTION) + prefix


def alt_path(tokenizer: Any, prompt: str, raw_prompt: str, render_off,
             spec: dict) -> dict | None:
    """The parenthesised reading of an "Answer:" item, or None if unusable.

    Requires the alt prompt to be exactly the primary prompt's tokens plus ONE
    bridge token (" ("), so P(bridge) can be read at the decide position, and
    every option to be one token after it.
    """
    alt_prompt = build_readout_prompt(raw_prompt, render_off, spec["alt_prefix"])
    bridge = spec["alt_prefix"][len(spec["prefix"]):]
    if prompt + bridge != alt_prompt:
        return None
    ctx = tokenizer(prompt, add_special_tokens=False).input_ids
    bridge_id = single_token_in_context(tokenizer, ctx, prompt, bridge)
    if bridge_id is None:
        return None
    tmap = option_token_map(tokenizer, alt_prompt, spec["options"], spec["alt_prefix"],
                            context_ids=ctx + [bridge_id])
    if not tmap["ok"]:
        return None
    return {"prompt": alt_prompt, "bridge_id": bridge_id, "tmap": tmap}


def option_probs(logprobs: np.ndarray, token_ids: dict[str, list[int]],
                 options: list[str]) -> np.ndarray:
    """Unnormalised probability of each option (sum over its surface forms)."""
    return np.array([float(np.exp(logprobs[token_ids[o]]).sum()) for o in options])


def score_options(logprobs: np.ndarray, token_ids: dict[str, list[int]],
                  options: list[str], alt: tuple | None = None) -> dict:
    """Renormalised option distribution from one next-token log-prob vector.

    `alt` = (bridge_logprob, alt_logprobs, alt_token_ids): the same options
    read after a bridge token (" (" after "Answer:"). Each option's probability
    is then the exact marginal over the two formats,

        P(" X" | "Answer:") + P(" (" | "Answer:") * P("X" | "Answer: ("),

    two disjoint continuations, so nothing is counted twice.
    """
    direct = option_probs(logprobs, token_ids, options)
    raw, extra = direct, {"mass_direct": float(direct.sum())}
    if alt is not None:
        bridge_lp, alt_lp, alt_ids = alt
        after = option_probs(alt_lp, alt_ids, options)
        via = float(np.exp(bridge_lp)) * after
        raw = direct + via
        extra.update(mass_paren=float(via.sum()), p_bridge=float(np.exp(bridge_lp)),
                     mass_after_paren=float(after.sum()))
    mass = float(raw.sum())
    probs = raw / mass if mass > 0 else np.full(len(options), 1.0 / len(options))
    k = int(np.argmax(probs))
    ent = float(-(probs * np.log(np.clip(probs, 1e-30, None))).sum())
    return {"probs": [float(p) for p in probs], "argmax": options[k],
            "p_max": float(probs[k]), "entropy": ent,
            "entropy_norm": ent / math.log(len(options)), "option_mass": mass,
            **extra}


# ---------------------------------------------------------------------------
# Model pass
# ---------------------------------------------------------------------------

def default_layers(num_layers: int) -> list[int]:
    """hidden_states indices: quarter, middle, three-quarter, and the last 4."""
    picks = {num_layers // 4, num_layers // 2, (3 * num_layers) // 4}
    picks.update(range(max(1, num_layers - 3), num_layers + 1))
    return sorted(picks)


def decide_pass(model, tokenizer, prompts: list[str], layers: list[int]):
    """One left-padded forward pass. Returns (logprobs (B, V) float32 numpy,
    hidden (B, len(layers), d) float16 numpy) at each row's last prompt token.

    Position ids are passed explicitly from the attention mask, so a padded
    row is computed exactly as it would be alone (a bare forward() would
    number positions from the first pad).
    """
    import torch

    enc = tokenizer(prompts, padding=True, return_tensors="pt",
                    add_special_tokens=False).to(model.device)
    mask = enc.attention_mask
    pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
    kwargs = dict(input_ids=enc.input_ids, attention_mask=mask, position_ids=pos,
                  output_hidden_states=True, use_cache=False)
    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        kwargs["logits_to_keep"] = 1
    with torch.no_grad():
        out = model(**kwargs)
    logits = out.logits[:, -1, :].float()
    logprobs = torch.log_softmax(logits, dim=-1).cpu().numpy()
    hidden = torch.stack([out.hidden_states[i][:, -1, :] for i in layers], dim=1)
    return logprobs, hidden.to(torch.float16).cpu().numpy()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _family_lookup(task: str, task_module) -> dict[str, str]:
    """sample_id -> category for mmlu_pro (the meta rows do not carry it).
    Best effort: an unavailable dataset leaves families empty, never fails."""
    if task != "mmlu_pro":
        return {}
    try:
        return {r["key"]: r["category"] for r in task_module.load_mmlu_pro("test")}
    except Exception as exc:  # noqa: BLE001
        logger.warning("mmlu_pro categories unavailable (%s); family left empty", exc)
        return {}


def run(args: argparse.Namespace) -> int:
    capture_dir, out_dir = Path(args.capture_dir), Path(args.out_dir)
    cfg = load_config(capture_dir)
    task = args.task or cfg.get("task")
    model_name = args.model or cfg.get("model_name")
    if not task or not model_name:
        raise SystemExit(f"{capture_dir}: config.json lacks task/model_name; pass "
                         "--task and --model")
    if not cfg.get("chat_template", True):
        raise SystemExit("capture was not chat-templated; the readout assumes it was")
    task_module = importlib.import_module(f"tasks.{task}")
    meta = load_meta(capture_dir)
    if args.max_rows:
        meta = meta[: args.max_rows]

    if args.check_only:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = None
    else:
        tokenizer, model = cap.load_model(model_name, args.attn_implementation)
    toggle = cap.detect_thinking_toggle(tokenizer, model_name)
    cap.verify_toggle_changes_prompt(tokenizer, model_name, toggle)

    def render_off(raw: str) -> str:
        return toggle.render(tokenizer, raw, False)

    candidates = prompt_candidates(task_module)
    families = _family_lookup(task, task_module)
    prov = cap.git_provenance()

    # Build every prompt and token map first (CPU), then run the model.
    items, flags, sources = [], Counter(), Counter()
    for row in meta:
        row = dict(row)
        if families and not row.get("category"):
            row["category"] = families.get(row["sample_id"])
        spec = answer_spec(task, row)
        rec: dict[str, Any] = {
            "sample_id": row["sample_id"], "task": task, "family": spec.get("family"),
            "kind": spec.get("kind"), "covered": spec["covered"],
            "exclude_reason": spec.get("exclude_reason"),
            "options": spec.get("options"), "gold": spec.get("gold"),
            "n_options": len(spec.get("options") or []) or None, "flags": [],
        }
        prompt = tmap = alt = None
        if spec["covered"]:
            name, raw = match_prompt(row, candidates, render_off)
            sources[name] += 1
            rec["prompt_source"] = name
            if name is None:
                rec["covered"], rec["exclude_reason"] = False, "prompt_hash_mismatch"
                rec["flags"].append("prompt_hash_mismatch")
            else:
                limit = cfg.get("max_prompt_len")
                if limit and (row.get("prompt_truncated")
                              or (row.get("prompt_len_off") or 0) >= int(limit)):
                    # The generations saw a cut prompt; the readout does not.
                    rec["flags"].append("capture_prompt_truncated")
                prompt = build_readout_prompt(raw, render_off, spec["prefix"])
                tmap = option_token_map(tokenizer, prompt, spec["options"],
                                        spec["prefix"])
                rec["option_forms"] = tmap["forms"]
                rec["option_token_ids"] = tmap["token_ids"]
                rec["prompt_len"] = len(tokenizer(prompt, add_special_tokens=False)
                                        .input_ids)
                if not tmap["ok"]:
                    rec["covered"] = False
                    rec["exclude_reason"] = f"option not one token: {tmap['failed']}"
                    rec["flags"].append("option_not_single_token")
                elif spec.get("alt_prefix"):
                    alt = alt_path(tokenizer, prompt, raw, render_off, spec)
                    if alt is None:
                        rec["flags"].append("alt_path_unavailable")
                    else:
                        rec["alt_option_forms"] = alt["tmap"]["forms"]
        flags.update(rec["flags"])
        items.append((rec, prompt, tmap, alt))

    covered = [i for i, (rec, *_) in enumerate(items) if rec["covered"]]
    logger.info("%s: %d rows, %d covered, prompt sources %s, flags %s", task,
                len(items), len(covered), dict(sources), dict(flags))
    if args.check_only:
        forms = Counter(json.dumps(r["option_forms"]) for r, *_ in items
                        if r.get("option_forms") and r["kind"] != "letter")
        forms.update(json.dumps({r["options"][0]: r["option_forms"][r["options"][0]]})
                     for r, *_ in items
                     if r.get("option_forms") and r["kind"] == "letter")
        print(json.dumps({"task": task, "n_rows": len(items), "n_covered": len(covered),
                          "prompt_source": {str(k): v for k, v in sources.items()},
                          "flags": dict(flags),
                          "exclude_reasons": dict(Counter(r["exclude_reason"] for r, *_
                                                          in items if not r["covered"])),
                          "option_forms": dict(forms)},
                         indent=2))
        return 0 if not flags.get("option_not_single_token") else 1

    import torch
    import transformers

    num_layers = int(getattr(model.config, "num_hidden_layers"))
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else default_layers(num_layers))
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "readout_version": READOUT_VERSION, "design": "D0 (LLM2Jev-style, training-free)",
        "capture_dir": str(capture_dir), "task": task, "model_name": model_name,
        "thinking": "off", "thinking_toggle": toggle.describe(),
        "readout_instruction": READOUT_INSTRUCTION,
        "prefixes": {"letter_mmlu_pro": PREFIX_COLON, "letter_bbh": PREFIX_PAREN,
                     "words": PREFIX_COLON},
        "alt_prefix": {"for": PREFIX_COLON, "via": PREFIX_PAREN,
                       "rule": "P(' X'|'Answer:') + P(' ('|'Answer:') * P('X'|'Answer: (')"},
        "hidden_layers": layers, "num_layers": num_layers,
        "batch_size": args.batch_size, "attn_implementation": args.attn_implementation,
        "dtype": str(next(model.parameters()).dtype),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "argv": list(sys.argv), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **prov,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    hidden_rows: dict[int, np.ndarray] = {}
    started = time.monotonic()
    for b in range(0, len(covered), args.batch_size):
        idx = covered[b: b + args.batch_size]
        logprobs, hidden = decide_pass(model, tokenizer, [items[i][1] for i in idx],
                                       layers)
        with_alt = [i for i in idx if items[i][3] is not None]
        alt_lp = {}
        if with_alt:
            lp2, _ = decide_pass(model, tokenizer,
                                 [items[i][3]["prompt"] for i in with_alt], layers[-1:])
            alt_lp = {i: lp2[k] for k, i in enumerate(with_alt)}
        for j, i in enumerate(idx):
            rec, _, tmap, alt = items[i]
            alt_args = None
            if alt is not None:
                alt_args = (logprobs[j][alt["bridge_id"]], alt_lp[i],
                            alt["tmap"]["token_ids"])
            s = score_options(logprobs[j], tmap["token_ids"], rec["options"], alt_args)
            top = int(np.argmax(logprobs[j]))
            rec.update(s)
            rec["s1_correct"] = s["argmax"] == rec["gold"]
            rec["top_token"] = tokenizer.decode([top])
            rec["top_token_prob"] = float(np.exp(logprobs[j][top]))
            hidden_rows[i] = hidden[j]
        if (b // args.batch_size) % 20 == 0:
            logger.info("%d/%d covered rows (%.1f rows/s)", b + len(idx), len(covered),
                        (b + len(idx)) / max(time.monotonic() - started, 1e-6))

    tmp = out_dir / "readout.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for rec, *_ in items:
            rec["git_commit"] = prov.get("git_commit")
            fh.write(json.dumps(rec) + "\n")
    tmp.replace(out_dir / "readout.jsonl")
    if covered:
        np.savez_compressed(
            out_dir / "decide_hidden.npz",
            hidden=np.stack([hidden_rows[i] for i in covered]).astype(np.float16),
            layers=np.array(layers), sample_id=np.array([items[i][0]["sample_id"]
                                                         for i in covered]))

    done = [items[i][0] for i in covered]
    by_family: dict[str, list] = {}
    for r in done:
        by_family.setdefault(str(r["family"]), []).append(r["s1_correct"])
    summary = {
        "task": task, "n_rows": len(items), "n_covered": len(done),
        "coverage": len(done) / len(items) if items else float("nan"),
        "s1_accuracy": float(np.mean([r["s1_correct"] for r in done])) if done else None,
        "mean_p_max": float(np.mean([r["p_max"] for r in done])) if done else None,
        "mean_option_mass": float(np.mean([r["option_mass"] for r in done])) if done else None,
        "min_option_mass": float(np.min([r["option_mass"] for r in done])) if done else None,
        "s1_accuracy_by_family": {k: [float(np.mean(v)), len(v)]
                                  for k, v in sorted(by_family.items())},
        "exclude_reasons": dict(Counter(r["exclude_reason"] for r, *_ in items
                                        if not r["covered"])),
        "prompt_source": {str(k): v for k, v in sources.items()},
        "flags": dict(flags), "seconds": round(time.monotonic() - started, 1),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **prov,
    }
    # Written last: its presence means readout.jsonl and the NPZ are complete.
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("S1 accuracy %.3f on %d covered rows (coverage %.1f%%); wrote %s",
                summary["s1_accuracy"] or float("nan"), len(done),
                100 * summary["coverage"], out_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--task", default=None, help="default: the capture config's task")
    p.add_argument("--model", default=None, help="default: the capture config's model")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--layers", default=None,
                   help="comma-separated hidden_states indices to save; default "
                        "quarter, middle, three-quarter and the last 4")
    p.add_argument("--max-rows", type=int, default=0, help="0 = all (smoke tests)")
    p.add_argument("--attn-implementation", default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--check-only", action="store_true",
                   help="tokenizer only: build prompts, run the option-token "
                        "checks, print a report, write nothing")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

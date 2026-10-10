#!/usr/bin/env python3
"""
rescore_confidence.py — recompute thinking-off confidence from stored responses.

Why: captures written before the B1 fix (rows without `confidence_version`)
scored each thinking-off answer from a left-padded batch with no attention
mask, so the model attended to its pad tokens and `confidence_off` tracked the
pad count (Spearman -0.72..-0.88 on the Qwen3 v3 captures). The responses
themselves are fine. This script re-scores them offline, one row at a time and
with no padding at all, using the capture's own scorer
(`capture_inference_thinking.sequence_confidence`) and its own prompt
construction (thinking toggle, chat template, prompt fitting). No generation.

For each `meta.shardNN.jsonl` in the capture dir it writes a SIDECAR

    confidence_off_v2.shardNN.jsonl   one row per meta row, same order:
        sample_id, confidence_off {mean_logprob, min_logprob, mean_entropy},
        confidence_version: 2, integrity fields and `flags`

and, last, `confidence_off_v2.summary.json` (use it as the cell's
output_check). The meta shards are never rewritten. Readers pick the sidecar
up through `utils.capture_io.resolve_confidence` (baseline_confidence.py's
`--confidence-source`, default auto = sidecar when present).

Reconstruction, per row, and how it is checked:

  * Prompt. Candidates are the task module's `format_prompt` and any legacy
    `PROMPT_TEMPLATE_V<k>` it keeps (BBH's pre-2026-10 prompt, which every v3
    BBH capture used). Each is rendered with the thinking-OFF toggle and the
    one whose sha256 equals the row's `prompt_hash` is used. No match: the
    row is kept, its confidence is null and it is flagged
    `prompt_hash_mismatch` -- scoring a prompt the model never saw would be a
    number about nothing. Readers fail loudly on a null.
  * Prompt length. Rows written before the B4 fix (no `prompt_truncated`
    field) were right-truncated by the tokenizer at max_prompt_len; newer rows
    went through `fit_prompt`. Whichever applies is reproduced. A token count
    different from the stored `prompt_len_off` is flagged
    `prompt_len_mismatch`.
  * Answer tokens. The stored response was decoded with
    skip_special_tokens=True, so the terminating EOS is gone from the text but
    counted in `n_tokens_off`. It is re-appended (tokenizer.eos_token_id) when
    the row was not truncated. `token_delta` = re-tokenized length minus
    `n_tokens_off` is recorded on every row; |delta| > 2 is flagged
    `token_count_mismatch` (the model emitted a non-canonical tokenization the
    text cannot reproduce). Such rows are still scored -- the delta says by
    how much the scored sequence differs from the generated one.

Usage (GPU node; Qwen3-8B fits one GPU):
    .venv/bin/python scripts/rescore_confidence.py \\
        --capture-dir shared/icr_capture/math500_thinking_qwen3v3

    # CPU, no model: run every integrity check and report, write nothing
    .venv/bin/python scripts/rescore_confidence.py --check-only \\
        --capture-dir shared/icr_capture/bbh_thinking_qwen3v3
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import scripts.capture_inference_thinking as cap  # noqa: E402
from utils.capture_io import (CONFIDENCE_SIDECAR_PREFIX,  # noqa: E402
                              shard_paths)

logger = logging.getLogger("rescore_confidence")

RESCORE_VERSION = cap.CONFIDENCE_VERSION          # 2
TOKEN_DELTA_FLAG = 2                              # |delta| above this is flagged
_LEGACY_TEMPLATE = re.compile(r"PROMPT_TEMPLATE_V\d+")
_SHARD_NUM = re.compile(r"\.shard(\d+)\.")


# ---------------------------------------------------------------------------
# Reconstruction (pure functions; tested without a model)
# ---------------------------------------------------------------------------

def prompt_candidates(task_module: Any) -> list[tuple[str, Callable[[str], str]]]:
    """(name, raw-prompt builder) pairs: the current format_prompt first, then
    every legacy PROMPT_TEMPLATE_V<k> the module keeps, oldest first."""
    cands: list[tuple[str, Callable[[str], str]]] = [
        ("format_prompt", lambda q: cap._prompt_default(task_module, {"question": q}))]
    for name in sorted(n for n in dir(task_module) if _LEGACY_TEMPLATE.fullmatch(n)):
        template = getattr(task_module, name)
        cands.append((name, lambda q, t=template: t.format(question=q.strip())))
    return cands


def match_prompt(row: dict, candidates, render_off) -> tuple[str | None, str | None]:
    """(candidate name, raw prompt) whose rendered thinking-off prompt hashes to
    the row's prompt_hash, or (None, None)."""
    for name, build in candidates:
        raw = build(row["question"])
        if cap.sha256(render_off(raw)) == row.get("prompt_hash"):
            return name, raw
    return None, None


def prompt_ids(raw: str, row: dict, render_off, tokenizer, max_prompt_len: int,
               add_special: bool) -> list[int]:
    """The prompt token ids the capture fed the model for this row."""
    if "prompt_truncated" in row:
        # B4-fixed capture: fit_prompt trims the user content from the left.
        fitted, _ = cap.fit_prompt(raw, render_off, tokenizer, max_prompt_len,
                                   add_special)
        return tokenizer(fitted, add_special_tokens=add_special).input_ids
    # Pre-B4 capture: the tokenizer right-truncated the rendered prompt.
    return tokenizer(render_off(raw), truncation=True, max_length=max_prompt_len,
                     add_special_tokens=add_special).input_ids


def answer_ids(row: dict, tokenizer) -> tuple[list[int], bool]:
    """Thinking-off answer token ids as generated, and whether EOS was appended.

    generate_batch decoded with skip_special_tokens=True and counted the EOS
    in n_tokens_off, so a row that stopped on EOS gets it back."""
    ids = tokenizer(row["response_off"] or "", add_special_tokens=False).input_ids
    eos = not row.get("truncated_off", False)
    if eos:
        ids = ids + [tokenizer.eos_token_id]
    return ids, eos


def rebuild_row(row: dict, *, candidates, render_off, tokenizer,
                max_prompt_len: int, add_special: bool) -> dict:
    """Everything about a row except the model's scores: the ids to score and
    the integrity record. `ids` is None when the prompt cannot be rebuilt."""
    rec: dict[str, Any] = {"sample_id": row["sample_id"], "flags": []}
    name, raw = match_prompt(row, candidates, render_off)
    rec["prompt_source"] = name
    rec["prompt_hash_match"] = name is not None
    if name is None:
        rec["flags"].append("prompt_hash_mismatch")
        rec["ids"] = None
        return rec
    p_ids = prompt_ids(raw, row, render_off, tokenizer, max_prompt_len, add_special)
    a_ids, eos = answer_ids(row, tokenizer)
    rec["prompt_len_rescored"] = len(p_ids)
    rec["prompt_len_off"] = row.get("prompt_len_off")
    if row.get("prompt_len_off") is not None and len(p_ids) != row["prompt_len_off"]:
        rec["flags"].append("prompt_len_mismatch")
    rec["n_tokens_rescored"] = len(a_ids)
    rec["n_tokens_off"] = row.get("n_tokens_off")
    rec["eos_appended"] = eos
    delta = (len(a_ids) - row["n_tokens_off"]
             if row.get("n_tokens_off") is not None else None)
    rec["token_delta"] = delta
    if delta is not None and abs(delta) > TOKEN_DELTA_FLAG:
        rec["flags"].append("token_count_mismatch")
    rec["ids"] = (p_ids, a_ids)
    return rec


# ---------------------------------------------------------------------------
# Capture config
# ---------------------------------------------------------------------------

def shard_config(capture_dir: Path, shard_num: int) -> dict:
    """The args that produced this shard: config.shardNN.json when the capture
    wrote one (B5), else the legacy config.json."""
    per = capture_dir / f"config.shard{shard_num:02d}.json"
    path = per if per.exists() else capture_dir / "config.json"
    if not path.exists():
        raise SystemExit(f"{capture_dir}: no config.shard{shard_num:02d}.json or "
                         "config.json -- cannot tell how prompts were built")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["_path"] = path.name
    return cfg


def _task_module(task: str):
    if task not in cap._TASK_REGISTRY:
        raise SystemExit(f"unknown task {task!r}")
    return importlib.import_module(f"tasks.{task}")


def _renderer(tokenizer, model_name: str, chat_template: bool):
    if not chat_template:
        return lambda raw: raw, None
    toggle = cap.detect_thinking_toggle(tokenizer, model_name)
    return (lambda raw: toggle.render(tokenizer, raw, False)), toggle


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_ids(model, p_ids: list[int], a_ids: list[int]) -> dict:
    """One unpadded row through the capture's own scorer."""
    import torch

    seq = torch.tensor([p_ids + a_ids], dtype=torch.long, device=model.device)
    return cap.sequence_confidence(model, seq, len(p_ids), [len(a_ids)])[0]


def _null_conf() -> dict:
    return {"mean_logprob": None, "min_logprob": None, "mean_entropy": None}


def rescore_shard(meta_path: Path, out_path: Path, *, tokenizer, model,
                  cfg: dict, task_module, log_every: int = 100) -> dict:
    """Rescore one meta shard into its sidecar. Returns the shard summary."""
    render_off, toggle = _renderer(tokenizer, cfg["model_name"],
                                   bool(cfg.get("chat_template")))
    candidates = prompt_candidates(task_module)
    add_special = not cfg.get("chat_template")
    max_prompt_len = int(cfg.get("max_prompt_len") or 1024)

    with open(meta_path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    out_rows, flags, sources, deltas = [], Counter(), Counter(), Counter()
    started = time.monotonic()
    for k, row in enumerate(rows):
        rec = rebuild_row(row, candidates=candidates, render_off=render_off,
                          tokenizer=tokenizer, max_prompt_len=max_prompt_len,
                          add_special=add_special)
        ids = rec.pop("ids")
        if model is None:
            conf = None
        elif ids is None:
            conf = _null_conf()
        else:
            conf = score_ids(model, *ids)
        out = {"sample_id": rec["sample_id"], "confidence_off": conf,
               "confidence_version": RESCORE_VERSION,
               **{k2: v for k2, v in rec.items() if k2 != "sample_id"}}
        out_rows.append(out)
        flags.update(rec["flags"])
        sources[rec["prompt_source"]] += 1
        if rec.get("token_delta") is not None:
            deltas[rec["token_delta"]] += 1
        if log_every and (k + 1) % log_every == 0:
            rate = (k + 1) / max(time.monotonic() - started, 1e-6)
            logger.info("%s: %d/%d rows (%.1f rows/s)", meta_path.name, k + 1,
                        len(rows), rate)

    if model is not None:
        tmp = out_path.with_name(out_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for out in out_rows:
                fh.write(json.dumps(out) + "\n")
        os.replace(tmp, out_path)            # never leave a half-written sidecar

    summary = {
        "meta": meta_path.name,
        "sidecar": out_path.name if model is not None else None,
        "n_rows": len(rows),
        "n_scored": sum(1 for o in out_rows
                        if o["confidence_off"] and o["confidence_off"]["mean_logprob"] is not None),
        "config_file": cfg["_path"],
        "thinking_toggle": toggle.describe() if toggle is not None else None,
        "prompt_source": dict(sources),
        "flags": dict(flags),
        "token_delta": {str(k): v for k, v in sorted(deltas.items())},
        "seconds": round(time.monotonic() - started, 1),
    }
    logger.info("%s: %d rows, prompt source %s, flags %s, token delta %s",
                meta_path.name, len(rows), dict(sources), dict(flags),
                summary["token_delta"])
    return summary


def sidecar_path(meta_path: Path) -> Path:
    m = _SHARD_NUM.search(meta_path.name)
    return meta_path.with_name(f"{CONFIDENCE_SIDECAR_PREFIX}.shard{m.group(1)}.jsonl")


def _complete(sidecar: Path, meta_path: Path) -> bool:
    if not sidecar.exists():
        return False
    def n(p):
        with open(p, encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    return n(sidecar) == n(meta_path)


def run(args: argparse.Namespace) -> int:
    capture_dir = Path(args.capture_dir)
    meta_paths = shard_paths(capture_dir, "meta.shard*.jsonl")
    if not meta_paths:
        raise SystemExit(f"no meta.shard*.jsonl under {capture_dir}")
    quarantined = shard_paths(capture_dir, "meta.shard*.jsonl.quarantined")
    if quarantined:
        logger.warning("%d quarantined shard(s) are not rescored: %s",
                       len(quarantined), [p.name for p in quarantined])

    cfgs = {p: shard_config(capture_dir, int(_SHARD_NUM.search(p.name).group(1)))
            for p in meta_paths}
    models = {c["model_name"] for c in cfgs.values()}
    tasks = {c["task"] for c in cfgs.values()}
    if len(models) != 1 or len(tasks) != 1:
        raise SystemExit(f"shards disagree on model/task: {models} / {tasks}")
    model_name = args.model or models.pop()
    task_module = _task_module(tasks.pop())

    if args.check_only:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = None
    else:
        tokenizer, model = cap.load_model(model_name, args.attn_implementation)

    shards = []
    for meta_path in meta_paths:
        out_path = sidecar_path(meta_path)
        if (model is not None and not args.overwrite
                and _complete(out_path, meta_path)):
            logger.info("%s exists and is complete; skipping (--overwrite to redo)",
                        out_path.name)
            shards.append({"meta": meta_path.name, "sidecar": out_path.name,
                           "skipped_existing": True})
            continue
        cfg = dict(cfgs[meta_path], model_name=model_name)
        shards.append(rescore_shard(meta_path, out_path, tokenizer=tokenizer,
                                    model=model, cfg=cfg, task_module=task_module,
                                    log_every=args.log_every))

    totals = Counter()
    for s in shards:
        totals.update(s.get("flags", {}))
    report = {
        "capture_dir": str(capture_dir),
        "model": model_name,
        "confidence_version": RESCORE_VERSION,
        "check_only": bool(args.check_only),
        "token_delta_flag_threshold": TOKEN_DELTA_FLAG,
        "eos_token_id": tokenizer.eos_token_id,
        "attn_implementation": args.attn_implementation,
        "shards": shards,
        "flags_total": dict(totals),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **cap.git_provenance(),
    }
    if model is not None:
        import torch
        import transformers
        report.update(torch_version=torch.__version__,
                      transformers_version=transformers.__version__,
                      dtype=str(next(model.parameters()).dtype))
        # Written last: its presence means every shard's sidecar is complete.
        path = capture_dir / f"{CONFIDENCE_SIDECAR_PREFIX}.summary.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logger.info("wrote %s", path)
    else:
        print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture-dir", required=True)
    p.add_argument("--model", default=None,
                   help="Override the model named in the capture config "
                        "(e.g. a local path to the same weights).")
    p.add_argument("--attn-implementation", default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--check-only", action="store_true",
                   help="Tokenizer only (CPU): rebuild every row, report the "
                        "integrity checks, write nothing.")
    p.add_argument("--overwrite", action="store_true",
                   help="Rescore shards whose sidecar is already complete.")
    p.add_argument("--log-every", type=int, default=100)
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

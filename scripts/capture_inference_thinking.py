#!/usr/bin/env python3
"""
capture_inference_thinking.py — paired thinking-off / thinking-on capture.

For each query this records:
  1. the **prefill hidden state** — the last-prompt-token activation at every
     layer, taken before a single token is generated. This is the probe's only
     input, so it must come from a clean forward pass over the prompt.
  2. the model's answer with thinking mode **off** and with thinking mode
     **on**, and whether each is correct. Those two bits become the label.

Usage:
    python scripts/capture_inference_thinking.py \\
        --task gsm8k --model Qwen/Qwen3-8B \\
        --out-dir shared/icr_capture/gsm8k_thinking_qwen3 \\
        --max-samples 500 --chat-template \\
        --max-response-len 2048 --max-response-len-thinking 8192

--max-response-len is required: its old default (320) silently truncated most
thinking-OFF answers on MATH-500 and invalidated a round of results.

Sharding, for fanning a capture across nodes via scripts/dispatch/:
    --shard-index 0 --shard-count 4     # takes samples 0, 4, 8, ... 

Each shard writes its own files, so shards never collide and a failed shard is
re-runnable on its own:

    <out-dir>/
      config.json                           # written by the first shard to start
      config.shard00.json                   # this shard's exact config, git
                                            # commit, and end-of-run summary
      meta.shard00.jsonl                    # one row per query
      activations_thinking_off.shard00.npz  # (N, num_layers+1, hidden_dim) fp16
      activations_thinking_on.shard00.npz

Read a capture back with utils.capture_io.load_capture(), which concatenates
shards in order.

Two details that matter for label quality:

  * **Thinking mode needs room.** Qwen3 emits a <think> block before its
    answer, routinely running to hundreds of tokens. Grading it under the
    thinking-off budget would score truncation, not reasoning, so
    --max-response-len-thinking defaults far higher and every row records
    whether generation hit the cap (`truncated_on`).
  * **The answer is what follows </think>.** Correctness is graded on the
    post-think text; the raw generation is kept for inspection. A thinking-on
    response whose <think> block never closes has no answer at all, so it is
    graded wrong and flagged (`unclosed_think_on`) rather than graded on the
    unfinished reasoning.

Meta-row fields added 2026-10 (absent on earlier captures; absence means the
old behaviour):

  dataset_index_global  index into the task split (after --max-samples).
                        `dataset_index` keeps its old meaning: the position
                        within this shard.
  prompt_truncated      the templated prompt exceeded --max-prompt-len, so the
                        START of the user content was dropped to make it fit.
                        (Before, the templated prompt was cut from the right,
                        silently dropping the assistant header.)
  has_reasoning_on      the thinking-on response carries a non-empty
                        reasoning trace. Some models ignore the toggle.
  unclosed_think_on     that trace never closed; correct_on is forced False.
  confidence_version    2 = confidence_off scored on the row with its left
                        padding stripped. Rows without the field were scored
                        by v1, which attended to pad tokens: invalid for any
                        row that was padded in its batch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

logger = logging.getLogger("capture")


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

def _correct_str(task_module: Any, generation: str, sample: dict) -> bool:
    # bbh resolves a bare "(B)" against the question's options; pass the
    # question to any grader that takes it, so capture grades as --regrade does.
    if "question" in inspect.signature(task_module.is_correct).parameters:
        return task_module.is_correct(generation, sample["answer"],
                                      question=sample["question"])
    return task_module.is_correct(generation, sample["answer"])


def _prompt_default(task_module: Any, sample: dict) -> str:
    if hasattr(task_module, "format_prompt"):
        return task_module.format_prompt(sample["question"])
    return sample["question"]


# task name -> (loader function name, correctness adapter, default split).
# Every entry must have a matching module in tasks/ — this repo ships its own
# task modules rather than importing them from anywhere else.
_TASK_REGISTRY: dict[str, tuple[str, Any, str]] = {
    "gsm8k": ("load_gsm8k", _correct_str, "test"),
    "lsat": ("load_lsat_logic", _correct_str, "test"),
    "math500": ("load_math500", _correct_str, "test"),
    "mmlu_pro": ("load_mmlu_pro", _correct_str, "test"),
    "bbh": ("load_bbh", _correct_str, "test"),
}

_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
# Granite 3.3 wraps its answer in <response>...</response> when thinking is on
# (its own system prompt instructs it to). Left in place, the tags ride along
# into the grader; unwrap them so every family reaches the grader in the same
# shape. Unclosed is deliberate -- a truncated response still yields its body.
_RESPONSE_OPEN = re.compile(r"<response>", re.IGNORECASE)
_RESPONSE_CLOSE = re.compile(r"</response>", re.IGNORECASE)
# gpt-oss speaks OpenAI's harmony format: reasoning goes to the "analysis"
# channel and the answer to the "final" channel, with no <think> tags in
# sight. Cut at the last final-channel marker or the grader sees the whole
# chain of thought and scores the reasoning instead of the answer.
_HARMONY_FINAL = re.compile(r"<\|channel\|>final<\|message\|>")
_HARMONY_END = re.compile(r"<\|(?:end|return|call)\|>")
_HARMONY_ANALYSIS = re.compile(r"<\|channel\|>analysis<\|message\|>")

# Version of the sequence_confidence scorer, recorded on every meta row that
# carries confidence_off. v1 (rows without the field) re-scored left-padded
# rows with no attention mask; v2 strips each row's padding first.
CONFIDENCE_VERSION = 2

# Below this share of thinking-on rows carrying a reasoning trace, the shard is
# reported at WARNING level: the "thinking-on" pass mostly did not think, so
# its labels compare two non-thinking runs. A warning, not a failure, so cells
# already queued keep their exit-code behaviour.
REASONING_TRACE_WARN = 0.8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def strip_thinking(text: str) -> str:
    """Return the answer portion — everything after the final </think>.

    A response truncated mid-thought has no closing tag; return it unchanged so
    the grader sees the truncation for what it is rather than an empty string.
    """
    finals = list(_HARMONY_FINAL.finditer(text))
    if finals:
        answer = text[finals[-1].end():]
        stop = _HARMONY_END.search(answer)
        return (answer[:stop.start()] if stop else answer).strip()

    matches = list(_THINK_CLOSE.finditer(text))
    answer = text[matches[-1].end():].strip() if matches else text.strip()

    opens = list(_RESPONSE_OPEN.finditer(answer))
    if opens:
        answer = answer[opens[-1].end():]
    closes = list(_RESPONSE_CLOSE.finditer(answer))
    if closes:
        answer = answer[:closes[0].start()]
    return answer.strip()


def _think_left_open(prompt: str) -> bool:
    """True if the rendered prompt opens a <think> block it never closes
    (templates that prefill '<think>' for the model)."""
    opens = list(_THINK_OPEN.finditer(prompt))
    if not opens:
        return False
    closes = list(_THINK_CLOSE.finditer(prompt))
    return not closes or closes[-1].start() < opens[-1].start()


def assess_reasoning(response: str, prompt: str = "") -> dict:
    """Does a thinking-on response carry a reasoning trace, and did it close?

    Returns {"has_reasoning": bool, "unclosed": bool}.

      * `<think>...</think>` with non-empty content -> reasoning, closed.
      * `<think>` and no closing tag -> reasoning, UNCLOSED: the response is
        all unfinished reasoning and has no answer to grade.
      * no tags at all -> no reasoning.
      * `</think>` with no `<think>`, or a prompt that itself leaves `<think>`
        open -> the reasoning starts at the first response token.
      * harmony (gpt-oss): an analysis channel is reasoning; it is unclosed
        if no final channel follows.
    """
    analysis = _HARMONY_ANALYSIS.search(response)
    if analysis is not None:
        final = _HARMONY_FINAL.search(response, analysis.end())
        end = final.start() if final else len(response)
        body = _HARMONY_END.split(response[analysis.end():end])[0]
        return {"has_reasoning": bool(body.strip()), "unclosed": final is None}

    if _think_left_open(prompt):
        start = 0
    else:
        first_open = _THINK_OPEN.search(response)
        first_close = _THINK_CLOSE.search(response)
        if first_open is not None and (first_close is None
                                       or first_open.start() < first_close.start()):
            start = first_open.end()
        elif first_close is not None:
            start = 0            # the opening tag was in the prompt or stripped
        else:
            return {"has_reasoning": False, "unclosed": False}
    close = _THINK_CLOSE.search(response, start)
    body = response[start: close.start() if close else len(response)]
    return {"has_reasoning": bool(_THINK_OPEN.sub("", body).strip()),
            "unclosed": close is None}


def git_provenance() -> dict:
    """Best-effort code provenance for the per-shard config. Never raises."""
    info: dict[str, Any] = {"git_commit": None, "git_dirty": None}
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT, capture_output=True,
            text=True, timeout=10, check=True).stdout.strip() or None
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=10,
            check=True).stdout
        info["git_dirty"] = bool(status.strip())
    except Exception as exc:  # noqa: BLE001 -- provenance must never kill a run
        logger.warning("could not record git commit: %s", exc)
    return info


# Families spell the thinking toggle three different ways, so resolve it per
# model rather than whitelisting names. An unrecognised name used to fall
# through silently, rendering identical prompts for thinking-off and
# thinking-on, which produces two identical captures and labels that mean
# nothing.
#
#   template kwarg   Qwen3, SmolLM3, Cogito -> enable_thinking=
#                    Granite 3.3            -> thinking=
#   system prompt    Nemotron-Nano          -> "detailed thinking on"/"off"
_TOGGLE_KWARGS = ("enable_thinking", "thinking")

# Model-name substring -> (system prompt when ON, system prompt when OFF).
# These are quoted from the vendor model card; the toggle is a literal string
# the model was tuned on, so it must match exactly.
_SYSTEM_TOGGLES = {
    "nemotron": ("detailed thinking on", "detailed thinking off"),
}

# Some families grade reasoning rather than switching it. gpt-oss takes
# reasoning_effort=low|medium|high, so "off" is the lowest setting rather than
# a true off -- less thinking, not none. Worth stating in any writeup: for
# these models the label is "more reasoning helps", not "reasoning helps".
_VALUE_TOGGLES = {
    "gpt-oss": ("reasoning_effort", "high", "low"),
}


class ThinkingToggle:
    """How to render a prompt with reasoning on or off for one model."""

    def __init__(self, kind: str, spec: Any):
        self.kind = kind          # "kwarg" | "system"
        self.spec = spec

    def describe(self) -> str:
        if self.kind == "kwarg":
            return f"chat-template kwarg {self.spec!r}"
        if self.kind == "kwarg_value":
            return "chat-template kwarg %s=%r/%r" % self.spec
        return f"system prompt {self.spec[0]!r}/{self.spec[1]!r}"

    def render(self, tokenizer: Any, raw_prompt: str, enable_thinking: bool) -> str:
        messages = [{"role": "user", "content": raw_prompt}]
        kwargs: dict[str, Any] = dict(add_generation_prompt=True, tokenize=False)
        if self.kind == "kwarg":
            kwargs[self.spec] = enable_thinking
        elif self.kind == "kwarg_value":
            name, on, off = self.spec
            kwargs[name] = on if enable_thinking else off
        else:
            on, off = self.spec
            messages.insert(0, {"role": "system",
                                "content": on if enable_thinking else off})
        return tokenizer.apply_chat_template(messages, **kwargs)


def detect_thinking_toggle(tokenizer: Any, model_name: str) -> ThinkingToggle:
    """Resolve how this model switches reasoning on and off."""
    lowered = model_name.lower()
    for needle, prompts in _SYSTEM_TOGGLES.items():
        if needle in lowered:
            return ThinkingToggle("system", prompts)
    for needle, spec in _VALUE_TOGGLES.items():
        if needle in lowered:
            return ThinkingToggle("kwarg_value", spec)

    template = getattr(tokenizer, "chat_template", None) or ""
    if isinstance(template, list):       # multi-template tokenizers
        template = " ".join(t.get("template", "") for t in template)
    for kwarg in _TOGGLE_KWARGS:
        if kwarg in template:
            return ThinkingToggle("kwarg", kwarg)

    raise ValueError(
        f"no thinking toggle found for {model_name!r}. Looked for chat-template "
        f"kwargs {_TOGGLE_KWARGS} and system-prompt families "
        f"{tuple(_SYSTEM_TOGGLES)}. Do not fall back to an untoggled template: "
        "it silently makes the thinking-off and thinking-on captures identical.")


def apply_chat_template_to_prompt(
    raw_prompt: str, tokenizer: Any, model_name: str, enable_thinking: bool = False,
    toggle: ThinkingToggle | None = None,
) -> str:
    """Render the prompt as a single user turn through the model's template."""
    if toggle is None:
        toggle = detect_thinking_toggle(tokenizer, model_name)
    return toggle.render(tokenizer, raw_prompt, enable_thinking)


def verify_toggle_changes_prompt(tokenizer: Any, model_name: str,
                                 toggle: ThinkingToggle) -> None:
    """Hard-fail if the toggle is inert. The failure is otherwise invisible."""
    probe = "What is 2 + 2?"
    off = toggle.render(tokenizer, probe, False)
    on = toggle.render(tokenizer, probe, True)
    if off == on:
        raise ValueError(
            f"{model_name!r}: {toggle.describe()} renders byte-identical "
            "prompts for reasoning on and off, so the capture would compare a "
            "model against itself. Refusing to run.")
    logger.info("thinking toggle verified (%s): off/on prompts differ",
                toggle.describe())


def build_prompt(
    sample: dict,
    task_module: Any,
    tokenizer: Any = None,
    model_name: str | None = None,
    chat_template: bool = False,
    enable_thinking: bool = False,
    toggle: Any = None,
) -> str:
    raw_prompt = _prompt_default(task_module, sample)
    if not chat_template:
        return raw_prompt
    assert tokenizer is not None and model_name is not None
    return apply_chat_template_to_prompt(
        raw_prompt, tokenizer, model_name, enable_thinking=enable_thinking,
        toggle=toggle,
    )


def fit_prompt(raw_prompt: str, render: Any, tokenizer: Any, max_len: int,
               add_special_tokens: bool) -> tuple[str, bool]:
    """Render `raw_prompt`; if it exceeds `max_len` tokens, shorten the USER
    CONTENT from the left until the rendered prompt fits.

    Returns (rendered_prompt, was_truncated).

    Truncating the rendered prompt instead -- what the tokenizer's
    `truncation=True` does -- cuts from the right and silently drops the
    template's generation prompt (`<|im_start|>assistant`, the thinking
    toggle's empty think block), so the model is asked to continue the
    question rather than answer it. Dropping the start of the question keeps
    the template intact and keeps the end of the question, where task prompts
    put the actual ask and the answer-format instruction.

    When nothing overflows this returns exactly `render(raw_prompt)`, so the
    tokens fed to the model are identical to the old behaviour.
    """
    def n_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=add_special_tokens).input_ids)

    rendered = render(raw_prompt)
    over = n_tokens(rendered) - max_len
    if over <= 0:
        return rendered, False

    ids = tokenizer(raw_prompt, add_special_tokens=False).input_ids
    keep = len(ids) - over
    # decode -> re-encode is not always length-preserving at the cut, so
    # shrink until it fits rather than trusting one subtraction.
    for _ in range(16):
        if keep <= 0:
            break
        rendered = render(tokenizer.decode(ids[-keep:], skip_special_tokens=False))
        over = n_tokens(rendered) - max_len
        if over <= 0:
            return rendered, True
        keep -= over
    raise ValueError(
        f"cannot fit prompt into --max-prompt-len {max_len}: the template "
        "alone (or nearly all of it) exceeds the limit. Raise --max-prompt-len.")


def load_model(model_name: str, attn_implementation: str = "sdpa"):
    """Load tokenizer + model for left-padded batched inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        # Left padding plus an explicit pad id is what lets the prefill state
        # for every row in a batch sit at the same position (-1).
        tokenizer.pad_token = tokenizer.eos_token

    dtype = (torch.bfloat16
             if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
             else torch.float16)
    logger.info("loading model: %s (%s, %s)", model_name, attn_implementation, dtype)
    # `torch_dtype` rather than the newer `dtype` alias: it is honored across
    # the whole transformers 4.x range this repo pins, where `dtype` only
    # exists from 4.56 on.
    def _load(attn: str):
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation=attn,
            torch_dtype=dtype,
            device_map="auto",
        )

    try:
        model = _load(attn_implementation)
    except ValueError as exc:
        # Not every architecture has an SDPA path -- gpt-oss, for one, refuses
        # it outright. Eager is slower but universally available, and a failed
        # load here costs a whole dispatched cell, so fall back rather than die.
        if attn_implementation == "eager" or "attention implementation" not in str(exc).lower():
            raise
        logger.warning("%s rejected attn_implementation=%r, falling back to "
                       "eager: %s", model_name, attn_implementation,
                       str(exc).split(".")[0])
        model = _load("eager")
    model.eval()
    return tokenizer, model


def prefill_hidden_states(model, input_ids, attention_mask) -> np.ndarray:
    """Last-prompt-token hidden state at every layer, before generation.

    Returns (B, num_layers+1, hidden_dim) float16 — index 0 is the embedding
    layer, matching HuggingFace's `hidden_states` convention.

    This is a plain forward pass, deliberately not `generate(...,
    output_hidden_states=True)`: that returns states for *every generated
    token*, indexed by step, which is both the wrong shape for a prefill probe
    and enormous. `use_cache=False` keeps the KV cache from being built for a
    pass whose output we throw away.
    """
    import torch

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    # Left padding puts the last *prompt* token at position -1 for every row.
    stacked = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1)
    return stacked.to(torch.float16).cpu().numpy()


def _left_pad_counts(attention_mask, prompt_len: int, n_rows: int) -> list[int]:
    """Leading-pad count per row, read from the batch's prompt attention mask.

    Read from the mask, never from token values: many tokenizers set pad = eos
    (or pad = bos), so a real prompt token can equal the pad id.
    """
    if attention_mask is None:
        if n_rows > 1:
            raise ValueError(
                "sequence_confidence needs the prompt attention_mask for a "
                "batch of more than one row: a left-padded row scored without "
                "it attends to its pad tokens (confidence v1 bug)")
        return [0]
    mask = attention_mask.detach().to("cpu").long()
    if mask.shape != (n_rows, prompt_len):
        raise ValueError(f"attention_mask shape {tuple(mask.shape)} does not "
                         f"match (rows, prompt_len) = ({n_rows}, {prompt_len})")
    pads = []
    for row in mask:
        n_real = int(row.sum())
        pad = prompt_len - n_real
        # Only contiguous LEFT padding can be stripped by slicing.
        if n_real == 0 or row[:pad].any() or not row[pad:].all():
            raise ValueError("attention_mask is not contiguous left padding; "
                             "the tokenizer must use padding_side='left'")
        pads.append(pad)
    return pads


def sequence_confidence(model, sequences, prompt_len: int, n_new: list[int],
                        pad_id: int | None = None,
                        attention_mask=None) -> list[dict]:
    """Mean log-prob and mean predictive entropy over each row's generated tokens.

    This is the baseline any reviewer asks for first: if the model can simply
    say it is unsure, a probe on its hidden state has to beat that to be worth
    a forward pass.

    `sequences` is generate()'s output for a LEFT-PADDED batch: (B, prompt_len
    + new). `attention_mask` is that batch's (B, prompt_len) prompt mask and is
    required when B > 1. Each row is scored with its padding sliced off, so it
    is scored exactly as if it had been generated alone. (Version 1 of this
    function sliced from column 0 with no mask, so the model attended to the
    pad tokens; mean log-prob then tracked the pad count, Spearman -0.72 to
    -0.88 on the v3 captures.) `pad_id` is unused and kept for call-site
    compatibility -- padding is read from the mask, not token values.

    Scored one row at a time on purpose. A batched forward materialises
    (batch, seq, vocab) logits -- at batch 16, 3k tokens and a 150k vocab that
    is ~20GB and OOMs the node. One row is ~1GB and the extra forwards are
    cheap next to thousands of sequential decode steps.
    """
    import torch
    import torch.nn.functional as F

    pads = _left_pad_counts(attention_mask, prompt_len, sequences.shape[0])
    out = []
    for i in range(sequences.shape[0]):
        n = int(n_new[i])
        if n <= 0:
            out.append({"mean_logprob": None, "min_logprob": None,
                        "mean_entropy": None})
            continue
        seq = sequences[i : i + 1, pads[i] : prompt_len + n]
        real_prompt = prompt_len - pads[i]
        with torch.no_grad():
            logits = model(input_ids=seq, use_cache=False).logits[0].float()
        # position t predicts token t+1, so the generated tokens are scored by
        # the logits at real_prompt-1 .. real_prompt+n-2
        pred = logits[real_prompt - 1 : real_prompt + n - 1]
        tgt = seq[0, real_prompt : real_prompt + n]
        logprobs = -F.cross_entropy(pred, tgt, reduction="none")
        lsm = F.log_softmax(pred, dim=-1)
        entropy = -(lsm.exp() * lsm).sum(dim=-1)
        out.append({
            "mean_logprob": float(logprobs.mean()),
            "min_logprob": float(logprobs.min()),
            "mean_entropy": float(entropy.mean()),
        })
        del logits, pred, lsm, entropy
    return out


def generate_batch(model, tokenizer, input_ids, attention_mask, max_new_tokens: int):
    """Greedy-decode a batch. Returns (texts, truncated_flags, n_new_tokens)."""
    import torch

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
    new_tokens = out.sequences[:, input_ids.shape[1]:]
    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    eos_ids = {tokenizer.eos_token_id}
    gen_cfg_eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(gen_cfg_eos, (list, tuple)):
        eos_ids.update(int(x) for x in gen_cfg_eos)
    elif gen_cfg_eos is not None:
        eos_ids.add(int(gen_cfg_eos))
    eos_ids.discard(None)

    truncated, lengths = [], []
    for row in new_tokens:
        hit = (torch.isin(row, torch.tensor(sorted(eos_ids), device=row.device))
               .nonzero())
        if hit.numel() > 0:
            truncated.append(False)
            lengths.append(int(hit[0].item()) + 1)
        else:
            truncated.append(True)
            lengths.append(int(row.shape[0]))
    return texts, truncated, lengths, out.sequences


# Above this share of thinking-OFF responses hitting the token cap, the capture
# is refused rather than published. correct_off defines both objectives, so a
# truncated off-pass does not measure capability -- it measures response length.
# See paper/results/metrics/truncation/README.md for the round of results this
# cost. 0.2 is a backstop, not a target: a healthy capture is near zero.
OFF_TRUNCATION_LIMIT = 0.2


def select_shard(dataset: list, shard_index: int, shard_count: int) -> list:
    """Round-robin slice, so every shard mixes easy and hard items evenly.

    A contiguous split would hand one shard all the long-reasoning tail if the
    dataset is ordered by difficulty, making shard runtimes wildly uneven.
    """
    if shard_count <= 1:
        return dataset
    return dataset[shard_index::shard_count]


def build_writer_config(model: Any, args: argparse.Namespace) -> dict:
    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    return {
        "model_name": args.model,
        "task": args.task,
        "split": args.split,
        "num_layers": num_layers,
        "hidden_dim": getattr(cfg, "hidden_size", None),
        "chat_template": bool(args.chat_template),
        "max_prompt_len": args.max_prompt_len,
        "max_response_len": args.max_response_len,
        "max_response_len_thinking": args.max_response_len_thinking,
        "batch_size": args.batch_size,
        "shard_count": args.shard_count,
    }


def build_shard_config(model: Any, args: argparse.Namespace,
                       toggle: Any) -> dict:
    """Everything that determined THIS shard's rows. config.json is written by
    whichever shard starts first, so it cannot show that two shards of one
    capture ran at different budgets, batch sizes, or commits; this can."""
    import torch
    import transformers

    cfg = build_writer_config(model, args)
    cfg.update({
        "thinking_toggle": toggle.describe() if toggle is not None else None,
        "shard_index": args.shard_index,
        "max_samples": args.max_samples,
        "attn_implementation": args.attn_implementation,
        "capture_logprobs": bool(args.capture_logprobs),
        "confidence_version": CONFIDENCE_VERSION if args.capture_logprobs else None,
        "prompt_truncation": "left, user content only, flagged per row",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "argv": list(sys.argv),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **git_provenance(),
    })
    return cfg


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def run_capture(args: argparse.Namespace) -> int:
    import torch

    tokenizer, model = load_model(args.model, args.attn_implementation)

    # Establish the toggle before spending a GPU-hour on it, and record which
    # kwarg was used so a capture is self-describing across model families.
    toggle = None
    if args.chat_template:
        toggle = detect_thinking_toggle(tokenizer, args.model)
        verify_toggle_changes_prompt(tokenizer, args.model, toggle)
    load_fn_name, is_correct_adapter, _ = _TASK_REGISTRY[args.task]

    try:
        task_module = importlib.import_module(f"tasks.{args.task}")
    except ImportError:
        logger.error("task module not found: tasks.%s", args.task)
        raise

    dataset = getattr(task_module, load_fn_name)(args.split)
    logger.info("loaded %s/%s: %d examples", args.task, args.split, len(dataset))

    if args.max_samples > 0:
        dataset = dataset[: args.max_samples]
    # Shard the indices alongside the rows so each row can record where it
    # sits in the split, not just where it sits in this shard.
    global_index = select_shard(list(range(len(dataset))),
                                args.shard_index, args.shard_count)
    dataset = select_shard(dataset, args.shard_index, args.shard_count)
    logger.info(
        "shard %d/%d: %d examples to process",
        args.shard_index, args.shard_count, len(dataset),
    )
    if not dataset:
        logger.warning("nothing to do for this shard")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"shard{args.shard_index:02d}"
    meta_path = out_dir / f"meta.{suffix}.jsonl"
    off_path = out_dir / f"activations_thinking_off.{suffix}.npz"
    on_path = out_dir / f"activations_thinking_on.{suffix}.npz"
    config_path = out_dir / "config.json"
    shard_config_path = out_dir / f"config.{suffix}.json"

    # config.json: unchanged legacy file, first shard to start wins. Readers
    # (utils.capture_io.load_config) only take shard-invariant keys from it.
    if not config_path.exists():
        capture_config = build_writer_config(model, args)
        capture_config["thinking_toggle"] = (
            toggle.describe() if toggle is not None else None)
        config_path.write_text(json.dumps(capture_config, indent=2))
    shard_config = build_shard_config(model, args, toggle)
    shard_config["status"] = "running"
    shard_config_path.write_text(json.dumps(shard_config, indent=2) + "\n")

    add_special = not args.chat_template

    def renderer(enable_thinking: bool):
        if not args.chat_template:
            return lambda raw: raw
        return lambda raw: toggle.render(tokenizer, raw, enable_thinking)

    render_off, render_on = renderer(False), renderer(True)

    acts_off, acts_on, meta_rows = [], [], []
    started = time.monotonic()

    for start in range(0, len(dataset), args.batch_size):
        batch = dataset[start: start + args.batch_size]
        raws = [_prompt_default(task_module, s) for s in batch]

        # The untruncated rendering, for prompt_hash: a row's identity should
        # not depend on --max-prompt-len.
        prompts_off = [render_off(r) for r in raws]
        fitted_off = [fit_prompt(r, render_off, tokenizer, args.max_prompt_len,
                                 add_special) for r in raws]
        fitted_on = [fit_prompt(r, render_on, tokenizer, args.max_prompt_len,
                                add_special) for r in raws]
        prompt_truncated = [a[1] or b[1] for a, b in zip(fitted_off, fitted_on)]
        if any(prompt_truncated):
            logger.warning(
                "%d/%d prompts in this batch exceed --max-prompt-len %d; the "
                "start of their user content was dropped (flagged "
                "prompt_truncated)", sum(prompt_truncated), len(batch),
                args.max_prompt_len)

        results = {}
        for mode, fitted, budget in (
            ("off", fitted_off, args.max_response_len),
            ("on", fitted_on, args.max_response_len_thinking),
        ):
            prompts = [f[0] for f in fitted]
            # No truncation here: fit_prompt already guaranteed the limit
            # without cutting the template.
            tokens = tokenizer(
                prompts, padding=True, return_tensors="pt",
                add_special_tokens=add_special,
            ).to(model.device)

            hidden = prefill_hidden_states(model, tokens.input_ids, tokens.attention_mask)
            texts, truncated, lengths, sequences = generate_batch(
                model, tokenizer, tokens.input_ids, tokens.attention_mask, budget
            )
            # Confidence on the thinking-OFF pass only. That is the pass a
            # router would actually have available, and it is the baseline the
            # probe has to beat: if the model can just say it is unsure, a
            # hidden-state probe needs to justify its forward pass.
            conf = [None] * len(texts)
            if args.capture_logprobs and mode == "off":
                conf = sequence_confidence(
                    model, sequences, tokens.input_ids.shape[1], lengths,
                    tokenizer.pad_token_id, attention_mask=tokens.attention_mask)
            results[mode] = {
                "prompts": prompts,
                "hidden": hidden, "texts": texts,
                "truncated": truncated, "lengths": lengths,
                "confidence": conf,
                "prompt_lens": tokens.attention_mask.sum(dim=1).tolist(),
            }
            del sequences
            del tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for i, sample in enumerate(batch):
            answer_off = strip_thinking(results["off"]["texts"][i])
            answer_on = strip_thinking(results["on"]["texts"][i])
            correct_off = bool(is_correct_adapter(task_module, answer_off, sample))
            reasoning_on = assess_reasoning(results["on"]["texts"][i],
                                            results["on"]["prompts"][i])
            # An unclosed <think> means the model never reached an answer;
            # strip_thinking returns the raw reasoning, and grading that can
            # score a number the model was still working through as correct.
            correct_on = (not reasoning_on["unclosed"]
                          and bool(is_correct_adapter(task_module, answer_on, sample)))
            gidx = global_index[start + i]
            conf_off = results["off"]["confidence"][i]

            acts_off.append(results["off"]["hidden"][i])
            acts_on.append(results["on"]["hidden"][i])
            row = {
                # Fallback id uses the split index: the within-shard position
                # collides across shards.
                "sample_id": sample.get("key", f"{args.task}-{gidx}"),
                "dataset_index": start + i,
                "dataset_index_global": gidx,
                "prompt_hash": sha256(prompts_off[i]),
                "question": sample["question"],
                "answer": sample["answer"],
                "difficulty": sample.get("difficulty"),
                "response_off": results["off"]["texts"][i],
                "response_on": results["on"]["texts"][i],
                "answer_off": answer_off,
                "answer_on": answer_on,
                "correct_off": correct_off,
                "correct_on": correct_on,
                "truncated_off": results["off"]["truncated"][i],
                "confidence_off": conf_off,
                "truncated_on": results["on"]["truncated"][i],
                "n_tokens_off": results["off"]["lengths"][i],
                "n_tokens_on": results["on"]["lengths"][i],
                "prompt_len_off": results["off"]["prompt_lens"][i],
                "prompt_len_on": results["on"]["prompt_lens"][i],
                "prompt_truncated": prompt_truncated[i],
                "has_reasoning_on": reasoning_on["has_reasoning"],
                "unclosed_think_on": reasoning_on["unclosed"],
            }
            if conf_off is not None:
                row["confidence_version"] = CONFIDENCE_VERSION
            meta_rows.append(row)

        done = len(meta_rows)
        rate = done / max(time.monotonic() - started, 1e-6)
        logger.info(
            "%d/%d  (%.2f samples/s, eta %.0f min)  off=%d%% on=%d%%",
            done, len(dataset), rate,
            (len(dataset) - done) / max(rate, 1e-6) / 60,
            round(100 * sum(r["correct_off"] for r in meta_rows) / done),
            round(100 * sum(r["correct_on"] for r in meta_rows) / done),
        )

    np.savez_compressed(off_path, activations=np.stack(acts_off).astype(np.float16))
    np.savez_compressed(on_path, activations=np.stack(acts_on).astype(np.float16))
    with open(meta_path, "w", encoding="utf-8") as fh:
        for row in meta_rows:
            fh.write(json.dumps(row) + "\n")

    n = len(meta_rows)
    n_off = sum(r["correct_off"] for r in meta_rows)
    n_on = sum(r["correct_on"] for r in meta_rows)
    helped = sum(1 for r in meta_rows if not r["correct_off"] and r["correct_on"])
    hurt = sum(1 for r in meta_rows if r["correct_off"] and not r["correct_on"])
    trunc = sum(r["truncated_on"] for r in meta_rows)
    trunc_off = sum(r["truncated_off"] for r in meta_rows)
    n_reasoning = sum(r["has_reasoning_on"] for r in meta_rows)
    n_unclosed = sum(r["unclosed_think_on"] for r in meta_rows)
    n_prompt_trunc = sum(r["prompt_truncated"] for r in meta_rows)

    shard_config["summary"] = {
        "n_samples": n,
        "accuracy_off": round(n_off / n, 4),
        "accuracy_on": round(n_on / n, 4),
        "off_truncation_rate": round(trunc_off / n, 4),
        "on_truncation_rate": round(trunc / n, 4),
        "has_reasoning_on_rate": round(n_reasoning / n, 4),
        "unclosed_think_on": n_unclosed,
        "prompt_truncated": n_prompt_trunc,
    }
    shard_config["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    shard_config["status"] = (
        "quarantined" if trunc_off / n > OFF_TRUNCATION_LIMIT else "complete")
    shard_config_path.write_text(json.dumps(shard_config, indent=2) + "\n")

    logger.info("wrote %s and activations %s", meta_path.name, np.stack(acts_off).shape)
    logger.info("thinking-OFF accuracy: %d/%d (%.1f%%)", n_off, n, 100 * n_off / n)
    logger.info("thinking-ON  accuracy: %d/%d (%.1f%%)", n_on, n, 100 * n_on / n)
    logger.info("helped (wrong->right): %d (%.1f%%)", helped, 100 * helped / n)
    logger.info("hurt   (right->wrong): %d (%.1f%%)", hurt, 100 * hurt / n)
    if trunc:
        logger.warning(
            "%d/%d thinking-on responses hit the %d-token cap — their labels "
            "reflect truncation, not reasoning; raise --max-response-len-thinking",
            trunc, n, args.max_response_len_thinking,
        )
    if n_unclosed:
        logger.warning(
            "%d/%d thinking-on responses never closed their reasoning block; "
            "graded correct_on=False (flagged unclosed_think_on)",
            n_unclosed, n)
    if n_prompt_trunc:
        logger.warning(
            "%d/%d prompts exceeded --max-prompt-len %d and had the start of "
            "their user content dropped (flagged prompt_truncated)",
            n_prompt_trunc, n, args.max_prompt_len)

    # A "thinking-on" pass that mostly did not think compares two non-thinking
    # runs; Nemotron-Nano on BBH emitted <think> in 7 of 540 rows. Warn only:
    # failing here would change the exit status of cells already queued.
    reasoning_rate = n_reasoning / n
    (logger.warning if reasoning_rate < REASONING_TRACE_WARN else logger.info)(
        "thinking-on responses with a reasoning trace: %d/%d (%.1f%%)%s",
        n_reasoning, n, 100 * reasoning_rate,
        "" if reasoning_rate >= REASONING_TRACE_WARN else
        f" -- below {REASONING_TRACE_WARN:.0%}: the thinking toggle is not "
        "taking effect on most rows, so on/off labels here do not measure "
        "what thinking adds")

    # Thinking-OFF truncation is the more dangerous of the two and used to go
    # unreported. correct_off defines BOTH objectives -- needs_thinking is
    # ~correct_off, and rescued conditions on it -- so a capped off-pass does
    # not measure "the model cannot do this without thinking", it measures
    # "the model did not finish writing in N tokens". At a 320-token cap on
    # MATH-500 that reached 75% of rows, correct_off became almost a pure
    # function of truncation (3.7% correct when capped vs 94.4% when not) and
    # the probe scored higher predicting truncation than predicting the label.
    if trunc_off:
        rate = trunc_off / n
        level = logger.error if rate > OFF_TRUNCATION_LIMIT else logger.warning
        level(
            "%d/%d (%.1f%%) thinking-OFF responses hit the %d-token cap. "
            "correct_off then reflects response LENGTH, not capability, and "
            "both needs_thinking and rescued inherit that confound. Raise "
            "--max-response-len until this is near zero before trusting any "
            "label from this capture.",
            trunc_off, n, 100 * rate, args.max_response_len,
        )
        if rate > OFF_TRUNCATION_LIMIT:
            # Fail the run, and make the failure stick.
            #
            # A non-zero exit alone is not enough. The dispatch worker checks
            # `output_check` BEFORE running a claimed cell (worker.py, the
            # resume path) and marks it skipped when those paths exist -- so a
            # capture that wrote its files and then exited non-zero would be
            # filed as failed once and laundered into "skipped" by the next
            # worker that touched it. Quarantining the meta file leaves
            # output_check unsatisfiable, so the cell stays failed until
            # someone re-runs it with a bigger budget.
            #
            # The data is not deleted: the activations stay put and the meta
            # rows move aside intact, so the truncation can be measured (that
            # is how the confound was diagnosed) without any downstream script
            # picking them up as a normal capture.
            quarantined = meta_path.with_suffix(meta_path.suffix + ".quarantined")
            meta_path.replace(quarantined)
            (out_dir / f"TRUNCATION_FAILURE.{suffix}.json").write_text(
                json.dumps({
                    "reason": "thinking-OFF truncation above limit",
                    "off_truncation_rate": round(rate, 4),
                    "limit": OFF_TRUNCATION_LIMIT,
                    "n_truncated": trunc_off,
                    "n_samples": n,
                    "max_response_len": args.max_response_len,
                    "task": args.task,
                    "model": args.model,
                    "shard": suffix,
                    "quarantined_meta": quarantined.name,
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            logger.error(
                "REFUSING to publish this capture: %s renamed to %s. "
                "correct_off is not trustworthy at this truncation rate, so "
                "every label derived from it would be confounded. Re-run with "
                "a larger --max-response-len.",
                meta_path.name, quarantined.name,
            )
            return 1
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Capture paired thinking-off/on inference with prefill states.")
    p.add_argument("--task", required=True, choices=list(_TASK_REGISTRY))
    p.add_argument("--split", default=None, help="Default: the task's own default split")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-samples", type=int, default=0, help="0 = the whole split")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-prompt-len", type=int, default=1024,
                   help="Token limit on the templated prompt. A prompt over it "
                        "has the start of its user content dropped (never the "
                        "template) and is flagged prompt_truncated.")
    p.add_argument("--max-response-len", type=int, required=True,
                   help="Token budget with thinking OFF. Required: the old "
                        "default of 320 truncated 75%% of MATH-500 answers, "
                        "and correct_off then measured length, not ability. "
                        "Size it so thinking-OFF truncation is near zero.")
    p.add_argument("--max-response-len-thinking", type=int, default=1536,
                   help="Token budget with thinking ON — must fit the <think> "
                        "block plus the answer, or labels measure truncation")
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--attn-implementation", default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--capture-logprobs", action="store_true",
                   help="Also record mean/min token log-probability and mean "
                        "predictive entropy of the thinking-OFF generation. "
                        "This is the confidence baseline the probe must beat. "
                        "Costs one extra forward pass per row.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.split is None:
        args.split = _TASK_REGISTRY[args.task][2]
    if not 0 <= args.shard_index < max(args.shard_count, 1):
        raise SystemExit(
            f"--shard-index {args.shard_index} out of range for "
            f"--shard-count {args.shard_count}")
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return run_capture(args)


if __name__ == "__main__":
    raise SystemExit(main())

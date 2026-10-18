"""
s1_cache.py — the System 1 serving loop's cache handling (issue #3, §2).

One prompt prefill feeds both paths:

  decide     run [invocation tokens + typed question] with the aLoRA on, read
             the decide-position logits, then put the cache back
             (utils/typed_question.score)
  escalate   adapter off, append the template's own thinking-on tail, decode

aLoRA leaves every prompt-token state at its base value, so the prefix cache
is the base model's. What the decide suffix changes must be undone:

  attention layers   grow by concatenation -> crop back to the prompt length
  GDN (linear) layers  conv and recurrent states are overwritten in place, and
                     transformers reports is_croppable=False -> clone at the
                     prompt end, copy back afterwards

Qwen3 (attention only) needs only the crop; Qwen3.5 needs both. Batch size 1.
"""

from __future__ import annotations

import contextlib

import torch

_ASSISTANT = "<|im_start|>assistant\n"


# ---------------------------------------------------------------------------
# Snapshot / restore
# ---------------------------------------------------------------------------

def _clone(d: dict) -> dict:
    return {i: None if t is None else t.clone() for i, t in d.items()}


def snapshot(cache) -> list[dict]:
    """Per-layer state needed to return ``cache`` to this exact point."""
    snap = []
    for layer in cache.layers:
        s = {}
        if hasattr(layer, "recurrent_states"):
            s["conv"] = _clone(layer.conv_states)
            s["recurrent"] = _clone(layer.recurrent_states)
            s["has_previous_state"] = dict(layer.has_previous_state)
        if hasattr(layer, "keys"):
            if getattr(layer, "is_sliding", False):
                raise NotImplementedError("sliding-window layers cannot be cropped back")
            s["length"] = layer.get_seq_length()
        if not s:
            raise NotImplementedError(f"unknown cache layer {type(layer).__name__}")
        snap.append(s)
    return snap


def _put(d: dict, saved: dict) -> None:
    for i, t in saved.items():
        if t is None:
            d[i] = None
        elif d[i] is not None and d[i].shape == t.shape:
            d[i].copy_(t)                      # keep the static address
        else:
            d[i] = t.clone()


def restore(cache, snap: list[dict]) -> None:
    """Return ``cache`` to the state captured by ``snapshot``. Reusable."""
    if len(snap) != len(cache.layers):
        raise ValueError("snapshot is from a different cache")
    for layer, s in zip(cache.layers, snap):
        if "conv" in s:
            _put(layer.conv_states, s["conv"])
            _put(layer.recurrent_states, s["recurrent"])
            layer.has_previous_state.update(s["has_previous_state"])
        if "length" in s:
            extra = layer.get_seq_length() - s["length"]
            if extra < 0:
                raise ValueError("cache is shorter than the snapshot")
            if extra:
                layer.crop(-extra)


# ---------------------------------------------------------------------------
# Prompt layout
# ---------------------------------------------------------------------------

def prompt_split(tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """(prefix_ids, escalate_ids) for one conversation.

    The prefix is the chat template up to and including the last
    ``<|im_start|>assistant\\n``; escalate_ids is whatever the template's own
    ``enable_thinking=True`` prompt adds after it (``<think>\\n`` on Qwen3.5,
    nothing on Qwen3, which opens ``<think>`` itself). Raises unless
    prefix + escalate tokenizes to exactly the vanilla thinking prompt, which
    is what makes "escalation is the untouched reasoner" testable.
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=True)
    cut = text.rfind(_ASSISTANT)
    if cut < 0:
        raise ValueError("chat template has no assistant header")
    cut += len(_ASSISTANT)
    enc = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    prefix, tail, full = enc(text[:cut]), enc(text[cut:]), enc(text)
    if prefix + tail != full:
        raise ValueError("prefix + escalate tail does not tokenize to the vanilla "
                         "thinking prompt (a merge across the cut)")
    return prefix, tail


# ---------------------------------------------------------------------------
# Forward steps
# ---------------------------------------------------------------------------

def adapter_off(model):
    """Context with every adapter disabled; a no-op on a plain model."""
    return model.disable_adapter() if hasattr(model, "disable_adapter") \
        else contextlib.nullcontext()


def _ids(x, device) -> torch.Tensor:
    return torch.tensor([list(x)], dtype=torch.long, device=device)


@torch.no_grad()
def prefill(model, prefix_ids):
    """Base-model prefill of the prompt. Returns (cache, logits at the last token)."""
    from transformers import DynamicCache
    cache = DynamicCache(config=model.config)
    with adapter_off(model):
        out = model(input_ids=_ids(prefix_ids, model.device), past_key_values=cache,
                    use_cache=True)
    return cache, out.logits[0, -1]


@torch.no_grad()
def escalate(model, cache, escalate_ids, last_logits: torch.Tensor,
             max_new_tokens: int, eos_ids=()) -> list[int]:
    """Greedy decode from the restored prompt cache with the adapter off.

    ``last_logits`` are the prefill's (from ``prefill``): Qwen3's thinking tail
    is empty, so its first generated token comes straight from them.
    Returns the generated ids (escalate_ids not included). Leaves the cache
    extended; snapshot first if the prompt state is needed again.
    """
    eos = set(eos_ids)
    out: list[int] = []
    with adapter_off(model):
        logits = last_logits
        if escalate_ids:
            logits = model(input_ids=_ids(escalate_ids, model.device),
                           past_key_values=cache, use_cache=True).logits[0, -1]
        for _ in range(max_new_tokens):
            tok = int(logits.argmax())
            out.append(tok)
            if tok in eos:
                break
            logits = model(input_ids=_ids([tok], model.device), past_key_values=cache,
                           use_cache=True).logits[0, -1]
    return out

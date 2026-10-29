"""
typed_question.py — the LLM-as-Jev bracketed-ID decide format (issue #3, §3).

Suffix appended after the cached prompt prefix:

    <invocation>{question}
    [1] option one
    [2] option two
    Best answer: [

Each legal continuation "k]" is scored as a whole token sequence (so "1]" is
not a prefix of "10]"), and the probabilities are renormalised over the legal
set: off-list mass cannot leak into the answer.
"""

from __future__ import annotations

import torch

from utils import s1_cache

STEM = "Best answer: ["


def _enc(tokenizer, s: str) -> list[int]:
    return tokenizer(s, add_special_tokens=False)["input_ids"]


def suffix_text(invocation: str, question: str, options: list[str]) -> str:
    lines = [f"[{k}] {o}" for k, o in enumerate(options, 1)]
    return f"{invocation}{question}\n" + "\n".join(lines) + f"\n{STEM}"


def build(tokenizer, invocation: str, question: str, options: list[str]) -> dict:
    """Token ids of the suffix and of each candidate "k]", checked in context (T4).

    Raises when the invocation tokens do not appear verbatim in the suffix
    (aLoRA would then never switch on) or when a candidate merges with the
    stem's "[" (its standalone tokens would not be what the model scores).
    """
    if len(options) < 2:
        raise ValueError("need at least two options")
    text = suffix_text(invocation, question, options)
    ids = _enc(tokenizer, text)
    inv = _enc(tokenizer, invocation)
    if ids[:len(inv)] != inv:
        raise ValueError(f"invocation {invocation!r} does not tokenize the same "
                         "at the start of the suffix")
    cands = []
    for k in range(1, len(options) + 1):
        full = _enc(tokenizer, text + f"{k}]")
        if full[:len(ids)] != ids:
            raise ValueError(f"candidate '{k}]' merges with the stem")
        cands.append(full[len(ids):])
    return {"text": text, "suffix_ids": ids, "invocation_ids": inv,
            "candidate_ids": cands}


@torch.no_grad()
def score(model, cache, q: dict) -> dict:
    """Score every candidate after the suffix; the cache is left at the prompt end.

    Candidate tokens carry no invocation sequence, so aLoRA's own offset
    search would switch the adapter off on them; the offset is passed
    explicitly instead, keeping the adapter on through the whole answer.
    """
    is_peft = hasattr(model, "disable_adapter")
    dev = model.device
    snap0 = s1_cache.snapshot(cache)
    try:
        out = model(input_ids=torch.tensor([q["suffix_ids"]], device=dev),
                    past_key_values=cache, use_cache=True)
        first = torch.log_softmax(out.logits[0, -1].float(), -1)
        snap1 = s1_cache.snapshot(cache)
        logp = []
        for cand in q["candidate_ids"]:
            lp = float(first[cand[0]])
            if len(cand) > 1:
                kw = {"alora_offsets": [len(cand) - 1]} if is_peft else {}
                o = model(input_ids=torch.tensor([cand[:-1]], device=dev),
                          past_key_values=cache, use_cache=True, **kw)
                rest = torch.log_softmax(o.logits[0].float(), -1)
                lp += float(sum(rest[i, t] for i, t in enumerate(cand[1:])))
                s1_cache.restore(cache, snap1)
            logp.append(lp)
    finally:
        s1_cache.restore(cache, snap0)
    logp = torch.tensor(logp)
    probs = torch.softmax(logp, 0)
    return {"logp": logp.tolist(), "probs": probs.tolist(),
            "argmax": int(probs.argmax()) + 1, "p_max": float(probs.max()),
            "legal_mass": float(logp.exp().sum())}

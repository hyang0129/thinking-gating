#!/usr/bin/env python3
"""
test_confidence.py — sequence_confidence must actually measure confidence.

Two kinds of test:

  * A tiny randomly-initialised Llama built from a config (no download). Enough
    to pin down the mechanics, above all the one that broke v1: a row taken
    from a LEFT-PADDED batch must score exactly as it does alone. v1 re-scored
    padded rows with no attention mask, so the model attended to the pads and
    mean log-prob tracked the pad count (Spearman -0.72..-0.88 on v3).
  * Real gpt2, only if it is already in the local HF cache (skipped otherwise,
    never downloaded). An untrained model assigns near-uniform probability to
    everything, so only a trained one can show greedy text outscoring random
    tokens.

    python -m pytest tests/test_confidence.py -q
    python tests/test_confidence.py
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.capture_inference_thinking import sequence_confidence  # noqa: E402

VOCAB = 64
PAD = EOS = 1          # pad == eos, as for many real tokenizers


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, pad_token_id=PAD, eos_token_id=EOS,
        bos_token_id=0, initializer_range=0.5)   # peaky, not near-uniform
    model = LlamaForCausalLM(cfg).float().eval()
    return model


def _close(a: dict, b: dict, tol: float = 1e-4) -> bool:
    return all(abs(a[k] - b[k]) <= tol for k in ("mean_logprob", "min_logprob",
                                                 "mean_entropy"))


def _padded_batch():
    """Two rows as generate() would return them from a left-padded batch.

    Row A's real prompt STARTS with the pad/eos id, so stripping padding by
    token value would eat a real token; only the attention mask is right.
    Row B is the longest prompt and gets no padding. Row A generated 5 tokens,
    row B 3 (then pad-filled after its EOS).
    """
    a_prompt, a_gen = [PAD, 7, 9], [11, 13, 15, 17, EOS]
    b_prompt, b_gen = [3, 4, 5, 6, 8, 10, 12], [20, 21, EOS]
    width = len(b_prompt)
    a_pad = width - len(a_prompt)
    seqs = torch.tensor([
        [PAD] * a_pad + a_prompt + a_gen,
        b_prompt + b_gen + [PAD] * (len(a_gen) - len(b_gen)),
    ])
    mask = torch.tensor([[0] * a_pad + [1] * len(a_prompt), [1] * width])
    return seqs, mask, width, (a_prompt, a_gen), (b_prompt, b_gen)


def test_padded_row_scores_identically_to_row_alone(tiny_model):
    seqs, mask, width, (ap, ag), (bp, bg) = _padded_batch()
    batched = sequence_confidence(tiny_model, seqs, width, [len(ag), len(bg)],
                                  PAD, attention_mask=mask)
    alone_a = sequence_confidence(tiny_model, torch.tensor([ap + ag]), len(ap),
                                  [len(ag)], PAD)[0]
    alone_b = sequence_confidence(tiny_model, torch.tensor([bp + bg]), len(bp),
                                  [len(bg)], PAD)[0]
    assert _close(batched[0], alone_a), (batched[0], alone_a)
    assert _close(batched[1], alone_b), (batched[1], alone_b)


def test_order_and_batch_mates_do_not_matter(tiny_model):
    seqs, mask, width, (ap, ag), (bp, bg) = _padded_batch()
    fwd = sequence_confidence(tiny_model, seqs, width, [len(ag), len(bg)],
                              attention_mask=mask)
    rev = sequence_confidence(tiny_model, seqs.flip(0), width,
                              [len(bg), len(ag)], attention_mask=mask.flip(0))
    assert _close(fwd[0], rev[1]) and _close(fwd[1], rev[0])


def test_the_padding_bug_is_real(tiny_model):
    """Teeth: scoring the padded row with its pads attended (v1) gives a
    different number, so the invariance test above can actually fail."""
    seqs, mask, width, (ap, ag), _ = _padded_batch()
    v1_style = sequence_confidence(tiny_model, seqs[:1], width, [len(ag)])[0]
    alone = sequence_confidence(tiny_model, torch.tensor([ap + ag]), len(ap),
                                [len(ag)])[0]
    assert abs(v1_style["mean_logprob"] - alone["mean_logprob"]) > 0.05


def test_batch_without_mask_is_refused(tiny_model):
    seqs, _, width, (_, ag), (_, bg) = _padded_batch()
    with pytest.raises(ValueError, match="attention_mask"):
        sequence_confidence(tiny_model, seqs, width, [len(ag), len(bg)], PAD)


def test_right_padding_is_refused(tiny_model):
    seqs, mask, width, (_, ag), (_, bg) = _padded_batch()
    with pytest.raises(ValueError, match="left padding"):
        sequence_confidence(tiny_model, seqs, width, [len(ag), len(bg)],
                            attention_mask=mask.flip(1))


def test_bounds_and_empty_generation(tiny_model):
    seqs, mask, width, (_, ag), _ = _padded_batch()
    out = sequence_confidence(tiny_model, seqs, width, [len(ag), 0],
                              attention_mask=mask)
    c = out[0]
    assert c["mean_logprob"] <= 0
    assert c["min_logprob"] <= c["mean_logprob"]
    assert 0 <= c["mean_entropy"] <= math.log(VOCAB) + 1e-5
    assert out[1] == {"mean_logprob": None, "min_logprob": None,
                      "mean_entropy": None}


def _cached_gpt2():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained("gpt2", local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gpt2 not in the local HF cache ({type(exc).__name__})")
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model


def test_trained_model_greedy_beats_random():
    tok, model = _cached_gpt2()
    prompt = tok("The capital of France is", return_tensors="pt")
    P = prompt.input_ids.shape[1]
    with torch.no_grad():
        gen = model.generate(**prompt, max_new_tokens=8, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    n_new = gen.shape[1] - P
    greedy = sequence_confidence(model, gen, P, [n_new], tok.pad_token_id)[0]

    torch.manual_seed(0)
    rand = gen.clone()
    rand[0, P:] = torch.randint(0, model.config.vocab_size, (n_new,))
    random_ = sequence_confidence(model, rand, P, [n_new], tok.pad_token_id)[0]

    assert greedy["mean_logprob"] > random_["mean_logprob"]
    assert greedy["mean_logprob"] <= 0
    assert 0 <= greedy["mean_entropy"] <= math.log(model.config.vocab_size)


def test_trained_model_batched_generation_matches_alone():
    """End to end on a trained model: left-padded batched greedy generation,
    scored with the batch mask, equals the same prompt generated and scored
    alone."""
    tok, model = _cached_gpt2()
    tok.padding_side = "left"
    prompts = ["The capital of France is",
               "Once upon a time, in a small village by the sea, there"]
    batch = tok(prompts, return_tensors="pt", padding=True)
    P = batch.input_ids.shape[1]
    with torch.no_grad():
        gen = model.generate(**batch, max_new_tokens=6, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    batched = sequence_confidence(model, gen, P, [6, 6], tok.pad_token_id,
                                  attention_mask=batch.attention_mask)

    alone_in = tok(prompts[0], return_tensors="pt")
    with torch.no_grad():
        alone_gen = model.generate(**alone_in, max_new_tokens=6, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
    Pa = alone_in.input_ids.shape[1]
    assert torch.equal(alone_gen[0, Pa:], gen[0, P:]), "greedy tokens diverged"
    alone = sequence_confidence(model, alone_gen, Pa, [6], tok.pad_token_id)[0]
    assert _close(batched[0], alone, tol=1e-3), (batched[0], alone)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

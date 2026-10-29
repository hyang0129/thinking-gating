"""
Serving-loop tests T1-T5 (issue #3, §2) on tiny random Qwen3.5 (GDN hybrid)
and Qwen3 (attention-only) configs, CPU. T6 (end to end on real checkpoints)
is GPU-only and lives in the dispatched check, not here.

Needs the .venv-s1 stack (transformers >= 5, peft >= 0.18); skipped under the
legacy .venv. T4's real-tokenizer cases skip unless the Qwen tokenizers are in
the local HF cache.
"""

import pytest

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
if int(transformers.__version__.split(".")[0]) < 5:
    pytest.skip("needs .venv-s1 (transformers >= 5)", allow_module_level=True)

import torch  # noqa: E402

from utils import s1_cache, typed_question  # noqa: E402

INV = [5, 6, 7]                       # invocation tokens; prompts avoid ids < 10
VOCAB = 128


def tiny(kind: str):
    torch.manual_seed(0)
    if kind == "qwen3_5":
        cfg = transformers.Qwen3_5TextConfig(
            vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=16, linear_value_head_dim=16)
        model = transformers.Qwen3_5ForCausalLM(cfg)
    else:
        cfg = transformers.Qwen3Config(
            vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16)
        model = transformers.Qwen3ForCausalLM(cfg)
    return model.float().eval()


def with_alora(model, seed: int = 1):
    cfg = peft.LoraConfig(r=4, lora_alpha=8, target_modules="all-linear",
                          alora_invocation_tokens=INV, task_type="CAUSAL_LM")
    pm = peft.get_peft_model(model, cfg)
    g = torch.Generator().manual_seed(seed)
    for n, p in pm.named_parameters():
        if "lora_B" in n:                   # B is zero at init: make it bite
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
    return pm.eval()


def ids(n: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(10, VOCAB, (n,), generator=g).tolist()


P, S, C = ids(12, 1), INV + ids(6, 2), ids(5, 3)
KINDS = ["qwen3_5", "qwen3"]


def step(model, cache, toks):
    return model(input_ids=torch.tensor([toks]), past_key_values=cache,
                 use_cache=True).logits[0]


@pytest.mark.parametrize("kind", KINDS)
def test_t1_snapshot_restore_is_exact(kind):
    pm = with_alora(tiny(kind))
    cache, _ = s1_cache.prefill(pm, P)
    snap = s1_cache.snapshot(cache)
    step(pm, cache, S)                                   # decide suffix, adapter on
    s1_cache.restore(cache, snap)
    with s1_cache.adapter_off(pm):
        got = step(pm, cache, C)
        fresh, _ = s1_cache.prefill(pm, P)
        want = step(pm, fresh, C)
    assert torch.equal(got, want)
    # and the cached path agrees with one uncached forward of P + C
    with s1_cache.adapter_off(pm):
        full = pm(input_ids=torch.tensor([P + C])).logits[0, len(P):]
    torch.testing.assert_close(got, full, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("kind", KINDS)
def test_t1_restore_is_reusable(kind):
    pm = with_alora(tiny(kind))
    cache, _ = s1_cache.prefill(pm, P)
    snap = s1_cache.snapshot(cache)
    outs = []
    for _ in range(3):
        outs.append(step(pm, cache, S))
        s1_cache.restore(cache, snap)
    assert all(torch.equal(outs[0], o) for o in outs[1:])


@pytest.mark.parametrize("kind", KINDS)
def test_t2_no_leak_onto_prompt(kind):
    base = tiny(kind)
    want = base(input_ids=torch.tensor([P])).logits
    pm = with_alora(tiny(kind))
    assert torch.equal(pm(input_ids=torch.tensor([P])).logits, want)
    # in one forward of P + S, every position before the invocation is base.
    # Compare at equal length: the chunked GDN kernel rounds differently by
    # sequence length, which is numerics, not a leak.
    want_ps = base(input_ids=torch.tensor([P + S])).logits[:, :len(P)]
    got = pm(input_ids=torch.tensor([P + S])).logits[:, :len(P)]
    assert torch.equal(got, want_ps)


@pytest.mark.parametrize("kind", KINDS)
def test_t3_adapter_activates_and_zero_b_is_base(kind):
    pm = with_alora(tiny(kind))
    cache, _ = s1_cache.prefill(pm, P)
    on = step(pm, cache, S)[-1]
    base_cache, _ = s1_cache.prefill(pm, P)
    with s1_cache.adapter_off(pm):
        off = step(pm, base_cache, S)[-1]
    assert (on - off).abs().max() > 1e-3
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.zero_()
    cache, _ = s1_cache.prefill(pm, P)
    assert torch.equal(step(pm, cache, S)[-1], off)


@pytest.mark.parametrize("kind", KINDS)
def test_t5_load_path_keeps_alora(kind, tmp_path):
    pm = with_alora(tiny(kind))
    cache, _ = s1_cache.prefill(pm, P)
    want = step(pm, cache, S)[-1]
    pm.save_pretrained(tmp_path)
    loaded = peft.PeftModel.from_pretrained(tiny(kind), tmp_path).eval()
    assert loaded.peft_config["default"].alora_invocation_tokens == INV
    cache, _ = s1_cache.prefill(loaded, P)
    got = step(loaded, cache, S)[-1]
    assert torch.equal(got, want)
    base_cache, _ = s1_cache.prefill(loaded, P)
    with s1_cache.adapter_off(loaded):
        assert (got - step(loaded, base_cache, S)[-1]).abs().max() > 1e-3


class _DigitTok:
    """Char-level stand-in tokenizer for the scoring tests (ids >= 10 only)."""

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [10 + (ord(c) % (VOCAB - 10)) for c in s]}


@pytest.mark.parametrize("kind", KINDS)
def test_score_matches_uncached_and_restores(kind):
    pm = with_alora(tiny(kind))
    tok = _DigitTok()
    q = typed_question.build(tok, "", "Q?", [f"opt{k}" for k in range(11)])
    q["suffix_ids"] = INV + q["suffix_ids"]        # invocation as raw ids
    assert any(len(c) > 1 for c in q["candidate_ids"])   # "10]", "11]"
    cache, _ = s1_cache.prefill(pm, P)
    snap_len = cache.get_seq_length()
    res = typed_question.score(pm, cache, q)
    assert cache.get_seq_length() == snap_len
    # uncached teacher forcing, adapter on from the invocation onward
    for k, cand in enumerate(q["candidate_ids"]):
        seq = P + q["suffix_ids"] + cand
        lp = torch.log_softmax(pm(input_ids=torch.tensor([seq])).logits[0].float(), -1)
        start = len(P) + len(q["suffix_ids"])
        want = sum(float(lp[start - 1 + i, t]) for i, t in enumerate(cand))
        assert abs(res["logp"][k] - want) < 1e-4
    assert abs(sum(res["probs"]) - 1) < 1e-6


@pytest.mark.parametrize("kind", KINDS)
def test_escalate_matches_uncached_greedy(kind):
    pm = with_alora(tiny(kind))
    tail = ids(2, 4)
    cache, last = s1_cache.prefill(pm, P)
    snap = s1_cache.snapshot(cache)
    step(pm, cache, S)                                   # a decide pass first
    s1_cache.restore(cache, snap)
    got = s1_cache.escalate(pm, cache, tail, last, max_new_tokens=8)
    seq = P + tail
    with s1_cache.adapter_off(pm):
        for _ in range(8):
            seq.append(int(pm(input_ids=torch.tensor([seq])).logits[0, -1].argmax()))
    assert got == seq[len(P) + len(tail):]


def _tokenizer(repo):
    try:
        return transformers.AutoTokenizer.from_pretrained(repo, local_files_only=True)
    except Exception:
        pytest.skip(f"{repo} tokenizer not in the local HF cache")


@pytest.mark.parametrize("repo,tail", [("Qwen/Qwen3.5-4B", "<think>\n"),
                                       ("Qwen/Qwen3-8B", "")])
def test_t4_prompt_split_is_the_vanilla_thinking_prompt(repo, tail):
    tok = _tokenizer(repo)
    msgs = [{"role": "user", "content": "What is 2 + 2?"}]
    prefix, esc = s1_cache.prompt_split(tok, msgs)
    assert tok.decode(esc) == tail
    assert tok.decode(prefix).endswith("<|im_start|>assistant\n")


@pytest.mark.parametrize("repo", ["Qwen/Qwen3.5-4B", "Qwen/Qwen3-8B"])
def test_t4_typed_question_tokens_in_context(repo):
    tok = _tokenizer(repo)
    for n in (2, 4, 10, 18):
        q = typed_question.build(tok, "<|decide|>", "Which?",
                                 [f"choice {k}" for k in range(n)])
        assert len(q["candidate_ids"]) == n
        assert len({tuple(c) for c in q["candidate_ids"]}) == n
        assert tok.decode(q["candidate_ids"][-1]) == f"{n}]"

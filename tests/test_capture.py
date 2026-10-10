#!/usr/bin/env python3
"""
test_capture.py — capture_inference_thinking.py, end to end on a tiny model.

No download and no GPU: the model is a 2-layer Llama built from a config with
random weights, and the tokenizer is a byte-level tokenizer built in memory
with a chat template that has an `enable_thinking` toggle. Generation is real;
only the thinking-on TEXT is scripted where a test needs a particular shape of
response (closed / unclosed / missing reasoning).

    python -m pytest tests/test_capture.py -q
"""
import json
import logging
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scripts.capture_inference_thinking as cap  # noqa: E402
from scripts.dispatch.cells import expand_manifest, load_manifest  # noqa: E402

REDO_MANIFESTS = ["capture_qwen3v3_redo.json", "capture_nemotronv3_redo.json"]

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>"
    "{% if enable_thinking is defined and not enable_thinking %}<think></think>"
    "{% endif %}{% endif %}")


# ---------------------------------------------------------------------------
# CLI backward compatibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("manifest", REDO_MANIFESTS)
def test_inflight_redo_manifest_args_still_parse(manifest):
    """The redo queues are in flight; a cluster that git-pulls mid-queue runs
    their exact argv against this parser."""
    cells = expand_manifest(load_manifest(ROOT / "configs/dispatch" / manifest))
    assert cells
    for cell in cells:
        assert cell["script"] == "scripts/capture_inference_thinking.py"
        args = cap.build_parser().parse_args(cell["args"])
        meta = cell["meta"]
        assert args.task == meta["task"]
        assert args.max_response_len == int(meta["off"])
        assert args.max_response_len_thinking == int(meta["think"])
        assert args.batch_size == int(meta["batch"])
        assert args.shard_index == int(meta["shard"])
        assert args.max_prompt_len == 1024
        assert args.capture_logprobs and args.chat_template


def test_max_response_len_is_required():
    with pytest.raises(SystemExit):
        cap.build_parser().parse_args(
            ["--task", "gsm8k", "--out-dir", "x", "--max-response-len-thinking", "8"])


# ---------------------------------------------------------------------------
# Reasoning-trace detection (B7, B7b)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("response,prompt,has,unclosed", [
    ("<think>\nlet me add\n</think>\n\n42", "", True, False),
    ("<think>\nlet me add 40 + 2 = 42 and then", "", True, True),
    ("The answer is 42.", "", False, False),
    ("<think>\n\n</think>\n\n42", "", False, False),        # empty trace
    ("reasoning here</think>42", "<|assistant|><think>", True, False),
    ("reasoning, cut off", "<|assistant|><think>\n", True, True),
    ("answer 42", "<|assistant|><think></think>", False, False),
    ("reasoning</think> 42", "", True, False),              # open tag stripped
    ("<|channel|>analysis<|message|>hmm<|end|>"
     "<|channel|>final<|message|>42", "", True, False),
    ("<|channel|>analysis<|message|>hmm, still", "", True, True),
])
def test_assess_reasoning(response, prompt, has, unclosed):
    assert cap.assess_reasoning(response, prompt) == {
        "has_reasoning": has, "unclosed": unclosed}


# ---------------------------------------------------------------------------
# Tiny model + tokenizer
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tok():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1}
    for ch in sorted(pre_tokenizers.ByteLevel.alphabet()):
        vocab[ch] = len(vocab)
    t = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>",
                                   eos_token="<eos>", padding_side="left")
    fast.add_tokens(["<think>", "</think>"])
    fast.chat_template = CHAT_TEMPLATE
    return fast


@pytest.fixture(scope="module")
def model(tok):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id, bos_token_id=tok.eos_token_id,
        initializer_range=0.5)
    return LlamaForCausalLM(cfg, ).float().eval()


def _toggle():
    return cap.ThinkingToggle("kwarg", "enable_thinking")


# ---------------------------------------------------------------------------
# Prompt fitting (B4)
# ---------------------------------------------------------------------------

def test_fit_prompt_short_prompt_is_untouched(tok):
    render = lambda raw: _toggle().render(tok, raw, False)  # noqa: E731
    raw = "What is 2 + 2?"
    fitted, truncated = cap.fit_prompt(raw, render, tok, 1024, False)
    assert fitted == render(raw) and not truncated
    # Same tokens the old `truncation=True, max_length=...` call produced.
    old = tok([fitted], truncation=True, max_length=1024,
              add_special_tokens=False).input_ids[0]
    assert tok([fitted], add_special_tokens=False).input_ids[0] == old


def test_fit_prompt_keeps_template_and_drops_start_of_question(tok):
    render = lambda raw: _toggle().render(tok, raw, False)  # noqa: E731
    suffix = "<|assistant|><think></think>"
    raw = "PREAMBLE " + "filler " * 200 + "QUESTION: what is 2 + 2?"
    limit = 120
    fitted, truncated = cap.fit_prompt(raw, render, tok, limit, False)
    assert truncated
    assert len(tok(fitted, add_special_tokens=False).input_ids) <= limit
    assert fitted.endswith(suffix), "generation prompt must survive"
    assert fitted.startswith("<|user|>")
    assert "QUESTION: what is 2 + 2?" in fitted and "PREAMBLE" not in fitted
    # What the old right-truncation did to the same prompt: header gone.
    old_ids = tok(render(raw), truncation=True, max_length=limit,
                  add_special_tokens=False).input_ids
    assert not tok.decode(old_ids).endswith(suffix)


def test_fit_prompt_refuses_when_template_alone_overflows(tok):
    render = lambda raw: _toggle().render(tok, raw, False)  # noqa: E731
    with pytest.raises(ValueError, match="max-prompt-len"):
        cap.fit_prompt("hello " * 50, render, tok, 10, False)


# ---------------------------------------------------------------------------
# run_capture end to end
# ---------------------------------------------------------------------------

QUESTIONS = [
    "Q0 add 1",
    "Q1 add two numbers please",
    "Q2 " + "very long context " * 40 + "then answer",   # overflows the limit
    "Q3 a",
    "Q4 subtract",
    "Q5 multiply these",
    "Q6 divide",
]
# Thinking-on texts scripted per row, by the question's own index: closed
# reasoning, unclosed reasoning, no reasoning.
SCRIPTED_ON = {0: "<think>work</think> 42", 1: "<think>still working 42",
               2: "42 straight away"}


@pytest.fixture
def fake_task(monkeypatch):
    mod = types.ModuleType("tasks.fakecap")
    mod.load_fake = lambda split: [
        {"question": q, "answer": "42", "difficulty": "easy"} for q in QUESTIONS]
    mod.format_prompt = lambda q: q
    mod.is_correct = lambda gen, ans: ans in gen
    monkeypatch.setitem(sys.modules, "tasks.fakecap", mod)
    monkeypatch.setitem(cap._TASK_REGISTRY, "fakecap",
                        ("load_fake", cap._correct_str, "test"))
    return mod


def _args(out_dir, **over):
    argv = ["--task", "gsm8k", "--model", "tiny-test", "--out-dir", str(out_dir),
            "--batch-size", "3", "--max-prompt-len", "80",
            "--max-response-len", "4", "--max-response-len-thinking", "5",
            "--chat-template", "--shard-index", "1", "--shard-count", "2",
            "--capture-logprobs"]
    args = cap.build_parser().parse_args(argv)
    args.task, args.split = "fakecap", "test"
    for k, v in over.items():
        setattr(args, k, v)
    return args


def _patch(monkeypatch, tok, model, script_on=True):
    monkeypatch.setattr(cap, "load_model", lambda name, attn: (tok, model))
    real = cap.generate_batch

    def scripted(model_, tok_, input_ids, attention_mask, max_new_tokens):
        texts, trunc, lengths, seqs = real(model_, tok_, input_ids,
                                           attention_mask, max_new_tokens)
        if script_on and max_new_tokens == 5:          # the thinking-on pass
            prompts = tok_.batch_decode(input_ids, skip_special_tokens=True)
            # A truncated prompt (Q2) has lost its "Q<n>" prefix: script it
            # as "no reasoning".
            texts = [SCRIPTED_ON[int(p.split("<|user|>Q")[1][0]) % 3]
                     if "<|user|>Q" in p else SCRIPTED_ON[2] for p in prompts]
        return texts, trunc, lengths, seqs

    monkeypatch.setattr(cap, "generate_batch", scripted)


def _meta(out_dir, shard=1):
    path = Path(out_dir) / f"meta.shard{shard:02d}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_capture_end_to_end(tmp_path, monkeypatch, tok, model, fake_task,
                                caplog):
    _patch(monkeypatch, tok, model)
    # A random model never emits EOS, so every off-response hits the cap;
    # lift the gate here (the quarantine path has its own test).
    monkeypatch.setattr(cap, "OFF_TRUNCATION_LIMIT", 1.01)
    caplog.set_level(logging.INFO, logger="capture")

    assert cap.run_capture(_args(tmp_path)) == 0
    rows = _meta(tmp_path)

    # B6: shard 1 of 2 holds split indices 1, 3, 5.
    assert [r["dataset_index"] for r in rows] == [0, 1, 2]
    assert [r["dataset_index_global"] for r in rows] == [1, 3, 5]
    assert [r["question"] for r in rows] == [QUESTIONS[i] for i in (1, 3, 5)]

    # B7 / B7b: Q1 unclosed, Q3 closed (3 % 3 = 0), Q5 no reasoning (5 % 3 = 2).
    by_q = {r["question"][:2]: r for r in rows}
    assert by_q["Q1"]["unclosed_think_on"] and not by_q["Q1"]["correct_on"]
    assert "42" in by_q["Q1"]["answer_on"]           # would have graded right
    assert by_q["Q3"]["has_reasoning_on"] and by_q["Q3"]["correct_on"]
    assert not by_q["Q5"]["has_reasoning_on"] and by_q["Q5"]["correct_on"]
    assert any("reasoning trace: 2/3" in m and "below 80%" in m
               for m in caplog.messages)

    # B1 bookkeeping.
    for r in rows:
        assert r["confidence_version"] == 2
        assert r["confidence_off"]["mean_logprob"] is not None
        assert r["prompt_truncated"] is False

    # B5: per-shard config with provenance and summary; legacy config.json too.
    shard_cfg = json.loads((tmp_path / "config.shard01.json").read_text())
    assert shard_cfg["status"] == "complete"
    assert shard_cfg["shard_index"] == 1
    assert shard_cfg["max_response_len"] == 4
    assert "git_commit" in shard_cfg and "git_dirty" in shard_cfg
    assert shard_cfg["confidence_version"] == 2
    assert shard_cfg["summary"]["n_samples"] == 3
    assert shard_cfg["summary"]["unclosed_think_on"] == 1
    assert abs(shard_cfg["summary"]["has_reasoning_on_rate"] - 2 / 3) < 1e-3
    legacy = json.loads((tmp_path / "config.json").read_text())
    assert legacy["task"] == "fakecap" and "shard_index" not in legacy


def test_run_capture_flags_overlong_prompt(tmp_path, monkeypatch, tok, model,
                                           fake_task):
    _patch(monkeypatch, tok, model)
    monkeypatch.setattr(cap, "OFF_TRUNCATION_LIMIT", 1.01)
    assert cap.run_capture(_args(tmp_path, shard_index=0)) == 0
    rows = _meta(tmp_path, shard=0)              # split indices 0, 2, 4, 6
    flags = {r["dataset_index_global"]: r["prompt_truncated"] for r in rows}
    assert flags == {0: False, 2: True, 4: False, 6: False}
    assert all(r["prompt_len_off"] <= 80 and r["prompt_len_on"] <= 80
               for r in rows)


def test_confidence_does_not_depend_on_batch_padding(tmp_path, monkeypatch,
                                                     tok, model, fake_task):
    """The call site, not just the function: the same rows captured at batch
    size 1 (no padding) and batch size 3 (left-padded) get the same
    confidence."""
    _patch(monkeypatch, tok, model, script_on=False)
    monkeypatch.setattr(cap, "OFF_TRUNCATION_LIMIT", 1.01)
    cap.run_capture(_args(tmp_path / "b1", batch_size=1))
    cap.run_capture(_args(tmp_path / "b3", batch_size=3))
    alone, batched = _meta(tmp_path / "b1"), _meta(tmp_path / "b3")
    assert len({r["prompt_len_off"] for r in batched}) > 1, "need padding"
    for a, b in zip(alone, batched):
        assert a["response_off"] == b["response_off"]
        for k in ("mean_logprob", "min_logprob", "mean_entropy"):
            assert abs(a["confidence_off"][k] - b["confidence_off"][k]) < 1e-4, k


def test_off_truncation_quarantine_still_fires(tmp_path, monkeypatch, tok,
                                               model, fake_task):
    _patch(monkeypatch, tok, model)
    assert cap.run_capture(_args(tmp_path)) == 1
    assert not (tmp_path / "meta.shard01.jsonl").exists()
    assert (tmp_path / "meta.shard01.jsonl.quarantined").exists()
    assert (tmp_path / "TRUNCATION_FAILURE.shard01.json").exists()
    shard_cfg = json.loads((tmp_path / "config.shard01.json").read_text())
    assert shard_cfg["status"] == "quarantined"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

#!/usr/bin/env python3
"""
test_readout.py — scripts/capture_readout.py, the D0 decision readout.

Two kinds of test:

  * No download, no GPU: the tiny byte-level tokenizer and random Llama of
    test_capture.py. They pin the plumbing -- answer specs, option-token maps
    and the in-context single-token check, restriction to option tokens with
    renormalisation, shapes, a left-padded row read exactly as if alone, and
    an end-to-end run over a fake BBH capture.
  * The real Qwen3-8B tokenizer, only if it is already in the local HF cache
    (skipped otherwise, never downloaded; tokenizer only, no model). Every
    option the v3 readout uses must be ONE token in its actual context, and
    the leading-space variants must be handled: " A" after "Answer:" (the
    no-space "A" merges into ":A" there), "A" after "Answer: (".

    python -m pytest tests/test_readout.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import scripts.capture_inference_thinking as cap  # noqa: E402
import scripts.capture_readout as ro  # noqa: E402
from test_capture import model, tok  # noqa: E402,F401

MMLU_Q = ("What is 2 + 2?\n\n(A) 3\n(B) 4\n(C) 5\n(D) 22")
BBH_LETTER_Q = ("Which is a fruit?\nOptions:\n(A) carrot\n(B) apple\n(C) leek")
BBH_YES_Q = ("Is the following sentence plausible? \"Messi kicked the ball.\"")
BBH_COUNT_Q = "I have a cat and two dogs. How many animals do I have?"


def _render(tok):  # noqa: F811
    toggle = cap.ThinkingToggle("kwarg", "enable_thinking")
    return lambda raw: toggle.render(tok, raw, False)


# ---------------------------------------------------------------------------
# Answer specs (pure)
# ---------------------------------------------------------------------------

def test_answer_spec_mmlu_pro_reads_only_the_listed_letters():
    s = ro.answer_spec("mmlu_pro", {"question": MMLU_Q, "answer": "B"})
    assert s["covered"] and s["options"] == ["A", "B", "C", "D"]
    assert s["prefix"] == "Answer:" and s["gold"] == "B" and s["kind"] == "letter"


def test_answer_spec_bbh_kinds():
    letter = ro.answer_spec("bbh", {"question": BBH_LETTER_Q, "answer": "(B)",
                                    "sample_id": "bbh-ruin_names-3"})
    assert letter["covered"] and letter["options"] == ["A", "B", "C"]
    assert letter["gold"] == "B" and letter["prefix"] == "Answer: ("
    assert letter["family"] == "ruin_names"

    yes = ro.answer_spec("bbh", {"question": BBH_YES_Q, "answer": "yes",
                                 "sample_id": "bbh-sports_understanding-0"})
    assert yes["covered"] and yes["options"] == ["Yes", "No"] and yes["gold"] == "Yes"

    valid = ro.answer_spec("bbh", {"question": "x", "answer": "invalid",
                                   "sample_id": "bbh-formal_fallacies-1"})
    assert valid["options"] == ["valid", "invalid"] and valid["gold"] == "invalid"

    for sub in ("object_counting", "multistep_arithmetic_two", "word_sorting",
                "dyck_languages"):
        s = ro.answer_spec("bbh", {"question": BBH_COUNT_Q, "answer": "3",
                                   "sample_id": f"bbh-{sub}-0"})
        assert not s["covered"] and "free-form" in s["exclude_reason"]


def test_answer_spec_refuses_a_gold_outside_the_options():
    s = ro.answer_spec("mmlu_pro", {"question": MMLU_Q, "answer": "H"})
    assert not s["covered"] and "not among options" in s["exclude_reason"]
    with pytest.raises(ValueError, match="verify format"):
        ro.answer_spec("gsm8k", {"question": "1+1", "answer": "2"})


def test_surface_forms_put_the_context_spacing_first():
    assert ro.surface_forms("A", "Answer:") == [" A", "A"]
    assert ro.surface_forms("A", "Answer: (") == ["A", " A"]
    assert ro.surface_forms("valid", "Answer:") == [" valid", "valid", " Valid", "Valid"]


# ---------------------------------------------------------------------------
# Option tokens and scoring
# ---------------------------------------------------------------------------

def test_option_token_map_checks_tokens_in_context(tok):  # noqa: F811
    render = _render(tok)
    # Byte-level tokenizer, no merges: "A" is one token, " A" is two.
    ctx = ro.build_readout_prompt("Q", render, ro.PREFIX_PAREN)
    m = ro.option_token_map(tok, ctx, ["A", "B", "C"], ro.PREFIX_PAREN)
    assert m["ok"] and m["forms"] == {"A": ["A"], "B": ["B"], "C": ["C"]}
    assert len({i for v in m["token_ids"].values() for i in v}) == 3
    for label, ids in m["token_ids"].items():
        assert tok.decode(ids) == label

    ctx = ro.build_readout_prompt("Q", render, ro.PREFIX_COLON)
    bad = ro.option_token_map(tok, ctx, ["Yes", "No"], ro.PREFIX_COLON)
    assert not bad["ok"] and bad["failed"] == ["Yes", "No"]


def test_score_options_restricts_and_renormalises():
    v = np.log(np.full(10, 1e-6))
    v[3], v[4], v[7] = np.log(0.30), np.log(0.10), np.log(0.20)   # 7 = B's 2nd form
    v[0] = np.log(0.39)                                          # not an option
    s = ro.score_options(v, {"A": [3], "B": [4, 7]}, ["A", "B"])
    assert s["probs"] == pytest.approx([0.5, 0.5])
    assert s["option_mass"] == pytest.approx(0.6)
    v[7] = np.log(0.02)
    s = ro.score_options(v, {"A": [3], "B": [4, 7]}, ["A", "B"])
    assert s["argmax"] == "A" and s["p_max"] == pytest.approx(0.30 / 0.42)
    assert sum(s["probs"]) == pytest.approx(1.0)
    assert 0 < s["entropy_norm"] < 1


def test_default_layers_middle_and_last_four():
    assert ro.default_layers(36) == [9, 18, 27, 33, 34, 35, 36]
    assert ro.default_layers(2) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Model pass (tiny random Llama)
# ---------------------------------------------------------------------------

def test_decide_pass_shapes_and_padding_invariance(tok, model):  # noqa: F811
    render = _render(tok)
    prompts = [ro.build_readout_prompt(q, render, ro.PREFIX_PAREN)
               for q in ("short", "a much longer question " * 4)]
    lp, hid = ro.decide_pass(model, tok, prompts, [0, 1, 2])
    assert lp.shape == (2, model.config.vocab_size)
    assert hid.shape == (2, 3, model.config.hidden_size) and hid.dtype == np.float16
    assert np.exp(lp).sum(axis=1) == pytest.approx([1.0, 1.0], abs=1e-4)
    for i, p in enumerate(prompts):
        lp1, hid1 = ro.decide_pass(model, tok, [p], [0, 1, 2])
        assert np.allclose(lp1[0], lp[i], atol=1e-4), "padded row differs from alone"
        assert np.allclose(hid1[0], hid[i], atol=1e-2)


def _fake_bbh_capture(path: Path, tok) -> list[dict]:  # noqa: F811
    import tasks.bbh as bbh

    render = _render(tok)
    rows = [
        {"sample_id": "bbh-ruin_names-0", "question": BBH_LETTER_Q, "answer": "(B)"},
        {"sample_id": "bbh-date_understanding-1",
         "question": "Today is?\nOptions:\n(A) Monday\n(B) Friday", "answer": "(A)"},
        {"sample_id": "bbh-object_counting-0", "question": BBH_COUNT_Q, "answer": "3"},
        {"sample_id": "bbh-sports_understanding-0", "question": BBH_YES_Q,
         "answer": "yes"},
        {"sample_id": "bbh-hyperbaton-2", "question": "Pick\nOptions:\n(A) x\n(B) y",
         "answer": "(A)", "prompt_hash": "0" * 64},
    ]
    for r in rows:
        r.setdefault("prompt_hash", cap.sha256(render(
            bbh.PROMPT_TEMPLATE_V1.format(question=r["question"].strip()))))
        r["prompt_len_off"] = 50
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({
        "model_name": "tiny-test", "task": "bbh", "chat_template": True,
        "max_prompt_len": 1024}))
    (path / "meta.shard00.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    return rows


def test_run_end_to_end_on_a_fake_capture(tmp_path, monkeypatch, tok, model):  # noqa: F811
    cap_dir, out = tmp_path / "bbh_thinking_tiny", tmp_path / "bbh_readout_tiny"
    rows = _fake_bbh_capture(cap_dir, tok)
    monkeypatch.setattr(cap, "load_model", lambda name, attn: (tok, model))
    assert ro.main(["--capture-dir", str(cap_dir), "--out-dir", str(out),
                    "--batch-size", "2", "--layers", "1,2"]) == 0

    recs = [json.loads(x) for x in (out / "readout.jsonl").read_text().splitlines()]
    assert [r["sample_id"] for r in recs] == [r["sample_id"] for r in rows]
    by = {r["sample_id"]: r for r in recs}
    covered = [r for r in recs if r["covered"]]
    assert [r["sample_id"] for r in covered] == ["bbh-ruin_names-0",
                                                 "bbh-date_understanding-1"]
    for r in covered:
        assert len(r["probs"]) == len(r["options"]) == r["n_options"]
        assert sum(r["probs"]) == pytest.approx(1.0)
        assert r["argmax"] in r["options"] and r["p_max"] == max(r["probs"])
        assert r["s1_correct"] == (r["argmax"] == r["gold"])
        assert 0 < r["option_mass"] <= 1 and r["prompt_source"] == "PROMPT_TEMPLATE_V1"
    assert by["bbh-object_counting-0"]["exclude_reason"].startswith("free-form")
    # " Yes" is two byte tokens here: flagged and left unscored, never guessed.
    assert "option_not_single_token" in by["bbh-sports_understanding-0"]["flags"]
    assert by["bbh-hyperbaton-2"]["exclude_reason"] == "prompt_hash_mismatch"

    with np.load(out / "decide_hidden.npz") as z:
        assert z["hidden"].shape == (2, 2, model.config.hidden_size)
        assert list(z["layers"]) == [1, 2]
        assert list(z["sample_id"]) == ["bbh-ruin_names-0", "bbh-date_understanding-1"]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_rows"] == 5 and summary["n_covered"] == 2
    assert summary["coverage"] == pytest.approx(0.4)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["hidden_layers"] == [1, 2] and cfg["thinking"] == "off"

    # The batch-1 run reads the same probabilities: batching changes nothing.
    out1 = tmp_path / "b1"
    assert ro.main(["--capture-dir", str(cap_dir), "--out-dir", str(out1),
                    "--batch-size", "1", "--layers", "1,2"]) == 0
    recs1 = [json.loads(x) for x in (out1 / "readout.jsonl").read_text().splitlines()]
    for a, b in zip(recs, recs1):
        if a["covered"]:
            assert a["probs"] == pytest.approx(b["probs"], abs=1e-4)


# ---------------------------------------------------------------------------
# Two answer formats: "Answer: Yes" and "Answer: (Yes"
# ---------------------------------------------------------------------------

def test_score_options_marginalises_over_the_paren_format():
    direct = np.log(np.full(8, 1e-9))
    direct[1], direct[2], direct[5] = np.log(0.01), np.log(0.03), np.log(0.90)  # 5 = " ("
    after = np.log(np.full(8, 1e-9))
    after[3], after[4] = np.log(0.6), np.log(0.2)
    s = ro.score_options(direct, {"Yes": [1], "No": [2]}, ["Yes", "No"],
                         (direct[5], after, {"Yes": [3], "No": [4]}))
    yes, no = 0.01 + 0.9 * 0.6, 0.03 + 0.9 * 0.2
    assert s["probs"] == pytest.approx([yes / (yes + no), no / (yes + no)])
    assert s["argmax"] == "Yes"            # the direct path alone would say "No"
    assert s["option_mass"] == pytest.approx(yes + no)
    assert s["mass_direct"] == pytest.approx(0.04)
    assert s["mass_paren"] == pytest.approx(0.72) and s["p_bridge"] == pytest.approx(0.9)
    alone = ro.score_options(direct, {"Yes": [1], "No": [2]}, ["Yes", "No"])
    assert alone["argmax"] == "No" and "mass_paren" not in alone


@pytest.fixture(scope="module")
def word_tok():
    """Byte-level tokenizer plus whole-word tokens, so " (", " Yes" and "Yes"
    are single tokens and the two-format path can be exercised end to end."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from test_capture import CHAT_TEMPLATE

    vocab = {"<pad>": 0, "<eos>": 1}
    for ch in sorted(pre_tokenizers.ByteLevel.alphabet()):
        vocab[ch] = len(vocab)
    t = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>",
                                   eos_token="<eos>", padding_side="left")
    fast.add_tokens(["<think>", "</think>", " (", " Yes", " No", "Yes", "No"])
    fast.chat_template = CHAT_TEMPLATE
    return fast


def test_alt_path_end_to_end_reads_both_formats(tmp_path, monkeypatch, word_tok):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(word_tok), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, pad_token_id=0, eos_token_id=1, bos_token_id=1,
        initializer_range=0.5)).float().eval()
    render = _render(word_tok)
    spec = ro.answer_spec("bbh", {"question": BBH_YES_Q, "answer": "No",
                                  "sample_id": "bbh-sports_understanding-0"})
    assert spec["alt_prefix"] == ro.PREFIX_PAREN
    prompt = ro.build_readout_prompt("Q", render, spec["prefix"])
    alt = ro.alt_path(word_tok, prompt, "Q", render, spec)
    assert alt is not None and word_tok.decode([alt["bridge_id"]]) == " ("
    assert alt["prompt"] == prompt + " ("

    import tasks.bbh as bbh
    cap_dir = tmp_path / "bbh_thinking_tiny"
    cap_dir.mkdir()
    rows = [{"sample_id": f"bbh-sports_understanding-{k}", "question": q, "answer": a,
             "prompt_len_off": 10,
             "prompt_hash": cap.sha256(render(bbh.PROMPT_TEMPLATE_V1.format(question=q)))}
            for k, (q, a) in enumerate([(BBH_YES_Q, "yes"), ("Is it plausible?", "no")])]
    (cap_dir / "config.json").write_text(json.dumps(
        {"model_name": "tiny-test", "task": "bbh", "chat_template": True,
         "max_prompt_len": 1024}))
    (cap_dir / "meta.shard00.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(cap, "load_model", lambda name, attn: (word_tok, m))
    out = tmp_path / "out"
    assert ro.main(["--capture-dir", str(cap_dir), "--out-dir", str(out),
                    "--batch-size", "2", "--layers", "2"]) == 0
    recs = [json.loads(x) for x in (out / "readout.jsonl").read_text().splitlines()]
    assert all(r["covered"] and "alt_path_unavailable" not in r["flags"] for r in recs)
    for r in recs:
        assert r["option_mass"] == pytest.approx(r["mass_direct"] + r["mass_paren"])
        assert r["mass_paren"] == pytest.approx(r["p_bridge"] * r["mass_after_paren"])
        assert sum(r["probs"]) == pytest.approx(1.0)
    # The same numbers by hand, from two plain forward passes.
    r = recs[0]
    name, raw = ro.match_prompt(rows[0], ro.prompt_candidates(bbh), render)
    p1 = ro.build_readout_prompt(raw, render, ro.PREFIX_COLON)
    lp1, _ = ro.decide_pass(m, word_tok, [p1], [2])
    lp2, _ = ro.decide_pass(m, word_tok, [p1 + " ("], [2])
    ids = {w: word_tok.convert_tokens_to_ids(w) for w in ("Yes", "No")}
    sp = {w: word_tok(p1 + " " + w, add_special_tokens=False).input_ids[-1]
          for w in ("Yes", "No")}
    bridge = word_tok(p1 + " (", add_special_tokens=False).input_ids[-1]
    raw_p = {w: np.exp(lp1[0][sp[w]]) + np.exp(lp1[0][ids[w]])
             + np.exp(lp1[0][bridge]) * (np.exp(lp2[0][ids[w]]) + np.exp(lp2[0][sp[w]]))
             for w in ("Yes", "No")}
    tot = raw_p["Yes"] + raw_p["No"]
    assert r["probs"] == pytest.approx([raw_p["Yes"] / tot, raw_p["No"] / tot], rel=1e-3)


# ---------------------------------------------------------------------------
# Real Qwen3-8B tokenizer (local cache only)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qwen_tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qwen/Qwen3-8B tokenizer not in the local HF cache: {exc}")


@pytest.mark.parametrize("task,row", [
    ("mmlu_pro", {"question": "Q?\n\n" + "\n".join(f"({chr(65 + i)}) opt {i}"
                                                  for i in range(10)), "answer": "J"}),
    ("bbh", {"question": BBH_LETTER_Q + "\n(D) d\n(E) e\n(F) f\n(G) g\n(H) h\n"
                                        "(I) i\n(J) j\n(K) k\n(L) l\n(M) m\n(N) n\n"
                                        "(O) o\n(P) p\n(Q) q\n(R) r",
             "answer": "(R)", "sample_id": "bbh-ruin_names-0"}),
    ("bbh", {"question": BBH_YES_Q, "answer": "No",
             "sample_id": "bbh-sports_understanding-0"}),
    ("bbh", {"question": "not ( True ) and ( True ) is", "answer": "False",
             "sample_id": "bbh-boolean_expressions-0"}),
    ("bbh", {"question": "Is the argument valid?\nOptions:\n- valid\n- invalid",
             "answer": "invalid", "sample_id": "bbh-formal_fallacies-0"}),
])
def test_qwen3_options_are_single_tokens_in_context(qwen_tok, task, row):
    import tasks.bbh as bbh
    import tasks.mmlu_pro as mmlu

    spec = ro.answer_spec(task, row)
    assert spec["covered"]
    toggle = cap.detect_thinking_toggle(qwen_tok, "Qwen/Qwen3-8B")
    render = lambda raw: toggle.render(qwen_tok, raw, False)  # noqa: E731
    raw = (mmlu.format_prompt(row["question"]) if task == "mmlu_pro"
           else bbh.PROMPT_TEMPLATE_V1.format(question=row["question"]))
    prompt = ro.build_readout_prompt(raw, render, spec["prefix"])
    assert prompt.endswith("</think>\n\n" + spec["prefix"])
    m = ro.option_token_map(qwen_tok, prompt, spec["options"], spec["prefix"])
    assert m["ok"], m["failed"]
    ctx = qwen_tok(prompt, add_special_tokens=False).input_ids
    for label in spec["options"]:
        primary = m["forms"][label][0]
        # Leading-space variant: the primary form carries the context's spacing.
        assert primary == ((" " + label) if spec["prefix"].endswith(":") else label)
        ids = qwen_tok(prompt + primary, add_special_tokens=False).input_ids
        assert ids[:len(ctx)] == ctx and len(ids) == len(ctx) + 1
        assert ids[-1] == m["token_ids"][label][0]
        assert qwen_tok.decode(m["token_ids"][label][:1]) == primary
    if spec["prefix"] == ro.PREFIX_COLON and spec["kind"] == "letter":
        # "Answer:" + "A" re-tokenizes the colon (":A"), so the no-space form
        # must NOT be read as an option there.
        assert all(forms == [" " + o] for o, forms in m["forms"].items())
    if spec["kind"] in ("yes_no", "true_false", "valid_invalid"):
        assert all(len(f) >= 2 for f in m["forms"].values())   # both spacings
    all_ids = [i for v in m["token_ids"].values() for i in v]
    assert len(all_ids) == len(set(all_ids))
    if spec["prefix"] == ro.PREFIX_COLON:
        # The parenthesised format: " (" is one token after "Answer:", and
        # every option is one token after it.
        alt = ro.alt_path(qwen_tok, prompt, raw, render, spec)
        assert alt is not None and qwen_tok.decode([alt["bridge_id"]]) == " ("
        assert all(f[0] == o for o, f in alt["tmap"]["forms"].items())
    else:
        assert spec["alt_prefix"] is None

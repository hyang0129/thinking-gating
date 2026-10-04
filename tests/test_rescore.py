#!/usr/bin/env python3
"""
test_rescore.py — scripts/rescore_confidence.py and the confidence readers.

No download, no GPU: the tiny byte-level tokenizer and random Llama of
test_capture.py, with generation restricted to printable ASCII so the stored
response text re-tokenizes to exactly the tokens that were generated, and the
EOS row of the output head tied to one character's so that some rows stop on
EOS and others hit the cap.

What must hold:
  * a capture made by the fixed scorer at batch size 1 re-scores to its own
    stored confidence (the re-score reproduces the capture's prompt, answer
    tokens and scorer);
  * a capture made at batch size 3 with the v1 scorer (pads attended) stores
    different values, and re-scoring it recovers the batch-1 values;
  * rows that cannot be rebuilt are kept and flagged, never dropped;
  * readers prefer the sidecar, refuse a partial one, and say what they used.

    python -m pytest tests/test_rescore.py -q
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
sys.path.insert(0, str(ROOT / "scripts"))
import scripts.capture_inference_thinking as cap  # noqa: E402
import scripts.rescore_confidence as rs  # noqa: E402
from baseline_confidence import confidence_features  # noqa: E402
from test_capture import _args, _meta, _patch, fake_task, tok  # noqa: E402,F401
from utils.capture_io import load_meta, resolve_confidence  # noqa: E402

KEYS = ("mean_logprob", "min_logprob", "mean_entropy")
ALLOWED = "abcdefghijklmnopqrstuvwxyz0123456789 .,"


@pytest.fixture(scope="module")
def model(tok):  # noqa: F811
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id, bos_token_id=tok.eos_token_id,
        initializer_range=0.5)
    m = LlamaForCausalLM(cfg).float().eval()
    allowed = {tok(c, add_special_tokens=False).input_ids[0] for c in ALLOWED}
    allowed.add(tok.eos_token_id)
    # ASCII only: arbitrary byte tokens decode to U+FFFD and cannot round-trip.
    m.generation_config.suppress_tokens = [i for i in range(len(tok))
                                           if i not in allowed]
    # EOS shadows "z", slightly louder: a row stops where it would write a z
    # (one of the three shard-1 rows does; the other two hit the cap).
    z = tok("z", add_special_tokens=False).input_ids[0]
    with torch.no_grad():
        m.lm_head.weight[tok.eos_token_id] = m.lm_head.weight[z] * 1.05
    return m


def v1_sequence_confidence(model, sequences, prompt_len, n_new, pad_id=None,
                           attention_mask=None):
    """The pre-fix scorer: slice from column 0, no mask, pads attended."""
    import torch.nn.functional as F

    out = []
    for i in range(sequences.shape[0]):
        n = int(n_new[i])
        seq = sequences[i:i + 1, :prompt_len + n]
        with torch.no_grad():
            logits = model(input_ids=seq, use_cache=False).logits[0].float()
        pred = logits[prompt_len - 1:prompt_len + n - 1]
        tgt = seq[0, prompt_len:prompt_len + n]
        lp = -F.cross_entropy(pred, tgt, reduction="none")
        lsm = F.log_softmax(pred, dim=-1)
        out.append({"mean_logprob": float(lp.mean()), "min_logprob": float(lp.min()),
                    "mean_entropy": float(-(lsm.exp() * lsm).sum(-1).mean())})
    return out


def _capture(path, monkeypatch, tok, model, batch_size, scorer=None):  # noqa: F811
    monkeypatch.setattr(cap, "OFF_TRUNCATION_LIMIT", 1.01)
    with monkeypatch.context() as m:
        if scorer is not None:
            m.setattr(cap, "sequence_confidence", scorer)
        assert cap.run_capture(_args(path, batch_size=batch_size,
                                     max_response_len=12)) == 0
    return _meta(path)


def _rescore(path, **kw):
    argv = ["--capture-dir", str(path), "--log-every", "0"]
    argv += [f"--{k.replace('_', '-')}" for k, v in kw.items() if v is True]
    assert rs.main(argv) == 0
    return [json.loads(line) for line in
            (Path(path) / "confidence_off_v2.shard01.jsonl").read_text().splitlines()]


def _close(a, b, tol=1e-4):
    return all(abs(a[k] - b[k]) <= tol for k in KEYS)


@pytest.fixture
def setup(monkeypatch, tok, model, fake_task):  # noqa: F811
    _patch(monkeypatch, tok, model, script_on=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# Equality with the capture's own scorer
# ---------------------------------------------------------------------------

def test_rescore_of_batch1_capture_equals_stored(tmp_path, setup, tok, model):  # noqa: F811
    rows = _capture(tmp_path, setup, tok, model, batch_size=1)
    # Both stopping paths are exercised: EOS (re-appended) and the cap.
    assert {r["truncated_off"] for r in rows} == {True, False}
    side = _rescore(tmp_path)
    assert [s["sample_id"] for s in side] == [r["sample_id"] for r in rows]
    for r, s in zip(rows, side):
        assert s["confidence_version"] == 2 and s["flags"] == []
        assert s["prompt_hash_match"] and s["prompt_source"] == "format_prompt"
        assert s["token_delta"] == 0 and s["n_tokens_rescored"] == r["n_tokens_off"]
        assert s["prompt_len_rescored"] == r["prompt_len_off"]
        assert s["eos_appended"] == (not r["truncated_off"])
        assert _close(s["confidence_off"], r["confidence_off"]), (s, r)
    summary = json.loads((tmp_path / "confidence_off_v2.summary.json").read_text())
    assert summary["shards"][0]["n_scored"] == len(rows)
    assert summary["flags_total"] == {} and "git_commit" in summary
    # The meta shard is never rewritten.
    assert _meta(tmp_path) == rows


def test_rescore_recovers_from_padded_batch_corruption(tmp_path, setup, tok, model):  # noqa: F811
    clean = _capture(tmp_path / "b1", setup, tok, model, batch_size=1)
    bad = _capture(tmp_path / "b3", setup, tok, model, batch_size=3,
                   scorer=v1_sequence_confidence)
    assert [r["response_off"] for r in bad] == [r["response_off"] for r in clean]
    padded = [i for i, r in enumerate(bad)
              if r["prompt_len_off"] < max(x["prompt_len_off"] for x in bad)]
    assert padded, "need at least one left-padded row"
    # Teeth: the v1 scorer's stored values are wrong on the padded rows ...
    for i in padded:
        assert not _close(bad[i]["confidence_off"], clean[i]["confidence_off"],
                          tol=1e-3)
    # ... and the re-score gets the clean values back on every row.
    side = _rescore(tmp_path / "b3")
    for s, c in zip(side, clean):
        assert s["flags"] == [] and _close(s["confidence_off"], c["confidence_off"])


# ---------------------------------------------------------------------------
# Integrity flags: rows are kept, never dropped
# ---------------------------------------------------------------------------

def _edit_meta(path, edit):
    meta_path = Path(path) / "meta.shard01.jsonl"
    rows = [json.loads(line) for line in meta_path.read_text().splitlines()]
    edit(rows)
    meta_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def test_flags_on_unrebuildable_rows(tmp_path, setup, tok, model):  # noqa: F811
    _capture(tmp_path, setup, tok, model, batch_size=1)

    def edit(rows):
        rows[0]["question"] += " (edited after capture)"     # prompt unknowable
        rows[1]["response_off"] += " extra words"            # 12 more tokens
        rows[2]["prompt_len_off"] += 1                       # recorded != rebuilt
    rows = _edit_meta(tmp_path, edit)
    side = _rescore(tmp_path)
    assert len(side) == len(rows)
    assert side[0]["flags"] == ["prompt_hash_mismatch"]
    assert side[0]["confidence_off"] == {k: None for k in KEYS}
    assert side[1]["flags"] == ["token_count_mismatch"] and side[1]["token_delta"] == 12
    assert side[1]["confidence_off"]["mean_logprob"] is not None
    assert side[2]["flags"] == ["prompt_len_mismatch"]
    # A reader fails loudly on the null rather than scoring a zero.
    conf, info = resolve_confidence(load_meta(tmp_path), tmp_path)
    assert info["flags"] == {"prompt_hash_mismatch": 1, "token_count_mismatch": 1,
                             "prompt_len_mismatch": 1}
    with pytest.raises(SystemExit, match="missing"):
        confidence_features(rows, conf)


def test_check_only_writes_nothing(tmp_path, setup, tok, model, monkeypatch, capsys):  # noqa: F811
    _capture(tmp_path, setup, tok, model, batch_size=1)
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        classmethod(lambda cls, *a, **k: tok))
    assert rs.main(["--capture-dir", str(tmp_path), "--check-only",
                    "--log-every", "0"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["check_only"] and report["flags_total"] == {}
    assert not list(tmp_path.glob("confidence_off_v2*"))


def test_complete_sidecar_is_skipped_unless_overwrite(tmp_path, setup, tok, model):  # noqa: F811
    _capture(tmp_path, setup, tok, model, batch_size=1)
    _rescore(tmp_path)
    sidecar = tmp_path / "confidence_off_v2.shard01.jsonl"
    sidecar.write_text(sidecar.read_text().replace('"confidence_version": 2',
                                                   '"confidence_version": 2, "x": 1'))
    _rescore(tmp_path)
    assert '"x": 1' in sidecar.read_text()
    _rescore(tmp_path, overwrite=True)
    assert '"x": 1' not in sidecar.read_text()


def test_legacy_prompt_template_is_matched_by_hash(tok):  # noqa: F811
    import types
    mod = types.ModuleType("legacy")
    mod.format_prompt = lambda q: f"NEW {q}"
    mod.PROMPT_TEMPLATE_V1 = "OLD {question}"
    render = lambda raw: cap.ThinkingToggle("kwarg", "enable_thinking").render(  # noqa: E731
        tok, raw, False)
    cands = rs.prompt_candidates(mod)
    assert [c[0] for c in cands] == ["format_prompt", "PROMPT_TEMPLATE_V1"]
    row = {"question": " q? ", "prompt_hash": cap.sha256(render("OLD q?"))}
    assert rs.match_prompt(row, cands, render) == ("PROMPT_TEMPLATE_V1", "OLD q?")
    row["prompt_hash"] = cap.sha256(render("NEW  q? "))
    assert rs.match_prompt(row, cands, render)[0] == "format_prompt"
    row["prompt_hash"] = "0" * 64
    assert rs.match_prompt(row, cands, render) == (None, None)


def test_bbh_keeps_the_v3_prompt_as_a_candidate():
    import tasks.bbh as bbh
    names = [c[0] for c in rs.prompt_candidates(bbh)]
    assert names == ["format_prompt", "PROMPT_TEMPLATE_V1"]


def test_legacy_rows_reproduce_right_truncation(tok):  # noqa: F811
    """Rows without `prompt_truncated` predate B4: the capture let the
    tokenizer right-truncate. Newer rows went through fit_prompt."""
    render = lambda raw: cap.ThinkingToggle("kwarg", "enable_thinking").render(  # noqa: E731
        tok, raw, False)
    raw = "PREAMBLE " + "filler " * 50 + "ask"
    legacy = rs.prompt_ids(raw, {}, render, tok, 40, False)
    assert legacy == tok(render(raw), add_special_tokens=False).input_ids[:40]
    fitted = rs.prompt_ids(raw, {"prompt_truncated": True}, render, tok, 40, False)
    assert len(fitted) <= 40 and fitted != legacy
    assert tok.decode(fitted).endswith("<|assistant|><think></think>")
    assert not tok.decode(legacy).endswith("<think></think>")


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def test_resolve_confidence_sources(tmp_path, setup, tok, model):  # noqa: F811
    bad = _capture(tmp_path, setup, tok, model, batch_size=3,
                   scorer=v1_sequence_confidence)
    # Make the rows look like a pre-fix capture: no confidence_version.
    _edit_meta(tmp_path, lambda rows: [r.pop("confidence_version") for r in rows])
    meta = load_meta(tmp_path)

    conf, info = resolve_confidence(meta, tmp_path, "auto")       # no sidecar yet
    assert info["source"] == "stored confidence_off"
    assert info["confidence_version_counts"] == {"1": len(meta)}
    assert conf == [r["confidence_off"] for r in bad]
    with pytest.raises(FileNotFoundError):
        resolve_confidence(meta, tmp_path, "v2")

    side = _rescore(tmp_path)
    for source in ("auto", "v2"):
        conf, info = resolve_confidence(meta, tmp_path, source)
        assert info["source"].startswith("confidence_off_v2")
        assert info["confidence_version"] == [2]
        assert conf == [s["confidence_off"] for s in side]
    conf, info = resolve_confidence(meta, tmp_path, "stored")      # forced old
    assert conf == [r["confidence_off"] for r in bad]

    # A partial sidecar is refused, not mixed with stored values.
    sidecar = tmp_path / "confidence_off_v2.shard01.jsonl"
    sidecar.write_text("".join(sidecar.read_text().splitlines(True)[:-1]))
    with pytest.raises(ValueError, match="incomplete"):
        resolve_confidence(meta, tmp_path, "auto")


def test_baseline_confidence_records_its_source(tmp_path, setup, tok, model):  # noqa: F811
    """End to end through baseline_confidence.main on a capture big enough to
    split: the output JSON names the sidecar, and --confidence-source stored
    reproduces the old numbers."""
    import baseline_confidence as bc

    rows = _capture(tmp_path, setup, tok, model, batch_size=3,
                    scorer=v1_sequence_confidence)
    # Replicate the 3 rows to 60 with distinct ids and a balanced target, in
    # the meta shard and (after re-scoring) the sidecar alike.
    big = []
    for k in range(20):
        for j, r in enumerate(rows):
            big.append(dict(r, sample_id=f"{r['sample_id']}-{k}",
                            correct_off=bool((k + j) % 2)))
    (tmp_path / "meta.shard01.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in big))
    rng = np.random.default_rng(0)
    side = [dict(sample_id=r["sample_id"], confidence_version=2, flags=[],
                 confidence_off={k: float(-abs(rng.normal())) for k in KEYS})
            for r in big]
    (tmp_path / "confidence_off_v2.shard01.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in side))
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps(
        {"sample_id": r["sample_id"], "correct_off": r["correct_off"],
         "correct_on": True, "label": "not_helped"}) + "\n" for r in big))

    outs = {}
    for source in ("auto", "stored"):
        out = tmp_path / f"conf_{source}.json"
        assert bc.main(["--capture-dir", str(tmp_path), "--labels", str(labels),
                        "--target", "needs_thinking", "--seeds", "42",
                        "--confidence-source", source, "--out-file", str(out)]) == 0
        outs[source] = json.loads(out.read_text())
    assert outs["auto"]["confidence_source"][0]["source"].startswith("confidence_off_v2")
    assert outs["stored"]["confidence_source"][0]["confidence_version_counts"] == {"2": 60}
    a = outs["auto"]["baselines"]["mean_logprob"]["test_auroc"]["mean"]
    s = outs["stored"]["baselines"]["mean_logprob"]["test_auroc"]["mean"]
    assert a != s
    # n_tokens_off does not depend on the confidence source.
    assert (outs["auto"]["baselines"]["n_tokens_off"]["test_auroc"]["mean"]
            == outs["stored"]["baselines"]["n_tokens_off"]["test_auroc"]["mean"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

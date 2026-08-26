"""End-to-end test of the label -> train -> transfer pipeline.

Builds a synthetic capture whose "thinking helped" outcome is genuinely
encoded in one layer's prefill state, then runs the three real scripts over it
as subprocesses. The point is to catch shape, alignment, and CLI breakage
before a multi-hour GPU capture, and to prove the probe recovers signal that is
actually there (and does not invent signal that is not).

Needs numpy, torch, and scikit-learn — run it in the project venv:
    .venv/bin/python tests/test_pipeline.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

SCRIPTS = _PROJECT_ROOT / "scripts"
N_LAYERS, HIDDEN, SIGNAL_LAYER = 6, 64, 3


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    proc = subprocess.run([sys.executable, str(SCRIPTS / script), *args],
                          capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise AssertionError(f"{script} failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return proc


def make_capture(out_dir: Path, n: int, seed: int, *, task: str = "gsm8k",
                 n_shards: int = 2, signal: float = 1.6, base_rate: float = 0.25):
    """Write a synthetic sharded capture with a known planted signal.

    `helped` is drawn from a logistic function of one direction in
    SIGNAL_LAYER, so a probe on that layer should score well above chance and a
    probe on an unrelated layer should not.
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    direction = np.random.default_rng(0).normal(size=HIDDEN)  # shared across tasks
    direction /= np.linalg.norm(direction)

    acts = rng.normal(size=(n, N_LAYERS, HIDDEN)).astype(np.float16)
    projection = acts[:, SIGNAL_LAYER, :].astype(np.float32) @ direction
    projection = (projection - projection.mean()) / projection.std()
    bias = np.log(base_rate / (1 - base_rate))
    p_helped = 1 / (1 + np.exp(-(signal * projection + bias)))
    helped = rng.random(n) < p_helped

    # "helped" means wrong without thinking and right with it. Everything else
    # is split across the other three outcomes so the graded label has content.
    correct_off = np.where(helped, False, rng.random(n) < 0.55)
    correct_on = np.where(helped, True, correct_off & (rng.random(n) > 0.08))
    difficulty = rng.choice(["easy", "medium", "hard"], size=n)

    rows = []
    for i in range(n):
        rows.append({
            "sample_id": f"{task}-{i}", "dataset_index": i,
            "prompt_hash": f"hash{i}", "question": f"q{i}", "answer": f"a{i}",
            "difficulty": str(difficulty[i]),
            "response_off": "...", "response_on": "...",
            "answer_off": "...", "answer_on": "...",
            "correct_off": bool(correct_off[i]), "correct_on": bool(correct_on[i]),
            "truncated_off": False, "truncated_on": bool(i % 37 == 0),
            "n_tokens_off": 50, "n_tokens_on": 300,
            "prompt_len_off": 40, "prompt_len_on": 42,
        })

    bounds = np.array_split(np.arange(n), n_shards)
    for shard, idx in enumerate(bounds):
        with open(out_dir / f"meta.shard{shard:02d}.jsonl", "w") as fh:
            for i in idx:
                fh.write(json.dumps(rows[i]) + "\n")
        for mode in ("off", "on"):
            np.savez_compressed(
                out_dir / f"activations_thinking_{mode}.shard{shard:02d}.npz",
                activations=acts[idx])
    (out_dir / "config.json").write_text(json.dumps({
        "model_name": "synthetic/test-model", "task": task, "split": "test",
        "num_layers": N_LAYERS - 1, "hidden_dim": HIDDEN, "shard_count": n_shards,
    }))
    return rows


# ---------------------------------------------------------------------------

def test_capture_io_concatenates_shards_in_order():
    from utils.capture_io import align_labels, load_capture
    with tempfile.TemporaryDirectory() as tmp:
        cap = Path(tmp) / "cap"
        rows = make_capture(cap, n=50, seed=1, n_shards=3)
        meta, acts = load_capture(cap, mode="off")
        assert len(meta) == 50 and acts.shape == (50, N_LAYERS, HIDDEN), acts.shape
        assert [m["sample_id"] for m in meta] == [r["sample_id"] for r in rows]

        # Labels in a shuffled order must still map to the right activation rows.
        shuffled = [{"sample_id": r["sample_id"]} for r in reversed(rows)]
        idx = align_labels(meta, shuffled)
        assert [meta[i]["sample_id"] for i in idx] == [s["sample_id"] for s in shuffled]


def test_capture_io_rejects_misaligned_shards():
    from utils.capture_io import load_capture
    with tempfile.TemporaryDirectory() as tmp:
        cap = Path(tmp) / "cap"
        make_capture(cap, n=20, seed=1, n_shards=2)
        (cap / "activations_thinking_off.shard01.npz").unlink()
        try:
            load_capture(cap, mode="off")
        except ValueError as exc:
            assert "shard" in str(exc), exc
            return
        raise AssertionError("accepted a capture with a missing activation shard")


def test_labels_match_the_definition():
    with tempfile.TemporaryDirectory() as tmp:
        cap, labels = Path(tmp) / "cap", Path(tmp) / "labels.jsonl"
        rows = make_capture(cap, n=200, seed=2)
        _run("generate_labels.py", "--capture-dir", str(cap), "--out-file", str(labels))

        produced = [json.loads(x) for x in labels.read_text().splitlines()]
        assert len(produced) == len(rows)
        by_id = {r["sample_id"]: r for r in rows}
        for lab in produced:
            src = by_id[lab["sample_id"]]
            expected = ("helped" if (not src["correct_off"] and src["correct_on"])
                        else "not_helped")
            assert lab["label"] == expected, (lab, src)
            if src["correct_off"] and not src["correct_on"]:
                assert lab["graded_label"] == "hurt"

        summary = json.loads((labels.with_suffix(".summary.json")).read_text())
        assert 0.05 < summary["base_rate_helped"] < 0.6, summary["base_rate_helped"]
        assert summary["oracle_accuracy"] >= summary["accuracy_thinking_on"]


def test_drop_truncated_removes_flagged_rows():
    with tempfile.TemporaryDirectory() as tmp:
        cap, labels = Path(tmp) / "cap", Path(tmp) / "labels.jsonl"
        make_capture(cap, n=100, seed=3)
        _run("generate_labels.py", "--capture-dir", str(cap),
             "--out-file", str(labels), "--drop-truncated")
        produced = [json.loads(x) for x in labels.read_text().splitlines()]
        assert produced and not any(lab["truncated_on"] for lab in produced)


def test_probe_recovers_planted_signal():
    """The probe should find real signal — and the metrics should be honest."""
    with tempfile.TemporaryDirectory() as tmp:
        cap, labels = Path(tmp) / "cap", Path(tmp) / "labels.jsonl"
        out = Path(tmp) / "probe"
        make_capture(cap, n=600, seed=4)
        _run("generate_labels.py", "--capture-dir", str(cap), "--out-file", str(labels))
        _run("run_experiment.py", "--capture-dir", str(cap), "--labels", str(labels),
             "--out-dir", str(out), "--method", "logreg",
             "--layer", str(SIGNAL_LAYER), "--seeds", "42", "1", "2")

        summary = json.loads((out / "aggregate_metrics.json").read_text())
        agg = summary["aggregate"]
        assert agg["test_auroc"]["mean"] > 0.65, agg["test_auroc"]
        assert agg["test_auroc"]["ci"][0] < agg["test_auroc"]["mean"] < agg["test_auroc"]["ci"][1]
        assert len(summary["per_seed"]) == 3

        # Routing can never beat knowing the answer in advance.
        assert agg["test_routed_accuracy"]["mean"] <= agg["oracle"]["mean"] + 1e-9
        for seed_metrics in summary["per_seed"]:
            test = seed_metrics["test"]
            assert 0.0 <= test["auroc"] <= 1.0
            assert test["n_positive"] > 0
            assert set(test["by_difficulty"]) <= {"easy", "medium", "hard"}

        for seed in (42, 1, 2):
            assert (out / f"seed_{seed}" / "checkpoint.json").exists()


def test_probe_finds_nothing_in_a_signal_free_layer():
    """The honesty check: a layer with no signal must not score well."""
    with tempfile.TemporaryDirectory() as tmp:
        cap, labels = Path(tmp) / "cap", Path(tmp) / "labels.jsonl"
        out = Path(tmp) / "probe"
        make_capture(cap, n=600, seed=5)
        _run("generate_labels.py", "--capture-dir", str(cap), "--out-file", str(labels))
        _run("run_experiment.py", "--capture-dir", str(cap), "--labels", str(labels),
             "--out-dir", str(out), "--method", "logreg",
             "--layer", str((SIGNAL_LAYER + 2) % N_LAYERS), "--seeds", "42", "1", "2")
        agg = json.loads((out / "aggregate_metrics.json").read_text())["aggregate"]
        assert agg["test_auroc"]["mean"] < 0.65, (
            f"found AUROC {agg['test_auroc']['mean']:.3f} in a layer with no planted "
            "signal — the evaluation is leaking")


def test_mlp_method_and_layer_sweep():
    with tempfile.TemporaryDirectory() as tmp:
        cap, labels = Path(tmp) / "cap", Path(tmp) / "labels.jsonl"
        out = Path(tmp) / "probe"
        make_capture(cap, n=400, seed=6)
        _run("generate_labels.py", "--capture-dir", str(cap), "--out-file", str(labels))
        _run("run_experiment.py", "--capture-dir", str(cap), "--labels", str(labels),
             "--out-dir", str(out), "--method", "mlp", "--layer-sweep",
             "--seeds", "42", "--max-epochs", "40", "--patience", "8")

        sweep = json.loads((out / "layer_sweep.json").read_text())
        assert len(sweep["val_auroc_by_layer"]) == N_LAYERS
        assert sweep["selected_on"] == "validation"
        # The chosen layer must be the validation argmax, not the test argmax.
        best_val = max(sweep["val_auroc_by_layer"],
                       key=lambda k: sweep["val_auroc_by_layer"][k])
        assert str(sweep["best_layer"]) == str(best_val), sweep
        assert (out / "seed_42" / "checkpoint.json").exists()
        ckpt = json.loads((out / "seed_42" / "checkpoint.json").read_text())
        assert ckpt["method"] == "mlp" and "state_dict" in ckpt and "scaler" in ckpt


def test_pooling_multiple_captures():
    """Two captures pooled into one training set, reported per task."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cap_a, lab_a = tmp_path / "a", tmp_path / "a.jsonl"
        cap_b, lab_b = tmp_path / "b", tmp_path / "b.jsonl"
        out = tmp_path / "pooled"
        make_capture(cap_a, n=300, seed=20, task="gsm8k")
        make_capture(cap_b, n=300, seed=21, task="lsat")
        for cap, lab in ((cap_a, lab_a), (cap_b, lab_b)):
            _run("generate_labels.py", "--capture-dir", str(cap), "--out-file", str(lab))

        _run("run_experiment.py", "--capture-dir", str(cap_a), str(cap_b),
             "--labels", str(lab_a), str(lab_b), "--out-dir", str(out),
             "--method", "logreg", "--layer", str(SIGNAL_LAYER), "--seeds", "42")

        summary = json.loads((out / "aggregate_metrics.json").read_text())
        assert summary["n_samples"] == 600, summary["n_samples"]
        assert sorted(summary["task"]) == ["gsm8k", "lsat"], summary["task"]
        assert len(summary["capture_dirs"]) == 2
        # The planted direction is shared, so pooling must still find it.
        assert summary["aggregate"]["test_auroc"]["mean"] > 0.6


def test_pooling_rejects_mismatched_counts():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cap, lab, out = tmp_path / "a", tmp_path / "a.jsonl", tmp_path / "o"
        make_capture(cap, n=100, seed=22)
        _run("generate_labels.py", "--capture-dir", str(cap), "--out-file", str(lab))
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "run_experiment.py"),
             "--capture-dir", str(cap), str(cap), "--labels", str(lab),
             "--out-dir", str(out), "--method", "logreg", "--seeds", "42"],
            capture_output=True, text=True, timeout=300)
        assert proc.returncode == 1, "accepted mismatched capture/label counts"
        assert "pair" in (proc.stdout + proc.stderr)


def test_transfer_applies_source_probe_without_refitting():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src_cap, src_labels = tmp_path / "gsm8k", tmp_path / "gsm8k.jsonl"
        tgt_cap, tgt_labels = tmp_path / "lsat", tmp_path / "lsat.jsonl"
        out = tmp_path / "probe"
        transfer_file = tmp_path / "transfer.json"

        # Same planted direction in both tasks: transfer should hold up.
        make_capture(src_cap, n=600, seed=7, task="gsm8k")
        make_capture(tgt_cap, n=300, seed=8, task="lsat")
        for cap, labels in ((src_cap, src_labels), (tgt_cap, tgt_labels)):
            _run("generate_labels.py", "--capture-dir", str(cap), "--out-file", str(labels))
        _run("run_experiment.py", "--capture-dir", str(src_cap), "--labels", str(src_labels),
             "--out-dir", str(out), "--method", "logreg",
             "--layer", str(SIGNAL_LAYER), "--seeds", "42", "1")

        _run("eval_transfer.py",
             "--probe", str(out / "seed_42" / "checkpoint.json"),
             str(out / "seed_1" / "checkpoint.json"),
             "--capture-dir", str(tgt_cap), "--labels", str(tgt_labels),
             "--source-metrics", str(out / "aggregate_metrics.json"),
             "--out-file", str(transfer_file))

        result = json.loads(transfer_file.read_text())
        assert result["target_task"] == "lsat"
        assert len(result["per_probe"]) == 2
        assert result["transfer_auroc"]["mean"] > 0.6, result["transfer_auroc"]
        assert "auroc_drop_pp" in result and "verdict" in result
        assert result["verdict"] in (
            "strong transfer", "moderate — likely label distribution shift",
            "transfer failure (format overfitting)")


def test_mlp_checkpoint_round_trips_through_transfer():
    """An MLP checkpoint must rebuild to the same scores it produced at training."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cap, labels, out = tmp_path / "cap", tmp_path / "l.jsonl", tmp_path / "probe"
        make_capture(cap, n=400, seed=9)
        _run("generate_labels.py", "--capture-dir", str(cap), "--out-file", str(labels))
        _run("run_experiment.py", "--capture-dir", str(cap), "--labels", str(labels),
             "--out-dir", str(out), "--method", "mlp", "--layer", str(SIGNAL_LAYER),
             "--seeds", "42", "--max-epochs", "40", "--patience", "8")

        from scripts.eval_transfer import load_probe
        from scripts.run_experiment import Standardizer, select_layer
        from utils.capture_io import load_capture

        score_fn, ckpt = load_probe(out / "seed_42" / "checkpoint.json")
        _, acts = load_capture(cap, mode="off")
        X = Standardizer.from_state(ckpt["scaler"]).transform(
            select_layer(acts, ckpt["layer"]))
        scores = score_fn(X)
        assert scores.shape == (acts.shape[0],)
        assert np.all((scores >= 0) & (scores <= 1))
        assert scores.std() > 1e-4, "rebuilt probe outputs a constant — weights lost"


# ---------------------------------------------------------------------------

def _main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = []
    for name, fn in tests:
        started = time.monotonic()
        try:
            fn()
            print(f"  PASS  {name}  ({time.monotonic() - started:.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001 — this is the test runner
            import traceback
            print(f"  FAIL  {name}: {exc}", flush=True)
            traceback.print_exc()
            failures.append(name)
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())

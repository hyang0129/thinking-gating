"""Unit tests for utils/metrics.py, on synthetic data with known answers, plus
end-to-end checks of the scripts that report those metrics.

    .venv/bin/python -m pytest tests/test_metrics.py -q
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils import metrics as M  # noqa: E402

SCRIPTS = _PROJECT_ROOT / "scripts"


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------

def test_auroc_known_values():
    y = np.array([0, 0, 1, 1])
    assert M.auroc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert M.auroc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert M.auroc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert np.isnan(M.auroc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_auroc_matches_sklearn_with_ties():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 200)
        s = rng.integers(0, 6, 200).astype(float)          # heavy ties
        assert M.auroc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_bootstrap_auroc_ci_brackets_estimate_and_shrinks_with_n():
    rng = np.random.default_rng(1)

    def sample(n):
        y = rng.integers(0, 2, n)
        return y, y + rng.normal(scale=1.5, size=n)
    y, s = sample(200)
    small = M.bootstrap_auroc_ci(y, s)
    assert small["ci"][0] < M.auroc(y, s) < small["ci"][1]
    y, s = sample(3000)
    big = M.bootstrap_auroc_ci(y, s, n_boot=500)
    assert (big["ci"][1] - big["ci"][0]) < (small["ci"][1] - small["ci"][0]) / 2


def test_bootstrap_is_seeded_and_reproducible():
    rng = np.random.default_rng(2)
    y, s = rng.integers(0, 2, 100), rng.normal(size=100)
    assert M.bootstrap_auroc_ci(y, s) == M.bootstrap_auroc_ci(y, s)


def test_stratified_bootstrap_keeps_every_stratum():
    y = np.r_[np.zeros(50), np.ones(5)]
    strata = np.r_[np.zeros(50), np.ones(5)]
    # Unstratified resamples of 5/55 positives sometimes lose the class;
    # stratified ones never do, so every replicate is defined.
    res = M.bootstrap_ci(lambda yy: yy.mean(), y, n_boot=300, strata=strata)
    assert res["n_boot"] == 300
    assert res["ci"][0] == res["ci"][1] == pytest.approx(5 / 55)


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------

def _two_scorers(n, rng, gap=0.6):
    """Scorer a has more signal than b, and their noise is shared (correlated)."""
    y = rng.integers(0, 2, n)
    shared = rng.normal(size=n)
    a = (1.0 + gap) * y + shared + 0.3 * rng.normal(size=n)
    b = 1.0 * y + shared + 0.3 * rng.normal(size=n)
    return y, a, b


def test_paired_bootstrap_ci_covers_true_difference():
    rng = np.random.default_rng(3)
    y, a, b = _two_scorers(200_000, rng)
    true_diff = M.auroc(y, a) - M.auroc(y, b)              # population value
    hits, trials = 0, 30
    for t in range(trials):
        ys, as_, bs = _two_scorers(300, np.random.default_rng(100 + t))
        res = M.paired_bootstrap_auroc_diff(ys, as_, bs, n_boot=300, seed=t)
        hits += res["ci"][0] <= true_diff <= res["ci"][1]
    assert hits / trials >= 0.8, f"coverage {hits}/{trials}"


def test_paired_interval_is_narrower_than_the_marginals():
    rng = np.random.default_rng(4)
    y, a, b = _two_scorers(400, rng)
    paired = M.paired_bootstrap_auroc_diff(y, a, b)
    ca, cb = M.bootstrap_auroc_ci(y, a)["ci"], M.bootstrap_auroc_ci(y, b)["ci"]
    assert paired["ci"][1] - paired["ci"][0] < (ca[1] - ca[0]) + (cb[1] - cb[0])
    assert paired["ci"][0] > 0 and paired["p_le_zero"] < 0.05


def test_paired_bootstrap_of_identical_scorers_is_exactly_zero():
    rng = np.random.default_rng(5)
    y, s = rng.integers(0, 2, 100), rng.normal(size=100)
    res = M.paired_bootstrap_auroc_diff(y, s, s, n_boot=200)
    assert res["estimate"] == 0.0 and res["ci"] == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Routed-accuracy curve and nAUC
# ---------------------------------------------------------------------------

def _outcomes(n, rng, p_off=0.6, p_resc=0.5, p_hurt=0.05):
    off = rng.random(n) < p_off
    on = np.where(off, rng.random(n) > p_hurt, rng.random(n) < p_resc)
    return off, on


def test_curve_endpoints_are_never_and_always_think():
    rng = np.random.default_rng(6)
    off, on = _outcomes(300, rng)
    curve = M.routed_accuracy_curve(off, on, rng.normal(size=300))
    assert curve["accuracy"][0] == pytest.approx(off.mean())
    assert curve["accuracy"][-1] == pytest.approx(on.mean())
    assert len(curve["fraction"]) == 301


def test_perfect_router_has_nauc_one_and_anti_router_negative():
    rng = np.random.default_rng(7)
    off, on = _outcomes(500, rng)
    gain = on.astype(float) - off.astype(float)
    for lo, hi in ((0.0, 0.5), (0.0, 1.0), (0.1, 0.3)):
        assert M.routed_nauc(off, on, gain, lo, hi)["nauc"] == pytest.approx(1.0)
    assert M.routed_nauc(off, on, -gain)["nauc"] < -0.5


def test_random_router_sits_on_the_chance_line():
    rng = np.random.default_rng(8)
    off, on = _outcomes(400, rng)
    naucs = [M.routed_nauc(off, on, rng.normal(size=400))["nauc"] for _ in range(200)]
    assert abs(np.mean(naucs)) < 0.03, np.mean(naucs)
    # A constant score is one tie block: its curve IS the chance line.
    flat = M.routed_nauc(off, on, np.zeros(400))
    assert flat["nauc"] == pytest.approx(0.0, abs=1e-12)
    assert flat["mean_accuracy"] == pytest.approx(flat["random_mean_accuracy"])


def test_curve_ties_are_deterministic_and_order_free():
    """B11: discrete scores (token counts) must not depend on row order."""
    rng = np.random.default_rng(9)
    off, on = _outcomes(200, rng)
    s = rng.integers(0, 4, 200).astype(float)
    a = M.routed_accuracy_curve(off, on, s)
    perm = rng.permutation(200)
    b = M.routed_accuracy_curve(off[perm], on[perm], s[perm])
    np.testing.assert_allclose(a["accuracy"], b["accuracy"], atol=1e-12)
    assert len(a["vertex_fraction"]) == len(np.unique(s)) + 1


def test_curve_area_is_exact():
    rng = np.random.default_rng(10)
    off, on = _outcomes(97, rng)
    curve = M.routed_accuracy_curve(off, on, rng.integers(0, 7, 97).astype(float))
    xs = np.linspace(0.0, 0.5, 200_001)
    ys = M.curve_accuracy_at(curve, xs)
    numeric = float(np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2))
    exact = M.curve_area(curve["vertex_fraction"], curve["vertex_accuracy"], 0.0, 0.5)
    assert exact == pytest.approx(numeric, abs=1e-8)


def test_min_fraction_is_read_off_the_exact_curve():
    """B12: the oracle needs exactly (rescued - hurt) escalations, not a grid point."""
    off = np.array([0] * 7 + [1] * 93, dtype=bool)
    on = off.copy()
    on[:5] = True                       # 5 rescued
    on[7:9] = False                     # 2 hurt
    curve = M.oracle_curve(off, on)
    assert M.min_fraction_for_accuracy(curve, on.mean()) == pytest.approx(0.03)
    # The 21-point grid can only say 0.05.
    grid = M.cost_accuracy_curve(off, on, on.astype(float) - off.astype(float))
    assert len(grid) == 21 and min(p["fraction_routed"] for p in grid
                                   if p["accuracy"] >= on.mean() - 1e-9) == 0.05


def test_random_and_oracle_at_matched_rate():
    off = np.array([0, 0, 1, 1], dtype=bool)
    on = np.array([1, 0, 1, 1], dtype=bool)
    assert M.random_accuracy_at(off, on, 0.5) == pytest.approx(0.625)
    assert M.curve_accuracy_at(M.oracle_curve(off, on), 0.25) == pytest.approx(0.75)


def test_paired_nauc_difference_oracle_vs_random():
    rng = np.random.default_rng(11)
    off, on = _outcomes(400, rng)
    gain = on.astype(float) - off.astype(float)
    res = M.paired_bootstrap_nauc_diff(off, on, gain, rng.normal(size=400), n_boot=200)
    assert res["estimate"] > 0.8 and res["ci"][0] > 0.5


# ---------------------------------------------------------------------------
# Within-group AUROC
# ---------------------------------------------------------------------------

def _group_detector(n_groups=10, per=60, seed=12):
    """Labels depend only on the group; the score is the group's base rate."""
    rng = np.random.default_rng(seed)
    rates = np.linspace(0.1, 0.9, n_groups)
    g = np.repeat(np.arange(n_groups), per)
    y = (rng.random(len(g)) < rates[g]).astype(int)
    return y, rates[g], g, rng


def test_pure_group_detector_is_at_chance_within_groups():
    y, s, g, rng = _group_detector()
    assert M.auroc(y, s) > 0.75
    st = M.within_group_stats(y, s, g)
    assert st["auroc"] == pytest.approx(0.5)            # all within-group pairs tie
    assert st["n_informative_groups"] == 10
    noisy = s + 1e-3 * rng.normal(size=len(s))
    assert abs(M.within_group_auroc(y, noisy, g) - 0.5) < 0.06


def test_within_group_keeps_real_query_level_signal():
    y, s, g, rng = _group_detector(seed=13)
    signal = y + 0.5 * rng.normal(size=len(y))
    assert M.within_group_auroc(y, signal, g) > 0.8


def test_one_group_degenerates_to_plain_auroc():
    """B8: a key that puts every row in one group silently returns plain AUROC."""
    rng = np.random.default_rng(14)
    y, s = rng.integers(0, 2, 200), rng.normal(size=200)
    st = M.within_group_stats(y, s, ["math500-test"] * 200)
    assert st["auroc"] == pytest.approx(M.auroc(y, s))
    assert st["n_groups"] == 1 and st["n_informative_groups"] == 1


def test_within_group_requires_explicit_groups():
    with pytest.raises(ValueError):
        M.within_group_auroc([0, 1], [0.1, 0.2], None)
    with pytest.raises(ValueError):
        M.within_group_auroc([0, 1], [0.1, 0.2], ["a"])


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def test_wilson_ci_known_values():
    lo, hi = M.wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    assert M.wilson_ci(0, 20)[0] == 0.0
    assert all(np.isnan(M.wilson_ci(0, 0)))


def test_precision_threshold_picks_lowest_qualifying_cutoff():
    conf = [0.9, 0.8, 0.7, 0.6]
    correct = [1, 1, 0, 1]
    t = M.precision_threshold(conf, correct, 0.95)
    assert t["tau"] == 0.8 and t["precision"] == 1.0 and t["coverage"] == 0.5
    assert M.precision_threshold(conf, correct, 0.75)["tau"] == 0.6
    assert M.precision_threshold([0.9, 0.8], [0, 0], 0.95)["tau"] == float("inf")


def test_confident_error_leakage_counts():
    conf = np.array([0.99, 0.97, 0.60, 0.98, 0.50, 0.40])
    correct = np.array([1, 0, 0, 0, 0, 1])
    depth = np.array([1, 1, 1, 3, 3, 3])
    cel = M.confident_error_leakage(conf, correct, tau=0.95, groups=depth)
    assert cel["overall"]["n_wrong"] == 4 and cel["overall"]["n_leaked"] == 2
    assert cel["by_group"]["1"]["cel"] == pytest.approx(0.5)
    assert cel["by_group"]["3"]["cel"] == pytest.approx(0.5)
    assert list(cel["by_group"]) == ["1", "3"]


def test_signed_overconfidence_calibrated_vs_overconfident():
    rng = np.random.default_rng(15)
    conf = rng.uniform(0.5, 1.0, 20_000)
    calibrated = rng.random(20_000) < conf
    over = rng.random(20_000) < (conf - 0.2)
    c = M.signed_overconfidence(conf, calibrated, n_boot=200)["overall"]
    assert abs(c["overconfidence"]) < 0.01 and c["ci"][0] < 0 < c["ci"][1]
    o = M.signed_overconfidence(conf, over, groups=np.where(conf > 0.75, "hi", "lo"),
                                n_boot=200)
    assert o["overall"]["overconfidence"] == pytest.approx(0.2, abs=0.01)
    assert o["overall"]["ci"][0] > 0.15
    assert set(o["by_group"]) == {"hi", "lo"}


def test_reliability_bins():
    rng = np.random.default_rng(16)
    conf = rng.uniform(0, 1, 50_000)
    correct = rng.random(50_000) < conf
    rel = M.reliability_bins(conf, correct, n_bins=10)
    assert sum(b["n"] for b in rel["bins"]) == 50_000
    assert rel["ece"] < 0.01
    q = M.reliability_bins(conf, correct, n_bins=5, strategy="quantile")
    assert all(abs(b["n"] - 10_000) <= 1 for b in q["bins"])
    edge = M.reliability_bins([1.0, 0.0], [1, 0], n_bins=4)
    assert edge["bins"][-1]["n"] == 1 and edge["bins"][0]["n"] == 1
    by = M.reliability_by_group(conf, correct, np.where(conf > 0.5, 2, 1))
    assert list(by) == ["1", "2"]


# ---------------------------------------------------------------------------
# Scripts: synthetic capture end-to-end
# ---------------------------------------------------------------------------

N_LAYERS, HIDDEN, SIGNAL_LAYER = 7, 32, 3
WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
         "nu xi omicron pi rho sigma tau upsilon").split()


def write_capture(out_dir: Path, task: str, n: int, seed: int, *, subtasks=None):
    """A sharded capture with a planted needs_thinking signal in SIGNAL_LAYER,
    real-looking question text, thinking-off confidence, and sample ids of the
    task modules' `<task>-<group>-<index>` form."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    direction = np.random.default_rng(0).normal(size=HIDDEN)
    direction /= np.linalg.norm(direction)
    acts = rng.normal(size=(n, N_LAYERS, HIDDEN)).astype(np.float32)
    with np.errstate(all="ignore"):   # macOS BLAS sets spurious FP flags; see test_pipeline
        proj = acts[:, SIGNAL_LAYER, :] @ direction
    assert np.isfinite(proj).all()
    proj = (proj - proj.mean()) / proj.std()
    wrong_off = rng.random(n) < 1 / (1 + np.exp(-(1.8 * proj - 0.3)))
    correct_off = ~wrong_off
    correct_on = np.where(correct_off, rng.random(n) > 0.05, rng.random(n) < 0.55)
    rows = []
    for i in range(n):
        group = subtasks[i % len(subtasks)] if subtasks else "test"
        rows.append({
            "sample_id": f"{task}-{group}-{i}", "dataset_index": i,
            "prompt_hash": f"h{task}{i}",
            "question": " ".join(rng.choice(WORDS, size=int(rng.integers(5, 25)))),
            "answer": "x", "difficulty": str(rng.choice(["easy", "medium", "hard"])),
            "response_off": "...", "response_on": "...",
            "answer_off": "...", "answer_on": "...",
            "correct_off": bool(correct_off[i]), "correct_on": bool(correct_on[i]),
            "truncated_off": False, "truncated_on": False,
            "confidence_off": {"mean_logprob": float(-abs(rng.normal())),
                               "min_logprob": float(-abs(rng.normal()) * 3),
                               "mean_entropy": float(abs(rng.normal()))},
            "n_tokens_off": int(rng.integers(20, 200)), "n_tokens_on": 400,
            "prompt_len_off": 40, "prompt_len_on": 42,
        })
    for shard, idx in enumerate(np.array_split(np.arange(n), 2)):
        with open(out_dir / f"meta.shard{shard:02d}.jsonl", "w") as fh:
            for i in idx:
                fh.write(json.dumps(rows[i]) + "\n")
        for mode in ("off", "on"):
            np.savez_compressed(out_dir / f"activations_thinking_{mode}.shard{shard:02d}.npz",
                                activations=acts[idx].astype(np.float16))
    (out_dir / "config.json").write_text(json.dumps({
        "model_name": "synthetic/test-model", "task": task, "split": "test",
        "num_layers": N_LAYERS - 1, "hidden_dim": HIDDEN, "shard_count": 2}))


def _run(script: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run([sys.executable, str(SCRIPTS / script), *args],
                          capture_output=True, text=True, timeout=600)
    if check and proc.returncode != 0:
        raise AssertionError(f"{script} failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return proc


@pytest.fixture(scope="module")
def bbh_like(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bbh")
    cap, lab = tmp / "cap", tmp / "labels.jsonl"
    write_capture(cap, "bbh", 480, seed=21, subtasks=["boolean_expressions",
                                                      "navigate", "web_of_lies"])
    _run("generate_labels.py", "--capture-dir", str(cap), "--out-file", str(lab))
    return tmp, cap, lab


def test_within_group_script_requires_an_informative_key(bbh_like):
    tmp, cap, lab = bbh_like
    proc = subprocess.run([sys.executable, str(SCRIPTS / "within_group_auroc.py"),
                           "--capture-dir", str(cap), "--labels", str(lab),
                           "--target", "needs_thinking"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0, "ran without an explicit --group-key"

    out = tmp / "wg.json"
    _run("within_group_auroc.py", "--capture-dir", str(cap), "--labels", str(lab),
         "--group-key", "sample_id:middle", "--target", "needs_thinking",
         "--seeds", "42", "1", "--n-boot", "100", "--out-file", str(out))
    res = json.loads(out.read_text())
    assert res["n_groups"] == 3 and res["n_informative_groups"] == 3
    assert res["layer"] == N_LAYERS // 2                 # middle layer from the shape
    assert len(res["test_split_groups"]) == 2

    # A key with one value for every row is exactly the B8 failure: refuse it.
    proc = _run("within_group_auroc.py", "--capture-dir", str(cap), "--labels", str(lab),
                "--group-key", "answer", "--target", "needs_thinking", check=False)
    assert proc.returncode == 2 and "contain both classes" in proc.stderr
    proc = _run("within_group_auroc.py", "--capture-dir", str(cap), "--labels", str(lab),
                "--group-key", "no_such_field", "--target", "needs_thinking", check=False)
    assert proc.returncode == 2


def test_probe_and_baselines_pair_on_identical_rows(bbh_like):
    tmp, cap, lab = bbh_like
    metrics = tmp / "metrics"
    probe = tmp / "probe"
    _run("run_experiment.py", "--capture-dir", str(cap), "--labels", str(lab),
         "--out-dir", str(probe), "--method", "logreg", "--target", "needs_thinking",
         "--layer", str(SIGNAL_LAYER), "--seeds", "42", "1")
    agg = json.loads((probe / "aggregate_metrics.json").read_text())
    seed = agg["per_seed"][0]["test"]
    assert -1.0 <= seed["routed_nauc"]["escalation_0_50"]["nauc"] <= 1.0
    assert seed["routed_curve"]["fraction"][0] == 0.0
    assert seed["routed_curve"]["fraction"][-1] == 1.0
    assert "random_at_matched_rate" in seed and "oracle_at_matched_rate" in seed
    assert seed["oracle_at_matched_rate"] >= seed["routed_accuracy"] - 1e-9
    assert "test_routed_nauc" in agg["aggregate"]
    preds = json.loads((probe / "predictions.json").read_text())
    assert set(preds["seeds"]) == {"42", "1"}
    assert len(preds["seeds"]["42"]["test"]["sample_id"]) == seed["n"]

    metrics.mkdir()
    shutil.copy(probe / "aggregate_metrics.json", metrics / "bbh__needs_thinking.json")
    shutil.copy(probe / "predictions.json", metrics / "bbh__needs_thinking.predictions.json")
    for kind in ("text", "confidence"):
        _run(f"baseline_{kind}.py", "--capture-dir", str(cap), "--labels", str(lab),
             "--target", "needs_thinking", "--seeds", "42", "1",
             "--out-file", str(metrics / "baselines" / f"{kind}__bbh__needs_thinking.json"))
        b = json.loads((metrics / "baselines" / f"{kind}__bbh__needs_thinking.json").read_text())
        assert all("val_auroc" in node for node in b["baselines"].values())
        assert (metrics / "baselines" / f"{kind}__bbh__needs_thinking.predictions.json").exists()

    from scripts.compare_baselines import rows_for
    rows = rows_for(metrics, n_boot=200)
    assert len(rows) == 1
    row = rows[0]
    assert row["text"][2] == "val" and row["confidence"][2] == "val"
    d = row["diff"]
    assert d["n_seeds"] == 2 and d["ci"][0] <= d["mean"] <= d["ci"][1]
    # The planted signal is in the activations only: the probe should win.
    assert d["mean"] > 0
    _run("compare_baselines.py", "--metrics-dir", str(metrics), "--n-boot", "100")


def test_compare_baselines_flags_legacy_test_selection():
    from scripts.compare_baselines import best
    legacy = {"a": {"test_auroc": {"mean": 0.6}, "test_auroc_bootstrap": {"ci": [0.5, 0.7]}},
              "b": {"test_auroc": {"mean": 0.7}, "test_auroc_bootstrap": {"ci": [0.6, 0.8]}}}
    assert best(legacy)[0] == "b" and best(legacy)[2] == "test"
    # With validation scores the choice follows val, even against test.
    legacy["a"]["val_auroc"] = {"mean": 0.9}
    legacy["b"]["val_auroc"] = {"mean": 0.5}
    name, node, selected_on = best(legacy)
    assert (name, selected_on) == ("a", "val") and node["mean"] == 0.6


def test_transfer_verdict_uses_bootstrap_interval(bbh_like):
    tmp, cap, lab = bbh_like
    probe = tmp / "probe_t"
    tgt_cap, tgt_lab = tmp / "tgt", tmp / "tgt.jsonl"
    write_capture(tgt_cap, "gsm8k", 240, seed=22)
    _run("generate_labels.py", "--capture-dir", str(tgt_cap), "--out-file", str(tgt_lab))
    _run("run_experiment.py", "--capture-dir", str(cap), "--labels", str(lab),
         "--out-dir", str(probe), "--method", "logreg", "--target", "needs_thinking",
         "--layers", str(SIGNAL_LAYER), str(SIGNAL_LAYER + 1), "--seeds", "42", "1")
    ckpt = json.loads((probe / "seed_42" / "checkpoint.json").read_text())
    assert ckpt["layers"] == [SIGNAL_LAYER, SIGNAL_LAYER + 1]
    out = tmp / "transfer.json"
    _run("eval_transfer.py", "--probe", str(probe / "seed_42" / "checkpoint.json"),
         str(probe / "seed_1" / "checkpoint.json"),
         "--capture-dir", str(tgt_cap), "--labels", str(tgt_lab),
         "--source-metrics", str(probe / "aggregate_metrics.json"), "--out-file", str(out))
    res = json.loads(out.read_text())
    assert res["verdict_ci"] == "transfer_auroc_bootstrap"
    boot, seed = res["transfer_auroc_bootstrap"]["ci"], res["transfer_auroc"]["ci"]
    assert boot[1] - boot[0] > seed[1] - seed[0]          # the seed spread is narrower
    from scripts.eval_transfer import VERDICT_NONE, VERDICTS
    assert res["verdict"] in VERDICTS
    if not boot[0] > 0.5:
        assert res["verdict"] == VERDICT_NONE


def test_results_table_prints_the_bootstrap_interval(bbh_like):
    tmp, cap, lab = bbh_like
    probe = tmp / "probe_rt"
    _run("run_experiment.py", "--capture-dir", str(cap), "--labels", str(lab),
         "--out-dir", str(probe), "--method", "logreg", "--target", "needs_thinking",
         "--layer", str(SIGNAL_LAYER), "--seeds", "42", "1")
    mdir = tmp / "rt"
    mdir.mkdir()
    shutil.copy(probe / "aggregate_metrics.json", mdir / "bbh__needs_thinking.json")
    shutil.copy(probe / "predictions.json", mdir / "bbh__needs_thinking.predictions.json")
    text = _run("results_table.py", "--metrics-dir", str(mdir)).stdout
    agg = json.loads((probe / "aggregate_metrics.json").read_text())["aggregate"]
    lo, hi = agg["test_auroc_bootstrap"]["ci"]
    assert "95% boot CI" in text and f"[{lo:.3f},{hi:.3f}]" in text
    s_lo, s_hi = agg["test_auroc"]["ci"]
    assert f"[{s_lo:.3f},{s_hi:.3f}]" not in text


# ---------------------------------------------------------------------------
# run_full_analysis.sh (B13)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def analysis_repo(tmp_path_factory):
    """A throwaway repo copy with two synthetic captures under shared/."""
    root = tmp_path_factory.mktemp("repo")
    for d in ("scripts", "utils"):
        shutil.copytree(_PROJECT_ROOT / d, root / d,
                        ignore=shutil.ignore_patterns("__pycache__"))
    write_capture(root / "shared/icr_capture/bbh_thinking_synth", "bbh", 300, seed=31,
                  subtasks=["boolean_expressions", "navigate", "web_of_lies"])
    write_capture(root / "shared/icr_capture/gsm8k_thinking_synth", "gsm8k", 240, seed=32)
    return root


def _analysis(root: Path, **env) -> subprocess.CompletedProcess:
    full_env = {**os.environ, "PY": sys.executable, "CAPTURE_SLUG": "synth",
                "TASKS": "bbh gsm8k", "TARGETS": "needs_thinking",
                "WITHIN_TARGETS": "needs_thinking", "SEEDS": "42 1", "REGRADE": "0",
                "OMP_NUM_THREADS": "2", **env}
    return subprocess.run(["bash", str(root / "scripts/run_full_analysis.sh")],
                          capture_output=True, text=True, timeout=900, env=full_env,
                          cwd=root)


def test_run_full_analysis_writes_to_output_and_promotes_only_on_request(analysis_repo):
    root = analysis_repo
    proc = _analysis(root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    mdir = root / "output/synth/metrics"
    for f in ("bbh__needs_thinking.json", "bbh__needs_thinking.predictions.json",
              "baselines/text__bbh__needs_thinking.json",
              "baselines/confidence__gsm8k__needs_thinking.predictions.json",
              "within_group__bbh__needs_thinking.json",
              "transfer__bbh_to_gsm8k__needs_thinking.json",
              "results_table.txt", "baseline_comparison.txt"):
        assert (mdir / f).exists(), f
    assert not (root / "paper").exists(), "analysis wrote into paper/ without PROMOTE"
    # Middle layer from the activation shape, not a hard-coded 18.
    agg = json.loads((mdir / "bbh__needs_thinking.json").read_text())
    assert agg["layer"] == N_LAYERS // 2

    # Second run: everything is up to date.
    proc = _analysis(root)
    assert proc.returncode == 0
    assert "probe bbh/needs_thinking — up to date" in proc.stdout

    # Re-graded labels must invalidate what was built from them.
    lab = root / "shared/labels/synth/bbh_labels.jsonl"
    future = time.time() + 5
    os.utime(lab, (future, future))
    proc = _analysis(root)
    assert proc.returncode == 0
    assert "probe bbh/needs_thinking\n" in proc.stdout
    assert "baseline text bbh/needs_thinking\n" in proc.stdout
    assert "probe gsm8k/needs_thinking — up to date" in proc.stdout

    # Promotion: copies once, then refuses to overwrite a changed file.
    proc = _analysis(root, PROMOTE="1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    pdir = root / "paper/results/metrics/synth"
    assert (pdir / "bbh__needs_thinking.json").read_bytes() == \
        (mdir / "bbh__needs_thinking.json").read_bytes()
    published = pdir / "results_table.txt"
    published.write_text("published version\n")
    sentinel = pdir / "bbh__needs_thinking.json"
    sentinel.write_text("{}\n")
    proc = _analysis(root, PROMOTE="1")
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert published.read_text() == "published version\n"
    assert sentinel.read_text() == "{}\n"            # nothing copied at all


def test_run_full_analysis_fails_fast(analysis_repo):
    root = analysis_repo
    proc = _analysis(root, TASKS="gsm8k")              # build (or confirm) a good state
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lab = root / "shared/labels/synth/gsm8k_labels.jsonl"
    # A labels file newer than its outputs that cannot be parsed: the probe step
    # must fail and stop the run, not print nothing and carry on. (It is also
    # newer than the capture meta, so the label step does not regenerate it.)
    saved = lab.read_text()
    try:
        lab.write_text("not json\n")
        future = time.time() + 10
        os.utime(lab, (future, future))
        proc = _analysis(root, TASKS="gsm8k")
        assert proc.returncode != 0
        assert "FAILED probe_gsm8k_needs_thinking" in proc.stdout
    finally:
        lab.write_text(saved)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

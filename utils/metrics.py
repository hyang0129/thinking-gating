"""
utils/metrics.py — every evaluation statistic the analysis scripts report.

One implementation per statistic, imported by run_experiment.py, the baseline
scripts, within_group_auroc.py, eval_transfer.py and compare_baselines.py, so
a number in two tables cannot come from two slightly different formulas.

Conventions
-----------
* ``correct_off`` / ``correct_on`` are per-row booleans for the cheap policy
  (thinking off, or a single-pass decision model) and the expensive one
  (thinking on, the reasoner). "Escalating" a row means taking its
  ``correct_on`` instead of its ``correct_off``.
* ``scores`` are router scores: higher means "escalate first".
* Every interval is a percentile bootstrap over **rows**. Quote these, never
  the seed-spread interval from ``mean_ci`` — on a few hundred test rows the
  seed spread is 1.6-8.8x too narrow (paper/results/metrics/decomposition/).
* Bootstraps are seeded and draw ``rng.integers(0, n, n)`` once per replicate,
  dropping replicates whose statistic is undefined. That is exactly what the
  pre-hoist code did, so re-running it reproduces the published intervals.

Contents
--------
AUROC            auroc, auprc, mean_ci, bootstrap_ci, bootstrap_auroc_ci,
                 mean_bootstrap_ci
paired           paired_bootstrap, paired_bootstrap_auroc_diff,
                 paired_bootstrap_nauc_diff
routing          routing_metrics, best_threshold, routed_accuracy_curve,
                 oracle_curve, random_accuracy_at, curve_accuracy_at,
                 curve_area, routed_nauc, min_fraction_for_accuracy,
                 cost_accuracy_curve
within-group     within_group_auroc, within_group_stats
calibration      wilson_ci, precision_threshold, confident_error_leakage,
                 signed_overconfidence, reliability_bins, reliability_by_group
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Callable

import numpy as np

NAN = float("nan")


def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


# ---------------------------------------------------------------------------
# AUROC and friends
# ---------------------------------------------------------------------------

def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks, ties sharing their average rank."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)            # one past the last 0-based position
    starts = ends - counts
    return ((starts + ends + 1) / 2.0)[inverse.reshape(-1)]


def auroc(y_true, scores) -> float:
    """Rank-based AUROC with ties counted half; nan when only one class is present."""
    y_true = np.asarray(y_true).astype(int)
    pos, neg = int(y_true.sum()), int((1 - y_true).sum())
    if pos == 0 or neg == 0:
        return NAN
    ranks = _average_ranks(np.asarray(scores, dtype=np.float64))
    return float((ranks[y_true == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def auprc(y_true, scores) -> float:
    """Average precision. The base rate is the trivial floor to compare against."""
    y_true = np.asarray(y_true).astype(int)
    if y_true.sum() == 0:
        return NAN
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    return float((precision * y_sorted).sum() / y_sorted.sum())


def mean_ci(values, confidence: float = 0.95) -> dict:
    """Mean with a normal-approximation CI over *seeds*; nans dropped.

    This is the seed-spread interval: how much a number moves when a fixed
    sample is re-split. It is not a population interval. Kept because
    aggregate_metrics.json has always carried it; never quote it as "95% CI".
    """
    vals = [float(v) for v in values if not _isnan(float(v))]
    if not vals:
        return {"mean": NAN, "ci": [NAN, NAN], "n": 0}
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return {"mean": mean, "ci": [mean, mean], "n": len(vals)}
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
    half = 1.96 * sd / math.sqrt(len(vals))
    return {"mean": mean, "ci": [mean - half, mean + half], "sd": sd, "n": len(vals)}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _resampler(n: int, rng: np.random.Generator, strata=None):
    if strata is None:
        return lambda: rng.integers(0, n, n)
    strata = np.asarray(strata)
    if len(strata) != n:
        raise ValueError(f"strata has {len(strata)} rows, data has {n}")
    blocks = [np.flatnonzero(strata == s) for s in sorted(set(strata.tolist()), key=str)]
    return lambda: np.concatenate([b[rng.integers(0, len(b), len(b))] for b in blocks])


def bootstrap_samples(stat_fn: Callable[..., float], *arrays, n_boot: int = 2000,
                      seed: int = 0, strata=None) -> np.ndarray:
    """Statistic on ``n_boot`` row-resamples; undefined (nan) replicates dropped.

    ``strata`` resamples within each stratum, preserving stratum sizes — use it
    when the estimand is defined per cell (family x depth) and a cell must not
    vanish from a replicate.
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("bootstrap arrays differ in length")
    if n < 2:
        return np.array([], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draw = _resampler(n, rng, strata)
    out = []
    for _ in range(n_boot):
        idx = draw()
        v = stat_fn(*(a[idx] for a in arrays))
        if not _isnan(v) and not (isinstance(v, (np.floating,)) and np.isnan(v)):
            out.append(float(v))
    return np.asarray(out, dtype=np.float64)


def _percentiles(samples: np.ndarray, confidence: float) -> list[float]:
    if len(samples) == 0:
        return [NAN, NAN]
    lo, hi = np.percentile(samples, [(1 - confidence) / 2 * 100,
                                     (1 + confidence) / 2 * 100])
    return [float(lo), float(hi)]


def bootstrap_ci(stat_fn: Callable[..., float], *arrays, n_boot: int = 2000,
                 seed: int = 0, confidence: float = 0.95, strata=None) -> dict:
    """Percentile-bootstrap CI of ``stat_fn(*arrays)`` over rows."""
    s = bootstrap_samples(stat_fn, *arrays, n_boot=n_boot, seed=seed, strata=strata)
    return {"ci": _percentiles(s, confidence), "n_boot": int(len(s))}


def bootstrap_auroc_ci(y, scores, n_boot: int = 2000, seed: int = 0,
                       confidence: float = 0.95) -> dict:
    """Percentile-bootstrap CI for AUROC over *test examples* — the one to quote.

    mean_ci over seeds answers a different and much narrower question (how
    much the number moves when a fixed sample is re-split), so on a few
    hundred test points it understates uncertainty severalfold.
    """
    y = np.asarray(y)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return {"ci": [NAN, NAN], "n_boot": 0}
    return bootstrap_ci(auroc, y, np.asarray(scores), n_boot=n_boot, seed=seed,
                        confidence=confidence)


def mean_bootstrap_ci(per_seed_cis: list[dict]) -> dict:
    """Average per-seed bootstrap intervals into one reported interval.

    The repo's convention for "one interval across seeds". The seeds' test
    sets overlap, so pooling their replicates as independent would be too
    narrow; averaging the bounds is the conservative choice.
    """
    los = [c["ci"][0] for c in per_seed_cis if not _isnan(c["ci"][0])]
    his = [c["ci"][1] for c in per_seed_cis if not _isnan(c["ci"][1])]
    if not los:
        return {"ci": [NAN, NAN], "n_seeds": 0}
    return {"ci": [sum(los) / len(los), sum(his) / len(his)], "n_seeds": len(los)}


# ---------------------------------------------------------------------------
# Paired bootstrap of a difference
# ---------------------------------------------------------------------------

def paired_bootstrap(diff_fn: Callable[..., float], *arrays, n_boot: int = 2000,
                     seed: int = 0, confidence: float = 0.95, strata=None) -> dict:
    """CI for a difference A − B computed on the *same* resampled rows.

    ``diff_fn(*arrays)`` must return the difference itself. Resampling rows
    jointly keeps the correlation between the two methods' errors, which is
    what makes a paired interval much narrower than two overlapping marginal
    ones — and the reason two marginal CIs that overlap say nothing about
    whether A beats B.

    Returns ``estimate`` (on the full sample), ``ci``, ``n_boot``, ``se`` and
    ``p_le_zero`` (share of replicates with A − B <= 0, a one-sided bootstrap
    p-value for "A beats B").
    """
    estimate = diff_fn(*(np.asarray(a) for a in arrays))
    s = bootstrap_samples(diff_fn, *arrays, n_boot=n_boot, seed=seed, strata=strata)
    return {
        "estimate": NAN if _isnan(estimate) else float(estimate),
        "ci": _percentiles(s, confidence),
        "n_boot": int(len(s)),
        "se": float(s.std(ddof=1)) if len(s) > 1 else NAN,
        "p_le_zero": float((s <= 0).mean()) if len(s) else NAN,
    }


def paired_bootstrap_auroc_diff(y, scores_a, scores_b, **kw) -> dict:
    """AUROC(a) − AUROC(b) on identical rows, paired bootstrap."""
    return paired_bootstrap(lambda yy, a, b: auroc(yy, a) - auroc(yy, b),
                            y, scores_a, scores_b, **kw)


def paired_bootstrap_nauc_diff(correct_off, correct_on, scores_a, scores_b,
                               lo: float = 0.0, hi: float = 0.5, **kw) -> dict:
    """ΔnAUC = nAUC(a) − nAUC(b) over escalation range [lo, hi], paired bootstrap.

    The pre-registered G2/G3 statistic (proposal §3).
    """
    def diff(off, on, a, b):
        return (routed_nauc(off, on, a, lo, hi)["nauc"]
                - routed_nauc(off, on, b, lo, hi)["nauc"])
    return paired_bootstrap(diff, correct_off, correct_on, scores_a, scores_b, **kw)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def routing_metrics(correct_off, correct_on, scores, threshold: float) -> dict:
    """Task accuracy if rows with score >= threshold are escalated."""
    think = np.asarray(scores) >= threshold
    routed = np.where(think, np.asarray(correct_on), np.asarray(correct_off))
    return {
        "threshold": float(threshold),
        "routed_accuracy": float(routed.mean()),
        "fraction_routed_to_thinking": float(think.mean()),
    }


def best_threshold(correct_off, correct_on, scores) -> float:
    """Threshold maximizing routed accuracy — choose it on validation only.

    Deterministic: candidates are scanned in ascending order and the first
    maximum wins, i.e. the ties go to the threshold that escalates the most.
    """
    candidates = np.unique(np.concatenate([[0.0, 1.0], np.asarray(scores, dtype=float)]))
    best, best_acc = 0.5, -1.0
    for t in candidates:
        acc = routing_metrics(correct_off, correct_on, scores, t)["routed_accuracy"]
        if acc > best_acc:
            best, best_acc = float(t), acc
    return best


def routed_accuracy_curve(correct_off, correct_on, scores) -> dict:
    """The exact routed-accuracy curve over every threshold.

    Rows are escalated in descending score order. Rows with *equal* scores are
    indistinguishable to the router, so inside a tie block the curve is the
    expectation over a random order — a straight line between the block's
    endpoints. That makes the curve a deterministic function of the scores
    (no dependence on row order or on argsort's tie-breaking), which matters
    for discrete scores such as token counts.

    Returns
      fraction, accuracy     length n+1, at every integer escalation count k
      vertex_fraction,       the curve's breakpoints (one per distinct score);
      vertex_accuracy,       the curve is linear between them, so these alone
      vertex_threshold       are exact. vertex_threshold[i] is the score cut-off
                             (escalate score >= t) that realises vertex i;
                             vertex 0 (nothing escalated) has threshold +inf.
    """
    off = np.asarray(correct_off, dtype=np.float64)
    on = np.asarray(correct_on, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    if not (len(off) == len(on) == n):
        raise ValueError("correct_off, correct_on and scores differ in length")
    if n == 0:
        raise ValueError("empty routing curve")
    if np.isnan(s).any():
        raise ValueError("router scores contain NaN")
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    gain = (on - off)[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s_sorted) != 0) + 1]
    ends = np.r_[starts[1:], n]
    lengths = ends - starts
    block_mean = np.add.reduceat(gain, starts) / lengths
    cum = np.r_[0.0, np.cumsum(np.repeat(block_mean, lengths))]
    base = off.sum()
    accuracy = (base + cum) / n
    return {
        "fraction": np.arange(n + 1) / n,
        "accuracy": accuracy,
        "vertex_fraction": np.r_[0, ends] / n,
        "vertex_accuracy": accuracy[np.r_[0, ends]],
        "vertex_threshold": np.r_[np.inf, s_sorted[starts]],
    }


def oracle_curve(correct_off, correct_on) -> dict:
    """The best any router can do: rescued rows first, hurt rows last."""
    gain = np.asarray(correct_on, dtype=float) - np.asarray(correct_off, dtype=float)
    return routed_accuracy_curve(correct_off, correct_on, gain)


def random_accuracy_at(correct_off, correct_on, fraction) -> float | np.ndarray:
    """Expected accuracy of escalating a uniformly random ``fraction`` of rows."""
    a_off = float(np.mean(correct_off))
    a_on = float(np.mean(correct_on))
    return a_off + np.asarray(fraction, dtype=float) * (a_on - a_off)


def curve_accuracy_at(curve: dict, fraction) -> float | np.ndarray:
    """Accuracy of a routed curve at an arbitrary escalation fraction."""
    return np.interp(fraction, curve["vertex_fraction"], curve["vertex_accuracy"])


def curve_area(vertex_fraction, vertex_accuracy, lo: float = 0.0, hi: float = 1.0) -> float:
    """Exact area under a piecewise-linear curve on [lo, hi]."""
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(f"bad escalation range [{lo}, {hi}]")
    vf = np.asarray(vertex_fraction, dtype=float)
    va = np.asarray(vertex_accuracy, dtype=float)
    xs = np.unique(np.r_[lo, hi, vf[(vf > lo) & (vf < hi)]])
    ys = np.interp(xs, vf, va)
    return float(np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2.0))


def routed_nauc(correct_off, correct_on, scores, lo: float = 0.0, hi: float = 0.5) -> dict:
    """Normalised area under the routed-accuracy curve on escalation range [lo, hi].

      mean_accuracy   area / (hi − lo): average accuracy over the range
      nauc            (area − random) / (oracle − random): 0 = no better than
                      escalating at random at every rate, 1 = the oracle.
                      nan when oracle == random (nothing to route for).

    Both references are computed on the same rows, so ΔnAUC between two
    routers on one workload is (Δarea) / (oracle − random) and pairs cleanly
    under the bootstrap. The proposal's G2 uses [0, 0.5].
    """
    width = hi - lo
    curve = routed_accuracy_curve(correct_off, correct_on, scores)
    orc = oracle_curve(correct_off, correct_on)
    a_router = curve_area(curve["vertex_fraction"], curve["vertex_accuracy"], lo, hi)
    a_oracle = curve_area(orc["vertex_fraction"], orc["vertex_accuracy"], lo, hi)
    r_lo, r_hi = random_accuracy_at(correct_off, correct_on, [lo, hi])
    a_random = float((r_lo + r_hi) / 2.0 * width)
    denom = a_oracle - a_random
    return {
        "range": [float(lo), float(hi)],
        "nauc": float((a_router - a_random) / denom) if denom > 1e-12 else NAN,
        "mean_accuracy": a_router / width,
        "random_mean_accuracy": a_random / width,
        "oracle_mean_accuracy": a_oracle / width,
    }


def min_fraction_for_accuracy(curve: dict, target: float, tol: float = 1e-9) -> float:
    """Smallest escalation fraction (over integer counts) reaching ``target``.

    Read off the exact curve, not a grid. 1.0 if it is never reached.
    """
    ok = np.flatnonzero(curve["accuracy"] >= target - tol)
    return float(curve["fraction"][ok[0]]) if len(ok) else 1.0


def cost_accuracy_curve(correct_off, correct_on, scores, n_points: int = 21) -> list[dict]:
    """A fixed grid sampled from the exact curve — kept for the JSON schema.

    Same points as the pre-hoist version (k = round(i * n / (n_points - 1))),
    but read off routed_accuracy_curve, so ties no longer depend on argsort.
    Use routed_accuracy_curve / routed_nauc for anything quantitative.
    """
    curve = routed_accuracy_curve(correct_off, correct_on, scores)
    n = len(curve["accuracy"]) - 1
    out = []
    for i in range(n_points):
        k = round(i * n / (n_points - 1))
        out.append({"fraction_routed": k / n if n else 0.0,
                    "accuracy": float(curve["accuracy"][k])})
    return out


# ---------------------------------------------------------------------------
# Within-group AUROC
# ---------------------------------------------------------------------------

def _require_groups(groups, n: int) -> np.ndarray:
    if groups is None:
        raise ValueError("within-group AUROC needs an explicit group array")
    g = np.asarray([str(x) for x in groups])
    if len(g) != n:
        raise ValueError(f"groups has {len(g)} rows, data has {n}")
    return g


def within_group_stats(y, scores, groups) -> dict:
    """Pooled concordance over (pos, neg) pairs drawn from the *same* group.

    Category identity cannot help on such a pair, so this is the score's
    signal net of "which group is this". Groups with only one class
    contribute no pairs; ``n_informative_groups`` counts the ones that do.
    With one informative group this is just that group's AUROC; with zero it
    is nan. A grouping that puts every row in one group makes it equal the
    ordinary AUROC — check ``n_informative_groups`` before quoting it.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    g = _require_groups(groups, len(y))
    conc = pairs = 0.0
    n_inf = 0
    uniq = np.unique(g)
    for grp in uniq:
        m = g == grp
        pos, neg = s[m][y[m] == 1], s[m][y[m] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        n_inf += 1
        diff = pos[:, None] - neg[None, :]
        conc += (diff > 0).sum() + 0.5 * (diff == 0).sum()
        pairs += diff.size
    return {"auroc": float(conc / pairs) if pairs else NAN,
            "n_groups": int(len(uniq)), "n_informative_groups": int(n_inf),
            "n_pairs": int(pairs)}


def within_group_auroc(y, scores, groups) -> float:
    """Pooled within-group AUROC (see within_group_stats)."""
    return within_group_stats(y, scores, groups)["auroc"]


# ---------------------------------------------------------------------------
# Calibration (proposal §3, H1-cal)
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, confidence: float = 0.95) -> list[float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n <= 0:
        return [NAN, NAN]
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def precision_threshold(conf, correct, target_precision: float = 0.95) -> dict:
    """Smallest τ such that P(correct | conf >= τ) >= target, on calibration rows.

    The proposal fixes τ on held-out depth-1 items at 95% accepted precision,
    then holds it fixed while depth grows. The smallest qualifying τ is the
    one with the most coverage. τ = +inf (accept nothing) if no cut-off
    reaches the target.
    """
    c = np.asarray(conf, dtype=np.float64)
    ok = np.asarray(correct).astype(bool)
    if len(c) == 0:
        raise ValueError("no calibration rows")
    order = np.argsort(-c, kind="mergesort")
    c_sorted, ok_sorted = c[order], ok[order]
    ends = np.r_[np.flatnonzero(np.diff(c_sorted) != 0) + 1, len(c)]
    n_acc = ends
    prec = np.cumsum(ok_sorted)[ends - 1] / n_acc
    good = np.flatnonzero(prec >= target_precision)
    if len(good) == 0:
        return {"tau": math.inf, "precision": NAN, "coverage": 0.0,
                "n": int(len(c)), "target_precision": target_precision}
    i = good[-1]                       # lowest qualifying threshold
    return {"tau": float(c_sorted[ends[i] - 1]), "precision": float(prec[i]),
            "coverage": float(n_acc[i] / len(c)), "n": int(len(c)),
            "target_precision": target_precision}


def confident_error_leakage(conf, correct, tau: float, groups=None,
                            confidence: float = 0.95) -> dict:
    """CEL = P(conf >= τ | wrong), overall and per group (e.g. per depth d).

    The share of the model's errors it would have accepted at the fixed
    threshold τ. Interval: Wilson, since τ is fixed in advance and CEL is a
    plain proportion of wrong rows.
    """
    c = np.asarray(conf, dtype=np.float64)
    ok = np.asarray(correct).astype(bool)

    def cell(mask):
        wrong = mask & ~ok
        n_wrong = int(wrong.sum())
        leaked = int((wrong & (c >= tau)).sum())
        return {"n": int(mask.sum()), "n_wrong": n_wrong, "n_leaked": leaked,
                "cel": leaked / n_wrong if n_wrong else NAN,
                "ci": wilson_ci(leaked, n_wrong, confidence)}

    out = {"tau": float(tau), "overall": cell(np.ones(len(c), dtype=bool))}
    if groups is not None:
        g = _require_groups(groups, len(c))
        out["by_group"] = {grp: cell(g == grp) for grp in _sorted_groups(g)}
    return out


def _sorted_groups(g: np.ndarray) -> list[str]:
    def key(x):
        try:
            return (0, float(x), x)
        except ValueError:
            return (1, 0.0, x)
    return sorted(set(g.tolist()), key=key)


def signed_overconfidence(conf, correct, groups=None, n_boot: int = 2000,
                          seed: int = 0, confidence: float = 0.95) -> dict:
    """E[conf] − accuracy, overall and per group, with a bootstrap CI.

    Positive = overconfident. Pass a combined key (e.g. "d=3|gold=yes") for
    per-cell and per-gold-label breakdowns.
    """
    c = np.asarray(conf, dtype=np.float64)
    ok = np.asarray(correct).astype(np.float64)

    def stat(cc, oo):
        return float(cc.mean() - oo.mean()) if len(cc) else NAN

    def cell(mask):
        cc, oo = c[mask], ok[mask]
        if len(cc) == 0:
            return {"n": 0, "mean_conf": NAN, "accuracy": NAN,
                    "overconfidence": NAN, "ci": [NAN, NAN]}
        return {"n": int(len(cc)), "mean_conf": float(cc.mean()),
                "accuracy": float(oo.mean()), "overconfidence": stat(cc, oo),
                "ci": bootstrap_ci(stat, cc, oo, n_boot=n_boot, seed=seed,
                                   confidence=confidence)["ci"]}

    out = {"overall": cell(np.ones(len(c), dtype=bool))}
    if groups is not None:
        g = _require_groups(groups, len(c))
        out["by_group"] = {grp: cell(g == grp) for grp in _sorted_groups(g)}
    return out


def reliability_bins(conf, correct, n_bins: int = 10, strategy: str = "uniform") -> dict:
    """Reliability diagram data plus ECE / MCE.

    ``uniform`` bins split [0, 1] evenly; ``quantile`` bins hold equal counts.
    Each bin reports n, mean confidence, accuracy and gap = conf − acc.
    """
    c = np.asarray(conf, dtype=np.float64)
    ok = np.asarray(correct).astype(np.float64)
    if len(c) == 0:
        return {"bins": [], "ece": NAN, "mce": NAN, "n": 0}
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(c, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = min(edges[0], 0.0), max(edges[-1], 1.0)
    else:
        raise ValueError(f"unknown binning strategy {strategy!r}")
    idx = np.clip(np.searchsorted(edges, c, side="right") - 1, 0, n_bins - 1)
    bins, ece, mce = [], 0.0, 0.0
    for b in range(n_bins):
        m = idx == b
        k = int(m.sum())
        if k == 0:
            bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": 0,
                         "mean_conf": NAN, "accuracy": NAN, "gap": NAN})
            continue
        mc, acc = float(c[m].mean()), float(ok[m].mean())
        gap = mc - acc
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": k,
                     "mean_conf": mc, "accuracy": acc, "gap": gap})
        ece += k / len(c) * abs(gap)
        mce = max(mce, abs(gap))
    return {"bins": bins, "ece": float(ece), "mce": float(mce), "n": int(len(c)),
            "strategy": strategy}


def reliability_by_group(conf, correct, groups, **kw) -> dict:
    """reliability_bins per group (e.g. per depth d)."""
    c = np.asarray(conf, dtype=np.float64)
    ok = np.asarray(correct)
    g = _require_groups(groups, len(c))
    return {grp: reliability_bins(c[g == grp], ok[g == grp], **kw)
            for grp in _sorted_groups(g)}

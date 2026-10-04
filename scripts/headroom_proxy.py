#!/usr/bin/env python3
"""
headroom_proxy.py — how much escalation does thinking-off confidence need,
against an oracle, to get within a tolerance of always-think accuracy?

This is the "G2 headroom proxy" from issue #1 §0, made reproducible. The first
version (an audit scratch script, 2026-10-04) read B1-corrupted
`confidence_off` values and the pre-regrade `correct_off`/`correct_on`; its
20.8% (confidence) vs 2.1% (oracle) is withdrawn. This one reads:

  * correctness from the LABEL files (`generate_labels.py --regrade`
    output), never from the meta rows, and
  * confidence through `utils.capture_io.resolve_confidence`, refusing
    confidence v1 (B1) unless --allow-v1 is passed.

Routers (each a per-row escalation score; escalate the top-k):
  conf_raw     -mean_logprob, no fitting
  conf_fit     5-fold cross-fitted linear regression of gain = on - off on
               [mean_logprob, min_logprob, mean_entropy, log1p(n_tokens_off)]
  conf_family  conf_fit's features + task one-hot + task x {mean_logprob,
               mean_entropy}: confidence plus the TRUE task family, a
               privileged router
  oracle       gain itself (escalates helped rows first, hurt rows last)

For each router, from the exact routed-accuracy curve (utils.metrics): nAUC
over escalation 0-50% (0 = random, 1 = oracle) and the mean routed accuracy
there; the smallest escalation reaching always-think accuracy minus 1, 0.5
and 0 pp; and accuracy at 10% and 20% escalation. The tolerance point alone
is fragile: on the regraded Qwen3 mix always-think beats never-think by only
0.6 pp, so "within 1 pp" needs no escalation at all. Per-task off/on
accuracy gaps are reported with bootstrap intervals. Intervals are a paired percentile
bootstrap over rows (the same resample for every router). The router scores
are cross-fitted once on the full data and held fixed across resamples, so
the intervals cover routing noise, not refitting noise.

    .venv/bin/python scripts/headroom_proxy.py --slug qwen3v3 \\
        --tasks gsm8k math500 mmlu_pro --out-file output/qwen3v3/headroom.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.capture_io import (CONFIDENCE_SOURCES, align_labels,  # noqa: E402
                              load_labels, load_meta, resolve_confidence)
from utils.metrics import (curve_accuracy_at, min_fraction_for_accuracy,  # noqa: E402
                           routed_accuracy_curve, routed_nauc)

logger = logging.getLogger("headroom_proxy")
ROUTERS = ("conf_raw", "conf_fit", "conf_family", "oracle")
PAIRS = (("conf_fit", "oracle"), ("conf_family", "conf_fit"),
         ("conf_raw", "conf_fit"), ("conf_family", "oracle"))


def load_rows(slug: str, tasks: list[str], source: str, allow_v1: bool,
              capture_root: str = "shared/icr_capture",
              label_root: str = "shared/labels"):
    task_col, off, on, feats, provenance = [], [], [], [], []
    for task in tasks:
        cap = Path(capture_root) / f"{task}_thinking_{slug}"
        lab_path = Path(label_root) / slug / f"{task}_labels.jsonl"
        meta = load_meta(cap)
        conf, info = resolve_confidence(meta, cap, source)
        v1 = int(info.get("confidence_version_counts", {}).get("1", 0))
        if v1 and not allow_v1:
            raise SystemExit(
                f"{cap}: {v1} row(s) carry confidence v1 (B1-corrupted) and no "
                "confidence_off_v2 sidecar is present. Run "
                "scripts/rescore_confidence.py, or pass --allow-v1 to reproduce "
                "the withdrawn numbers.")
        labels = load_labels(lab_path)
        summary_path = lab_path.with_name(lab_path.stem + ".summary.json")
        summary = (json.loads(summary_path.read_text())
                   if summary_path.exists() else {})
        idx = align_labels(meta, labels)
        for lab, i in zip(labels, idx):
            c = conf[i] or {}
            task_col.append(task)
            off.append(int(bool(lab["correct_off"])))
            on.append(int(bool(lab["correct_on"])))
            feats.append([c.get("mean_logprob"), c.get("min_logprob"),
                          c.get("mean_entropy"),
                          np.log1p(meta[i]["n_tokens_off"])])
        provenance.append({"task": task, "capture_dir": str(cap),
                           "labels": str(lab_path), "n": len(labels),
                           "regraded": summary.get("regraded"),
                           "grader_version": summary.get("grader_version"),
                           "confidence": info})
    X = np.array(feats, dtype=float)
    return np.array(task_col), np.array(off), np.array(on), X, provenance


def clean_features(X: np.ndarray) -> tuple[np.ndarray, dict]:
    """Null/inf confidence is reported and clipped, never silently dropped."""
    bad = ~np.isfinite(X)
    report = {"nonfinite_per_feature": bad.sum(0).tolist()}
    X = np.nan_to_num(X, nan=0.0, neginf=-50.0, posinf=50.0)
    X[:, 1] = np.clip(X[:, 1], -50, 0)
    return X, report


def cross_fit(X, y, seed=0, folds=5):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold

    p = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        p[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
    return p


TOLERANCES = (0.01, 0.005, 0.0)
BUDGETS = (0.1, 0.2)


def router_stats(score, off, on, tolerances=TOLERANCES, budgets=BUDGETS) -> dict:
    """Every statistic for one router on one set of rows, from the exact
    tie-averaged routed-accuracy curve (utils.metrics)."""
    curve = routed_accuracy_curve(off, on, score)
    out = {}
    nauc = routed_nauc(off, on, score, 0.0, 0.5)
    out["nauc_0_50"] = nauc["nauc"]
    out["mean_routed_acc_0_50"] = nauc["mean_accuracy"]
    acc_on = float(np.mean(on))
    for tol in tolerances:
        f = min_fraction_for_accuracy(curve, acc_on - tol)
        # min_fraction_for_accuracy returns 1.0 when never reached
        out[f"escalation_to_always_think_minus_{tol:g}"] = f
    for b in budgets:
        out[f"acc_at_{b:g}_escalation"] = float(curve_accuracy_at(curve, b))
    return out


def evaluate(scores, off, on, rows=None):
    if rows is None:
        rows = np.arange(len(off))
    return {name: router_stats(s[rows], off[rows], on[rows])
            for name, s in scores.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slug", default="qwen3v3")
    p.add_argument("--tasks", nargs="+", default=["gsm8k", "math500", "mmlu_pro"])
    p.add_argument("--confidence-source", default="auto", choices=CONFIDENCE_SOURCES)
    p.add_argument("--allow-v1", action="store_true",
                   help="Accept B1-corrupted stored confidence (reproduction only).")
    p.add_argument("--label-root", default="shared/labels",
                   help="Labels are read from <label-root>/<slug>/<task>_labels.jsonl")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-file", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    T, off, on, X, prov = load_rows(args.slug, args.tasks, args.confidence_source,
                                    args.allow_v1, label_root=args.label_root)
    X, feat_report = clean_features(X)
    gain = (on - off).astype(float)
    fams = list(args.tasks)
    F = np.stack([(T == f).astype(float) for f in fams], 1)
    Xcf = np.c_[X, F, X[:, :1] * F, X[:, 2:3] * F]
    scores = {"conf_raw": -X[:, 0], "conf_fit": cross_fit(X, gain, args.seed),
              "conf_family": cross_fit(Xcf, gain, args.seed), "oracle": gain.copy()}

    point = evaluate(scores, off, on)
    stat_keys = list(point["oracle"])
    rng = np.random.default_rng(args.seed)
    n = len(off)
    boots = {name: {k: [] for k in stat_keys} for name in ROUTERS}
    fam_boot = {f: [] for f in fams}
    for _ in range(args.n_boot):
        rows = rng.integers(0, n, n)
        res = evaluate(scores, off, on, rows)
        for name in ROUTERS:
            for k in stat_keys:
                boots[name][k].append(res[name][k])
        for f in fams:
            r = rows[T[rows] == f]
            fam_boot[f].append(on[r].mean() - off[r].mean() if len(r) else np.nan)

    def ci(vals):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        return [float(x) for x in np.percentile(v, [2.5, 97.5])] if len(v) else [None, None]

    routers = {name: {k: {"point": point[name][k], "ci": ci(boots[name][k])}
                      for k in stat_keys} for name in ROUTERS}
    diffs = {}
    for a, b in PAIRS:
        for k in stat_keys:
            d = np.asarray(boots[a][k]) - np.asarray(boots[b][k])
            diffs[f"{a} - {b}: {k}"] = {
                "point": point[a][k] - point[b][k], "ci": ci(d),
                "p_le_zero": float(np.mean(d[np.isfinite(d)] <= 0))}

    per_task = {}
    for f in fams:
        m = T == f
        per_task[f] = {"n": int(m.sum()), "acc_off": float(off[m].mean()),
                       "acc_on": float(on[m].mean()),
                       "gap_on_minus_off": float(on[m].mean() - off[m].mean()),
                       "gap_ci": ci(fam_boot[f]),
                       "helped": float(((on - off)[m] == 1).mean()),
                       "hurt": float(((on - off)[m] == -1).mean())}
    out = {
        "slug": args.slug, "tasks": fams, "n": int(n), "n_boot": args.n_boot,
        "seed": args.seed, "acc_off": float(off.mean()), "acc_on": float(on.mean()),
        "helped": float((gain == 1).mean()), "hurt": float((gain == -1).mean()),
        "per_task": per_task, "features": feat_report, "routers": routers,
        "paired_differences": diffs, "provenance": prov,
        "note": ("exact tie-averaged routed-accuracy curves (utils.metrics); "
                 "nAUC over escalation 0-50%: 0 = random, 1 = oracle; "
                 "escalation_to_always_think_minus_t is 1.0 when never reached; "
                 "router scores cross-fitted once (5-fold) and held fixed, the "
                 "bootstrap resamples rows only"),
    }
    print(f"n={n} acc_off={off.mean():.4f} acc_on={on.mean():.4f} "
          f"helped={(gain == 1).mean():.4f} hurt={(gain == -1).mean():.4f}")
    for f, d in per_task.items():
        print(f"  {f:9s} n={d['n']:5d} off={d['acc_off']:.3f} on={d['acc_on']:.3f} "
              f"gap={d['gap_on_minus_off']:+.3f} [{d['gap_ci'][0]:+.3f}, {d['gap_ci'][1]:+.3f}] "
              f"helped={d['helped']:.3f} hurt={d['hurt']:.3f}")
    for k in stat_keys:
        print(k)
        for name in ROUTERS:
            r = routers[name][k]
            print(f"  {name:12s} {r['point']:.4f} [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]")
    for key, d in diffs.items():
        print(f"{key:70s} {d['point']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]")
    if args.out_file:
        Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_file).write_text(json.dumps(out, indent=2) + "\n")
        logger.info("wrote %s", args.out_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

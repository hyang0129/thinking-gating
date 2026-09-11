#!/usr/bin/env python3
"""
within_group_auroc.py — the pooled within-group AUROC that stratify_check.py
cannot give you.

stratify_check.py reports one AUROC per group, but a BBH subtask has ~20 rows
and a test split of 4, so most per-group intervals span [0, 1] and "how many
groups clear chance" says more about power than about the probe. The right
statistic is one AUROC over *all* (positive, negative) test pairs that share a
group. Category identity cannot help on such a pair by construction, so this
is the probe's query-level signal net of "which subtask is this".

Same probe as run_experiment.py (logreg, one layer, identical splits and
seeds); only the evaluation differs. Reports both the ordinary test AUROC and
the within-group one, with a percentile bootstrap over test rows for each.

    python scripts/within_group_auroc.py \\
        --capture-dir shared/icr_capture/bbh_thinking_qwen3v3 \\
        --labels shared/labels/qwen3v3/bbh_labels.jsonl \\
        --group-from sample_id --target rescued \\
        --out-file paper/results/metrics/qwen3v3/within_group__bbh__rescued.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import (Standardizer, auroc, mean_ci, select_layer,   # noqa: E402
                            stratified_split, train_logreg, predict_sklearn)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.capture_io import align_labels, load_capture, load_labels    # noqa: E402

logger = logging.getLogger("within_group")


def group_of(row: dict, spec: str) -> str:
    if spec == "sample_id":
        # BBH sample ids look like "<subtask>/<n>"; MATH ids carry the subject
        # the same way. Everything before the last "/" is the group.
        sid = str(row["sample_id"])
        return sid.rsplit("/", 1)[0] if "/" in sid else sid.rsplit("-", 1)[0]
    if spec.startswith("meta:"):
        return str(row.get(spec[5:]))
    raise ValueError(spec)


def within_group_auroc(y, s, g) -> float:
    """Concordance over (pos, neg) pairs drawn from the same group."""
    conc = pairs = 0.0
    for grp in np.unique(g):
        m = g == grp
        pos, neg = s[m][y[m] == 1], s[m][y[m] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        diff = pos[:, None] - neg[None, :]
        conc += (diff > 0).sum() + 0.5 * (diff == 0).sum()
        pairs += diff.size
    return float(conc / pairs) if pairs else float("nan")


def bootstrap(fn, y, s, g, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        v = fn(y[idx], s[idx], g[idx])
        if not np.isnan(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture-dir", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--group-from", default="sample_id")
    p.add_argument("--target", default="rescued",
                   choices=["helped", "needs_thinking", "rescued"])
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    p.add_argument("--out-file", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    meta, acts = load_capture(Path(args.capture_dir), mode="off")
    labels = load_labels(Path(args.labels))
    idx = align_labels(meta, labels)
    acts = acts[idx]
    rows = [meta[i] for i in idx]
    groups = np.array([group_of(r, args.group_from) for r in rows])
    correct_off = np.array([bool(l["correct_off"]) for l in labels])
    correct_on = np.array([bool(l["correct_on"]) for l in labels])

    if args.target == "helped":
        y = np.array([1 if l["label"] == "helped" else 0 for l in labels])
        keep = np.arange(len(y))
    elif args.target == "needs_thinking":
        y = (~correct_off).astype(int)
        keep = np.arange(len(y))
    else:
        keep = np.flatnonzero(~correct_off)
        y = correct_on[keep].astype(int)
    X = select_layer(acts[keep], args.layer)
    g = groups[keep]
    logger.info("target=%s n=%d groups=%d base rate %.3f",
                args.target, len(y), len(np.unique(g)), y.mean())

    overall, within, over_ci, with_ci = [], [], [], []
    for seed in args.seeds:
        tr, va, te = stratified_split(y, seed)
        scaler = Standardizer().fit(X[tr])
        model, _ = train_logreg(scaler.transform(X[tr]), y[tr],
                                scaler.transform(X[va]), y[va], seed=seed)
        s = predict_sklearn(model, scaler.transform(X[te]))
        o, w = auroc(y[te], s), within_group_auroc(y[te], s, g[te])
        overall.append(o); within.append(w)
        over_ci.append(bootstrap(lambda yy, ss, gg: auroc(yy, ss), y[te], s, g[te]))
        with_ci.append(bootstrap(within_group_auroc, y[te], s, g[te]))
        logger.info("seed %-3d overall %.3f  within-group %.3f", seed, o, w)

    def agg(vals, cis):
        m = mean_ci(vals)
        return {"mean": m["mean"], "ci": [float(np.mean([c[0] for c in cis])),
                                         float(np.mean([c[1] for c in cis]))]}
    out = {"target": args.target, "capture_dir": args.capture_dir,
           "group_from": args.group_from, "layer": args.layer, "seeds": args.seeds,
           "n": int(len(y)), "n_groups": int(len(np.unique(g))),
           "base_rate": float(y.mean()),
           "test_auroc_bootstrap": agg(overall, over_ci),
           "within_group_auroc_bootstrap": agg(within, with_ci)}
    logger.info("overall      %.3f [%.3f, %.3f]", out["test_auroc_bootstrap"]["mean"],
                *out["test_auroc_bootstrap"]["ci"])
    logger.info("within-group %.3f [%.3f, %.3f]", out["within_group_auroc_bootstrap"]["mean"],
                *out["within_group_auroc_bootstrap"]["ci"])
    if args.out_file:
        Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_file).write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
within_group_auroc.py — the pooled within-group AUROC that stratify_check.py
cannot give you.

stratify_check.py reports one AUROC per group, but a BBH subtask has ~20 rows
and a test split of 4, so most per-group intervals span [0, 1] and "how many
groups clear chance" says more about power than about the probe. The right
statistic is one AUROC over *all* (positive, negative) test pairs that share a
group. Category identity cannot help on such a pair by construction, so this
is the probe's query-level signal net of "which group is this".

Same probe as run_experiment.py (logreg, one layer, identical splits and
seeds); only the evaluation differs. Reports both the ordinary test AUROC and
the within-group one, with a percentile bootstrap over test rows for each.

The group key is explicit and required (--group-key):

    <field>            a field of the label row, else of the capture's meta
                       row (e.g. subtask, subject, family, depth, difficulty).
                       Every row must have it.
    sample_id:middle   the middle of a `<task>-<group>-<index>` sample id.
                       Right for BBH (`bbh-boolean_expressions-0` ->
                       `boolean_expressions`), which is how the published BBH
                       numbers were grouped. WRONG for math500, gsm8k,
                       mmlu_pro and lsat: their middle segment is the split
                       name, so every row lands in one group — this script
                       now refuses that instead of quietly returning the
                       ordinary AUROC (bug B8; the old default did exactly
                       that, and its docstring wrongly said MATH ids carry the
                       subject).

It exits non-zero unless at least two groups contain both classes, and it
reports how many groups are informative (both classes present) overall and
in each seed's test split. Groups without both classes contribute no pairs.

    python scripts/within_group_auroc.py \\
        --capture-dir shared/icr_capture/bbh_thinking_qwen3v3 \\
        --labels shared/labels/qwen3v3/bbh_labels.jsonl \\
        --group-key sample_id:middle --target rescued \\
        --out-file output/qwen3v3/metrics/within_group__bbh__rescued.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import (Standardizer, select_layer,                  # noqa: E402
                            stratified_split, train_logreg, predict_sklearn)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.capture_io import align_labels, load_capture, load_labels    # noqa: E402
from utils.metrics import (auroc, bootstrap_ci, mean_ci,                # noqa: E402
                           within_group_auroc, within_group_stats)

logger = logging.getLogger("within_group")

SAMPLE_ID_MIDDLE = "sample_id:middle"


def group_of(label_row: dict, meta_row: dict, key: str) -> str:
    """The group of one row under an explicit key. Raises if it is missing."""
    if key == SAMPLE_ID_MIDDLE:
        parts = str(label_row["sample_id"]).split("-")
        if len(parts) < 3:
            raise ValueError(f"sample_id {label_row['sample_id']!r} is not "
                             "<task>-<group>-<index>")
        return "-".join(parts[1:-1])
    for row in (label_row, meta_row):
        if key in row and row[key] is not None:
            return str(row[key])
    raise ValueError(f"row {label_row.get('sample_id')!r} has no {key!r} field "
                     "in its label or meta row")


def informative_groups(y: np.ndarray, g: np.ndarray) -> list[str]:
    """Groups that contain both classes — the only ones that contribute pairs."""
    return [grp for grp in np.unique(g)
            if (y[g == grp] == 1).any() and (y[g == grp] == 0).any()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture-dir", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--group-key", required=True,
                   help="Label/meta field to group by, or 'sample_id:middle'")
    p.add_argument("--target", default="rescued",
                   choices=["helped", "needs_thinking", "rescued"])
    p.add_argument("--layer", type=int, default=None,
                   help="Layer to probe (default: the middle layer, n_layers // 2, "
                        "as run_experiment.py)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out-file", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    meta, acts = load_capture(Path(args.capture_dir), mode="off")
    labels = load_labels(Path(args.labels))
    idx = align_labels(meta, labels)
    acts = acts[idx]
    rows = [meta[i] for i in idx]
    try:
        groups = np.array([group_of(lab, row, args.group_key)
                           for lab, row in zip(labels, rows)])
    except (KeyError, ValueError) as exc:
        logger.error("cannot group by %r: %s", args.group_key, exc)
        return 2
    correct_off = np.array([bool(lab["correct_off"]) for lab in labels])
    correct_on = np.array([bool(lab["correct_on"]) for lab in labels])

    if args.target == "helped":
        y = np.array([1 if lab["label"] == "helped" else 0 for lab in labels])
        keep = np.arange(len(y))
    elif args.target == "needs_thinking":
        y = (~correct_off).astype(int)
        keep = np.arange(len(y))
    else:
        keep = np.flatnonzero(~correct_off)
        y = correct_on[keep].astype(int)
    if args.layer is None:
        args.layer = acts.shape[1] // 2
    X = select_layer(acts[keep], args.layer)
    g = groups[keep]

    n_groups = int(len(np.unique(g)))
    inf = informative_groups(y, g)
    logger.info("target=%s n=%d layer=%d groups=%d informative=%d base rate %.3f",
                args.target, len(y), args.layer, n_groups, len(inf), y.mean())
    if len(inf) < 2:
        logger.error("only %d group(s) under --group-key %r contain both classes "
                     "(of %d groups) — the within-group AUROC would just be the "
                     "ordinary AUROC (or undefined). Pick a key that actually "
                     "partitions this task.", len(inf), args.group_key, n_groups)
        return 2

    overall, within, over_ci, with_ci, test_info = [], [], [], [], []
    for seed in args.seeds:
        tr, va, te = stratified_split(y, seed)
        scaler = Standardizer().fit(X[tr])
        model, _ = train_logreg(scaler.transform(X[tr]), y[tr],
                                scaler.transform(X[va]), y[va], seed=seed)
        s = predict_sklearn(model, scaler.transform(X[te]))
        o = auroc(y[te], s)
        stats = within_group_stats(y[te], s, g[te])
        w = stats["auroc"]
        overall.append(o)
        within.append(w)
        test_info.append({"seed": seed, "n_test": int(len(te)),
                          "n_groups": stats["n_groups"],
                          "n_informative_groups": stats["n_informative_groups"],
                          "n_pairs": stats["n_pairs"]})
        if stats["n_informative_groups"] < 2:
            logger.warning("seed %d: only %d informative group(s) in the test split",
                           seed, stats["n_informative_groups"])
        over_ci.append(bootstrap_ci(lambda yy, ss, gg: auroc(yy, ss),
                                    y[te], s, g[te], n_boot=args.n_boot)["ci"])
        with_ci.append(bootstrap_ci(within_group_auroc, y[te], s, g[te],
                                    n_boot=args.n_boot)["ci"])
        logger.info("seed %-3d overall %.3f  within-group %.3f  (%d informative "
                    "test groups, %d pairs)", seed, o, w,
                    stats["n_informative_groups"], stats["n_pairs"])

    def agg(vals, cis):
        m = mean_ci(vals)
        return {"mean": m["mean"], "ci": [float(np.nanmean([c[0] for c in cis])),
                                         float(np.nanmean([c[1] for c in cis]))]}
    out = {"target": args.target, "capture_dir": args.capture_dir,
           "group_key": args.group_key, "layer": args.layer, "seeds": args.seeds,
           "n": int(len(y)), "n_groups": n_groups,
           "n_informative_groups": len(inf),
           "base_rate": float(y.mean()),
           "test_split_groups": test_info,
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

#!/usr/bin/env python3
"""
baseline_confidence.py — can the model's own thinking-off uncertainty do the
probe's job?

This is the strongest cheap competitor to the prefill probe. A router that
first runs the thinking-off pass gets, for free, the model's token log-probs
and predictive entropy over its own answer plus the answer's length. If those
scalars predict `rescued` as well as an 8B prefill probe does, the probe adds
nothing over "run it once, and think again when the model looks unsure".

Note what this baseline is allowed to see that the probe is not: it reads the
thinking-OFF *generation*, so it is a post-hoc router (decide after one cheap
pass) where the probe is a pre-hoc one (decide before generating anything).
That is the honest comparison, and it should be stated wherever the two are
put side by side.

Runs on the identical splits, seeds, target and metric as run_experiment.py
and baseline_text.py — it imports their split, AUROC and scoring code — so the
numbers are directly comparable to aggregate_metrics.json.

Baselines:
  mean_logprob   mean token log-prob of the thinking-off answer
  min_logprob    the least likely token in the answer
  mean_entropy   mean predictive entropy over the answer's tokens
  n_tokens_off   answer length in tokens (the truncation-confound detector)
  confidence_lr  logistic regression over the four scalars, fit on train

Scalar baselines need no fitting; their sign is chosen on TRAIN so the test
AUROC cannot be flattered by picking the direction after the fact.

Requires captures made with --capture-logprobs (every v3 capture). Fails
loudly, rather than silently scoring zeros, if the field is missing.

Usage:
    python scripts/baseline_confidence.py \\
        --capture-dir shared/icr_capture/math500_thinking_qwen3v3 \\
        --labels shared/labels/qwen3v3/math500_labels.jsonl --target rescued
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_text import build_target, score_scalar                 # noqa: E402
from run_experiment import (auroc, bootstrap_auroc_ci, mean_ci,       # noqa: E402
                            mean_bootstrap_ci, stratified_split)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.capture_io import align_labels, load_labels, load_meta    # noqa: E402

logger = logging.getLogger("baseline_confidence")

SCALARS = ("mean_logprob", "min_logprob", "mean_entropy", "n_tokens_off")


def confidence_features(meta_rows: list[dict]) -> np.ndarray:
    """(N, 4) array of the thinking-off confidence scalars, in SCALARS order."""
    feats = np.full((len(meta_rows), len(SCALARS)), np.nan, dtype=np.float64)
    missing = 0
    for i, row in enumerate(meta_rows):
        conf = row.get("confidence_off") or {}
        for j, name in enumerate(SCALARS[:3]):
            val = conf.get(name)
            if val is None:
                missing += 1
            else:
                feats[i, j] = float(val)
        n_tok = row.get("n_tokens_off")
        feats[i, 3] = float(n_tok) if n_tok is not None else np.nan
    if missing:
        raise SystemExit(
            f"{missing} confidence value(s) missing — this capture was not made "
            "with --capture-logprobs, so this baseline cannot be run on it")
    bad = np.isnan(feats).any(axis=1)
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} row(s) have NaN confidence features")
    return feats


def score_logreg(X, y, seeds):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    per_seed, cis = [], []
    for seed in seeds:
        tr, _, te = stratified_split(y, seed)
        scaler = StandardScaler().fit(X[tr])           # fit on train only
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(scaler.transform(X[tr]), y[tr])
        s = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        per_seed.append(auroc(y[te], s))
        cis.append(bootstrap_auroc_ci(y[te], s))
    return per_seed, cis


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture-dir", required=True, nargs="+")
    p.add_argument("--labels", required=True, nargs="+")
    p.add_argument("--target", default="rescued",
                   choices=["helped", "needs_thinking", "rescued"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    p.add_argument("--out-file", default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    labels, rows = [], []
    for cap, lab in zip(args.capture_dir, args.labels):
        meta = load_meta(Path(cap))
        li = load_labels(Path(lab))
        idx = align_labels(meta, li)
        labels.extend(li)
        rows.extend(meta[i] for i in idx)

    y, keep = build_target(labels, args.target)
    X = confidence_features([rows[i] for i in keep])
    logger.info("target=%s  n=%d  base rate %.3f", args.target, len(y), y.mean())

    results = {}
    scored = {name: score_scalar(X[:, j], y, args.seeds)
              for j, name in enumerate(SCALARS)}
    scored["confidence_lr"] = score_logreg(X, y, args.seeds)
    for name, (ps, cis) in scored.items():
        agg = mean_ci(ps)
        boot = mean_bootstrap_ci(cis)
        results[name] = {"test_auroc": agg, "test_auroc_bootstrap": boot}
        logger.info("%-14s AUROC %.3f  bootstrap [%.3f, %.3f]",
                    name, agg["mean"], boot["ci"][0], boot["ci"][1])

    out = {"target": args.target, "n": int(len(y)),
           "base_rate": float(y.mean()), "seeds": args.seeds,
           "capture_dirs": args.capture_dir, "features": list(SCALARS),
           "sees_thinking_off_generation": True,
           "baselines": results}
    if args.out_file:
        Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_file).write_text(json.dumps(out, indent=2) + "\n")
        logger.info("wrote %s", args.out_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

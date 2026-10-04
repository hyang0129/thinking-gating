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
AUROC cannot be flattered by picking the direction after the fact. Output
shape (val_auroc for selection, a .predictions.json beside --out-file) is
the same as baseline_text.py's.

Requires captures made with --capture-logprobs (every v3 capture). Fails
loudly, rather than silently scoring zeros, if the field is missing.

Which confidence: captures without `confidence_version: 2` were scored with
their left padding attended (B1) and are invalid. `--confidence-source auto`
(the default) uses the re-scored `confidence_off_v2.shard*.jsonl` sidecar
written by scripts/rescore_confidence.py whenever the capture has one, and
records the source in the output JSON (`confidence_source`); `stored` forces
the values written at capture time.

Usage:
    python scripts/baseline_confidence.py \\
        --capture-dir shared/icr_capture/math500_thinking_qwen3v3 \\
        --labels shared/labels/qwen3v3/math500_labels.jsonl --target rescued
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_text import (build_target, predictions_path, score_scalar,  # noqa: E402
                           seed_record, summarize, write_results)
from run_experiment import stratified_split                           # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.capture_io import (CONFIDENCE_SOURCES, align_labels,      # noqa: E402
                              load_labels, load_meta, resolve_confidence)

logger = logging.getLogger("baseline_confidence")

SCALARS = ("mean_logprob", "min_logprob", "mean_entropy", "n_tokens_off")


def confidence_features(meta_rows: list[dict],
                        confidences: list[dict | None] | None = None) -> np.ndarray:
    """(N, 4) array of the thinking-off confidence scalars, in SCALARS order.

    `confidences` (one dict per row, from utils.capture_io.resolve_confidence)
    overrides the rows' stored `confidence_off`; n_tokens_off always comes
    from the meta row."""
    feats = np.full((len(meta_rows), len(SCALARS)), np.nan, dtype=np.float64)
    missing = 0
    for i, row in enumerate(meta_rows):
        conf = (confidences[i] if confidences is not None
                else row.get("confidence_off")) or {}
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
            f"{missing} confidence value(s) missing — the capture was not made "
            "with --capture-logprobs, or a re-scored row could not be rebuilt "
            "(prompt_hash_mismatch in the sidecar); this baseline cannot be "
            "run on it")
    bad = np.isnan(feats).any(axis=1)
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} row(s) have NaN confidence features")
    return feats


def score_logreg(X, y, seeds):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    out = []
    for seed in seeds:
        tr, va, te = stratified_split(y, seed)
        scaler = StandardScaler().fit(X[tr])           # fit on train only
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(scaler.transform(X[tr]), y[tr])
        s_va = clf.predict_proba(scaler.transform(X[va]))[:, 1]
        s_te = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        out.append(seed_record(seed, y, va, te, s_va, s_te))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture-dir", required=True, nargs="+")
    p.add_argument("--labels", required=True, nargs="+")
    p.add_argument("--target", default="rescued",
                   choices=["helped", "needs_thinking", "rescued"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    p.add_argument("--out-file", default=None)
    p.add_argument("--confidence-source", default="auto", choices=CONFIDENCE_SOURCES,
                   help="auto (default): the re-scored confidence_off_v2 sidecar "
                        "when a capture has one, else the stored values. v2: "
                        "require the sidecar. stored: force the values written "
                        "at capture time (v1 rows are B1-corrupted).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    labels, rows, confs, sources = [], [], [], []
    for cap, lab in zip(args.capture_dir, args.labels):
        meta = load_meta(Path(cap))
        conf, info = resolve_confidence(meta, Path(cap), args.confidence_source)
        sources.append(info)
        stale = int(info.get("confidence_version_counts", {}).get("1", 0))
        (logger.warning if stale else logger.info)(
            "%s: confidence from %s%s", cap, info["source"],
            f" -- {stale} row(s) are confidence v1 (B1: scored with pad tokens "
            "attended); run scripts/rescore_confidence.py" if stale else "")
        li = load_labels(Path(lab))
        idx = align_labels(meta, li)
        labels.extend(li)
        rows.extend(meta[i] for i in idx)
        confs.extend(conf[i] for i in idx)

    y, keep = build_target(labels, args.target)
    X = confidence_features([rows[i] for i in keep], [confs[i] for i in keep])
    logger.info("target=%s  n=%d  base rate %.3f", args.target, len(y), y.mean())

    results = {}
    scored = {name: score_scalar(X[:, j], y, args.seeds)
              for j, name in enumerate(SCALARS)}
    scored["confidence_lr"] = score_logreg(X, y, args.seeds)
    for name, records in scored.items():
        results[name] = summarize(records)
        agg, boot = results[name]["test_auroc"], results[name]["test_auroc_bootstrap"]
        logger.info("%-14s AUROC %.3f  bootstrap [%.3f, %.3f]  (val %.3f)",
                    name, agg["mean"], boot["ci"][0], boot["ci"][1],
                    results[name]["val_auroc"]["mean"])

    out = {"target": args.target, "n": int(len(y)),
           "base_rate": float(y.mean()), "seeds": args.seeds,
           "capture_dirs": args.capture_dir, "features": list(SCALARS),
           "sees_thinking_off_generation": True,
           "confidence_source": sources,
           "baselines": results}
    if args.out_file:
        write_results(args.out_file, out, scored, [labels[i] for i in keep], y,
                      args.target, "baseline_confidence")
        logger.info("wrote %s (+ %s)", args.out_file, predictions_path(args.out_file).name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

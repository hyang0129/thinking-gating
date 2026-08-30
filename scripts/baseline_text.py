#!/usr/bin/env python3
"""
baseline_text.py — what does the prefill state buy over cheap text features?

The probe needs a forward pass through an 8B model. That cost is only
justified if the prefill state beats predictors that read the raw question
and nothing else. This script runs those predictors on the *identical*
splits, seeds, target and metric as scripts/run_experiment.py, so the numbers
are directly comparable to its aggregate_metrics.json.

Baselines:
  length_chars   question length in characters (a single scalar feature)
  length_words   question length in whitespace tokens
  tfidf          TF-IDF over word 1-2 grams -> logistic regression
  tfidf_char     TF-IDF over char 3-5 grams -> logistic regression

Usage:
    python scripts/baseline_text.py \\
        --capture-dir shared/icr_capture/math500_thinking_qwen3 \\
        --labels shared/math500_labels.jsonl --target rescued
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import (auroc, bootstrap_auroc_ci, mean_ci,      # noqa: E402
                            mean_bootstrap_ci, stratified_split)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.capture_io import align_labels, load_labels, load_meta    # noqa: E402

logger = logging.getLogger("baseline_text")


def build_target(labels, target):
    correct_off = np.array([bool(l["correct_off"]) for l in labels])
    correct_on = np.array([bool(l["correct_on"]) for l in labels])
    if target == "helped":
        y = np.array([1 if l["label"] == "helped" else 0 for l in labels])
        keep = np.arange(len(y))
    elif target == "needs_thinking":
        y = (~correct_off).astype(int)
        keep = np.arange(len(y))
    elif target == "rescued":
        keep = np.flatnonzero(~correct_off)
        y = correct_on[keep].astype(int)
    else:
        raise ValueError(target)
    return y, keep


def score_scalar(values, y, seeds):
    """A single scalar feature needs no fitting -- AUROC reads it directly.

    Sign is chosen on TRAIN, so the test number cannot be flattered by
    flipping the comparison after the fact.
    """
    per_seed, cis = [], []
    for seed in seeds:
        tr, _, te = stratified_split(y, seed)
        sign = 1.0 if auroc(y[tr], values[tr]) >= 0.5 else -1.0
        s = sign * values[te]
        per_seed.append(auroc(y[te], s))
        cis.append(bootstrap_auroc_ci(y[te], s))
    return per_seed, cis


def score_tfidf(texts, y, seeds, analyzer, ngram):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    per_seed, cis = [], []
    for seed in seeds:
        tr, va, te = stratified_split(y, seed)
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram,
                              min_df=2, sublinear_tf=True)
        Xtr = vec.fit_transform([texts[i] for i in tr])   # fit on train only
        Xte = vec.transform([texts[i] for i in te])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y[tr])
        s = clf.predict_proba(Xte)[:, 1]
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

    labels, texts = [], []
    for cap, lab in zip(args.capture_dir, args.labels):
        meta = load_meta(Path(cap))
        li = load_labels(Path(lab))
        idx = align_labels(meta, li)
        labels.extend(li)
        texts.extend([meta[i]["question"] for i in idx])

    y, keep = build_target(labels, args.target)
    texts = [texts[i] for i in keep]
    logger.info("target=%s  n=%d  base rate %.3f", args.target, len(y), y.mean())

    chars = np.array([float(len(t)) for t in texts])
    words = np.array([float(len(t.split())) for t in texts])

    results = {}
    for name, (ps, cis) in {
        "length_chars": score_scalar(chars, y, args.seeds),
        "length_words": score_scalar(words, y, args.seeds),
        "tfidf_word": score_tfidf(texts, y, args.seeds, "word", (1, 2)),
        "tfidf_char": score_tfidf(texts, y, args.seeds, "char_wb", (3, 5)),
    }.items():
        agg = mean_ci(ps)
        boot = mean_bootstrap_ci(cis)
        results[name] = {"test_auroc": agg, "test_auroc_bootstrap": boot}
        logger.info("%-14s AUROC %.3f  bootstrap [%.3f, %.3f]",
                    name, agg["mean"], boot["ci"][0], boot["ci"][1])

    out = {"target": args.target, "n": int(len(y)),
           "base_rate": float(y.mean()), "seeds": args.seeds,
           "capture_dirs": args.capture_dir, "baselines": results}
    if args.out_file:
        Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_file).write_text(json.dumps(out, indent=2) + "\n")
        logger.info("wrote %s", args.out_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

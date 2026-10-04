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

Each baseline's node carries `test_auroc` (seed mean), `test_auroc_bootstrap`
(the interval to quote) and `val_auroc` (what compare_baselines.py selects the
best baseline on). With --out-file, per-row val/test scores by sample_id are
written beside it as `<out-file stem>.predictions.json`.

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
from run_experiment import predictions_record, stratified_split      # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.capture_io import align_labels, load_labels, load_meta    # noqa: E402
from utils.metrics import (auroc, bootstrap_auroc_ci, mean_bootstrap_ci,  # noqa: E402
                           mean_ci)

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


def seed_record(seed, y, va, te, val_scores, test_scores) -> dict:
    """One seed of one baseline: val/test AUROC, test bootstrap CI, raw scores.

    The validation AUROC is what compare_baselines.py selects "the best
    baseline" on; the test AUROC is only ever reported, never selected on.
    """
    return {"seed": seed, "val_row": va, "test_row": te,
            "val_score": np.asarray(val_scores, dtype=float),
            "test_score": np.asarray(test_scores, dtype=float),
            "val_auroc": auroc(y[va], val_scores),
            "test_auroc": auroc(y[te], test_scores),
            "test_ci": bootstrap_auroc_ci(y[te], test_scores)}


def score_scalar(values, y, seeds) -> list[dict]:
    """A single scalar feature needs no fitting -- AUROC reads it directly.

    Sign is chosen on TRAIN, so neither the validation nor the test number
    can be flattered by flipping the comparison after the fact.
    """
    out = []
    for seed in seeds:
        tr, va, te = stratified_split(y, seed)
        sign = 1.0 if auroc(y[tr], values[tr]) >= 0.5 else -1.0
        out.append(seed_record(seed, y, va, te, sign * values[va], sign * values[te]))
    return out


def score_tfidf(texts, y, seeds, analyzer, ngram) -> list[dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    out = []
    for seed in seeds:
        tr, va, te = stratified_split(y, seed)
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram,
                              min_df=2, sublinear_tf=True)
        Xtr = vec.fit_transform([texts[i] for i in tr])   # fit on train only
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y[tr])
        s_va = clf.predict_proba(vec.transform([texts[i] for i in va]))[:, 1]
        s_te = clf.predict_proba(vec.transform([texts[i] for i in te]))[:, 1]
        out.append(seed_record(seed, y, va, te, s_va, s_te))
    return out


def summarize(records: list[dict]) -> dict:
    """The per-baseline JSON node: seed-mean test AUROC, its bootstrap CI,
    and the seed-mean validation AUROC used for selection."""
    return {"test_auroc": mean_ci([r["test_auroc"] for r in records]),
            "test_auroc_bootstrap": mean_bootstrap_ci([r["test_ci"] for r in records]),
            "val_auroc": mean_ci([r["val_auroc"] for r in records])}


def predictions_path(out_file) -> Path:
    """`x.json` -> `x.predictions.json`, the per-row scores beside a result."""
    return Path(out_file).with_suffix(".predictions.json")


def write_results(out_file, out: dict, scored: dict[str, list[dict]], rows, y,
                  target: str, source: str) -> None:
    """Write the summary JSON and, beside it, the per-row predictions."""
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2) + "\n")
    correct_off = np.array([bool(r["correct_off"]) for r in rows])
    correct_on = np.array([bool(r["correct_on"]) for r in rows])
    per_seed = {}
    for name, records in scored.items():
        for r in records:
            node = per_seed.setdefault(r["seed"], {
                "val": {"row": list(map(int, r["val_row"])), "scores": {}},
                "test": {"row": list(map(int, r["test_row"])), "scores": {}}})
            node["val"]["scores"][name] = r["val_score"]
            node["test"]["scores"][name] = r["test_score"]
    preds = predictions_record(rows, y, correct_off, correct_on, per_seed,
                               target=target, source=source)
    predictions_path(out_file).write_text(json.dumps(preds) + "\n")


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

    scored = {
        "length_chars": score_scalar(chars, y, args.seeds),
        "length_words": score_scalar(words, y, args.seeds),
        "tfidf_word": score_tfidf(texts, y, args.seeds, "word", (1, 2)),
        "tfidf_char": score_tfidf(texts, y, args.seeds, "char_wb", (3, 5)),
    }
    results = {}
    for name, records in scored.items():
        results[name] = summarize(records)
        agg, boot = results[name]["test_auroc"], results[name]["test_auroc_bootstrap"]
        logger.info("%-14s AUROC %.3f  bootstrap [%.3f, %.3f]  (val %.3f)",
                    name, agg["mean"], boot["ci"][0], boot["ci"][1],
                    results[name]["val_auroc"]["mean"])

    out = {"target": args.target, "n": int(len(y)),
           "base_rate": float(y.mean()), "seeds": args.seeds,
           "capture_dirs": args.capture_dir, "baselines": results}
    if args.out_file:
        write_results(args.out_file, out, scored, [labels[i] for i in keep], y,
                      args.target, "baseline_text")
        logger.info("wrote %s (+ %s)", args.out_file, predictions_path(args.out_file).name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

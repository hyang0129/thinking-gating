#!/usr/bin/env python3
"""
generate_labels.py — turn paired thinking-off/on correctness into probe labels.

    python scripts/generate_labels.py \\
        --capture-dir shared/icr_capture/gsm8k_thinking_qwen3 \\
        --out-file shared/gsm8k_thinking_labels.jsonl

Binary label, exactly as the experiment design specifies:

    helped      correct_off == False and correct_on == True
    not_helped  everything else (both right, both wrong, or right -> wrong)

The graded label keeps the four cases apart for analysis — in particular
`hurt` (right -> wrong), which is invisible in the binary view but is the
thing a router most needs to avoid.

One caveat this script surfaces rather than hides: if a thinking-on generation
hit its token cap, its answer was cut off mid-reasoning, so a `not_helped`
label may be measuring the budget instead of the model. Those rows are counted
in the report and flagged per-row as `truncated_on`; `--drop-truncated`
excludes them entirely.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.capture_io import load_capture, load_config  # noqa: E402

logger = logging.getLogger("labels")

HELPED = "helped"
NOT_HELPED = "not_helped"


def graded_label(correct_off: bool, correct_on: bool) -> str:
    if not correct_off and correct_on:
        return "helped"
    if correct_off and not correct_on:
        return "hurt"
    return "no_change_correct" if correct_off else "no_change_wrong"


def build_label_row(row: dict) -> dict:
    correct_off = bool(row["correct_off"])
    correct_on = bool(row["correct_on"])
    return {
        "sample_id": row["sample_id"],
        "prompt_hash": row.get("prompt_hash"),
        "label": HELPED if (not correct_off and correct_on) else NOT_HELPED,
        "graded_label": graded_label(correct_off, correct_on),
        "correct_off": correct_off,
        "correct_on": correct_on,
        "difficulty": row.get("difficulty"),
        "truncated_on": bool(row.get("truncated_on", False)),
        "n_tokens_on": row.get("n_tokens_on"),
        "n_tokens_off": row.get("n_tokens_off"),
    }


def report(labels: list[dict], config: dict) -> dict:
    """Print and return the base rates that contextualize every later AUROC."""
    n = len(labels)
    graded = Counter(lab["graded_label"] for lab in labels)
    helped = sum(1 for lab in labels if lab["label"] == HELPED)
    truncated = sum(1 for lab in labels if lab["truncated_on"])

    summary = {
        "n": n,
        "base_rate_helped": helped / n if n else 0.0,
        "graded_counts": dict(graded),
        "accuracy_thinking_off": sum(lab["correct_off"] for lab in labels) / n if n else 0.0,
        "accuracy_thinking_on": sum(lab["correct_on"] for lab in labels) / n if n else 0.0,
        "oracle_accuracy": (
            sum(1 for lab in labels if lab["correct_off"] or lab["correct_on"]) / n
            if n else 0.0),
        "truncated_on": truncated,
        "by_difficulty": {},
        "model": config.get("model_name"),
        "task": config.get("task"),
    }

    for difficulty in sorted({lab["difficulty"] for lab in labels}, key=str):
        subset = [lab for lab in labels if lab["difficulty"] == difficulty]
        summary["by_difficulty"][str(difficulty)] = {
            "n": len(subset),
            "base_rate_helped": sum(1 for x in subset if x["label"] == HELPED) / len(subset),
            "accuracy_thinking_off": sum(x["correct_off"] for x in subset) / len(subset),
            "accuracy_thinking_on": sum(x["correct_on"] for x in subset) / len(subset),
        }

    logger.info("labelled %d samples from %s", n, config.get("task", "?"))
    logger.info("  thinking OFF accuracy : %.1f%%", 100 * summary["accuracy_thinking_off"])
    logger.info("  thinking ON  accuracy : %.1f%%", 100 * summary["accuracy_thinking_on"])
    logger.info("  oracle (best of both) : %.1f%%", 100 * summary["oracle_accuracy"])
    logger.info("  base rate 'helped'    : %.1f%%  (%d samples)",
                100 * summary["base_rate_helped"], helped)
    for name, count in sorted(graded.items()):
        logger.info("    %-18s %4d  (%.1f%%)", name, count, 100 * count / n)
    for difficulty, stats in summary["by_difficulty"].items():
        logger.info("  difficulty %-6s n=%-4d helped=%.1f%%  off=%.1f%% on=%.1f%%",
                    difficulty, stats["n"], 100 * stats["base_rate_helped"],
                    100 * stats["accuracy_thinking_off"],
                    100 * stats["accuracy_thinking_on"])
    if truncated:
        logger.warning(
            "  %d/%d thinking-on responses were truncated — their labels may "
            "reflect the token budget rather than reasoning", truncated, n)
    if helped == 0:
        logger.error("  no positive examples: a probe cannot be trained on this capture")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture-dir", required=True)
    p.add_argument("--out-file", required=True)
    p.add_argument("--drop-truncated", action="store_true",
                   help="Exclude rows whose thinking-on generation hit the token cap")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    capture_dir = Path(args.capture_dir)
    meta, activations = load_capture(capture_dir, mode="off")
    config = load_config(capture_dir)
    logger.info("capture: %d rows, activations %s", len(meta), activations.shape)

    labels = [build_label_row(row) for row in meta]
    if args.drop_truncated:
        before = len(labels)
        labels = [lab for lab in labels if not lab["truncated_on"]]
        logger.info("dropped %d truncated row(s)", before - len(labels))

    summary = report(labels, config)

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fh:
        for lab in labels:
            fh.write(json.dumps(lab) + "\n")
    summary_file = out_file.with_suffix(".summary.json")
    summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s and %s", out_file, summary_file.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

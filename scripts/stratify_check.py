#!/usr/bin/env python3
"""
stratify_check.py — is the probe predicting the query, or just its category?

    # BBH: does the signal survive within each of the 27 subtasks?
    python scripts/stratify_check.py \\
        --capture-dir shared/icr_capture/bbh_thinking_qwen3 \\
        --labels shared/bbh_labels.jsonl --group-from sample_id \\
        --out-dir output/bbh_subtask_check

    # MATH-500: does it survive within each subject?
    python scripts/stratify_check.py \\
        --capture-dir shared/icr_capture/math500_thinking_qwen3 \\
        --labels shared/math500_labels.jsonl --group-from task-field:subject \\
        --task math500 --out-dir output/math500_subject_check

Why this exists. A benchmark is usually a bundle of categories, and category
identity often predicts correctness almost perfectly — the model is uniformly
good at boolean_expressions and uniformly bad at dyck_languages. A probe can
therefore score a high overall AUROC by learning "which category is this",
which is not the query-level signal being claimed and does not exist outside
that benchmark.

Stratifying by *difficulty* does not catch this: BBH scored 0.838 overall with
a worst-difficulty stratum of 0.765, and still turned out to be mostly a
subtask detector once grouped by subtask instead. The tell was that it
transferred to other tasks at exactly chance.

This rewrites the grouping into the labels' `difficulty` field and defers to
run_experiment.py, so the numbers come from the same code path as every other
result rather than a parallel implementation.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def group_from_sample_id(sample_id: str) -> str:
    """`bbh-causal_judgement-7` -> `causal_judgement`; `gsm8k-test-3` -> `test`.

    Task modules build keys as `<task>-<group>-<index>`, so dropping the first
    and last segments recovers whatever the middle one encodes.
    """
    parts = sample_id.split("-")
    return "-".join(parts[1:-1]) if len(parts) > 2 else sample_id


def group_from_task_field(labels: list[dict], task: str, field: str) -> dict[str, str]:
    """Rejoin a dataset field the capture didn't carry, keyed by sample_id."""
    import importlib

    module = importlib.import_module(f"tasks.{task}")
    loader = next(getattr(module, n) for n in dir(module) if n.startswith("load_"))
    rows = loader()
    by_key = {r["key"]: r for r in rows}
    missing = [lab["sample_id"] for lab in labels if lab["sample_id"] not in by_key]
    if missing:
        raise SystemExit(
            f"{len(missing)} label(s) have no matching dataset row "
            f"(first: {missing[0]!r}) — the capture and the dataset disagree")
    return {lab["sample_id"]: str(by_key[lab["sample_id"]][field]).replace(" ", "_")
            for lab in labels}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture-dir", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--group-from", default="sample_id",
                   help="'sample_id' (middle segment) or 'task-field:<field>'")
    p.add_argument("--task", default=None, help="Task module, for task-field grouping")
    p.add_argument("--target", default="needs_thinking")
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--method", default="logreg")
    p.add_argument("--seeds", nargs="+", default=["42", "1", "2", "3", "4"])
    args = p.parse_args(argv)

    labels_path = Path(args.labels)
    labels = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]

    if args.group_from.startswith("task-field:"):
        if not args.task:
            raise SystemExit("--task is required for task-field grouping")
        mapping = group_from_task_field(labels, args.task, args.group_from.split(":", 1)[1])
        for lab in labels:
            lab["difficulty"] = mapping[lab["sample_id"]]
    else:
        for lab in labels:
            lab["difficulty"] = group_from_sample_id(lab["sample_id"])

    counts = collections.Counter(lab["difficulty"] for lab in labels)
    print(f"{len(labels)} rows across {len(counts)} group(s)")
    degenerate = [g for g in counts
                  if len({lab["correct_off"] for lab in labels if lab["difficulty"] == g}) == 1]
    if degenerate:
        # These are the groups that make category identity so predictive: the
        # model is uniformly right or uniformly wrong across the whole group.
        print(f"  {len(degenerate)} group(s) where thinking-off is all-right or "
              f"all-wrong — category identity alone settles them: {degenerate[:8]}")

    grouped_labels = labels_path.with_name(labels_path.stem + "_grouped.jsonl")
    with open(grouped_labels, "w", encoding="utf-8") as fh:
        for lab in labels:
            fh.write(json.dumps(lab) + "\n")

    cmd = [sys.executable, str(_PROJECT_ROOT / "scripts" / "run_experiment.py"),
           "--capture-dir", args.capture_dir, "--labels", str(grouped_labels),
           "--out-dir", args.out_dir, "--method", args.method,
           "--target", args.target, "--layer", str(args.layer),
           "--seeds", *[str(s) for s in args.seeds]]
    print(f"\nrunning: {' '.join(cmd[1:])}\n")
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())

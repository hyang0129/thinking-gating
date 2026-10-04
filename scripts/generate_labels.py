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

`--regrade` recomputes correct_off / correct_on from the stored raw
generations (`response_off` / `response_on` in the capture meta) with the
CURRENT task grader, so a grader fix reaches an existing capture without a GPU:

    python scripts/generate_labels.py --regrade \\
        --capture-dir shared/icr_capture/bbh_thinking_qwen3v3 \\
        --out-file shared/bbh_thinking_labels.jsonl

Regrading also applies one rule the capture-time grader lacked: a thinking-on
response that opened <think> and never closed it ran out of budget inside its
reasoning, so it has no final answer. It is graded wrong and flagged
`unclosed_think_on`; before, the grader's fallbacks scored the unfinished trace.
The stored grades are kept alongside as `correct_off_stored` /
`correct_on_stored`, and every row records the `grader_version` (a hash of the
task module's source) that produced it. Without `--regrade` the stored grades
are used as-is and `grader_version` is null. Regrading is opt-in so that
existing callers keep getting exactly the labels they got before; that includes
run_full_analysis.sh and the synthetic captures in tests/test_pipeline.py,
whose responses are placeholders. The task module comes from --task, else
config.json's `task`, else the capture-dir prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import logging
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.capture_io import load_capture, load_config  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

logger = logging.getLogger("labels")

HELPED = "helped"
NOT_HELPED = "not_helped"


def graded_label(correct_off: bool, correct_on: bool) -> str:
    if not correct_off and correct_on:
        return "helped"
    if correct_off and not correct_on:
        return "hurt"
    return "no_change_correct" if correct_off else "no_change_wrong"


# Capture-dir task prefixes that predate the one-name-per-task rule.
_TASK_ALIASES = {"mmlupro": "mmlu_pro", "gsm8k_full": "gsm8k", "gsm8kfull": "gsm8k",
                 "lsat_long": "lsat"}


def resolve_task(config: dict, capture_dir: Path, override: str | None = None) -> str:
    """Task module name: --task, else config.json's `task`, else the dir prefix."""
    task = override or config.get("task")
    if not task:
        prefix = capture_dir.name.split("_thinking")[0]
        task = _TASK_ALIASES.get(prefix, prefix)
    return task


def grader_version(task_module) -> str:
    """Short, stable id of the grader: '<task>:<sha256 of its module source>[:12]'."""
    source = Path(inspect.getsourcefile(task_module)).read_bytes()
    name = task_module.__name__.rsplit(".", 1)[-1]
    return f"{name}:{hashlib.sha256(source).hexdigest()[:12]}"


def has_unclosed_think(text: str | None) -> bool:
    """True if the last <think> is never closed — generation stopped mid-reasoning."""
    if not text:
        return False
    lower = text.lower()
    start = lower.rfind("<think>")
    return start != -1 and "</think>" not in lower[start:]


def regrade_row(row: dict, task_module, strip_thinking) -> dict:
    """Recompute correct_off/on from the raw responses with the current grader.

    Returns a copy of `row` with correct_* replaced and the stored grades kept
    as correct_*_stored. A response with an unclosed <think> is graded wrong.
    """
    for key in ("response_off", "response_on", "answer"):
        if key not in row:
            raise KeyError(f"meta row {row.get('sample_id')!r} has no {key!r}; cannot regrade")
    takes_question = "question" in inspect.signature(task_module.is_correct).parameters
    out = dict(row)
    for mode in ("off", "on"):
        response = row[f"response_{mode}"] or ""
        unclosed = has_unclosed_think(response)
        if unclosed:
            correct = False
        else:
            kwargs = {"question": row.get("question")} if takes_question else {}
            correct = bool(task_module.is_correct(strip_thinking(response), row["answer"], **kwargs))
        out[f"correct_{mode}_stored"] = bool(row.get(f"correct_{mode}"))
        out[f"correct_{mode}"] = correct
        out[f"unclosed_think_{mode}"] = unclosed
    return out


def build_label_row(row: dict, version: str | None = None) -> dict:
    correct_off = bool(row["correct_off"])
    correct_on = bool(row["correct_on"])
    label = {
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
        "truncated_off": bool(row.get("truncated_off", False)),
        "grader_version": version,
    }
    if "correct_off_stored" in row:  # regraded
        label.update({
            "correct_off_stored": row["correct_off_stored"],
            "correct_on_stored": row["correct_on_stored"],
            "unclosed_think_on": bool(row.get("unclosed_think_on", False)),
            "unclosed_think_off": bool(row.get("unclosed_think_off", False)),
        })
    return label


def regrade_flips(labels: list[dict]) -> dict:
    """How the regrade moved each grade and the binary label, vs. the stored grades."""
    flips: Counter = Counter()
    for lab in labels:
        for mode in ("off", "on"):
            old, new = lab[f"correct_{mode}_stored"], lab[f"correct_{mode}"]
            if old != new:
                flips[f"{mode}_{'false_to_true' if new else 'true_to_false'}"] += 1
        old_helped = (not lab["correct_off_stored"]) and lab["correct_on_stored"]
        if old_helped != (lab["label"] == HELPED):
            flips["label_changed"] += 1
        if any(lab[f"correct_{m}_stored"] != lab[f"correct_{m}"] for m in ("off", "on")):
            flips["rows_changed"] += 1
    return {key: flips.get(key, 0) for key in (
        "off_false_to_true", "off_true_to_false", "on_false_to_true",
        "on_true_to_false", "rows_changed", "label_changed")}


def report(labels: list[dict], config: dict, version: str | None = None) -> dict:
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
        "truncated_off": sum(1 for lab in labels if lab.get("truncated_off")),
        "by_difficulty": {},
        "model": config.get("model_name"),
        "task": config.get("task"),
        "grader_version": version,
        "regraded": bool(labels) and "correct_off_stored" in labels[0],
    }
    if summary["regraded"]:
        summary["unclosed_think_on"] = sum(1 for lab in labels if lab["unclosed_think_on"])
        summary["regrade_flips"] = regrade_flips(labels)
        stored_off = sum(lab["correct_off_stored"] for lab in labels)
        stored_on = sum(lab["correct_on_stored"] for lab in labels)
        summary["stored_accuracy_thinking_off"] = stored_off / n if n else 0.0
        summary["stored_accuracy_thinking_on"] = stored_on / n if n else 0.0

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
    if summary["regraded"]:
        logger.info("  regraded with %s: %d unclosed <think> on; flips %s",
                    version, summary["unclosed_think_on"], summary["regrade_flips"])
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
    p.add_argument("--regrade", action="store_true",
                   help="Recompute correct_off/on from the stored responses with the "
                        "current task grader (unclosed <think> on => wrong)")
    p.add_argument("--task", default=None,
                   help="Task module for --regrade (default: config.json 'task', "
                        "else the capture-dir prefix)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    capture_dir = Path(args.capture_dir)
    meta, activations = load_capture(capture_dir, mode="off")
    config = load_config(capture_dir)
    logger.info("capture: %d rows, activations %s", len(meta), activations.shape)

    version = None
    if args.regrade:
        task = resolve_task(config, capture_dir, args.task)
        task_module = importlib.import_module(f"tasks.{task}")
        from capture_inference_thinking import strip_thinking
        version = grader_version(task_module)
        logger.info("regrading %d rows with tasks.%s (%s)", len(meta), task, version)
        meta = [regrade_row(row, task_module, strip_thinking) for row in meta]

    labels = [build_label_row(row, version) for row in meta]
    if args.drop_truncated:
        before = len(labels)
        labels = [lab for lab in labels if not lab["truncated_on"]]
        logger.info("dropped %d truncated row(s)", before - len(labels))

    summary = report(labels, config, version)

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

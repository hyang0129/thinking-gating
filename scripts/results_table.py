#!/usr/bin/env python3
"""
results_table.py — one table across every probe and transfer run.

    python scripts/results_table.py --metrics-dir paper/results/metrics
    python scripts/results_table.py --metrics-dir paper/results/metrics --format csv

Reads the aggregate JSON that run_experiment.py and eval_transfer.py write and
renders them together, so a claim in the paper can be checked against the run
that produced it instead of a number retyped from a log.

Columns, and why each is here:

    AUROC           mean ± 95% CI over seeds, on held-out test splits.
    strat           the confound check — the *worst* within-difficulty AUROC.
                    The helped rate climbs steeply with difficulty, so a probe
                    that only detects hard questions still scores well overall.
                    A probe is only credible if this column, not just AUROC,
                    stays above chance.
    never/always    task accuracy routing nothing / everything to thinking.
    oracle          accuracy of a router that already knows the answer.
    routed          accuracy of the probe's routing, threshold set on validation.
    waste           share of thinking compute that buys nothing: 1 − the
                    smallest routed fraction that still matches always-think.

A probe is only interesting when routed lands strictly above max(never, always)
or when waste is large — otherwise "always think" is the better policy and the
probe is a solution without a problem.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def fmt(value: float, digits: int = 3) -> str:
    return "  -  " if value is None or math.isnan(value) else f"{value:.{digits}f}"


def fmt_ci(node: dict) -> str:
    if not node or math.isnan(node.get("mean", float("nan"))):
        return "  -  "
    lo, hi = node.get("ci", [float("nan")] * 2)
    return f"{node['mean']:.3f} [{lo:.3f},{hi:.3f}]"


def worst_stratum(agg: dict) -> tuple[str, float]:
    """Lowest within-difficulty AUROC, which is the honest summary of it."""
    by_diff = agg.get("test_auroc_by_difficulty") or {}
    scored = [(k, v["mean"]) for k, v in by_diff.items()
              if v and not math.isnan(v.get("mean", float("nan")))]
    if not scored:
        return "-", float("nan")
    return min(scored, key=lambda kv: kv[1])


def load_probe_rows(metrics_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(metrics_dir.glob("*.json")):
        if path.name.startswith("transfer__"):
            continue
        data = json.loads(path.read_text())
        agg = data.get("aggregate")
        if not agg:
            continue
        task = data.get("task")
        task = "+".join(task) if isinstance(task, list) else (task or path.stem)
        stratum, stratum_auc = worst_stratum(agg)
        waste = agg.get("min_routed_for_always_think_accuracy", {})
        rows.append({
            "run": path.stem, "task": task, "target": data.get("target", "helped"),
            "n": data.get("n_samples"), "base_rate": data.get("base_rate_helped"),
            "auroc": agg["test_auroc"], "worst_stratum": stratum,
            "worst_stratum_auroc": stratum_auc,
            "never": agg["baseline_never_think"]["mean"],
            "always": agg["baseline_always_think"]["mean"],
            "oracle": agg["oracle"]["mean"],
            "routed": agg["test_routed_accuracy"]["mean"],
            "waste": 1 - waste["mean"] if waste and not math.isnan(waste.get("mean", float("nan"))) else float("nan"),
        })
    return rows


def load_transfer_rows(metrics_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(metrics_dir.glob("transfer__*.json")):
        data = json.loads(path.read_text())
        rows.append({
            "run": path.stem,
            "source": data.get("source_task", "?"),
            "target": data.get("target_task", "?"),
            "objective": data.get("target_objective", ""),
            "source_auroc": data.get("source_test_auroc", float("nan")),
            "transfer_auroc": data.get("transfer_auroc", {}),
            "drop_pp": data.get("auroc_drop_pp", float("nan")),
            "verdict": data.get("verdict", ""),
        })
    return rows


def render_probes(rows: list[dict]) -> str:
    if not rows:
        return "(no probe runs found)\n"
    head = (f"{'task':<16}{'target':<16}{'n':>6}{'base':>7}  {'AUROC [95% CI]':<22}"
            f"{'worst stratum':<22}{'never':>7}{'always':>8}{'oracle':>8}{'routed':>8}{'waste':>7}")
    out = [head, "-" * len(head)]
    for r in sorted(rows, key=lambda r: (r["task"], r["target"])):
        strat = f"{r['worst_stratum']}:{fmt(r['worst_stratum_auroc'])}"
        out.append(
            f"{r['task']:<16}{r['target']:<16}{r['n'] or 0:>6}"
            f"{fmt(r['base_rate'], 2):>7}  {fmt_ci(r['auroc']):<22}{strat:<22}"
            f"{fmt(r['never']):>7}{fmt(r['always']):>8}{fmt(r['oracle']):>8}"
            f"{fmt(r['routed']):>8}{fmt(r['waste'], 2):>7}")
    return "\n".join(out) + "\n"


def render_transfers(rows: list[dict]) -> str:
    if not rows:
        return ""
    head = (f"{'source':<18}{'target':<18}{'objective':<16}{'src AUROC':>10}"
            f"  {'transfer AUROC':<22}{'drop pp':>9}  verdict")
    out = ["", "Transfer (no retraining: source scaler and threshold reused)",
           head, "-" * len(head)]
    for r in sorted(rows, key=lambda r: (r["source"], r["target"])):
        out.append(
            f"{r['source']:<18}{r['target']:<18}{r['objective']:<16}"
            f"{fmt(r['source_auroc']):>10}  {fmt_ci(r['transfer_auroc']):<22}"
            f"{r['drop_pp']:>9.1f}  {r['verdict']}")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics-dir", default="paper/results/metrics")
    p.add_argument("--format", default="text", choices=["text", "csv"])
    p.add_argument("--out", default=None, help="Write here instead of stdout")
    args = p.parse_args(argv)

    metrics_dir = Path(args.metrics_dir)
    if not metrics_dir.is_dir():
        print(f"no such metrics dir: {metrics_dir}", file=sys.stderr)
        return 1

    probes = load_probe_rows(metrics_dir)
    transfers = load_transfer_rows(metrics_dir)

    if args.format == "csv":
        stream = open(args.out, "w", newline="") if args.out else sys.stdout
        writer = csv.writer(stream)
        writer.writerow(["kind", "task", "target", "n", "base_rate", "auroc",
                         "auroc_lo", "auroc_hi", "worst_stratum",
                         "worst_stratum_auroc", "never", "always", "oracle",
                         "routed", "waste"])
        for r in probes:
            writer.writerow(["probe", r["task"], r["target"], r["n"], r["base_rate"],
                             r["auroc"]["mean"], r["auroc"]["ci"][0], r["auroc"]["ci"][1],
                             r["worst_stratum"], r["worst_stratum_auroc"], r["never"],
                             r["always"], r["oracle"], r["routed"], r["waste"]])
        for r in transfers:
            writer.writerow(["transfer", f"{r['source']}->{r['target']}", r["objective"],
                             "", "", r["transfer_auroc"].get("mean"),
                             *(r["transfer_auroc"].get("ci") or ["", ""]),
                             "", "", "", "", "", "", r["drop_pp"]])
        if args.out:
            stream.close()
            print(f"wrote {args.out}")
        return 0

    text = render_probes(probes) + render_transfers(transfers)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

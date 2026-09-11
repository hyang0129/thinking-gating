#!/usr/bin/env python3
"""
compare_baselines.py — the probe next to every baseline, one row per (task, target).

Reads what run_full_analysis.sh writes under paper/results/metrics/<slug>/ and
renders the comparison the paper actually needs: the prefill probe's bootstrap
CI beside the best text baseline and the best thinking-off confidence
baseline, on identical splits. Everything is `test_auroc_bootstrap.ci`; the
seed-spread interval is never shown here.

    python scripts/compare_baselines.py --metrics-dir paper/results/metrics/qwen3v3
    python scripts/compare_baselines.py --metrics-dir paper/results/metrics/qwen3v3 --format csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def ci(node: dict) -> str:
    lo, hi = node["ci"]
    return f"{node['mean']:.3f} [{lo:.3f}, {hi:.3f}]"


def point_and_interval(entry: dict) -> dict:
    """Mean over seeds from test_auroc, interval from the bootstrap node.

    run_experiment.py and the baseline scripts store the seed-mean under
    test_auroc and only the percentile interval under test_auroc_bootstrap;
    the seed-spread interval in test_auroc is never used here.
    """
    return {"mean": entry["test_auroc"]["mean"],
            "ci": entry["test_auroc_bootstrap"]["ci"]}


def best(baselines: dict) -> tuple[str, dict]:
    name = max(baselines, key=lambda k: baselines[k]["test_auroc"]["mean"])
    return name, point_and_interval(baselines[name])


def rows_for(metrics_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(metrics_dir.glob("*__*.json")):
        if path.name.startswith("transfer__"):
            continue
        data = json.loads(path.read_text())
        agg = data.get("aggregate")
        if not agg:
            continue
        task, target = path.stem.split("__", 1)
        row = {"task": task, "target": target, "n": data.get("n_samples"),
               "base_rate": data.get("base_rate_helped"),
               "probe": point_and_interval(agg)}
        for kind in ("text", "confidence"):
            bpath = metrics_dir / "baselines" / f"{kind}__{task}__{target}.json"
            if bpath.exists():
                name, node = best(json.loads(bpath.read_text())["baselines"])
                row[kind] = (name, node)
        rows.append(row)
    return rows


def render_text(rows: list[dict]) -> str:
    out = [f"{'task':<10}{'target':<16}{'n':>6}{'base':>7}  "
           f"{'prefill probe':<24}{'best text':<38}{'best confidence':<38}"]
    out.append("-" * len(out[0]))
    for r in rows:
        text = f"{r['text'][0]} {ci(r['text'][1])}" if "text" in r else "-"
        conf = f"{r['confidence'][0]} {ci(r['confidence'][1])}" if "confidence" in r else "-"
        base = f"{r['base_rate']:.3f}" if r.get("base_rate") is not None else "-"
        out.append(f"{r['task']:<10}{r['target']:<16}{str(r['n']):>6}{base:>7}  "
                   f"{ci(r['probe']):<24}{text:<38}{conf:<38}")
    return "\n".join(out)


def render_csv(rows: list[dict]) -> str:
    out = ["task,target,n,base_rate,probe_mean,probe_lo,probe_hi,"
           "text_name,text_mean,text_lo,text_hi,conf_name,conf_mean,conf_lo,conf_hi"]
    for r in rows:
        cells = [r["task"], r["target"], str(r["n"]),
                 f"{r['base_rate']:.4f}" if r.get("base_rate") is not None else "",
                 f"{r['probe']['mean']:.4f}", f"{r['probe']['ci'][0]:.4f}", f"{r['probe']['ci'][1]:.4f}"]
        for kind in ("text", "confidence"):
            if kind in r:
                name, node = r[kind]
                cells += [name, f"{node['mean']:.4f}", f"{node['ci'][0]:.4f}", f"{node['ci'][1]:.4f}"]
            else:
                cells += ["", "", "", ""]
        out.append(",".join(cells))
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics-dir", required=True)
    p.add_argument("--format", default="text", choices=["text", "csv"])
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    rows = rows_for(Path(args.metrics_dir))
    if not rows:
        print(f"no probe metrics under {args.metrics_dir}", file=sys.stderr)
        return 1
    text = render_text(rows) if args.format == "text" else render_csv(rows)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

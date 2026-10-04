#!/usr/bin/env python3
"""
compare_baselines.py — the probe next to every baseline, one row per (task, target).

Reads what run_full_analysis.sh writes under <metrics-dir> (by default
output/<slug>/metrics/) and renders the comparison the paper needs: the
prefill probe's bootstrap CI beside the best text baseline and the best
thinking-off confidence baseline, on identical splits. Every interval is a
bootstrap over test rows; the seed-spread interval is never shown here.

**Which baseline is "best" is decided on validation.** Each baseline file
records the seed-mean validation AUROC (`val_auroc`) and the best is its
argmax; the test AUROC is then reported, never selected on. Files written
before `val_auroc` existed (the published v3/Nemotron baselines) can only be
ranked on test — those rows are marked `*` ("selected on test") and are
optimistic for the baseline by the selection effect.

**Probe − best baseline, paired.** When per-row predictions exist for both
(`<task>__<target>.predictions.json` from run_experiment.py, and
`baselines/<kind>__<task>__<target>.predictions.json` from the baseline
scripts), the difference in test AUROC is bootstrapped on the *identical* test
rows (matched by sample_id), per seed, and the per-seed estimates and interval
bounds are averaged — the same convention as test_auroc_bootstrap. "best"
here is the val-selected best across text and confidence baselines together.
Two marginal intervals that overlap say nothing about whether one method beats
the other; this one does.

    python scripts/compare_baselines.py --metrics-dir output/qwen3v3/metrics
    python scripts/compare_baselines.py --metrics-dir output/qwen3v3/metrics --format csv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.metrics import paired_bootstrap_auroc_diff  # noqa: E402

NAN = float("nan")


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


def _val(entry: dict) -> float:
    v = (entry.get("val_auroc") or {}).get("mean")
    return NAN if v is None else float(v)


def best(baselines: dict) -> tuple[str, dict, str]:
    """(name, {mean, ci}, selected_on). Val-selected when every baseline in
    the file has a validation AUROC; otherwise falls back to test."""
    vals = {k: _val(v) for k, v in baselines.items()}
    if vals and all(not math.isnan(v) for v in vals.values()):
        name = max(sorted(vals), key=lambda k: vals[k])
        selected_on = "val"
    else:
        name = max(sorted(baselines), key=lambda k: baselines[k]["test_auroc"]["mean"])
        selected_on = "test"
    node = point_and_interval(baselines[name])
    node["val"] = vals.get(name, NAN)
    return name, node, selected_on


def predictions_path(path: Path) -> Path:
    return path.with_suffix(".predictions.json")


def load_predictions(path: Path) -> dict | None:
    pp = predictions_path(path)
    return json.loads(pp.read_text()) if pp.exists() else None


def paired_difference(probe_preds: dict, base_preds: dict, name: str,
                      n_boot: int = 2000) -> dict | None:
    """Seed-averaged paired bootstrap of AUROC(probe) − AUROC(baseline `name`)
    on identical test rows. None if the rows cannot be matched."""
    per_seed = []
    for seed, p_node in probe_preds["seeds"].items():
        b_node = base_preds["seeds"].get(seed)
        if b_node is None:
            continue
        pt, bt = p_node["test"], b_node["test"]
        if name not in bt.get("scores", {}):
            return None
        b_score = dict(zip(bt["sample_id"], bt["scores"][name]))
        if set(b_score) != set(pt["sample_id"]):
            # Different test rows: the splits disagree, so pairing is invalid.
            return None
        y = np.asarray(pt["y"])
        a = np.asarray(pt["score"], dtype=float)
        b = np.asarray([b_score[sid] for sid in pt["sample_id"]], dtype=float)
        res = paired_bootstrap_auroc_diff(y, a, b, n_boot=n_boot)
        res["seed"] = int(seed)
        per_seed.append(res)
    good = [r for r in per_seed if not math.isnan(r["ci"][0])]
    if not good:
        return None
    return {"baseline": name,
            "mean": float(np.mean([r["estimate"] for r in good])),
            "ci": [float(np.mean([r["ci"][0] for r in good])),
                   float(np.mean([r["ci"][1] for r in good]))],
            "p_le_zero": float(np.mean([r["p_le_zero"] for r in good])),
            "n_seeds": len(good)}


def rows_for(metrics_dir: Path, n_boot: int = 2000) -> list[dict]:
    rows = []
    for path in sorted(metrics_dir.glob("*__*.json")):
        if path.name.startswith("transfer__") or path.name.endswith(".predictions.json"):
            continue
        data = json.loads(path.read_text())
        agg = data.get("aggregate")
        if not agg:
            continue
        task, target = path.stem.split("__", 1)
        row = {"task": task, "target": target, "n": data.get("n_samples"),
               "base_rate": data.get("base_rate_helped"),
               "probe": point_and_interval(agg)}
        candidates = []          # (val, kind, name, file) for the overall best
        for kind in ("text", "confidence"):
            bpath = metrics_dir / "baselines" / f"{kind}__{task}__{target}.json"
            if bpath.exists():
                baselines = json.loads(bpath.read_text())["baselines"]
                name, node, selected_on = best(baselines)
                row[kind] = (name, node, selected_on)
                candidates.append((node["val"], kind, name, bpath, selected_on))
        probe_preds = load_predictions(path)
        if probe_preds and candidates and all(c[4] == "val" for c in candidates):
            _, kind, name, bpath, _ = max(candidates, key=lambda c: (c[0], c[1]))
            base_preds = load_predictions(bpath)
            if base_preds:
                diff = paired_difference(probe_preds, base_preds, name, n_boot=n_boot)
                if diff:
                    diff["kind"] = kind
                    row["diff"] = diff
        rows.append(row)
    return rows


def _baseline_cell(r: dict, kind: str) -> str:
    if kind not in r:
        return "-"
    name, node, selected_on = r[kind]
    return f"{name}{'*' if selected_on == 'test' else ''} {ci(node)}"


def _diff_cell(r: dict) -> str:
    d = r.get("diff")
    if not d:
        return "-"
    return f"{d['mean']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] vs {d['baseline']}"


def render_text(rows: list[dict]) -> str:
    out = [f"{'task':<10}{'target':<16}{'n':>6}{'base':>7}  "
           f"{'prefill probe':<24}{'best text (val-sel.)':<38}"
           f"{'best confidence (val-sel.)':<38}{'probe - best, paired':<44}"]
    out.append("-" * len(out[0]))
    for r in rows:
        base = f"{r['base_rate']:.3f}" if r.get("base_rate") is not None else "-"
        out.append(f"{r['task']:<10}{r['target']:<16}{str(r['n']):>6}{base:>7}  "
                   f"{ci(r['probe']):<24}{_baseline_cell(r, 'text'):<38}"
                   f"{_baseline_cell(r, 'confidence'):<38}{_diff_cell(r):<44}")
    if any(r.get(k) and r[k][2] == "test" for r in rows for k in ("text", "confidence")):
        out.append("* selected on test: this baseline file has no val_auroc "
                   "(written before validation selection existed)")
    return "\n".join(out)


def render_csv(rows: list[dict]) -> str:
    out = ["task,target,n,base_rate,probe_mean,probe_lo,probe_hi,"
           "text_name,text_selected_on,text_mean,text_lo,text_hi,"
           "conf_name,conf_selected_on,conf_mean,conf_lo,conf_hi,"
           "diff_vs,diff_mean,diff_lo,diff_hi,diff_p_le_zero"]
    for r in rows:
        cells = [r["task"], r["target"], str(r["n"]),
                 f"{r['base_rate']:.4f}" if r.get("base_rate") is not None else "",
                 f"{r['probe']['mean']:.4f}", f"{r['probe']['ci'][0]:.4f}", f"{r['probe']['ci'][1]:.4f}"]
        for kind in ("text", "confidence"):
            if kind in r:
                name, node, selected_on = r[kind]
                cells += [name, selected_on, f"{node['mean']:.4f}",
                          f"{node['ci'][0]:.4f}", f"{node['ci'][1]:.4f}"]
            else:
                cells += ["", "", "", "", ""]
        d = r.get("diff")
        cells += ([d["baseline"], f"{d['mean']:.4f}", f"{d['ci'][0]:.4f}",
                   f"{d['ci'][1]:.4f}", f"{d['p_le_zero']:.4f}"] if d else ["", "", "", "", ""])
        out.append(",".join(cells))
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics-dir", required=True)
    p.add_argument("--format", default="text", choices=["text", "csv"])
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    rows = rows_for(Path(args.metrics_dir), n_boot=args.n_boot)
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

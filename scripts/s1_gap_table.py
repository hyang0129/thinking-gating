#!/usr/bin/env python3
"""
s1_gap_table.py — Phase 0 step 1: S1 (decision readout) vs no-think vs think.

Joins a D0 readout (scripts/capture_readout.py) with the REGRADED labels of
the same capture (scripts/generate_labels.py --regrade) and reports, per task
and per family (BBH subtask, MMLU-Pro category):

  1. Tier table: accuracy of S1 / no-think / think on the SAME covered items,
     percentile-bootstrap CIs, n, the random-guess floor (mean 1/n_options),
     and the paired S1 -> think and S1 -> no-think gaps.
  2. S1 calibration of p_max: ECE, a reliability table, mean p_max on right vs
     wrong, AUROC(p_max -> S1 correct).
  3. Routing S1 -> think on S1's own confidence (escalate lowest p_max first):
     the exact routed-accuracy curve, nAUC over 0-50% escalation vs random and
     oracle (bootstrap CI), the escalation needed to come within 1 and 3 pp of
     always-think, and the same for a privileged family-oracle router (each
     family escalated in order of its true mean gain, random within family)
     with the paired confidence - family-oracle ΔnAUC and the pooled
     within-family AUROC of the confidence score: the family-detector check.
  4. Three tiers, S1 -> no-think -> think, under a two-threshold policy on
     p_max (p >= τ_hi: S1; τ_lo <= p < τ_hi: no-think; else think). Thresholds
     chosen to minimise mean generated tokens subject to accuracy >= always-think
     - δ, both in-sample (optimistic) and 5-fold cross-fitted; plus the oracle
     split (cheapest tier that is right).

Every statistic comes from utils/metrics.py.

    .venv/bin/python scripts/s1_gap_table.py \\
        --task mmlu_pro shared/icr_capture/mmlu_pro_readout_qwen3v3 <labels>/mmlu_pro_labels.jsonl \\
        --task bbh shared/icr_capture/bbh_readout_qwen3v3 <labels>/bbh_labels.jsonl \\
        --out-json output/qwen3v3/s1_gap/s1_gap.json --out-md output/qwen3v3/s1_gap/s1_gap.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402

from utils import metrics as M  # noqa: E402
from utils.capture_io import load_labels  # noqa: E402

N_BOOT = 2000
DELTAS = (0.01, 0.03)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_task(task: str, readout_dir: Path, labels_file: Path,
              families: dict[str, str] | None = None) -> dict:
    """Covered readout rows joined to regraded labels, as aligned arrays."""
    readout = [json.loads(x) for x in (Path(readout_dir) / "readout.jsonl")
               .read_text(encoding="utf-8").splitlines() if x.strip()]
    labels = {r["sample_id"]: r for r in load_labels(labels_file)}
    if not all(r.get("grader_version") for r in labels.values()):
        raise SystemExit(f"{labels_file}: not regraded (no grader_version); run "
                         "generate_labels.py --regrade")
    missing = [r["sample_id"] for r in readout if r["sample_id"] not in labels]
    if missing:
        raise SystemExit(f"{task}: {len(missing)} readout rows have no label, "
                         f"e.g. {missing[:3]}")
    cov = [r for r in readout if r["covered"]]
    fam = []
    for r in cov:
        f = r.get("family") or (families or {}).get(r["sample_id"])
        fam.append(str(f) if f else None)
    lab = [labels[r["sample_id"]] for r in cov]
    return {
        "task": task, "n_rows": len(readout), "n_covered": len(cov),
        "excluded": {k: sum(1 for r in readout if not r["covered"]
                            and r["exclude_reason"] == k)
                     for k in sorted({str(r["exclude_reason"]) for r in readout
                                      if not r["covered"]})},
        "sample_id": [r["sample_id"] for r in cov],
        "family": fam, "kind": [r["kind"] for r in cov],
        "s1": np.array([bool(r["s1_correct"]) for r in cov]),
        "off": np.array([bool(x["correct_off"]) for x in lab]),
        "on": np.array([bool(x["correct_on"]) for x in lab]),
        "p_max": np.array([float(r["p_max"]) for r in cov]),
        "floor": np.array([1.0 / r["n_options"] for r in cov]),
        "option_mass": np.array([float(r["option_mass"]) for r in cov]),
        "tok_off": np.array([float(x["n_tokens_off"]) for x in lab]),
        "tok_on": np.array([float(x["n_tokens_on"]) for x in lab]),
        "trunc_on": np.array([bool(x.get("truncated_on")) for x in lab]),
        "trunc_off": np.array([bool(x.get("truncated_off")) for x in lab]),
        "unclosed_on": np.array([bool(x.get("unclosed_think_on")) for x in lab]),
        "grader_version": sorted({x["grader_version"] for x in lab}),
        "commits": sorted({str(r.get("git_commit")) for r in cov}),
    }


def subset(d: dict, mask: np.ndarray, name: str) -> dict:
    out = {"task": name}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v[mask]
        elif isinstance(v, list) and len(v) == len(mask):
            out[k] = [x for x, m in zip(v, mask) if m]
    return out


def concat(parts: list[dict], name: str) -> dict:
    out = {"task": name}
    for k, v in parts[0].items():
        if isinstance(v, np.ndarray):
            out[k] = np.concatenate([p[k] for p in parts])
        elif k in ("sample_id", "kind"):
            out[k] = sum((p[k] for p in parts), [])
    out["family"] = sum(([f"{p['task']}:{f}" if f else None for f in p["family"]]
                         for p in parts), [])
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _mean(a):
    return float(np.mean(a)) if len(a) else math.nan


def acc_ci(x: np.ndarray, seed: int = 0) -> dict:
    x = x.astype(float)
    return {"acc": _mean(x), "ci": M.bootstrap_ci(_mean, x, n_boot=N_BOOT,
                                                   seed=seed)["ci"]}


def tier_row(d: dict) -> dict:
    gap = lambda a, b: float(np.mean(a) - np.mean(b))  # noqa: E731
    s1, off, on = (d[k].astype(float) for k in ("s1", "off", "on"))
    return {
        "n": int(len(s1)), "floor": _mean(d["floor"]),
        "s1": acc_ci(s1), "no_think": acc_ci(off), "think": acc_ci(on),
        "gap_think_minus_s1": M.paired_bootstrap(gap, on, s1, n_boot=N_BOOT),
        "gap_nothink_minus_s1": M.paired_bootstrap(gap, off, s1, n_boot=N_BOOT),
        "gap_think_minus_nothink": M.paired_bootstrap(gap, on, off, n_boot=N_BOOT),
    }


def calibration(d: dict) -> dict:
    p, ok = d["p_max"], d["s1"]
    rel = M.reliability_bins(p, ok, n_bins=10, strategy="uniform")
    au = M.auroc(ok, p)
    return {
        "ece": rel["ece"], "mce": rel["mce"], "bins": rel["bins"],
        "ece_quantile": M.reliability_bins(p, ok, n_bins=10, strategy="quantile")["ece"],
        "mean_p_max": _mean(p), "accuracy": _mean(ok),
        "overconfidence": M.signed_overconfidence(p, ok, n_boot=N_BOOT)["overall"],
        "mean_p_max_right": _mean(p[ok]), "mean_p_max_wrong": _mean(p[~ok]),
        "auroc_p_max_correct": au,
        "auroc_ci": M.bootstrap_auroc_ci(ok, p, n_boot=N_BOOT)["ci"],
    }


def _random_fraction(a0: float, a1: float, target: float) -> float:
    if a0 >= target - 1e-9:
        return 0.0
    if a1 <= a0 or a1 < target - 1e-9:
        return 1.0
    return float((target - a0) / (a1 - a0))


def family_gain_scores(d: dict) -> np.ndarray | None:
    """Privileged router: each row scored by its family's true mean gain."""
    fam = d["family"]
    if any(f is None for f in fam):
        return None
    gain = d["on"].astype(float) - d["s1"].astype(float)
    means = {f: float(gain[[g == f for g in fam]].mean()) for f in set(fam)}
    return np.array([means[f] for f in fam])


def routing(d: dict, cheap_key: str = "s1") -> dict:
    """Route cheap tier -> think on the S1 confidence (low p_max first)."""
    cheap, on = d[cheap_key], d["on"]
    score = -d["p_max"]
    a0, a1 = _mean(cheap), _mean(on)
    curve = M.routed_accuracy_curve(cheap, on, score)
    orc = M.oracle_curve(cheap, on)
    nauc = M.routed_nauc(cheap, on, score, 0.0, 0.5)
    nauc_ci = M.bootstrap_ci(lambda c, o, s: M.routed_nauc(c, o, s)["nauc"],
                             cheap, on, score, n_boot=N_BOOT)["ci"]
    out = {
        "acc_cheap": a0, "acc_think": a1,
        "nauc": nauc["nauc"], "nauc_ci": nauc_ci,
        "mean_acc_0_50": nauc["mean_accuracy"],
        "random_mean_acc_0_50": nauc["random_mean_accuracy"],
        "oracle_mean_acc_0_50": nauc["oracle_mean_accuracy"],
        "acc_at": {f"{int(f * 100)}%": float(M.curve_accuracy_at(curve, f))
                   for f in (0.1, 0.2, 0.3, 0.5)},
        "curve": {"fraction": [float(x) for x in curve["vertex_fraction"]],
                  "accuracy": [float(x) for x in curve["vertex_accuracy"]]},
        "escalation_to_target": {},
    }
    fam_scores = family_gain_scores(d)
    if fam_scores is not None:
        fcurve = M.routed_accuracy_curve(cheap, on, fam_scores)
        out["family_oracle"] = {
            "n_families": len(set(d["family"])),
            "nauc": M.routed_nauc(cheap, on, fam_scores)["nauc"],
            "mean_acc_0_50": M.routed_nauc(cheap, on, fam_scores)["mean_accuracy"],
            "delta_nauc_conf_minus_family": M.paired_bootstrap_nauc_diff(
                cheap, on, score, fam_scores, n_boot=N_BOOT),
            "within_family_auroc_conf_vs_s1_wrong": M.within_group_stats(
                ~cheap, score, d["family"]),
            "within_family_auroc_conf_vs_helped": M.within_group_stats(
                (~cheap) & on, score, d["family"]),
        }
    for delta in DELTAS:
        target = a1 - delta
        row = {"target": target,
               "confidence": M.min_fraction_for_accuracy(curve, target),
               "oracle": M.min_fraction_for_accuracy(orc, target),
               "random": _random_fraction(a0, a1, target)}
        if fam_scores is not None:
            row["family_oracle"] = M.min_fraction_for_accuracy(fcurve, target)
        out["escalation_to_target"][f"{int(delta * 100)}pp"] = row
    return out


# ---------------------------------------------------------------------------
# Three tiers
# ---------------------------------------------------------------------------

def three_tier_eval(d: dict, idx: np.ndarray, lo: float, hi: float) -> dict:
    p = d["p_max"][idx]
    t_s1, t_nt = p >= hi, (p >= lo) & (p < hi)
    t_th = ~(t_s1 | t_nt)
    correct = np.where(t_s1, d["s1"][idx], np.where(t_nt, d["off"][idx], d["on"][idx]))
    tokens = np.where(t_s1, 0.0, np.where(t_nt, d["tok_off"][idx], d["tok_on"][idx]))
    return {"acc": float(correct.mean()), "tokens": float(tokens.mean()),
            "frac_s1": float(t_s1.mean()), "frac_nothink": float(t_nt.mean()),
            "frac_think": float(t_th.mean())}


def _grid(p: np.ndarray, n: int = 41) -> np.ndarray:
    q = np.unique(np.quantile(p, np.linspace(0, 1, n)))
    return np.r_[q, np.inf]


def best_policy(d: dict, idx: np.ndarray, delta: float) -> tuple[float, float] | None:
    """(τ_lo, τ_hi) with the fewest mean tokens reaching acc >= always-think - δ."""
    target = float(d["on"][idx].mean()) - delta
    grid = _grid(d["p_max"][idx])
    best = None
    for i, lo in enumerate(grid):
        for hi in grid[i:]:
            r = three_tier_eval(d, idx, lo, hi)
            if r["acc"] >= target - 1e-12 and (best is None or r["tokens"] < best[0]):
                best = (r["tokens"], lo, hi)
    return None if best is None else (best[1], best[2])


def three_tier(d: dict, k_folds: int = 5, seed: int = 0) -> dict:
    n = len(d["s1"])
    allidx = np.arange(n)
    oracle = np.where(d["s1"], 0, np.where(d["off"], 1, np.where(d["on"], 2, 3)))
    out = {
        "always": {"s1": {"acc": _mean(d["s1"]), "tokens": 0.0},
                   "no_think": {"acc": _mean(d["off"]), "tokens": _mean(d["tok_off"])},
                   "think": {"acc": _mean(d["on"]), "tokens": _mean(d["tok_on"])}},
        "oracle_split": {"s1": float((oracle == 0).mean()),
                         "no_think": float((oracle == 1).mean()),
                         "think": float((oracle == 2).mean()),
                         "none_right": float((oracle == 3).mean())},
        "policies": {},
    }
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k_folds)
    for delta in DELTAS:
        key = f"{int(delta * 100)}pp"
        ins = best_policy(d, allidx, delta)
        res = {"in_sample": None, "cross_fitted": None}
        if ins is not None:
            res["in_sample"] = {"tau_lo": float(ins[0]), "tau_hi": float(ins[1]),
                                **three_tier_eval(d, allidx, *ins)}
        # Cross-fitted: thresholds chosen on the other folds, applied here.
        parts = []
        for f in folds:
            train = np.setdiff1d(allidx, f)
            pol = best_policy(d, train, delta)
            if pol is None:
                pol = (np.inf, np.inf)             # always think
            r = three_tier_eval(d, f, *pol)
            parts.append((len(f), r))
        tot = sum(w for w, _ in parts)
        res["cross_fitted"] = {k: sum(w * r[k] for w, r in parts) / tot
                               for k in parts[0][1]}
        out["policies"][key] = res
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(x, d=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}"


def _ci(r):
    return f"{_pct(r['acc'])} [{_pct(r['ci'][0], 0)}, {_pct(r['ci'][1], 0)}]"


def _gap(g):
    return f"{100 * g['estimate']:+.1f} [{100 * g['ci'][0]:+.0f}, {100 * g['ci'][1]:+.0f}]"


def render_md(res: dict) -> str:
    L = []
    L.append("### Tier table (accuracy %, 95% bootstrap CI; gaps paired, pp)\n")
    L.append("| set | n | floor | S1 | no-think | think | think − S1 | no-think − S1 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, t in res["tiers"].items():
        L.append(f"| {name} | {t['n']} | {_pct(t['floor'], 0)} | {_ci(t['s1'])} | "
                 f"{_ci(t['no_think'])} | {_ci(t['think'])} | "
                 f"{_gap(t['gap_think_minus_s1'])} | {_gap(t['gap_nothink_minus_s1'])} |")
    L.append("\n### Per family (S1 / no-think / think %, think − S1 pp)\n")
    for task, fams in res["families"].items():
        L.append(f"**{task}**\n")
        L.append("| family | n | floor | S1 | no-think | think | think − S1 |")
        L.append("|---|---|---|---|---|---|---|")
        for f, t in sorted(fams.items(), key=lambda kv: -kv[1]["gap_think_minus_s1"]["estimate"]):
            L.append(f"| {f} | {t['n']} | {_pct(t['floor'], 0)} | {_pct(t['s1']['acc'], 0)} | "
                     f"{_pct(t['no_think']['acc'], 0)} | {_pct(t['think']['acc'], 0)} | "
                     f"{100 * t['gap_think_minus_s1']['estimate']:+.0f} |")
        L.append("")
    L.append("### S1 calibration (p_max)\n")
    L.append("| set | ECE | mean p_max | acc | p_max right / wrong | AUROC(p_max→correct) |")
    L.append("|---|---|---|---|---|---|")
    for name, c in res["calibration"].items():
        L.append(f"| {name} | {c['ece']:.3f} | {c['mean_p_max']:.3f} | {c['accuracy']:.3f} | "
                 f"{c['mean_p_max_right']:.3f} / {c['mean_p_max_wrong']:.3f} | "
                 f"{c['auroc_p_max_correct']:.3f} [{c['auroc_ci'][0]:.2f}, {c['auroc_ci'][1]:.2f}] |")
    L.append("\n### Routing S1 → think on S1 p_max\n")
    L.append("| set | nAUC 0–50% [CI] | mean acc 0–50%: conf / random / oracle | "
             "esc. to think−1pp: conf / fam-oracle / oracle / random | "
             "esc. to think−3pp: conf / fam-oracle / oracle / random | ΔnAUC conf − fam-oracle | within-family AUROC (S1 wrong) |")
    L.append("|---|---|---|---|---|---|---|")
    for name, r in res["routing"].items():
        e1, e3 = r["escalation_to_target"]["1pp"], r["escalation_to_target"]["3pp"]
        fo = r.get("family_oracle")
        fmt = lambda e: " / ".join([_pct(e["confidence"], 0), _pct(e.get("family_oracle"), 0),  # noqa: E731
                                    _pct(e["oracle"], 0), _pct(e["random"], 0)])
        dn = (f"{fo['delta_nauc_conf_minus_family']['estimate']:+.3f} "
              f"[{fo['delta_nauc_conf_minus_family']['ci'][0]:+.2f}, "
              f"{fo['delta_nauc_conf_minus_family']['ci'][1]:+.2f}]") if fo else "–"
        wf = (f"{fo['within_family_auroc_conf_vs_s1_wrong']['auroc']:.3f}") if fo else "–"
        L.append(f"| {name} | {r['nauc']:.3f} [{r['nauc_ci'][0]:.2f}, {r['nauc_ci'][1]:.2f}] | "
                 f"{_pct(r['mean_acc_0_50'])} / {_pct(r['random_mean_acc_0_50'])} / "
                 f"{_pct(r['oracle_mean_acc_0_50'])} | {fmt(e1)} | {fmt(e3)} | {dn} | {wf} |")
    L.append("\n### Three tiers: S1 → no-think → think (two thresholds on p_max)\n")
    L.append("| set | policy | acc | mean gen tokens | % S1 / no-think / think |")
    L.append("|---|---|---|---|---|")
    for name, t in res["three_tier"].items():
        for k in ("s1", "no_think", "think"):
            a = t["always"][k]
            L.append(f"| {name} | always {k} | {_pct(a['acc'])} | {a['tokens']:.0f} | |")
        for key, pol in t["policies"].items():
            for mode in ("in_sample", "cross_fitted"):
                r = pol[mode]
                if r is None:
                    continue
                L.append(f"| {name} | ≥ think−{key}, {mode.replace('_', '-')} | {_pct(r['acc'])} | "
                         f"{r['tokens']:.0f} | {_pct(r['frac_s1'], 0)} / "
                         f"{_pct(r['frac_nothink'], 0)} / {_pct(r['frac_think'], 0)} |")
        o = t["oracle_split"]
        L.append(f"| {name} | oracle split | | | {_pct(o['s1'], 0)} / {_pct(o['no_think'], 0)} / "
                 f"{_pct(o['think'], 0)} (none right {_pct(o['none_right'], 0)}) |")
    L.append("\n### Caveats data\n")
    for name, c in res["caveats"].items():
        L.append(f"- {name}: " + ", ".join(f"{k} {v}" for k, v in c.items()))
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", nargs=3, action="append", required=True,
                   metavar=("NAME", "READOUT_DIR", "LABELS"))
    p.add_argument("--mmlu-categories", default=None,
                   help="optional JSON {sample_id: category} when the readout "
                        "rows carry no MMLU-Pro family")
    p.add_argument("--bbh-lettered-only", action="store_true",
                   help="drop BBH word-answer items (yes/no, true/false, "
                        "valid/invalid) everywhere -- for readout v1, whose "
                        "option mass on them is ~0")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args(argv)

    cats = (json.loads(Path(args.mmlu_categories).read_text())
            if args.mmlu_categories else None)
    data = {name: load_task(name, Path(rd), Path(lf),
                            cats if name == "mmlu_pro" else None)
            for name, rd, lf in args.task}
    if args.bbh_lettered_only and "bbh" in data:
        b = data["bbh"]
        keep = np.array(b["kind"]) == "letter"
        dropped = int((~keep).sum())
        data["bbh"] = {**b, **subset(b, keep, "bbh"), "n_covered": int(keep.sum()),
                       "excluded": {**b["excluded"],
                                    "word answer (--bbh-lettered-only)": dropped}}

    sets = dict(data)
    if "bbh" in data and not args.bbh_lettered_only:
        b = data["bbh"]
        kinds = np.array(b["kind"])
        sets["bbh/lettered"] = subset(b, kinds == "letter", "bbh/lettered")
        sets["bbh/binary"] = subset(b, kinds != "letter", "bbh/binary")
    if len(data) > 1:
        sets["pooled"] = concat(list(data.values()), "pooled")

    res = {"inputs": {n: {"readout": rd, "labels": lf} for n, rd, lf in args.task},
           "tiers": {}, "families": {}, "calibration": {}, "routing": {},
           "three_tier": {}, "caveats": {}}
    for name, d in sets.items():
        res["tiers"][name] = tier_row(d)
        res["calibration"][name] = calibration(d)
        res["routing"][name] = routing(d)
        res["three_tier"][name] = three_tier(d)
    for name, d in data.items():
        fams = d["family"]
        if any(f is None for f in fams):
            continue
        res["families"][name] = {
            f: {**tier_row(subset(d, np.array([g == f for g in fams]), f))}
            for f in sorted(set(fams))}
        res["caveats"][name] = {
            "rows": d["n_rows"], "covered": d["n_covered"],
            "coverage": round(d["n_covered"] / d["n_rows"], 3),
            "excluded": d["excluded"],
            "think_truncated_%": round(100 * float(d["trunc_on"].mean()), 1),
            "think_unclosed_%": round(100 * float(d["unclosed_on"].mean()), 1),
            "nothink_truncated_%": round(100 * float(d["trunc_off"].mean()), 1),
            "mean_option_mass": round(float(d["option_mass"].mean()), 3),
            "min_option_mass": round(float(d["option_mass"].min()), 3),
            "grader": d["grader_version"], "readout_commit": d["commits"],
        }
    for name, d in data.items():
        if name not in res["caveats"]:
            res["caveats"][name] = {
                "rows": d["n_rows"], "covered": d["n_covered"],
                "think_truncated_%": round(100 * float(d["trunc_on"].mean()), 1),
                "think_unclosed_%": round(100 * float(d["unclosed_on"].mean()), 1),
                "mean_option_mass": round(float(d["option_mass"].mean()), 3),
                "grader": d["grader_version"], "readout_commit": d["commits"]}

    out_json, out_md = Path(args.out_json), Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=1, default=float) + "\n")
    out_md.write_text(render_md(res))
    print(render_md(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

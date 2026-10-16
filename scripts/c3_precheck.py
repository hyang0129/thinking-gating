#!/usr/bin/env python3
"""
c3_precheck.py — Phase 0 step 0a (issue #3): first C3 evidence on existing data.

Does a learned think-flag beat the model's own confidence at deciding when to
escalate S1 -> think? On the Qwen3-8B D0 readout (training-free, untrained
head), every router below is cross-fitted (5 folds, stratified by family x
label) and scored out-of-fold on the same pooled rows:

  conf          -p_max                                    (no fitting)
  best_conf     logistic on log-odds (p_max, margin, entropy)   the C3 comparator
  family        out-of-fold family mean of y              non-privileged family router
  family_conf   logistic on (one-hot family, logit p_max, family x logit p_max)  per-family tau
  text          TF-IDF word 1-2 grams on the question -> logistic
  d1_L<k>       logistic on the decide-position hidden state, layer k (D1)
  d1_L<k>+conf  same, plus the best_conf features
  d1_auto       D1 with its layer chosen by inner-CV AUROC inside each training fold

Confidence features enter as log-odds: p_max saturates near 1 (mean 0.94), and
a linear model on the raw value cannot order 0.9999 against 0.999999, which
raw -p_max ranking does.

Flag target y = 1[gain = +1] = S1 wrong and think right. Routing curves use
cheap = S1, expensive = think. Reported per router: nAUC over 0-50%
escalation (bootstrap CI), AUROC(y), pooled within-family AUROC(y); and, for
every router against conf, best_conf and family_conf, the paired ΔnAUC and the
paired within-family ΔAUROC (bootstrap stratified by family).

Informative, not a gate: D0 is untrained, and n is ~1.5k (the plan wants 4k).
The primary D1 layer is fixed in advance at --primary-layer (18, the middle
of Qwen3-8B); the other captured layers are secondary.

    .venv/bin/python scripts/c3_precheck.py \\
        --task mmlu_pro shared/icr_capture/mmlu_pro_readout_qwen3v3_v2 \\
            shared/icr_capture/mmlu_pro_thinking_qwen3v3 shared/labels/qwen3v3/mmlu_pro_labels.jsonl \\
        --task bbh shared/icr_capture/bbh_readout_qwen3v3_v2 \\
            shared/icr_capture/bbh_thinking_qwen3v3 shared/labels/qwen3v3/bbh_labels.jsonl \\
        --out-json output/qwen3v3/c3_precheck/c3_precheck.json \\
        --out-md output/qwen3v3/c3_precheck/c3_precheck.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402

from s1_gap_table import concat, load_task, subset  # noqa: E402
from utils import metrics as M  # noqa: E402
from utils.capture_io import load_meta  # noqa: E402

N_BOOT = 2000
K_FOLDS = 5
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(name: str, readout_dir: Path, capture_dir: Path, labels: Path) -> dict:
    d = load_task(name, readout_dir, labels)
    rows = {}
    for x in (readout_dir / "readout.jsonl").read_text().splitlines():
        if x.strip():
            r = json.loads(x)
            rows[r["sample_id"]] = r
    q = {m["sample_id"]: m["question"] for m in load_meta(capture_dir)}
    z = np.load(readout_dir / "decide_hidden.npz")
    pos = {s: i for i, s in enumerate(z["sample_id"].tolist())}
    ids = d["sample_id"]
    missing = [s for s in ids if s not in pos or s not in q]
    if missing:
        raise SystemExit(f"{name}: {len(missing)} rows lack hidden state or question, "
                         f"e.g. {missing[:3]}")
    probs = [np.sort(np.asarray(rows[s]["probs"], float))[::-1] for s in ids]
    d["margin"] = np.array([p[0] - (p[1] if len(p) > 1 else 0.0) for p in probs])
    d["entropy"] = np.array([float(rows[s]["entropy_norm"]) for s in ids])
    d["hidden"] = z["hidden"][[pos[s] for s in ids]].astype(np.float32)
    d["question"] = [q[s] for s in ids]
    d["layers"] = z["layers"].tolist()
    return d


def pooled(parts: list[dict]) -> dict:
    out = concat(parts, "pooled")           # arrays, sample_id, kind, family
    out["hidden"] = np.concatenate([p["hidden"] for p in parts])
    out["question"] = sum((p["question"] for p in parts), [])
    out["layers"] = parts[0]["layers"]
    return out


# ---------------------------------------------------------------------------
# Cross-fitted routers (higher score = escalate first)
# ---------------------------------------------------------------------------

def folds(y: np.ndarray, fam: list[str], seed: int) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold
    strata = np.array([f"{f}|{int(v)}" for f, v in zip(fam, y)])
    # Strata smaller than K fall back to the label alone.
    counts = {s: (strata == s).sum() for s in set(strata)}
    strata = np.array([s if counts[s] >= K_FOLDS else f"_|{s[-1]}" for s in strata])
    fold = np.empty(len(y), int)
    skf = StratifiedKFold(K_FOLDS, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(skf.split(np.zeros(len(y)), strata)):
        fold[te] = k
    return fold


def _logreg(X, y, Xte, dense: bool = True):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = LogisticRegressionCV(Cs=CS, cv=3, scoring="roc_auc", max_iter=5000)
    model = make_pipeline(StandardScaler(), clf) if dense else clf
    model.fit(X, y)
    return model.predict_proba(Xte)[:, 1]


def oof(fit_predict, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    s = np.empty(len(y))
    for k in range(K_FOLDS):
        tr, te = fold != k, fold == k
        s[te] = fit_predict(tr, te)
    return s


def routers(d: dict, y: np.ndarray, fold: np.ndarray, primary: int,
            all_layers: bool) -> dict[str, np.ndarray]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    fam = np.array(d["family"])
    eps = 1e-12
    lp = np.log(d["p_max"] + eps) - np.log1p(-d["p_max"] + eps)
    conf_X = np.c_[lp, np.log(d["margin"] + eps), np.log(d["entropy"] + eps)]
    fams = sorted(set(fam))
    onehot = (fam[:, None] == np.array(fams)[None, :]).astype(float)
    famconf_X = np.c_[onehot, lp, onehot * lp[:, None]]

    def family_mean(tr, te):
        means = {f: y[tr][fam[tr] == f].mean() for f in fams if (fam[tr] == f).any()}
        prior = y[tr].mean()
        return np.array([means.get(f, prior) for f in fam[te]])

    def text(tr, te):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        Xtr = vec.fit_transform([d["question"][i] for i in np.flatnonzero(tr)])
        Xte = vec.transform([d["question"][i] for i in np.flatnonzero(te)])
        return _logreg(Xtr, y[tr], Xte, dense=False)

    out = {
        "conf": -d["p_max"],
        "best_conf": oof(lambda tr, te: _logreg(conf_X[tr], y[tr], conf_X[te]), y, fold),
        "family": oof(family_mean, y, fold),
        "family_conf": oof(lambda tr, te: _logreg(famconf_X[tr], y[tr], famconf_X[te]),
                           y, fold),
        "text": oof(text, y, fold),
    }
    layers = d["layers"] if all_layers else [primary]
    for L in layers:
        H = d["hidden"][:, d["layers"].index(L)]
        out[f"d1_L{L}"] = oof(lambda tr, te: _logreg(H[tr], y[tr], H[te]), y, fold)
        if L == primary:
            HC = np.c_[H, conf_X]
            out[f"d1_L{L}+conf"] = oof(lambda tr, te: _logreg(HC[tr], y[tr], HC[te]),
                                       y, fold)
    if all_layers:
        def auto(tr, te):
            from sklearn.linear_model import LogisticRegressionCV
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            best = None
            for i in range(len(d["layers"])):
                m = make_pipeline(StandardScaler(), LogisticRegressionCV(
                    Cs=CS, cv=3, scoring="roc_auc", max_iter=5000))
                m.fit(d["hidden"][tr, i], y[tr])
                inner = next(iter(m[-1].scores_.values())).mean(axis=0).max()
                if best is None or inner > best[0]:
                    best = (inner, m, i)
            return best[1].predict_proba(d["hidden"][te, best[2]])[:, 1]
        out["d1_auto"] = oof(auto, y, fold)
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(d: dict, y: np.ndarray, scores: dict[str, np.ndarray],
             comparators=("conf", "best_conf", "family_conf")) -> dict:
    cheap, on, fam = d["s1"], d["on"], d["family"]
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "base_rate": float(y.mean()),
           "n_families": len(set(fam)), "acc_s1": float(cheap.mean()),
           "acc_think": float(on.mean()), "routers": {}}
    for name, s in scores.items():
        r = {
            "nauc": M.routed_nauc(cheap, on, s)["nauc"],
            "nauc_ci": M.bootstrap_ci(lambda c, o, x: M.routed_nauc(c, o, x)["nauc"],
                                      cheap, on, s, n_boot=N_BOOT, strata=fam)["ci"],
            "auroc": M.auroc(y, s),
            "auroc_ci": M.bootstrap_auroc_ci(y, s, n_boot=N_BOOT)["ci"],
            "within_family_auroc": M.within_group_stats(y, s, fam),
        }
        for c in comparators:
            if c == name:
                continue
            r[f"dnauc_vs_{c}"] = M.paired_bootstrap_nauc_diff(
                cheap, on, s, scores[c], n_boot=N_BOOT, strata=fam)
            r[f"dwithin_auroc_vs_{c}"] = M.paired_bootstrap(
                lambda yy, a, b, g: (M.within_group_auroc(yy, a, g)
                                     - M.within_group_auroc(yy, b, g)),
                y, s, scores[c], np.array(fam), n_boot=N_BOOT, strata=fam)
        out["routers"][name] = r
    return out


def _d(x):
    return f"{x['estimate']:+.3f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]"


def render_md(res: dict) -> str:
    L = []
    for name, e in res["sets"].items():
        L.append(f"### {name}: n {e['n']}, y = S1 wrong & think right: "
                 f"{e['n_pos']} ({100 * e['base_rate']:.1f}%), {e['n_families']} families, "
                 f"S1 {100 * e['acc_s1']:.1f}% → think {100 * e['acc_think']:.1f}%\n")
        cmp = ("conf", "best_conf", "family_conf")
        L.append("| router | nAUC 0–50% [CI] | AUROC(y) [CI] | within-fam AUROC | "
                 + " | ".join(f"ΔnAUC vs {c} | Δwithin-fam AUROC vs {c}" for c in cmp) + " |")
        L.append("|---" * (4 + 2 * len(cmp)) + "|")
        for r, m in e["routers"].items():
            cells = [_d(m[k]) if k in m else "–"
                     for c in cmp for k in (f"dnauc_vs_{c}", f"dwithin_auroc_vs_{c}")]
            L.append(f"| {r} | {m['nauc']:.3f} [{m['nauc_ci'][0]:.2f}, {m['nauc_ci'][1]:.2f}] | "
                     f"{m['auroc']:.3f} [{m['auroc_ci'][0]:.2f}, {m['auroc_ci'][1]:.2f}] | "
                     f"{m['within_family_auroc']['auroc']:.3f} | " + " | ".join(cells) + " |")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", nargs=4, action="append", required=True,
                   metavar=("NAME", "READOUT_DIR", "CAPTURE_DIR", "LABELS"))
    p.add_argument("--primary-layer", type=int, default=18)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args(argv)

    data = {n: load(n, Path(r), Path(c), Path(lf)) for n, r, c, lf in args.task}
    sets = {"pooled": pooled(list(data.values()))} if len(data) > 1 else {}
    sets.update(data)
    if "bbh" in data:
        b = data["bbh"]
        keep = np.array(b["kind"]) == "letter"
        sets["bbh/lettered"] = {**subset(b, keep, "bbh/lettered"),
                                "hidden": b["hidden"][keep], "layers": b["layers"],
                                "question": [q for q, k in zip(b["question"], keep) if k]}

    res = {"inputs": [list(t) for t in args.task], "primary_layer": args.primary_layer,
           "k_folds": K_FOLDS, "seed": args.seed, "n_boot": N_BOOT, "sets": {}}
    for name, d in sets.items():
        if any(f is None for f in d["family"]):
            raise SystemExit(f"{name}: rows without a family")
        y = (~d["s1"]) & d["on"]
        fold = folds(y, d["family"], args.seed)
        scores = routers(d, y, fold, args.primary_layer, all_layers=(name == "pooled"))
        res["sets"][name] = evaluate(d, y, scores)
        print(f"done {name}", file=sys.stderr)

    out_json, out_md = Path(args.out_json), Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=1, default=float) + "\n")
    out_md.write_text(render_md(res))
    print(render_md(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

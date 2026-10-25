#!/usr/bin/env python3
"""
c3_analysis.py — the pre-registered C3 test (issue #3, §8). CPU.

Escalation S1 -> think on the eval pool, cheap = D2 final head correct (s1),
expensive = mean thinking-on correctness (s2_mean, 4 samples). Routers, all
fitted on the train pool (out-of-fold S1, 1 S2 sample) and scored on eval:

  conf          -p_max of the D2 answer head
  conf_logit    logistic on log-odds (p_max, margin, entropy)
  best_conf     whichever of conf / conf_logit has the higher nAUC on a 20%
                held-out slice of train: THE C3 COMPARATOR
  family_conf   per-family tau: logistic on (family one-hot, logit p_max,
                family x logit p_max)
  text          TF-IDF on the context -> logistic
  d1            logistic on the base decide-position state (D0 hidden.npz),
                layer chosen on the same held-out slice
  flag          the D2 flag aLoRA, P(Yes)

Primary: paired-bootstrap ΔnAUC(flag − best_conf), escalation 0-50%, pooled,
2000 resamples stratified by family. Pass: lower bound > 0 AND flag beats
family_conf with a pooled within-family ΔnAUC lower bound > 0 (within-family
= scores replaced by their percentile rank inside each family, so every
family is escalated at the same rate and family identity cannot help).
Null: CI covers 0 with half-width <= 0.05. Underpowered: half-width > 0.05.
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

from utils import metrics as M  # noqa: E402

N_BOOT = 2000
EPS = 1e-12


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def conf_feats(rows):
    p = np.array([r["p_max"] for r in rows])
    probs = [np.sort(np.asarray(r["probs"]))[::-1] for r in rows]
    margin = np.array([q[0] - (q[1] if len(q) > 1 else 0) for q in probs])
    ent = np.array([-(q * np.log(q + EPS)).sum() / np.log(len(q)) for q in probs])
    lp = np.log(p + EPS) - np.log1p(-p + EPS)
    return np.c_[lp, np.log(margin + EPS), np.log(ent + EPS)], lp


def logreg(Xtr, ytr, Xte, dense=True):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = LogisticRegressionCV(Cs=[1e-4, 1e-3, 1e-2, 1e-1, 1.0], cv=3,
                               scoring="roc_auc", max_iter=5000)
    m = make_pipeline(StandardScaler(), clf) if dense else clf
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def within_rank(scores, fam):
    """Percentile rank of each score inside its family (ties averaged)."""
    from scipy.stats import rankdata
    out = np.empty(len(scores))
    fam = np.asarray(fam)
    for f in np.unique(fam):
        m = fam == f
        out[m] = rankdata(scores[m]) / m.sum()
    return out


def nauc(off, on, s):
    return M.routed_nauc(off, on, s)["nauc"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flag-dir", default="shared/s1/flag")
    ap.add_argument("--records-dir", default="shared/s1/records")
    ap.add_argument("--pred-dir", default="shared/s1/pred/qwen3.5-4b")
    ap.add_argument("--out", default="output/qwen3.5-4b/c3/c3_analysis.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    R = lambda p: _PROJECT_ROOT / p  # noqa: E731

    tr = load_jsonl(R(args.flag_dir) / "flag_labels_train.jsonl")
    ev = load_jsonl(R(args.flag_dir) / "flag_labels_eval.jsonl")
    ctx = {r["id"]: r["context"] for p in ("train", "eval")
           for r in load_jsonl(R(args.records_dir) / f"{p}.jsonl")}
    flag = {x["id"].rsplit(":flag", 1)[0]: x["probs"][0]
            for x in load_jsonl(R(args.pred_dir) / "flag_eval/predictions.jsonl")}
    ev = [r for r in ev if r["id"] in flag]

    rng = np.random.default_rng(args.seed)
    hold = rng.random(len(tr)) < 0.2
    ytr = np.array([r["y"] for r in tr])
    s1_tr = np.array([r["s1"] for r in tr], float)
    s2_tr = np.array([r["s2_mean"] for r in tr])
    s1 = np.array([r["s1"] for r in ev], float)
    s2 = np.array([r["s2_mean"] for r in ev])
    fam = np.array([r["family"] for r in ev])
    fam_tr = np.array([r["family"] for r in tr])
    y_ev = ((1 - s1) * s2 > 0.5)

    Xc_tr, lp_tr = conf_feats(tr)
    Xc_ev, lp_ev = conf_feats(ev)
    S = {"conf": -np.array([r["p_max"] for r in ev])}
    S["conf_logit"] = logreg(Xc_tr, ytr, Xc_ev)
    val_logit = logreg(Xc_tr[~hold], ytr[~hold], Xc_tr[hold])
    v_conf = nauc(s1_tr[hold], s2_tr[hold], -np.exp(lp_tr[hold]))
    v_logit = nauc(s1_tr[hold], s2_tr[hold], val_logit)
    best = "conf_logit" if v_logit > v_conf else "conf"
    S["best_conf"] = S[best]

    fams = sorted(set(fam_tr))
    oh = lambda f: (np.asarray(f)[:, None] == np.array(fams)[None]).astype(float)  # noqa: E731
    S["family_conf"] = logreg(np.c_[oh(fam_tr), lp_tr, oh(fam_tr) * lp_tr[:, None]], ytr,
                              np.c_[oh(fam), lp_ev, oh(fam) * lp_ev[:, None]])

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
    Xt_tr = vec.fit_transform([ctx[r["id"]] for r in tr])
    S["text"] = logreg(Xt_tr, ytr, vec.transform([ctx[r["id"]] for r in ev]), dense=False)

    d1_info = None
    hid = {}
    for d in sorted(R(args.pred_dir).glob("d0_*")):
        if (d / "hidden.npz").exists():
            z = np.load(d / "hidden.npz")
            layers = z["layers"].tolist()
            for i, h in zip(z["id"], z["hidden"]):
                hid[str(i)] = h
    if hid and all(r["id"] in hid for r in tr + ev):
        H_tr = np.stack([hid[r["id"]] for r in tr]).astype(np.float32)
        H_ev = np.stack([hid[r["id"]] for r in ev]).astype(np.float32)
        vals = []
        for li in range(H_tr.shape[1]):
            v = logreg(H_tr[~hold, li], ytr[~hold], H_tr[hold, li])
            vals.append(nauc(s1_tr[hold], s2_tr[hold], v))
        li = int(np.argmax(vals))
        S["d1"] = logreg(H_tr[:, li], ytr, H_ev[:, li])
        d1_info = {"layer": layers[li], "val_nauc_by_layer": dict(zip(layers, vals))}
    S["flag"] = np.array([flag[r["id"]] for r in ev])

    def block(mask):
        out = {"n": int(mask.sum()), "acc_s1": float(s1[mask].mean()),
               "acc_think": float(s2[mask].mean()), "y_rate": float(y_ev[mask].mean()),
               "routers": {}}
        f = fam[mask]
        for k, s in S.items():
            ss = s[mask]
            r = {"nauc": nauc(s1[mask], s2[mask], ss),
                 "nauc_ci": M.bootstrap_ci(lambda a, b, c: nauc(a, b, c), s1[mask], s2[mask], ss,
                                           n_boot=N_BOOT, strata=f)["ci"],
                 "within_family_auroc": M.within_group_stats(y_ev[mask], ss, f)["auroc"],
                 "acc_at": {p: float(M.curve_accuracy_at(
                     M.routed_accuracy_curve(s1[mask], s2[mask], ss), p / 100)) for p in (10, 20, 30)}}
            if k != "flag":
                r["dnauc_flag_minus"] = M.paired_bootstrap_nauc_diff(
                    s1[mask], s2[mask], S["flag"][mask], ss, n_boot=N_BOOT, strata=f)
                r["dnauc_within_flag_minus"] = M.paired_bootstrap(
                    lambda a, b, x, z, g: nauc(a, b, within_rank(x, g)) - nauc(a, b, within_rank(z, g)),
                    s1[mask], s2[mask], S["flag"][mask], ss, f, n_boot=N_BOOT, strata=f)
            out["routers"][k] = r
        return out

    res = {"n_train": len(tr), "n_eval": len(ev), "best_conf_choice": best,
           "best_conf_val_nauc": {"conf": v_conf, "conf_logit": v_logit}, "d1": d1_info,
           "pooled": block(np.ones(len(ev), bool)),
           "by_task": {t: block(np.array([r["task"] == t for r in ev]))
                       for t in sorted({r["task"] for r in ev})}}
    prim = res["pooled"]["routers"]["best_conf"]["dnauc_flag_minus"]
    fam_w = res["pooled"]["routers"]["family_conf"]["dnauc_within_flag_minus"]
    half = (prim["ci"][1] - prim["ci"][0]) / 2
    fam_d = res["pooled"]["routers"]["family_conf"]["dnauc_flag_minus"]
    if prim["ci"][0] > 0 and fam_w["ci"][0] > 0 and fam_d["estimate"] > 0:
        verdict = "pass"
    elif prim["ci"][1] < 0:
        verdict = "fail: flag below own confidence"
    elif prim["ci"][0] > 0:
        verdict = "fail: flag beats confidence but not per-family tau within families (family detector)"
    elif half > 0.05:
        verdict = "underpowered"
    else:
        verdict = "null: confidence suffices"
    res["primary"] = {"dnauc_flag_minus_best_conf": prim, "half_width": half,
                      "within_family_dnauc_flag_minus_family_conf": fam_w, "verdict": verdict}
    out = R(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({"verdict": verdict, "primary": prim, "within_family_vs_family_conf": fam_w,
                      "best_conf": best}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

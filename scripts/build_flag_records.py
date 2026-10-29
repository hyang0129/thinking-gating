#!/usr/bin/env python3
"""
build_flag_records.py — think-flag labels and records (issue #3, steps 6-7).

Per typed record (shared/s1/records/{train,eval}.jsonl):

  s1   D2 correct: out-of-fold heads on train (cross-fitted, §5), the final
       head on eval
  s2   thinking-on correctness, mean over the captured samples (1 on train,
       4 on eval). Verify items ("Is the final answer X?") take S2's answer
       to the underlying problem: right when X is gold and S2 solved it, or
       when X is not gold and S2's answer is not X.
  y    1[S1 wrong and S2 right] on train (the flag target, §5); the eval row
       keeps s1 and s2_mean for the routing curves.

Writes flag_{train,eval}.jsonl: the record's context, a flag question that
restates the typed question, options [Yes, No], gold 0 iff y, plus labels.
Also flag_labels_{train,eval}.jsonl ({id, task, family, kind, s1, s2_mean,
n_s2, y, p_max}) for the analysis. CPU only.
"""

from __future__ import annotations

import argparse
import collections
import glob
import importlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

FLAG_Q = "Would working this out step by step change your answer?"


def flag_question(rec: dict) -> str:
    opts = "\n".join(f"- {o}" for o in rec["options"])
    return f"{rec['question']}\n{opts}\n{FLAG_Q}"


def s2_rows(pattern: str) -> dict[str, dict]:
    out = {}
    for f in glob.glob(str(_PROJECT_ROOT / pattern)):
        for line in open(f):
            r = json.loads(line)
            out[r["sample_id"]] = r
    return out


def s2_correct(rec: dict, row: dict) -> list[bool]:
    """Per-sample S2 correctness on this record's decision."""
    if rec["kind"] != "verify":
        return [bool(s["correct"]) for s in row["samples"]]
    mod = importlib.import_module(f"tasks.{rec['task']}")
    out = []
    for s in row["samples"]:
        if rec["candidate_is_gold"]:
            out.append(bool(s["correct"]))
        else:
            says_x = (not s["truncated"] and not s.get("unclosed")
                      and mod.is_correct(s["answer"], rec["candidate"]))
            out.append(not says_x)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slug", default="qwen3.5-4b")
    p.add_argument("--records-dir", default="shared/s1/records")
    p.add_argument("--pred-dir", default="shared/s1/pred/qwen3.5-4b")
    p.add_argument("--s2-dir", default="shared/s2")
    p.add_argument("--out-dir", default="shared/s1/flag")
    args = p.parse_args(argv)
    root = _PROJECT_ROOT
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for pool in ("train", "eval"):
        recs = [json.loads(l) for l in open(root / args.records_dir / f"{pool}.jsonl")]
        pred_glob = (f"{args.pred_dir}/d2_oof-fold*/predictions.jsonl" if pool == "train"
                     else f"{args.pred_dir}/d2_final-eval/predictions.jsonl")
        preds = {}
        for f in glob.glob(str(root / pred_glob)):
            for line in open(f):
                x = json.loads(line)
                preds[x["id"]] = x
        s2 = s2_rows(f"{args.s2_dir}/{pool}_{args.slug}/*/shard*/s2.jsonl")
        miss = collections.Counter()
        n = collections.Counter()
        with open(out / f"flag_{pool}.jsonl", "w") as fr, \
                open(out / f"flag_labels_{pool}.jsonl", "w") as fl:
            for r in recs:
                key = r["id"].split(":", 1)[1]
                if r["id"] not in preds:
                    miss["no_s1"] += 1
                    continue
                if key not in s2:
                    miss["no_s2"] += 1
                    continue
                c2 = s2_correct(r, s2[key])
                s1 = bool(preds[r["id"]]["correct"])
                s2m = sum(c2) / len(c2)
                y = (not s1) and s2m > 0.5 if pool == "eval" else (not s1) and c2[0]
                fam = f"{r['task']}:{r['family']}"
                fl.write(json.dumps({"id": r["id"], "task": r["task"], "family": fam,
                                     "kind": r["kind"], "s1": s1, "s2_mean": s2m,
                                     "n_s2": len(c2), "y": bool(y),
                                     "p_max": preds[r["id"]]["p_max"],
                                     "probs": preds[r["id"]]["probs"]}) + "\n")
                fr.write(json.dumps({**{k: r[k] for k in ("task", "kind", "context")},
                                     "id": r["id"] + ":flag", "family": fam,
                                     "question": flag_question(r), "options": ["Yes", "No"],
                                     "gold": 0 if y else 1, "candidate": None,
                                     "candidate_is_gold": None}) + "\n")
                n["rows"] += 1
                n["y"] += bool(y)
                n["s1"] += s1
                n["hurt"] += s1 and s2m < 0.5
        summary[pool] = {"n": n["rows"], "y_rate": n["y"] / max(n["rows"], 1),
                         "s1_acc": n["s1"] / max(n["rows"], 1),
                         "hurt_rate": n["hurt"] / max(n["rows"], 1),
                         "missing": dict(miss), "n_preds": len(preds), "n_s2_rows": len(s2)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

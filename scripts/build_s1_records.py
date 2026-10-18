#!/usr/bin/env python3
"""
build_s1_records.py — write the typed-question records for step 5 (issue #3).

    shared/s1/records/{eval,train}.jsonl     our pools (utils/s1_pools.py)
    shared/s1/records/dv7_{train,development,test}.jsonl   Kev decision-v7
    shared/s1/records/summary.json

Verify items take their hard negative from the no-think capture
(shared/s2/nothink_{pool}_<slug>/<task>/shard*/s2.jsonl); a pool whose
no-think capture is missing falls back to numeric perturbations and the
summary says so. CPU only.

    .venv-s1/bin/python scripts/build_s1_records.py --slug qwen3.5-4b \\
        --kev-revision cc4bac803e73112689ec327ffa481c519cbc7a05
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.s1_pools import POOL_VERSION, load_pool  # noqa: E402
from utils.s1_records import from_decision_v7, from_task  # noqa: E402

TASKS = ("mmlu_pro", "bbh", "gsm8k", "math500", "lsat")


def nothink_rows(pool: str, task: str, slug: str) -> dict[str, dict] | None:
    files = glob.glob(str(_PROJECT_ROOT / f"shared/s2/nothink_{pool}_{slug}/{task}/shard*/s2.jsonl"))
    if not files:
        return None
    out = {}
    for f in files:
        for line in open(f):
            r = json.loads(line)
            out[r["sample_id"]] = r["samples"][0]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slug", required=True)
    p.add_argument("--kev-revision", required=True)
    p.add_argument("--out-dir", default="shared/s1/records")
    args = p.parse_args(argv)
    from huggingface_hub import hf_hub_download

    out = _PROJECT_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = {"pool_version": POOL_VERSION, "slug": args.slug,
               "kev_revision": args.kev_revision, "files": {}}
    for pool in ("eval", "train"):
        n, kinds, neg, missing_nt = 0, collections.Counter(), collections.Counter(), []
        with open(out / f"{pool}.jsonl", "w") as f:
            for task in TASKS:
                items = load_pool(pool, task)
                if not items:
                    continue
                nt = nothink_rows(pool, task, args.slug)
                if nt is None and task in ("gsm8k", "math500", "bbh"):
                    missing_nt.append(task)
                for it in items:
                    r = from_task(task, it, (nt or {}).get(it["key"]))
                    if r is None:
                        continue
                    f.write(json.dumps(r) + "\n")
                    n += 1
                    kinds[f"{task}:{r['kind']}"] += 1
                    if r["candidate_is_gold"] is False:
                        neg[task] += 1
        summary["files"][f"{pool}.jsonl"] = {"n": n, "kinds": dict(kinds),
                                             "verify_negatives": dict(neg),
                                             "nothink_missing": missing_nt}
    for split in ("train", "development", "test"):
        src = hf_hub_download("jaredpalmer/kev-suites", f"v7/decision-v7/{split}.jsonl",
                              repo_type="dataset", revision=args.kev_revision)
        n = 0
        with open(out / f"dv7_{split}.jsonl", "w") as f:
            for line in open(src):
                for q in from_decision_v7(json.loads(line)):
                    f.write(json.dumps(q) + "\n")
                    n += 1
        summary["files"][f"dv7_{split}.jsonl"] = {"n": n}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

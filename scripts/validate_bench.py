#!/usr/bin/env python3
"""
validate_bench.py — flag any cross-model bench result trained on a partial capture.

The dispatch queue has no cross-batch dependencies, so a bench cell that
starts while its capture is still writing will train on whatever shards exist
and report a perfectly confident number from partial data. Two results were
produced that way before the shard guard existed (gpt-oss/gsm8k at 990/1319,
qwen3-14b/math500 at 250/500), and nothing in the metrics JSON marks them as
suspect -- only n_samples gives it away.

Run this before quoting any number out of output/xm/.

Usage:
    python scripts/validate_bench.py
"""
import json, os, glob

EXPECTED = {"gsm8k": 1319, "math500": 500, "mmlu_pro": 1000}
bad = []
for p in sorted(glob.glob("output/xm/*/aggregate_metrics.json")):
    slug_task_target = os.path.basename(os.path.dirname(p))
    slug, task, target = slug_task_target.split("__")
    d = json.load(open(p))
    n = d["n_samples"]
    exp = EXPECTED[task]
    # 'rescued' legitimately subsets, so only the full-set targets are checked
    if target in ("needs_thinking", "helped") and n != exp:
        bad.append((slug_task_target, n, exp))
        print("INVALID  %-44s n=%-5d expected %d" % (slug_task_target, n, exp))
if not bad:
    print("all full-set results match their expected capture size")
else:
    print()
    print("stale dirs to remove:")
    for name, _, _ in bad:
        slug, task, _ = name.split("__")
        for t in ("needs_thinking", "helped", "rescued"):
            print("  output/xm/%s__%s__%s" % (slug, task, t))
        print("  shared/xm/%s__%s_labels.jsonl" % (slug, task))
        break

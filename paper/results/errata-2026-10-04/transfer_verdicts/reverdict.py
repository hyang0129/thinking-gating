"""Re-apply eval_transfer.py's fixed (B9) verdict rule to the PUBLISHED transfer files.
Uses the per-probe auroc_bootstrap intervals already stored in each file; no retraining."""
import json, sys
from pathlib import Path
MAIN = Path("/Users/hong/Documents/code-projects/thinking-gating")
sys.path.insert(0, str(MAIN))
from utils.metrics import mean_bootstrap_ci
from scripts.eval_transfer import (VERDICT_UNDEFINED, VERDICT_NONE, VERDICT_STRONG, VERDICT_PARTIAL, VERDICT_WEAK)
for slug in ("qwen3v3", "nemotronv3"):
    for f in sorted((MAIN / "paper/results/metrics" / slug).glob("transfer__*.json")):
        d = json.loads(f.read_text())
        boot = mean_bootstrap_ci([r["auroc_bootstrap"] for r in d["per_probe"]])
        src, drop = d["source_test_auroc"], d["auroc_drop_pp"]
        lo = boot["ci"][0]
        if src < 0.55: v = VERDICT_UNDEFINED
        elif not lo > 0.5: v = VERDICT_NONE
        elif drop < 5: v = VERDICT_STRONG
        elif drop <= 15: v = VERDICT_PARTIAL
        else: v = VERDICT_WEAK
        flag = "FLIP" if v != d["verdict"] else "same"
        print(f"{slug:<11}{f.stem:<45}{d['transfer_auroc']['mean']:.3f} boot[{boot['ci'][0]:.3f},{boot['ci'][1]:.3f}]  {flag}: {d['verdict']} -> {v}")

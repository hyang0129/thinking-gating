"""Before (stored v1) vs after (v2 sidecar) confidence-baseline AUROCs, same regraded labels, same splits."""
import json, sys
from pathlib import Path
before, after = Path(sys.argv[1]), Path(sys.argv[2])
rows = {}
for f in sorted(after.glob("confidence__*__*.json")):
    if f.name.endswith(".predictions.json"): continue
    _, task, target = f.stem.split("__")
    a = json.loads(f.read_text()); b = json.loads((before / f.name).read_text())
    for name in a["baselines"]:
        na, nb = a["baselines"][name], b["baselines"][name]
        rows[f"{task}/{target}/{name}"] = {
            "before_v1": [nb["test_auroc"]["mean"], *nb["test_auroc_bootstrap"]["ci"]],
            "after_v2": [na["test_auroc"]["mean"], *na["test_auroc_bootstrap"]["ci"]],
            "val_before": nb["val_auroc"]["mean"], "val_after": na["val_auroc"]["mean"],
            "source_after": a["confidence_source"][0]["source"]}
        print(f"{task:9s}{target:15s}{name:14s} v1 {nb['test_auroc']['mean']:.3f} -> v2 {na['test_auroc']['mean']:.3f} "
              f"[{na['test_auroc_bootstrap']['ci'][0]:.3f}, {na['test_auroc_bootstrap']['ci'][1]:.3f}]")
if len(sys.argv) > 3: Path(sys.argv[3]).write_text(json.dumps(rows, indent=2) + "\n")

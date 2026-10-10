import json, sys
from pathlib import Path
sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
from compare_baselines import best, load_predictions, paired_difference
M = Path(sys.argv[1]); out = {}
for task in ["gsm8k", "math500", "mmlu_pro", "bbh"]:
    for target in ["needs_thinking", "rescued"]:
        pp = load_predictions(M / f"{task}__{target}.json")
        for kind in ("text", "confidence"):
            bpath = M / "baselines" / f"{kind}__{task}__{target}.json"
            d = json.loads(bpath.read_text())
            name, node, sel = best(d["baselines"])
            diff = paired_difference(pp, load_predictions(bpath), name)
            out[f"{task}/{target}/{kind}"] = {"best": name, "selected_on": sel,
                "auroc": node.get("mean"), "ci": node.get("ci"),
                "source": d.get("confidence_source"), "probe_minus_best": diff}
            print(f"{task:9s}{target:15s}{kind:11s}{name:14s} AUROC {node['mean']:.3f} [{node['ci'][0]:.3f},{node['ci'][1]:.3f}]  probe-base {diff['mean']:+.3f} [{diff['ci'][0]:+.3f},{diff['ci'][1]:+.3f}]")
if len(sys.argv) > 2: Path(sys.argv[2]).write_text(json.dumps(out, indent=2) + "\n")

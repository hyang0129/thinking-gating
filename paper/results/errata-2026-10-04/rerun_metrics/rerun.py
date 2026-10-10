"""Re-run the v3 analysis with the fixed analysis layer on the PUBLISHED labels.
Reads captures/labels from the main checkout; writes only into this scratch dir."""
import os
import shutil
import subprocess
from pathlib import Path

WT = Path("/Users/hong/Documents/code-projects/thinking-gating/.claude/worktrees/agent-afbd0b4a543305334")
MAIN = Path("/Users/hong/Documents/code-projects/thinking-gating")
PY = str(MAIN / ".venv/bin/python")
S = Path(__file__).resolve().parent
ENV = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
SEEDS = ["42", "1", "2", "3", "4"]


def run(args, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        rc = subprocess.run([PY, *args], cwd=MAIN, env=ENV, stdout=fh, stderr=subprocess.STDOUT).returncode
    return rc


def slug_run(slug, tasks, layer_args=()):
    M = S / slug / "metrics"
    (M / "baselines").mkdir(parents=True, exist_ok=True)
    for task in tasks:
        cap = f"shared/icr_capture/{task}_thinking_{slug}"
        lab = f"shared/labels/{slug}/{task}_labels.jsonl"
        for target in ("rescued", "needs_thinking", "helped"):
            out = S / slug / f"probe_{task}_{target}"
            if not (out / "aggregate_metrics.json").exists():
                print("probe", slug, task, target, flush=True)
                rc = run([str(WT / "scripts/run_experiment.py"), "--capture-dir", cap, "--labels", lab,
                          "--out-dir", str(out), "--method", "logreg", "--target", target,
                          *layer_args, "--seeds", *SEEDS], S / slug / "logs" / f"probe_{task}_{target}.log")
                assert rc == 0, (slug, task, target)
            shutil.copy(out / "aggregate_metrics.json", M / f"{task}__{target}.json")
            shutil.copy(out / "predictions.json", M / f"{task}__{target}.predictions.json")
            if target == "helped":
                continue
            for kind in ("text", "confidence"):
                b = M / "baselines" / f"{kind}__{task}__{target}.json"
                if not b.exists():
                    print("baseline", kind, slug, task, target, flush=True)
                    rc = run([str(WT / f"scripts/baseline_{kind}.py"), "--capture-dir", cap, "--labels", lab,
                              "--target", target, "--seeds", *SEEDS, "--out-file", str(b)],
                             S / slug / "logs" / f"baseline_{kind}_{task}_{target}.log")
                    assert rc == 0, (kind, slug, task, target)
    run([str(WT / "scripts/compare_baselines.py"), "--metrics-dir", str(M), "--out", str(M / "baseline_comparison.txt")],
        S / slug / "logs" / "compare.log")
    run([str(WT / "scripts/results_table.py"), "--metrics-dir", str(M), "--out", str(M / "results_table.txt")],
        S / slug / "logs" / "table.log")


slug_run("qwen3v3", ["gsm8k", "math500", "mmlu_pro", "bbh"])
slug_run("nemotronv3", ["bbh", "lsat"])            # default = middle layer (16)
for task in ("bbh", "lsat"):                       # published layer 18, to isolate the layer change
    for target in ("rescued", "needs_thinking", "helped"):
        out = S / "nemotronv3_L18" / f"probe_{task}_{target}"
        if not (out / "aggregate_metrics.json").exists():
            print("probe L18", task, target, flush=True)
            run([str(WT / "scripts/run_experiment.py"), "--capture-dir", f"shared/icr_capture/{task}_thinking_nemotronv3",
                 "--labels", f"shared/labels/nemotronv3/{task}_labels.jsonl", "--out-dir", str(out),
                 "--method", "logreg", "--target", target, "--layer", "18", "--seeds", *SEEDS],
                S / "nemotronv3_L18" / "logs" / f"{task}_{target}.log")

for target in ("rescued", "needs_thinking"):
    rc = run([str(WT / "scripts/within_group_auroc.py"), "--capture-dir", "shared/icr_capture/bbh_thinking_qwen3v3",
              "--labels", "shared/labels/qwen3v3/bbh_labels.jsonl", "--group-key", "sample_id:middle",
              "--target", target, "--out-file", str(S / "qwen3v3" / f"within_group__bbh__{target}.json")],
             S / "qwen3v3" / "logs" / f"within_bbh_{target}.log")
    print("bbh within-group", target, "rc", rc, flush=True)
    rc = run([str(WT / "scripts/within_group_auroc.py"), "--capture-dir", "shared/icr_capture/math500_thinking_qwen3v3",
              "--labels", "shared/labels/qwen3v3/math500_labels.jsonl", "--group-key", "sample_id:middle",
              "--target", target], S / "qwen3v3" / "logs" / f"within_math500_{target}.log")
    print("math500 within-group (should refuse)", target, "rc", rc, flush=True)
print("DONE", flush=True)

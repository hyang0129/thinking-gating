"""Compare the scratch re-run against the published metrics."""
import json
import sys
from pathlib import Path

MAIN = Path("/Users/hong/Documents/code-projects/thinking-gating")
WT = Path("/Users/hong/Documents/code-projects/thinking-gating/.claude/worktrees/agent-afbd0b4a543305334")
S = Path(__file__).resolve().parent
sys.path.insert(0, str(WT))
from scripts.compare_baselines import best  # noqa: E402

PUB = MAIN / "paper/results/metrics"


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def probe_diffs(slug, rerun_dir, label):
    print(f"\n== {label}: probe aggregates vs published {slug} ==")
    for pub in sorted((PUB / slug).glob("*__*.json")):
        if pub.name.startswith(("transfer__", "within_group__")):
            continue
        task, target = pub.stem.split("__", 1)
        new_path = rerun_dir / f"probe_{task}_{target}" / "aggregate_metrics.json"
        if not new_path.exists():
            continue
        a, b = json.loads(pub.read_text())["aggregate"], json.loads(new_path.read_text())["aggregate"]
        out = []
        for key in ("test_auroc", "test_routed_accuracy", "fraction_routed",
                    "min_routed_for_always_think_accuracy"):
            if not close(a[key]["mean"], b[key]["mean"]):
                out.append(f"{key} {a[key]['mean']:.4f}->{b[key]['mean']:.4f}")
        if not all(close(x, y) for x, y in zip(a["test_auroc_bootstrap"]["ci"], b["test_auroc_bootstrap"]["ci"])):
            out.append("boot ci {:.3f},{:.3f}->{:.3f},{:.3f}".format(*a["test_auroc_bootstrap"]["ci"], *b["test_auroc_bootstrap"]["ci"]))
        n50 = b["test_routed_nauc"]["escalation_0_50"]["mean"]
        print(f"  {task:<9}{target:<15} {'identical' if not out else '; '.join(out)}   [new nAUC0-50 {n50:+.3f}]")


def baseline_diffs(slug):
    print(f"\n== {slug}: baselines (test numbers, selection) ==")
    for pub in sorted((PUB / slug / "baselines").glob("*.json")):
        new = S / slug / "metrics" / "baselines" / pub.name
        if not new.exists():
            continue
        a, b = json.loads(pub.read_text())["baselines"], json.loads(new.read_text())["baselines"]
        same = all(close(a[k]["test_auroc"]["mean"], b[k]["test_auroc"]["mean"])
                   and all(close(x, y) for x, y in zip(a[k]["test_auroc_bootstrap"]["ci"],
                                                       b[k]["test_auroc_bootstrap"]["ci"]))
                   for k in a)
        old_name, old_node, _ = best(a)
        new_name, new_node, sel = best(b)
        flag = "" if old_name == new_name else f"  ** best changes {old_name} {old_node['mean']:.3f} -> {new_name} {new_node['mean']:.3f} (val {new_node['val']:.3f})"
        print(f"  {pub.stem:<38} test numbers {'identical' if same else 'DIFFER'}; best({sel}) {new_name}{flag}")


def within(slug):
    print(f"\n== {slug}: within-group ==")
    for pub in sorted((PUB / slug).glob("within_group__*.json")):
        new = S / slug / pub.name
        if new.exists():
            a, b = json.loads(pub.read_text()), json.loads(new.read_text())
            print(f"  {pub.stem}: published {a['within_group_auroc_bootstrap']['mean']:.4f} {a['within_group_auroc_bootstrap']['ci']} "
                  f"-> rerun {b['within_group_auroc_bootstrap']['mean']:.4f} {b['within_group_auroc_bootstrap']['ci']} "
                  f"(informative groups {b['n_informative_groups']}/{b['n_groups']}); overall "
                  f"{a['test_auroc_bootstrap']['mean']:.4f}->{b['test_auroc_bootstrap']['mean']:.4f}")


probe_diffs("qwen3v3", S / "qwen3v3", "qwen3v3 (layer 18 = middle)")
probe_diffs("nemotronv3", S / "nemotronv3_L18", "nemotronv3 at published layer 18")
probe_diffs("nemotronv3", S / "nemotronv3", "nemotronv3 at middle layer 16")
baseline_diffs("qwen3v3")
baseline_diffs("nemotronv3")
within("qwen3v3")
for slug in ("qwen3v3", "nemotronv3"):
    print(f"\n== {slug}: baseline_comparison.txt (rerun) ==")
    print((S / slug / "metrics" / "baseline_comparison.txt").read_text())

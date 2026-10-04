# Agent Instructions — thinking-gating

This file guides coding agents (Claude Code reads it through the `CLAUDE.md`
shim; Codex reads it directly). Human handoff context is in
`.agent-work/HANDOFF.md`; cluster procedure in `.agent-work/EMPIRE_AI_SETUP.md`.

**Self-containment (non-negotiable):** this repo owns everything it runs — its
task modules (`tasks/`), its dispatch tooling (`scripts/gpu_dispatch.py`,
`scripts/launch_jupyter.py`, `utils/jupyter_exec.py`), and its own virtualenv
(`.venv/`, built by `scripts/setup_env.sh`). Do not symlink, `sys.path`-inject,
or import from a sibling checkout, and do not install into a shared or system
interpreter. If a script needs something new, add it here and list it in
`requirements.txt`.

## Current State (2026-10-04)

**One line:** the repo is pivoting to **System One decision models** — can a
single-pass decision model estimate the value of computation it cannot perform,
better than its own confidence? Plan: GitHub issue #1
(`gh issue view 1 -R hyang0129/thinking-gating --comments`) and
`paper/proposal_system_one_gating.md`. The old prefill-probe line is **closed
as a negative result** (`paper/negative_result.md`), with an
"Errata (2026-10-04)" section at its top listing the bugs found in the
cleanup audit and what they change.

Do not reuse the old framing that "no evaluated method uses target-model
prefill state for thinking-mode routing": it is false (Self-Route, arXiv
2505.20664, routes think/no-think from hidden states).

### What carries over, and where

| piece | where | status |
|---|---|---|
| Paired thinking-off/on capture | `scripts/capture_inference_thinking.py` | Fixed in cleanup: B1 confidence padding (rows now carry `confidence_version: 2`), B3 `--max-response-len` is **required**, B4 over-long prompts are trimmed from the front of the user content and flagged `prompt_truncated` (never right-truncated past the assistant header), B5 per-shard `config.shardNN.json`, B6 `dataset_index_global`, B7 an unclosed `<think>` is graded wrong and flagged `unclosed_think_on`, B7b `has_reasoning_on` per row with a WARNING below 80%. |
| Confidence re-score (B1) | `scripts/rescore_confidence.py`, manifest `configs/dispatch/rescore_confidence_qwen3v3.json` | Re-scores stored thinking-off responses one unpadded row at a time with the capture's own prompt builder and scorer (GPU, no generation). Rebuilds each prompt by `prompt_hash` (BBH v3 used `PROMPT_TEMPLATE_V1`), checks prompt and answer token counts, and writes a `confidence_off_v2.shardNN.jsonl` sidecar plus `confidence_off_v2.summary.json`. Meta shards are never rewritten. `--check-only` runs the integrity checks on CPU with the tokenizer alone. |
| Truncation gate | same script | Above 20% thinking-OFF truncation a shard's meta is renamed `meta.shardNN.jsonl.quarantined`, `TRUNCATION_FAILURE.shardNN.json` is written, and the script **exits 1**. |
| Task modules + graders | `tasks/{gsm8k,math500,mmlu_pro,bbh,lsat}.py` | Graders fixed (B2 bbh, B16 math500, B17 mmlu_pro, B18 gsm8k); lsat audited clean. Regression cases in `tests/test_graders.py`. |
| Labels | `scripts/generate_labels.py` | `--regrade` recomputes `correct_off`/`correct_on` from the stored responses with the current grader (no GPU), keeps `*_stored`, records `grader_version`. |
| Statistics | `utils/metrics.py` | AUROC + percentile bootstrap, paired bootstrap of a difference (AUROC, nAUC), exact routed-accuracy curve + nAUC, pooled within-group AUROC, CEL(d) / signed overconfidence / reliability bins. Scripts import from here; do not re-implement. |
| Baselines | `scripts/baseline_text.py`, `scripts/baseline_confidence.py`, `scripts/compare_baselines.py` | Best baseline is now selected on **validation**, with a paired-bootstrap probe − baseline difference on identical test rows. `baseline_confidence.py --confidence-source auto` (default) reads the re-scored sidecar when present (`utils.capture_io.resolve_confidence`) and records the source in its output; `stored` forces capture-time values. |
| Within-group control | `scripts/within_group_auroc.py --group-key <field>\|sample_id:middle` | Key is explicit and required; refuses unless ≥ 2 groups hold both classes (B8). `sample_id:middle` is right for BBH only. |
| Probe / transfer | `scripts/run_experiment.py`, `scripts/eval_transfer.py` | The prefill probe becomes one baseline. Transfer verdicts now use the bootstrap interval over target rows (B9). |
| Full analysis | `scripts/run_full_analysis.sh` | Fail-fast; writes only `output/<slug>/`; regrades labels by default (`REGRADE=0` to keep stored grades); `PROMOTE=1` copies into `paper/results/metrics/<slug>/` and refuses to change existing files unless `FORCE=1`; `LAYER` defaults to the middle layer from the activation shape (18 for Qwen3-8B, 16 for Nemotron-8B). |
| Dispatch | `scripts/dispatch/`, `scripts/gpu_dispatch.py`, `scripts/launch_jupyter.py`, `scripts/watch_and_dispatch.py` | Fixed D1–D4, D6, D7; see **Cell + Worker Dispatch**. |
| Per-group AUROC (archival) | `scripts/stratify_check.py` | Kept only because `paper/results/metrics/decomposition/README.md` reproduces through `configs/dispatch/stratify_rescued.json`. Use `within_group_auroc.py` for new work. |

Not built yet (proposal §6 and the issue list the pieces): synthetic
generators with length controls, decision-model readout (LLM2Jev / Kev), the
Jev client, typed-question LoRA flag training, the wall-clock harness. That
work needs Qwen3.5 / Kev / vLLM / peft and a newer `transformers` than the
`<5` pin here, so it gets a **separate environment** (`requirements-s1.txt`
and its own venv — not created yet). Do not upgrade `.venv` in place: the
cluster redo captures run in it.

### Cluster state (checked 2026-10-04)

- **No SLURM jobs.** The watcher died 2026-09-12 on
  `node 'alphagpu52' not found in config` (bare hostname instead of the
  `<host>-<port>` key; fixed in c878020).
- Both redo queues are **0% done — never started**:
  `shared/dispatch/capture_nemotronv3_redo` (12 cells: gsm8k @2048,
  math500 @4096, mmlu_pro @2048 batch 8) and `shared/dispatch/capture_qwen3v3_redo`
  (4 cells: lsat @4096). Their manifests are frozen; the cell argv is tested
  to still parse (`tests/test_capture.py`).
- They will run on the **fixed** capture code when next launched — the
  cluster checkout must `git pull` this branch once it is merged. Relaunching
  means a Jupyter allocation (`launch_jupyter.py`, autonomous) plus the watcher
  or a worker dispatch (**needs approval**).
- The failed cells of the original v3 queues sit in `<root>/retired/`;
  `queue.py expand` no longer resurrects them (see below).

### Methodology rules that hold

- **Quote the bootstrap CI** (`test_auroc_bootstrap.ci`); the seed-spread CI
  is 1.6–8.8× too narrow.
- **Text and own-confidence baselines are mandatory**, selected on validation,
  compared paired on identical rows.
- **Use the pooled within-group AUROC** for multi-category benchmarks;
  per-group AUROCs are underpowered. BBH looked like 0.82 and was 0.49.
- **Check the truncation rate of the pass that defines the label**, and the
  rate of thinking-on rows that actually contain a reasoning trace.
- **Sample size binds.** `rescued` at n≈100 has a ±0.25 interval.

## Data layout

```
shared/icr_capture/{task}_thinking_{slug}/        # one capture, N shards
  config.json                    # first shard's args (legacy, kept)
  config.shardNN.json            # per-shard args, git commit, versions, end status
  meta.shardNN.jsonl             # one row per query: sample_id, question, answer,
                                 #   response_off/on, answer_off/on, correct_off/on,
                                 #   truncated_off/on, n_tokens_off/on, difficulty,
                                 #   confidence_off (with --capture-logprobs); new
                                 #   captures add has_reasoning_on, unclosed_think_on,
                                 #   dataset_index_global, confidence_version
  activations_thinking_off.shardNN.npz   # prefill states, (n, layers+1, hidden)
  activations_thinking_on.shardNN.npz
  TRUNCATION_FAILURE.shardNN.json        # present only if the gate quarantined the shard
  confidence_off_v2.shardNN.jsonl        # re-scored thinking-off confidence (B1),
  confidence_off_v2.summary.json         #   from scripts/rescore_confidence.py
    ↓ generate_labels.py [--regrade]
shared/labels/{slug}/{task}_labels.jsonl  (+ .summary.json)
    ↓ run_experiment.py / baseline_*.py / within_group_auroc.py / eval_transfer.py
output/{slug}/...                         # working results
    ↓ run_full_analysis.sh PROMOTE=1
paper/results/metrics/{slug}/             # provenance record
```

`utils/capture_io.py` loads and aligns shards; never read `meta.jsonl` by hand.

**Labels.** `needs_thinking = ~correct_off`; `helped = ~correct_off & correct_on`;
`rescued = correct_on` on rows with `correct_off == False`; `graded_label` also
keeps `hurt` (right → wrong). Train on the thinking-**off** prefill only
(`--prefill-mode off`, the default); thinking-on prefill is for analysis.

**Capture dir names** are `{task}_thinking_{model-slug}` with `{task}` the task
module name exactly (`gsm8k`, `math500`, `mmlu_pro`, `bbh`, `lsat`). Vary the
model slug, never the task name. Pre-v3 dirs spelled tasks differently
(`gsm8k_full`, `mmlupro`, `lsat_long`) and same-named dirs can differ
(`gsm8k_thinking_qwen3` is a 500-row pilot) — resolve legacy dirs through the
alias table in `run_full_analysis.sh`, never by guessing.

Large artifacts on Empire AI live in `/raid0/think-gating/`; `scp` **data**
back after long runs (data is gitignored; the no-`scp` rule is about code).

**Task contract** (`tasks/__init__.py`): `load_<task>(split) -> list[dict]`
with `question`, `answer`, `key`, `difficulty`; `format_prompt(question)`;
`is_correct(generation, answer[, question=])`. `_TASK_REGISTRY` in the capture
script lists only tasks with a module here.

## Environment & Dispatch

Full procedure in `.agent-work/EMPIRE_AI_SETUP.md`. The rules below are the ones
you must know *before* running anything — do not skip them because a command
looks harmless.

### Where are you running? (decide first, every session)

| Context | How to tell | What you may do there |
|---------|-------------|-----------------------|
| **Local machine** (this Mac) | No `squeue`; `torch.cuda.is_available()` is False | Write code, generate labels, train probes on CPU (slow), analyze results. **No GPU work** — do not try to load Qwen3-8B or run capture here. |
| **Empire AI login node** (`alpha1`) | `squeue --me` works; hostname `alpha1*` | Orchestration only: `git`, `gh`, `gpu_dispatch.py`, `launch_jupyter.py`, file inspection, `scripts/setup_env.sh`. **Never train or infer here** — no capture runs, no model loads, no pytest against a real model. |
| **GPU node** (`alphagpuNN`) | Never your shell's context — you only ever reach it through its Jupyter kernel | All real compute. Get there via `gpu_dispatch.py run` (batch) or `utils/jupyter_exec.py` through a tunnel (quick checks). |

GPU nodes are not reachable from a laptop or dev container at all, and even
from the login node direct SSH is unreliable (host-key rotation) and expensive
(~60s of PAM setup per channel, plus orphan processes against the `TasksMax`
cap). That is why every interaction — health probes, dispatch, status, kill —
goes through the node's Jupyter kernel instead.

### Local / Interactive
- `bash scripts/setup_env.sh` once, then `source .venv/bin/activate`
- `python scripts/capture_inference_thinking.py --help` for full options
- CPU is fine for label generation, probe training, and analysis; capture needs a GPU node

### Empire AI: deploying code
**Deploy with git. Never `scp` code or edit files directly on the cluster.**

```bash
# after committing + pushing from here
ssh empire-ai 'cd ~/LLM_research/thinking-gating && git pull --ff-only'
# deploying a branch before merge
ssh empire-ai 'cd ~/LLM_research/thinking-gating && git fetch && git checkout <branch>'
```

Why it matters: a `scp`'d or hand-edited file leaves the remote checkout dirty
and untracked. The next `git pull` conflicts, and — worse — a dispatched run can
execute code that exists in no commit, producing numbers you cannot reproduce or
trace to a diff. Reserve direct copies for throwaway scratch, never for code
that generates logged results.

### Empire AI: what needs approval

**Never submit or kill SLURM jobs without explicit user approval**, with the one
guarded exception below. Ask in a concrete form and wait for an answer:

> "I want to dispatch the GSM8K capture (500 samples) to alphagpu04. Yes/No?"

Requires approval every time:
- `gpu_dispatch.py run` — any capture, training, or eval dispatch
- Starting `watch_and_dispatch.py` — it calls `gpu_dispatch.py run` for you
- Raw `sbatch` / `srun` — always, no exceptions (it bypasses the guarded launcher's caps)

**Forbidden outright** — never do these, approval or not, unless the user
explicitly and specifically asks in the moment:
- `scancel`, `scontrol` cancel/suspend, `gpu_dispatch.py kill`, or killing remote PIDs. Agents do not cancel other people's (or their own) jobs on a shared cluster.
- Editing the caps in `scripts/launch_jupyter.py`, or working around a refusal from it by any other route
- Running compute on the login node
- Installing into a shared/system interpreter instead of this repo's `.venv`

#### Autonomous exception: launching Jupyter nodes
You **may** start a Jupyter allocation without asking, but **only** through
`scripts/launch_jupyter.py`. It enforces the caps in code and has no cancel path
by design:

- refuses at **≥ 12 RUNNING** `jupyter_*` jobs
- refuses at **≥ 12 jobs total**, any state (PENDING counts)
- refuses a port already served by a running jupyter job

```bash
ssh empire-ai 'cd ~/LLM_research/thinking-gating && python scripts/launch_jupyter.py 8882'
ssh empire-ai 'cd ~/LLM_research/thinking-gating && python scripts/launch_jupyter.py 8882 --dry-run'
```

Ports follow the 88xx convention. A non-zero exit means a cap was hit — report
it, do not route around it. **Give every node a distinct port** (8882, 8883,
8884, …): the launcher only refuses a port serving a *RUNNING* job, so a second
launch on a port that is still PENDING slips through and collides.

### Empire AI: dispatch hygiene
- **`sync-jupyter` first, every time.** `configs/nodes.json` goes stale as allocations come and go; `python scripts/gpu_dispatch.py sync-jupyter` rebuilds it from live `squeue`. Dispatch reaches only nodes with a live Jupyter allocation registered there. Nodes are keyed `<hostname>-<port>` (e.g. `alphagpu52-8882`); `run --node` wants that key, not the bare hostname.
- **Run `gpu_dispatch.py` itself with `.venv/bin/python`.** Its Jupyter transport imports `requests`/`websocket-client`; under the login node's system 3.9 the import fails and every node reports `unreachable`, which reads like a cluster outage but is not.
- **Quote a dispatched command containing flags** — `run` takes `nargs="+"`, so a bare `--root` after the command is parsed as gpu_dispatch's own argument.
- **Set `OMP_NUM_THREADS` for CPU work run directly on the login node** (192 cores, torch grabs them all: 315s vs 22s on the same tests). Dispatched cells already get this.
- **Name `.venv/bin/python` in the dispatched command.** `gpu_dispatch.py run` passes the command through verbatim, so a bare `python` silently picks up the node default and you get `ModuleNotFoundError` — or worse, a different transformers version.
- **Commit before dispatching.** A run whose code is not in a commit is not reproducible. Each attempt records `git_commit`/`git_dirty` in its result and log header.
- **A timed-out dispatch may have launched anyway.** Before re-dispatching anything, run `gpu_dispatch.py jobs --all` and wait ≥ 2 minutes. Duplicate captures silently double-append to the shard's meta file.
- **Job manifest:** `shared/gpu_jobs.json` (relative to `project_root`). Job logs: `shared/logs/<job_id>.log`.
- **Fan-out work goes through the cell queue, not many `gpu_dispatch.py run` calls.** See **Cell + Worker Dispatch** below. A single `run` is right for a one-off; a sweep is a manifest plus N workers.
- **Don't guess whether data exists — check.** `wc -l <capture-dir>/meta.shard*.jsonl`, the NPZ shapes, `config.shardNN.json`'s end status, and any `TRUNCATION_FAILURE.*.json` tell you what a capture actually produced; a job that appeared to finish may have OOM'd mid-run.

### Cell + Worker Dispatch (sweeps, batches, anything that fans out)

`scripts/dispatch/` is a coordinator-free work queue. Workers on any number of
nodes race to claim **cells** from a shared directory via atomic `rename(2)`.

**The worker never changes.** A cell fully describes its own work, so new
training, inference, or data-generation sweeps mean writing a manifest — never
editing `worker.py`. Four cell kinds cover everything:

| kind | payload | use it for |
|------|---------|-----------|
| `python_script` | `script` + `args` | running any script in `scripts/` |
| `python_code` | `code` | a few lines of inline Python, no file needed |
| `call` | `target: "module:function"` + `args`/`kwargs` | putting **any importable function** on the cluster; its return value is captured to `results/<cell_id>.json` |
| `shell` | `command` | escape hatch |

Workflow:

```bash
# 1. write a manifest (copy configs/dispatch/example_probe_sweep.json), then preview
python scripts/dispatch/queue.py expand configs/dispatch/my_sweep.json --dry-run

# 2. queue it (idempotent — re-expanding never re-runs finished cells)
python scripts/dispatch/queue.py expand configs/dispatch/my_sweep.json

# 3. dispatch N workers onto N nodes (job submission — needs approval)
python scripts/gpu_dispatch.py run --desc "my_sweep worker" \
    ".venv/bin/python scripts/dispatch/worker.py --root shared/dispatch/my_sweep"

# 4. watch, then triage
python scripts/dispatch/queue.py status --root shared/dispatch/my_sweep
python scripts/dispatch/queue.py logs   --root shared/dispatch/my_sweep --cell <id>
python scripts/dispatch/queue.py retry  --root shared/dispatch/my_sweep --all
```

A manifest expands by `grid` (cartesian product), `zip` (lockstep), and
`exclude`, with `{name}` templating across every field:

```json
{
  "name": "gsm8k_probe", "kind": "python_script",
  "script": "scripts/run_experiment.py", "cell_id_hash": true,
  "constants": {"out": "output/{name}/{method}_seed{seed}"},
  "grid": {"method": ["mlp", "logreg"], "seed": [42, 1, 2, 3, 4]},
  "args": ["--capture-dir", "shared/icr_capture/gsm8k_thinking_qwen3v3",
           "--labels", "shared/labels/qwen3v3/gsm8k_labels.jsonl",
           "--method", "{method}", "--seeds", "{seed}", "--out-dir", "{out}"],
  "output_check": ["{out}/aggregate_metrics.json"],
  "timeout_s": 7200, "max_attempts": 2
}
```

Semantics worth relying on:
- **`output_check` is the resume mechanism.** Present before the run → cell is skipped. Missing after exit 0 → cell is **failed**, not quietly completed. Always set it; a script that exits 0 having written nothing is the failure mode this catches.
- **Isolation.** Each cell is a subprocess in its own process group; a segfault or OOM kills the cell, not the worker. `gpu_dispatch.py kill` signals the whole process group.
- **Resumable and re-entrant.** Re-launching workers over a partly-drained queue is the normal recovery path. Cells from a crashed worker return to pending once its heartbeat goes stale (5 min); `queue.py gc` forces it.
- **`max_attempts > 1`** re-queues on failure so another node can try — **except** a terminal failure: if the attempt leaves a `terminal_markers` file (default `TRUNCATION_FAILURE.*.json`, the capture gate's marker) for its own shard next to an `output_check` path, the cell fails for good. Re-running a quarantined shard at the same budget only burns GPU; raise the budget in a new manifest.
- **Cell identity.** Ids come from the template/grid, not the args. Re-expanding an *edited* manifest is refused before anything is written (`--replace` overrides). New manifests should set `"cell_id_hash": true`, which appends an args fingerprint so an edit yields new ids. Existing manifests do not, so their ids never move.
- **Retired cells stay retired.** `expand` also scans `<root>/retired/` and will not recreate a retired cell (`--allow-resurrect` overrides). Re-expanding `capture_qwen3v3.json` / `capture_nemotronv3.json` is therefore safe but pointless — those manifests are archival.
- **Shutdown is clean.** SIGTERM releases the in-flight cell back to pending immediately. A worker that loses its claim to GC kills its copy and exits 3; cells must be idempotent because a double run is still possible.

**The watcher** (`scripts/watch_and_dispatch.py --roots <root> ...`) polls until
every root with work has a live worker or is drained, so a second allocation
that lands later gets its own worker; liveness comes from heartbeats, so a dead
worker's stale claim no longer blocks a root. It `git fetch`es and refuses to
dispatch while the checkout is behind upstream. Stop it with
`touch shared/dispatch/STOP_WATCH`. Starting it needs approval (above).

Run the tests after touching anything under `scripts/dispatch/`:
`python3 tests/test_dispatch.py` (stdlib only, no GPU, 45 tests).

### Empire AI: reaching a GPU node interactively
For quick verification (is CUDA visible? did the checkpoint land?), use a
tunnel + Jupyter kernel rather than asking the user to run cells by hand:

```bash
# 1. tunnel localhost -> GPU node (pick an unused local port)
ssh -f -N -L 18882:alphagpuXX:8882 empire-ai

# 2. run code against it
GPUNODE=localhost GPUNODEPORT=18882 .venv/bin/python utils/jupyter_exec.py \
    "import torch; print(torch.cuda.get_device_name(0))"
```

Or from Python:

```python
import os
os.environ["GPUNODE"], os.environ["GPUNODEPORT"] = "localhost", "18882"
from utils.jupyter_exec import JupyterExecutor

with JupyterExecutor() as jup:
    result = jup.run("import torch; print(torch.cuda.is_available())")
    print(result.status, result.stdout)   # status: "ok" | "error" | "timeout"
```

`GPUNODE`/`GPUNODEPORT` may also live in a gitignored `.env` at the repo root;
env vars win over it. Add `--keep-kernel` for work that must survive an SSH drop.

This is the same transport `gpu_dispatch.py` uses, and it is read-only — it runs
no SLURM commands, so it needs no approval.

### Answering "what's running on the cluster?"
Correlate three sources, then report:

```bash
ssh empire-ai 'squeue --me --format="%.18i %.9P %.30j %.8T %.10M %R %N"'   # allocations (name = jupyter_empire_<port>)
ssh empire-ai 'cat ~/LLM_research/thinking-gating/shared/gpu_jobs.json'    # our dispatched jobs (filter status=="running")
ssh empire-ai 'cd ~/LLM_research/thinking-gating && .venv/bin/python scripts/gpu_dispatch.py status'  # live GPU util + VRAM
```

A `gpu_jobs.json` entry maps to an allocation by `node_name` (`alphagpuNN-PPPP`),
whose port matches the `jupyter_empire_<port>` SLURM job name. An allocation with
no running manifest entry is an **idle Jupyter node** — say so rather than
implying work is in flight.

## Script reference (real CLIs)

```bash
# capture (GPU node only; --max-response-len is required — size it so
# thinking-OFF truncation is near zero; known-good budgets are in
# configs/dispatch/capture_qwen3v3.json and the *_redo.json manifests)
.venv/bin/python scripts/capture_inference_thinking.py --task math500 \
    --model Qwen/Qwen3-8B --chat-template --max-response-len 2048 \
    --max-response-len-thinking 4096 --capture-logprobs \
    --out-dir /raid0/think-gating/math500_thinking_qwen3v3 \
    [--max-samples N] [--shard-index i --shard-count k] [--batch-size 8]

# labels (CPU); --regrade re-applies the current grader to stored responses
.venv/bin/python scripts/generate_labels.py --regrade \
    --capture-dir shared/icr_capture/math500_thinking_qwen3v3 \
    --out-file shared/labels/qwen3v3/math500_labels.jsonl [--drop-truncated]

# probe (CPU); writes aggregate_metrics.json, predictions.json, seed_<N>/{metrics,checkpoint}.json
.venv/bin/python scripts/run_experiment.py \
    --capture-dir shared/icr_capture/math500_thinking_qwen3v3 \
    --labels shared/labels/qwen3v3/math500_labels.jsonl \
    --target {rescued|needs_thinking|helped} --method {logreg|mlp} \
    --seeds 42 1 2 3 4 --out-dir output/qwen3v3/probe_math500_rescued [--layer L]

# baselines on the identical splits, then the comparison table (M = output/<slug>/metrics)
.venv/bin/python scripts/baseline_text.py       --capture-dir ... --labels ... --target rescued --out-file M/baselines/text__math500__rescued.json
.venv/bin/python scripts/baseline_confidence.py --capture-dir ... --labels ... --target rescued --out-file M/baselines/confidence__math500__rescued.json
.venv/bin/python scripts/compare_baselines.py --metrics-dir M --out M/baseline_comparison.txt

# controls
.venv/bin/python scripts/within_group_auroc.py --capture-dir ... --labels ... \
    --group-key sample_id:middle --target rescued --out-file M/within_group__bbh__rescued.json
.venv/bin/python scripts/eval_transfer.py --probe output/.../seed_*/checkpoint.json \
    --capture-dir <target capture> --labels <target labels> \
    --source-metrics output/.../aggregate_metrics.json --out-file M/transfer__src_to_tgt__rescued.json

# all of the above for one model slug
CAPTURE_SLUG=qwen3v3 TASKS="gsm8k math500 mmlu_pro bbh" bash scripts/run_full_analysis.sh
```

The probe is logistic regression (or a small MLP) on one layer's prefill
state; 60/20/20 stratified splits per seed, 5 seeds, early stopping on
validation loss. There is no contrastive probe and no k-fold CV.

## Testing & Validation

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m pytest tests -q   # needs numpy, scikit-learn, torch, transformers
python3 tests/test_dispatch.py                          # stdlib only
```

`test_confidence.py` / `test_capture.py` run a tiny random Llama on CPU (and
gpt2 when cached). The math-verify oracle cases skip unless `math_verify` is
installed.

**Smoke test.** Step 1 needs a GPU node — dispatch it with approval, never run
it locally or on the login node:

```bash
ssh empire-ai 'cd ~/LLM_research/thinking-gating && .venv/bin/python scripts/gpu_dispatch.py sync-jupyter && \
  .venv/bin/python scripts/gpu_dispatch.py run --desc "gsm8k smoke" \
    ".venv/bin/python scripts/capture_inference_thinking.py \
        --task gsm8k --max-samples 100 --max-response-len 1024 \
        --model Qwen/Qwen3-8B --out-dir /raid0/think-gating/gsm8k_smoke --chat-template"'

# then confirm it actually produced rows and was not quarantined
ssh empire-ai 'wc -l /raid0/think-gating/gsm8k_smoke/meta.shard*.jsonl; ls /raid0/think-gating/gsm8k_smoke/'
```

Steps 2–3 are CPU-only (after copying the capture back):

```bash
.venv/bin/python scripts/generate_labels.py --regrade \
    --capture-dir /tmp/gsm8k_smoke --out-file /tmp/gsm8k_smoke_labels.jsonl
.venv/bin/python scripts/run_experiment.py --capture-dir /tmp/gsm8k_smoke \
    --labels /tmp/gsm8k_smoke_labels.jsonl --seeds 42 --max-epochs 5 \
    --out-dir /tmp/gsm8k_probe_smoke
```

**Before believing any number:**
- **Truncation first.** Thinking-OFF truncation must be near zero (the gate
  quarantines and exits 1 above 20%, but 8% still contaminates labels); also
  check thinking-ON truncation, `unclosed_think_on`, and the
  `has_reasoning_on` rate — a "thinking" run that does not think compares two
  non-thinking runs.
- **Regrade** with the current graders and record `grader_version`.
- AUROC above 0.5 and below oracle; quote `test_auroc_bootstrap.ci`, never
  `test_auroc.ci`.
- Beat the **text and confidence baselines** on identical rows (paired
  difference from `compare_baselines.py`). An 8B forward pass that cannot beat
  TF-IDF on the raw question is not a result.
- For multi-category data, the **pooled within-group AUROC** with an explicit
  `--group-key`; if signal holds only across groups, it is a category detector.

## Common Pitfalls

- Don't reach outside this repo for code, data loaders, or a Python environment.
- Don't submit or kill cluster jobs without approval, or run compute on the login node.
- Don't leak test labels into training; evaluate on held-out splits only.
- Don't train one probe on mixed tasks and call it transfer.
- Don't use thinking-on outputs or activations as features for predicting the
  value of thinking (circular).
- Don't quote a confidence-baseline number from a capture without
  `confidence_version: 2` (B1) or a complete `confidence_off_v2` sidecar;
  check `confidence_source` in the baseline's output JSON.
- Do commit before dispatching, and dispatch `.venv/bin/python`.
- Do log base rates and n beside every AUROC.

## Paper / Results

`paper/results/` exists and is the provenance record: metrics JSON copied
verbatim from cluster runs, one file per run, plus a README per group
explaining how to read it (`baselines/`, `decomposition/`, `tuning/`,
`truncation/`, `qwen3v3/`, `nemotronv3/`). **A number in the paper must trace
to a file there.** Working metrics land in `output/` first; promote to
`paper/results/` when a run is one you would cite (`run_full_analysis.sh
PROMOTE=1`), and write the README entry at the same time — the READMEs are
where the caveats live, and the caveats are the load-bearing part. Never
overwrite a promoted file to "fix" it; add a new run and an erratum.

**Never edit a `.bib` file directly.** Agents hallucinate references. Add a
citation in the section text with enough context (title, authors, venue, year)
for a human to verify, and wait for explicit approval before any bibliography
insertion. Numbers quoted in the paper come from the saved metrics JSON/CSV in
`output/`, never retyped from a chat message or a log scroll.

---

**Last updated:** 2026-10-04 (cleanup for the System One pivot)
**Questions/blockers?** See `.agent-work/HANDOFF.md`.

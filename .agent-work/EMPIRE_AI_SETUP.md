# Empire AI Setup & Dispatch Conventions

This repo is **self-contained**. It ships its own task modules (`tasks/`), its own
dispatch tooling (`scripts/gpu_dispatch.py`, `scripts/launch_jupyter.py`,
`utils/jupyter_exec.py`), and it runs out of **its own virtualenv**. Nothing is
symlinked, copied, or imported from a sibling checkout at run time — if a script
needs something, it lives in this repo and is listed in `requirements.txt`.

## Pre-Dispatch Setup (One-Time)

### 1. Clone on Empire AI
```bash
ssh empire-ai 'mkdir -p ~/LLM_research && cd ~/LLM_research && \
  git clone <repo-url> thinking-gating'

# Already cloned? Just refresh:
ssh empire-ai 'cd ~/LLM_research/thinking-gating && git fetch && git pull origin main'
```

### 2. Build the repo's own venv
`scripts/setup_env.sh` creates `.venv/` inside the checkout and installs
`requirements.txt` into it. Run it on the **login node** — pip needs no GPU, and
the venv lives in the shared home directory so every GPU node sees it.

```bash
ssh empire-ai 'cd ~/LLM_research/thinking-gating && bash scripts/setup_env.sh'
```

The script prints the resolved interpreter plus the installed torch/transformers
versions. Two things to check in that output:

- `transformers` is **≥ 4.51** — Qwen3 support and the `enable_thinking`
  chat-template flag both landed there, and the whole paired-capture design
  depends on it.
- torch reports `cuda=True` when the check runs on a GPU node. On the login node
  `cuda=False` is expected and fine.

No conda, no `pip install --user`, no shared site-packages. Every dispatched
command runs `.venv/bin/python` explicitly, so the environment cannot drift.

### 3. Point the dispatcher at this repo's venv
`scripts/gpu_dispatch.py` reads `configs/nodes.json`, which is gitignored
(`sync-jupyter` rewrites its `nodes` list from live SLURM state). Seed it once
from the tracked template:

```bash
ssh empire-ai 'cd ~/LLM_research/thinking-gating && \
  cp -n configs/nodes.example.json configs/nodes.json'
```

Then edit `defaults` in `configs/nodes.json` to this checkout:

```json
"defaults": {
  "python": "/home/YOUR_USER/LLM_research/thinking-gating/.venv/bin/python",
  "project_root": "/home/YOUR_USER/LLM_research/thinking-gating",
  "job_manifest": "shared/gpu_jobs.json",
  "jupyter_password": "123"
}
```

`project_root` is what every dispatched job `cd`s into, and `python` is the
interpreter written into node entries by `sync-jupyter`. Both must point at this
repo — not at any other project's checkout or interpreter.

### 4. Register live Jupyter allocations
Dispatch runs entirely over each node's Jupyter kernel (SSH dispatch is not
supported — the login node spends ~60s per SSH channel on PAM setup and leaks
orphan processes against the `TasksMax=512` cap). So a node is only usable once
it has a running `jupyter_*` allocation:

```bash
# Launch one (guarded launcher: hard caps, no cancel path)
ssh empire-ai 'cd ~/LLM_research/thinking-gating && python scripts/launch_jupyter.py 8882'

# Reconcile configs/nodes.json against live SLURM state
ssh empire-ai 'cd ~/LLM_research/thinking-gating && python scripts/gpu_dispatch.py sync-jupyter'

# Confirm
ssh empire-ai 'cd ~/LLM_research/thinking-gating && python scripts/gpu_dispatch.py status'
```

## SSH Convenience
Add to `~/.ssh/config`:
```
Host empire-ai
    HostName alpha1.empire-ai.org
    User YOUR_USERNAME
    ControlMaster auto
    ControlPersist 60m
    ControlPath ~/.ssh/control_%h_%p_%r
```

Then `ssh empire-ai 'cmd'` reuses connections.

## Deploying Code Changes

**Use git. Never `scp` code to the cluster or edit files there directly.**

```bash
# after committing + pushing from the local repo
ssh empire-ai 'cd ~/LLM_research/thinking-gating && git pull --ff-only'

# deploying a branch before merge
ssh empire-ai 'cd ~/LLM_research/thinking-gating && git fetch && git checkout <branch>'
```

A `scp`'d or hand-edited file leaves the remote checkout dirty and untracked: the
next `git pull` conflicts, and a dispatched run can execute code that exists in
no commit — results you cannot reproduce or trace to a diff. Direct copies are
for throwaway scratch only, never for code that generates logged numbers.

Data moves the other way: `scp` captures and checkpoints *back* from the cluster,
since `shared/` and `output/` are gitignored.

## Standard Dispatch Pattern

### Before every dispatch
1. `git status` clean and pushed, then `git pull --ff-only` on the cluster — the run must correspond to a commit.
2. `python scripts/gpu_dispatch.py sync-jupyter` — `configs/nodes.json` goes stale as SLURM allocations come and go, and dispatch only reaches nodes registered there.
3. `python scripts/gpu_dispatch.py status` — confirm a node with enough free VRAM.
4. Get explicit user approval for the dispatch itself (see Rules below).
5. Write `.venv/bin/python` in the command, not bare `python`.

If a dispatch call times out, **it may still have launched**. Check
`gpu_dispatch.py jobs --all` and wait at least 2 minutes before re-dispatching —
a duplicate capture silently double-appends to `meta.jsonl`.

Always invoke `.venv/bin/python` in the dispatched command. `gpu_dispatch.py run`
passes the command through verbatim, so a bare `python` would resolve to
whatever the node's default interpreter happens to be.

### Capture (Example)
```bash
ssh empire-ai 'cd ~/LLM_research/thinking-gating && python scripts/gpu_dispatch.py run \
    --min-vram 20 \
    --desc "gsm8k thinking capture" \
    .venv/bin/python scripts/capture_inference_thinking.py \
        --task gsm8k \
        --model Qwen/Qwen3-8B \
        --max-samples 500 \
        --out-dir /raid0/think-gating/gsm8k_thinking_qwen3 \
        --chat-template'

# Monitor:
ssh empire-ai 'cd ~/LLM_research/thinking-gating && python scripts/gpu_dispatch.py jobs --all'
```

### Training (Example)
```bash
ssh empire-ai 'cd ~/LLM_research/thinking-gating && python scripts/gpu_dispatch.py run \
    --desc "gsm8k mlp probe, 5 seeds" \
    .venv/bin/python scripts/run_experiment.py \
        --activations /raid0/think-gating/gsm8k_thinking_qwen3/activations_thinking_off.npz \
        --labels /raid0/think-gating/gsm8k_thinking_labels.jsonl \
        --method mlp \
        --seeds 42 1 2 3 4 \
        --out-dir /raid0/think-gating/probes/gsm8k_mlp'
```

## Data Paths

| Context | Path | Notes |
|---------|------|-------|
| Local (macBook) | `shared/icr_capture/` | Relative to repo root; can be a symlink to an external volume if large |
| Empire AI | `/raid0/think-gating/` | Fast node-local SSD; sync back after capture with `scp -r` |
| Checkout | `~/LLM_research/thinking-gating/` | Code + `.venv/` in shared home; data does **not** live here |
| Job logs | `shared/logs/<job_id>.log` | Relative to `project_root`; written by dispatched jobs |

## Jupyter (Interactive)

```bash
# Launch a single Jupyter node (guarded launcher, no approval needed)
ssh empire-ai 'cd ~/LLM_research/thinking-gating && python scripts/launch_jupyter.py 8882'

# Dry-run to preview the sbatch line
ssh empire-ai 'cd ~/LLM_research/thinking-gating && python scripts/launch_jupyter.py 8882 --dry-run'

# Tunnel from macBook to the Jupyter node
ssh -f -N -L 18882:alphagpuXX:8882 empire-ai

# Run code against it
GPUNODE=localhost GPUNODEPORT=18882 .venv/bin/python utils/jupyter_exec.py \
    "import torch; print(torch.cuda.get_device_name(0))"

# Longer interactive work: keep the kernel alive across an SSH drop
GPUNODE=localhost GPUNODEPORT=18882 .venv/bin/python utils/jupyter_exec.py \
    --keep-kernel -f scratch/probe_check.py
```

`GPUNODE`/`GPUNODEPORT` can also live in a `.env` at the repo root (gitignored);
environment variables win over the file. `jupyter_exec.py` also takes `--file`,
`--stream`, and `--timeout`, and exits non-zero when the cell errors.

The launcher enforces hard caps in one place (`MAX_ACTIVE_JUPYTER`,
`MAX_TOTAL_JOBS`, both 12) and has no cancel path by design. It submits
`~/rit_rc_scripts/empire_jupyter_lab.sh`, which is a per-user cluster script and
the one piece of this workflow that lives outside the repo.

A non-zero exit means a cap was hit — report the refusal, never route around it
with raw `sbatch` or by editing the caps.

**Give every node a distinct 88xx port** (8882, 8883, 8884, …). The launcher only
refuses a port already serving a *RUNNING* job, so a second launch on a port
whose job is still PENDING slips past the check and collides.

After launching, register it: `python scripts/gpu_dispatch.py sync-jupyter`.

## Rules

### ❌ Forbidden
- Train/infer on the login node (alpha1) — use `gpu_dispatch.py`
- Direct edits or `scp` of code to Empire AI — use git
- Job submission without approval (except Jupyter via the guarded launcher)
- Raw `sbatch` / `srun` — always, since it bypasses the launcher's caps
- `scancel`, `scontrol` cancel/suspend, `gpu_dispatch.py kill`, or killing remote
  PIDs — agents do not cancel jobs on a shared cluster, ever, without the user
  asking for that specific kill in the moment
- Editing the caps in `scripts/launch_jupyter.py`, or working around one of its
  refusals by another route
- Force-push to git
- Installing into a system/shared interpreter, or reaching into another repo's
  venv or modules — this repo runs from its own `.venv` only

Approval means asking concretely and waiting for an answer:

> "I want to dispatch the GSM8K capture (500 samples) to alphagpu04. Yes/No?"

### ✅ Required
- All code committed before dispatch
- `sync-jupyter` before every dispatch session
- `.venv/bin/python` (not bare `python`) in every dispatched command
- Explicit user approval for SLURM job submission (except guarded Jupyter)
- Multi-seed runs + CI reporting in results
- Sync data back with `scp` after long-running jobs
- Verify a finished job actually produced data (`wc -l .../meta.jsonl`, NPZ
  shapes) rather than assuming — a job can exit after OOM-ing mid-run

One `gpu_dispatch.py run` is right for a one-off job. Anything that fans out
into many runs goes through the cell queue below.

## Cell + Worker Dispatch

For sweeps, batches, and anything else that fans out, use the work queue in
`scripts/dispatch/` rather than dispatching each run by hand.

Workers on any number of nodes race to claim cells from a shared directory via
atomic `rename(2)` — no coordinator, no lock server. **The worker is generic and
never changes**: a cell describes its own work (`python_script`, `python_code`,
`call` an importable function, or `shell`), so a new sweep is a new manifest.

### Running a sweep

```bash
# 1. Preview what the manifest expands to
python scripts/dispatch/queue.py expand configs/dispatch/my_sweep.json --dry-run

# 2. Queue it. Idempotent: re-running never re-queues finished or claimed cells,
#    so this is also how you append new work to a live queue.
python scripts/dispatch/queue.py expand configs/dispatch/my_sweep.json
#    → cells land in shared/dispatch/<manifest name>/pending/

# 3. Commit, deploy, sync nodes, then dispatch one worker per node.
#    This is job submission — get approval first.
ssh empire-ai 'cd ~/LLM_research/thinking-gating && git pull --ff-only && \
  python scripts/gpu_dispatch.py sync-jupyter'

for node in alphagpu04-8882 alphagpu07-8883; do
  ssh empire-ai "cd ~/LLM_research/thinking-gating && \
    python scripts/gpu_dispatch.py run --node $node --desc 'my_sweep worker' \
      .venv/bin/python scripts/dispatch/worker.py --root shared/dispatch/my_sweep"
done

# 4. Monitor and triage
ssh empire-ai 'cd ~/LLM_research/thinking-gating && \
  python scripts/dispatch/queue.py status --root shared/dispatch/my_sweep'
ssh empire-ai 'cd ~/LLM_research/thinking-gating && \
  python scripts/dispatch/queue.py logs --root shared/dispatch/my_sweep --cell <cell_id> --tail 50'
ssh empire-ai 'cd ~/LLM_research/thinking-gating && \
  python scripts/dispatch/queue.py retry --root shared/dispatch/my_sweep --all'
```

Add workers at any time — they just start claiming. A worker exits when the
queue drains; pass `--wait 600` to keep it alive while more cells are being
added.

### Queue layout

```
shared/dispatch/<name>/
├── pending/<prio>__<cell_id>.json     # claimable
├── claimed/<worker_id>/…               # in flight, plus a heartbeat file
├── done/<prio>__<cell_id>.json         # cell + result record
├── failed/<prio>__<cell_id>.json       # cell + result record (error tail included)
├── logs/<cell_id>.attempt<N>.log       # full stdout+stderr, one per attempt
└── results/<cell_id>.json              # return value of `call` cells
```

Dispatch roots live under `shared/`, which is gitignored — the queue is runtime
state, not code. It survives across worker restarts, so a drained-then-refilled
queue is the normal way to run a long campaign.

### Operational notes

- **Set `output_check` on every cell.** It doubles as the resume mechanism (present before the run → skipped) and the correctness guard (missing after exit 0 → recorded as **failed**, since scripts do exit 0 having written nothing).
- **Cells must be idempotent** — a cell can run twice after a node dies.
- **Dead workers.** A worker killed by SLURM leaves its cell in `claimed/`; once its heartbeat is 5 minutes stale, the next worker's startup GC re-queues it. Force it with `queue.py gc --root …`.
- **Stale workers in `status`** are the signal that an allocation died mid-cell.
- **Per-cell env** for cluster quirks — set these on cells that load models:
  `"env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}` (weights are
  NFS-cached; `from_pretrained` otherwise makes a network call) and
  `"CUDA_MPS_PIPE_DIRECTORY": "/no/such/path"` on nodes where another user's MPS
  daemon blocks the runtime. `worker.py --env KEY=VAL` applies them queue-wide.
- **Tests:** `python3 tests/test_dispatch.py` — stdlib only, no GPU, ~15s. Run it after touching `scripts/dispatch/`.

## Answering "What's Running on the Cluster?"

Three sources, correlated:

```bash
# 1. live SLURM allocations (job name encodes the port: jupyter_empire_<port>)
ssh empire-ai 'squeue --me --format="%.18i %.9P %.30j %.8T %.10M %R %N"'

# 2. our dispatched jobs (filter status == "running"; fields: node_name, command, pid, description, started_at)
ssh empire-ai 'cat ~/LLM_research/thinking-gating/shared/gpu_jobs.json'

# 3. per-node GPU util + VRAM used/total, and how many of our jobs each is running
ssh empire-ai 'cd ~/LLM_research/thinking-gating && python scripts/gpu_dispatch.py status'
```

A manifest entry maps to an allocation through `node_name` (`alphagpuNN-PPPP`),
whose port matches the `jupyter_empire_<port>` SLURM job name.

| Signal | Meaning |
|--------|---------|
| Manifest entry `status == "running"` + matching allocation | Active dispatched job |
| Allocation with no running manifest entry | **Idle Jupyter node** — capacity, not work in flight |
| Manifest entry whose node has no allocation | The allocation ended; `jobs --all` re-probes and marks such jobs `unknown` rather than `running` |

Report idle nodes as idle. An allocation is not evidence that anything is
computing.

## Troubleshooting

### `ModuleNotFoundError` inside a dispatched job
The job used the wrong interpreter. Check that the command starts with
`.venv/bin/python`, and that `defaults.python` in `configs/nodes.json` points at
this checkout's venv. Re-verify with:
```bash
ssh empire-ai 'cd ~/LLM_research/thinking-gating && .venv/bin/python -c \
  "import transformers, datasets; print(transformers.__version__, datasets.__version__)"'
```

### Qwen3 fails to load / `enable_thinking` ignored
`transformers` is too old. Rebuild the venv:
```bash
ssh empire-ai 'cd ~/LLM_research/thinking-gating && bash scripts/setup_env.sh --recreate'
```

### `no candidate nodes have jupyter_url configured`
No live Jupyter allocation is registered. Launch one, then `sync-jupyter`
(step 4 above).

### Job Stuck in PENDING
```bash
ssh empire-ai 'squeue -j JOB_ID --format="%.10i %.9P %.8j %.8u %.2t %.10M %.6D %R"'
ssh empire-ai 'sinfo --Node --state=idle'
```

### OOM During Capture
- Reduce `--max-prompt-len` or `--max-response-len`
- Check node VRAM: `python scripts/gpu_dispatch.py status`

### Stale Checkout
```bash
ssh empire-ai 'cd ~/LLM_research/thinking-gating && git status'

# If dirty, stash and pull
ssh empire-ai 'cd ~/LLM_research/thinking-gating && git stash && git pull origin main'
```
Note that `configs/nodes.json` and `.venv/` are gitignored, so neither shows up
as dirty and neither survives a fresh clone — re-run steps 2–4 after cloning
somewhere new.

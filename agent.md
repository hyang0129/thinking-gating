# Agent Instructions — Thinking-Mode Gating Experiment

This document guides agents (Claude Code, subagents) working on the thinking-gating repo. For human handoff context, see `.agent-work/HANDOFF.md`.

## Project Overview

**Goal:** Build and evaluate a probe that predicts whether thinking mode (extended reasoning) will improve query outcome, using only the prefill hidden state (last-prompt-token activation).

**Paper context:** Pathway 1 of the prefill-applications line of work. Named gap in HRBench: no evaluated method uses target-model prefill state for thinking-mode routing.

**Self-containment (non-negotiable):** this repo owns everything it runs — its task modules (`tasks/`), its dispatch tooling (`scripts/gpu_dispatch.py`, `scripts/launch_jupyter.py`, `utils/jupyter_exec.py`), and its own virtualenv (`.venv/`, built by `scripts/setup_env.sh`). Do not symlink, `sys.path`-inject, or import from a sibling checkout, and do not install into a shared or system interpreter. If a script needs something new, add it here and list it in `requirements.txt`.

## Current State (2026-08-21)

✅ **Phase 1 complete:**
- Repo bootstrapped at `/Users/hong/Documents/code-projects/thinking-gating/`
- `scripts/capture_inference_thinking.py` handles:
  - Paired inference (thinking-off + thinking-on)
  - Prefill-only extraction (last-prompt-token hidden state)
  - Metadata storage for downstream label generation
- Configs locked: GSM8K (primary) + LSAT Logic Games (secondary transfer test)
- Requirements + .gitignore in place

✅ **Self-contained (2026-08-21):**
- `tasks/gsm8k.py` + `tasks/lsat.py` written in-repo (loaders, prompts, graders, difficulty buckets)
- `scripts/gpu_dispatch.py`, `scripts/launch_jupyter.py`, `utils/jupyter_exec.py` vendored in
- `scripts/setup_env.sh` builds the repo's own `.venv`; `configs/nodes.example.json` templates the dispatch config
- `requirements.txt` re-pinned — `transformers >= 4.51` is a hard floor for Qwen3 + `enable_thinking`

⏳ **Phase 2 in progress:**
- Label generation (`generate_labels.py`) — needs writing
- Probe training (`run_experiment.py`) — needs writing
- Transfer evaluation (`eval_transfer.py`) — needs writing
- Template ablation (`template_ablation.py`) — optional, lower priority

## Architecture & Key Decisions

### Data Flow
```
Raw dataset (GSM8K) 
  ↓ capture_inference_thinking.py
shared/icr_capture/gsm8k_thinking_qwen3/
  ├── config.json
  ├── meta.jsonl (correctness_off, correctness_on per query)
  ├── activations_thinking_off.npz (N, num_layers, hidden_dim)
  └── activations_thinking_on.npz (N, num_layers, hidden_dim)
  ↓ generate_labels.py
shared/gsm8k_thinking_labels.jsonl
  ├── label: "helped" | "not_helped"
  ├── difficulty: "easy" | "medium" | "hard"
  └── (graded: "hurt" for analysis only)
  ↓ run_experiment.py
output/gsm8k_probe/
  ├── seed_42/ (checkpoint, metrics)
  ├── seed_1/, seed_2/, seed_3/, seed_4/
  └── aggregate_metrics.json (mean ± CI)
  ↓ eval_transfer.py
output/gsm8k_to_lsat_transfer.json (AUROC + drop analysis)
```

### Probe Design
- **Input:** prefill hidden state (last prompt token), shape (num_layers, hidden_dim) = (32, 4096) for Qwen3-8B
- **Output:** binary classification ("helped" vs "not_helped")
- **Candidate architectures:**
  - MLP (simple baseline): (hidden_dim) → [256, 64] → (1)
  - Contrastive (optional): embed to (128,) then classify
- **Training:** Adam, early stopping on validation loss, 5-fold CV with 5 seeds each

### Label Schema
```json
{
  "idx": 0,
  "prompt_hash": "abc123...",
  "label": "helped",          // "helped" | "not_helped"
  "correct_off": true,        // thinking-off correctness
  "correct_on": false,        // thinking-on correctness
  "difficulty": "hard",       // stratification variable
  "graded_label": "helped"    // optional: "helped" | "hurt" | "no_change"
}
```

**Label construction (binary):**
- `"helped"` iff correct_off == False AND correct_on == True
- `"not_helped"` iff (both correct, both wrong, or right→wrong flip)

**Why difficulty matters:** Prevent probe from learning "hard queries benefit from thinking" instead of query-specific signals. Always evaluate AUROC stratified by difficulty.

### Experiment Design (Anti-Confound)
1. **Same-dataset train/val/test:** 60/20/20 split, 5 independent seeds → report AUROC ± 95% CI
2. **Cross-task transfer:** Train on GSM8K, zero-shot eval on LSAT (no retraining). <5pp drop = good.
3. **Minimal-pair template test:** Render same query under 2 templates, check if probe predictions drift aligns with correctness label drift.
4. **Baselines:** "Always think", "Never think", oracle accuracy to contextualize probe performance.

### Task Modules
Local to this repo, one contract (see `tasks/__init__.py`):

- `load_<task>(split) -> list[dict]` with keys `question`, `answer`, `key`, `difficulty`
- `format_prompt(question) -> str` — raw prompt, before any chat template
- `is_correct(generation, answer) -> bool`

Shipped: `tasks/gsm8k.py` (`openai/gsm8k`, primary) and `tasks/lsat.py` (`hails/agieval-lsat-ar`, transfer-only). `_TASK_REGISTRY` in the capture script must only list tasks with a matching module here — adding a task means writing the module, not pointing elsewhere.

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
- **`sync-jupyter` first, every time.** `configs/nodes.json` goes stale as allocations come and go; `python scripts/gpu_dispatch.py sync-jupyter` rebuilds it from live `squeue`. Dispatch reaches only nodes with a live Jupyter allocation registered there.
- **Name `.venv/bin/python` in the dispatched command.** `gpu_dispatch.py run` passes the command through verbatim, so a bare `python` silently picks up the node default and you get `ModuleNotFoundError` — or worse, a different transformers version.
- **Commit before dispatching.** A run whose code is not in a commit is not reproducible.
- **A timed-out dispatch may have launched anyway.** Before re-dispatching anything, run `gpu_dispatch.py jobs --all` and wait ≥ 2 minutes. Duplicate captures silently double-append to `meta.jsonl`.
- **Job manifest:** `shared/gpu_jobs.json` (relative to `project_root`). Job logs: `shared/logs/<job_id>.log`.
- **Fan-out work goes through the cell queue, not many `gpu_dispatch.py run` calls.** See **Cell + Worker Dispatch** below. A single `run` is right for a one-off; a sweep is a manifest plus N workers.
- **Don't guess whether data exists — check.** `wc -l <capture-dir>/meta.jsonl` and the NPZ shapes tell you what a capture actually produced; a job that appeared to finish may have OOM'd mid-run.

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
# 1. write a manifest (see configs/dispatch/example_probe_sweep.json), then preview
python scripts/dispatch/queue.py expand configs/dispatch/my_sweep.json --dry-run

# 2. queue it (idempotent — re-expanding never re-runs finished cells)
python scripts/dispatch/queue.py expand configs/dispatch/my_sweep.json

# 3. dispatch N workers onto N nodes (job submission — needs approval)
python scripts/gpu_dispatch.py run --desc "my_sweep worker" \
    .venv/bin/python scripts/dispatch/worker.py --root shared/dispatch/my_sweep

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
  "script": "scripts/run_experiment.py",
  "constants": {"out": "output/{name}/{method}_seed{seed}"},
  "grid": {"method": ["mlp", "contrastive"], "seed": [42, 1, 2, 3, 4]},
  "args": ["--method", "{method}", "--seeds", "{seed}", "--out-dir", "{out}"],
  "output_check": ["{out}/metrics.json"],
  "timeout_s": 7200, "max_attempts": 2
}
```

Semantics worth relying on:
- **`output_check` is the resume mechanism.** Present before the run → cell is skipped. Missing after exit 0 → cell is **failed**, not quietly completed. Always set it; a script that exits 0 having written nothing is the failure mode this catches.
- **Isolation.** Each cell is a subprocess in its own process group; a segfault or OOM kills the cell, not the worker.
- **Resumable and re-entrant.** Re-launching workers over a partly-drained queue is the normal recovery path. Cells from a crashed worker return to pending once its heartbeat goes stale (5 min); `queue.py gc` forces it.
- **`max_attempts > 1`** re-queues on failure so another node can try.
- **Shutdown is clean.** SIGTERM releases the in-flight cell back to pending immediately.

Write cells that are idempotent — a cell may run more than once.

Run the tests after touching anything under `scripts/dispatch/`:
`python3 tests/test_dispatch.py` (stdlib only, no GPU, ~15s).

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
ssh empire-ai 'cd ~/LLM_research/thinking-gating && python scripts/gpu_dispatch.py status'  # live GPU util + VRAM
```

A `gpu_jobs.json` entry maps to an allocation by `node_name` (`alphagpuNN-PPPP`),
whose port matches the `jupyter_empire_<port>` SLURM job name. An allocation with
no running manifest entry is an **idle Jupyter node** — say so rather than
implying work is in flight.

### Data Paths
- Relative paths: `shared/icr_capture/`, `shared/`, `output/` (relative to repo root, on local or cluster alike)
- Absolute paths: `/raid0/think-gating/` on Empire AI for large artifacts; `scp` **data** back after long runs (data is gitignored, never committed — the no-`scp` rule is about code going the other way)

## Writing New Scripts

### Template: Label Generation (`generate_labels.py`)
```python
"""
generate_labels.py — Convert paired thinking-off/on correctness into binary labels.

Input: meta.jsonl with correct_off, correct_on per sample
Output: labels.jsonl with label, difficulty, graded_label

Usage:
    python scripts/generate_labels.py \\
        --meta-file shared/icr_capture/gsm8k_thinking_qwen3/meta.jsonl \\
        --activations-dir shared/icr_capture/gsm8k_thinking_qwen3 \\
        --task gsm8k \\
        --out-file shared/gsm8k_thinking_labels.jsonl
"""
```

**Steps:**
1. Parse meta.jsonl line-by-line
2. For each row, extract correct_off, correct_on
3. Compute label: "helped" if correct_off=False AND correct_on=True, else "not_helped"
4. Read `difficulty` straight from the meta row (the capture script carries it through from the task module); fall back to re-loading the dataset only for captures written before that field existed
5. Optionally compute graded label (hurt, helped, no_change)
6. Write to JSONL
7. Report base rates (% "helped", % "hurt", etc.)

### Template: Training (`run_experiment.py`)
```python
"""
run_experiment.py — Train and evaluate thinking-mode probes.

Input: activations NPZ, labels JSONL
Output: trained probes, metrics JSON

Usage:
    python scripts/run_experiment.py \\
        --activations shared/icr_capture/gsm8k_thinking_qwen3/activations_thinking_off.npz \\
        --labels shared/gsm8k_thinking_labels.jsonl \\
        --method mlp \\
        --seeds 42 1 2 3 4 \\
        --out-dir output/gsm8k_probe
"""
```

**Steps:**
1. Load activations + labels
2. For each seed: split into 60/20/20 train/val/test
3. Instantiate probe (MLP or contrastive)
4. Train with early stopping on val loss
5. Evaluate on test (AUROC, stratified by difficulty)
6. Save checkpoint + metrics
7. Aggregate across seeds: mean AUROC ± 95% CI
8. Compare to baselines

### Template: Transfer Evaluation (`eval_transfer.py`)
```python
"""
eval_transfer.py — Zero-shot cross-task transfer of trained probes.

Usage:
    python scripts/eval_transfer.py \\
        --probe output/gsm8k_probe/seed_42/checkpoint.pt \\
        --test-activations shared/icr_capture/lsat_thinking_qwen3/activations_thinking_off.npz \\
        --test-labels shared/lsat_thinking_labels.jsonl \\
        --out-file output/gsm8k_to_lsat_transfer.json
"""
```

**Steps:**
1. Load trained probe (from GSM8K)
2. Load test activations + labels (LSAT)
3. Forward pass, compute AUROC
4. Compare to GSM8K test AUROC, report drop
5. Output results

## Common Pitfalls

### ❌ Don't
- Reach outside this repo for code, data loaders, or a Python environment — it is self-contained by design
- Submit or kill cluster jobs without approval, or run compute on the login node — see **Environment & Dispatch** above for the full rules; they are not optional
- Leak test labels during training (stratified eval must happen on held-out test set)
- Train a single probe on mixed GSM8K + LSAT data (defeats transfer test purpose)
- Ignore difficulty stratification (hard queries naturally benefit from thinking more)
- Use thinking-on activations for training the "thinking helps" predictor (logical circularity — train on thinking-off prefill only)

### ✅ Do
- Commit before dispatching, and dispatch `.venv/bin/python` — an uncommitted or wrong-interpreter run is a wasted GPU hour
- Always report confidence intervals (5-fold × 5 seeds = 25 runs)
- Stratify evaluation by difficulty even if not training on it
- Save probe checkpoints + hyperparams for reproducibility
- Log base rates (% "helped" in training data) to contextualize AUROCs
- Test on held-out test split first, then transfer to LSAT

## Testing & Validation

### Smoke Test
Env first, once per machine: `bash scripts/setup_env.sh && source .venv/bin/activate`.

**Step 1 needs a GPU node** — dispatch it, don't run it locally or on the login
node (and get approval for the dispatch first):

```bash
ssh empire-ai 'cd ~/LLM_research/thinking-gating && python scripts/gpu_dispatch.py sync-jupyter && \
  python scripts/gpu_dispatch.py run --desc "gsm8k smoke" \
    .venv/bin/python scripts/capture_inference_thinking.py \
        --task gsm8k --max-samples 100 \
        --model Qwen/Qwen3-8B --out-dir /raid0/think-gating/gsm8k_smoke --chat-template'

# then confirm it actually produced rows before moving on
ssh empire-ai 'wc -l /raid0/think-gating/gsm8k_smoke/meta.jsonl'
```

Steps 2–3 are CPU-only and run anywhere (locally, after `scp`-ing the capture
back):

```bash
python scripts/generate_labels.py \
    --meta-file /tmp/gsm8k_smoke/meta.jsonl \
    --out-file /tmp/gsm8k_smoke_labels.jsonl

python scripts/run_experiment.py \
    --activations /tmp/gsm8k_smoke/activations_thinking_off.npz \
    --labels /tmp/gsm8k_smoke_labels.jsonl \
    --seeds 42 \
    --max-epochs 5 \
    --out-dir /tmp/gsm8k_probe_smoke
```

### Validation Checks
- Probe AUROC >0.50 (better than random)
- Probe AUROC <oracle AUROC (ceiling check)
- Base rate of "helped" is ~20–25% (sanity check on label construction)
- Difficulty stratification shows AUC pattern (easy <medium <hard or vice versa)

## Paper / Results

Final results will live in `paper/` (structure TBD, but likely):
- `paper/results/` — figures, tables, metrics CSVs
- `paper/sections/` — draft sections on thinking-mode gating
- `paper/macros.tex` — if integrated into a LaTeX paper

For now, store raw metrics in `output/` and tabulate in analysis notebooks.

**Never edit a `.bib` file directly.** Agents hallucinate references. Add a
citation in the section text with enough context (title, authors, venue, year)
for a human to verify, and wait for explicit approval before any bibliography
insertion. Numbers quoted in the paper come from the saved metrics JSON/CSV in
`output/`, never retyped from a chat message or a log scroll.

---

**Last updated:** 2026-08-21 (Hong Yang)  
**Questions/blockers?** See `.agent-work/HANDOFF.md` for contact info and next steps.

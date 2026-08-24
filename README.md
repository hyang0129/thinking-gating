# thinking-gating

Distilled codebase for learning when thinking mode helps: use prefill activations (last-prompt-token hidden state) to predict whether extended reasoning will improve query outcome.

## Experiment Overview

**Goal:** Train a small probe on prefill hidden states to predict "will thinking mode improve this query?"

**Key Design:**
- **Tasks:** GSM8K (primary) + LSAT Logic Games (secondary for transfer test)
- **Labels:** Binary ("thinking_helped" = wrong→correct with thinking; "thinking_no_help" = everything else)
- **Probe:** MLP or contrastive on last-prompt-token hidden state (prefill)
- **Confounds:** Train/val/test on same dataset (60/20/20); cross-task transfer to LSAT; template ablation for format robustness

## Setup

The repo is self-contained: its own venv, its own task modules, its own dispatch
tooling. Nothing is imported from a sibling checkout.

```bash
bash scripts/setup_env.sh      # creates ./.venv and installs requirements.txt
source .venv/bin/activate
```

The same two commands work on Empire AI. See
[.agent-work/EMPIRE_AI_SETUP.md](.agent-work/EMPIRE_AI_SETUP.md) for cluster
dispatch.

Note: `transformers >= 4.51` is a hard floor — Qwen3 support and the
`enable_thinking` chat-template flag that the paired capture depends on both
landed there.

## Quick Start

### 1. Capture thinking-mode pairs
```bash
python scripts/capture_inference_thinking.py \
    --task gsm8k \
    --model Qwen/Qwen3-8B \
    --out-dir shared/icr_capture/gsm8k_thinking_qwen3 \
    --max-samples 500 \
    --chat-template
```

### 2. Generate labels
```bash
python scripts/generate_labels.py \
    --capture-dir shared/icr_capture/gsm8k_thinking_qwen3 \
    --out-labels shared/gsm8k_thinking_labels.jsonl
```

### 3. Train probe
```bash
python scripts/run_experiment.py \
    --labels shared/gsm8k_thinking_labels.jsonl \
    --method mlp \
    --seeds 42 1 2 3 4 \
    --out-dir output/gsm8k_probe
```

### 4. Evaluate transfer
```bash
python scripts/eval_transfer.py \
    --train-probe output/gsm8k_probe/seed_42 \
    --test-capture shared/icr_capture/lsat_thinking_qwen3 \
    --test-labels shared/lsat_thinking_labels.jsonl
```

## File Structure

```
thinking-gating/
├── scripts/
│   ├── setup_env.sh                    # Creates ./.venv, installs requirements
│   ├── capture_inference_thinking.py   # Thinking on/off toggle + prefill-only
│   ├── generate_labels.py              # Paired runs → thinking_helped labels
│   ├── run_experiment.py               # Probe training (MLP / contrastive)
│   ├── eval_transfer.py                # Cross-task transfer eval
│   ├── template_ablation.py            # Minimal-pair confound test
│   ├── gpu_dispatch.py                 # Multi-node GPU job dispatch (Empire AI)
│   ├── launch_jupyter.py               # Guarded Jupyter/SLURM launcher
│   └── dispatch/                       # Cell + worker queue (fan-out work)
│       ├── worker.py                   #   the one generic worker
│       ├── queue.py                    #   CLI: expand / status / logs / retry
│       ├── cells.py                    #   cell schema + manifest expansion
│       └── claim.py                    #   atomic claim queue primitives
├── tasks/
│   ├── gsm8k.py                        # Loader + prompt + grader (primary)
│   └── lsat.py                         # Loader + prompt + grader (transfer)
├── utils/
│   └── jupyter_exec.py                 # Jupyter kernel transport for dispatch
├── tests/
│   └── test_dispatch.py                # Stdlib-only, no GPU: python3 tests/test_dispatch.py
├── configs/
│   ├── datasets/
│   │   ├── gsm8k_thinking.json
│   │   └── lsat_thinking.json
│   ├── methods/
│   │   ├── mlp.json
│   │   └── contrastive.json
│   ├── dispatch/
│   │   └── example_probe_sweep.json    # Sweep manifest template
│   └── nodes.example.json              # Template for gitignored configs/nodes.json
├── data/
│   └── (icr_capture format: activations, labels, metadata)
└── paper/
    └── (analysis notebooks, figures, draft sections)
```

## Running Sweeps

Anything that fans out — multi-seed, multi-method, per-dataset batches — goes
through the cell queue in [scripts/dispatch/](scripts/dispatch/). Workers on any
number of nodes claim cells from a shared directory via atomic `rename(2)`.

**The worker is generic and never changes.** A cell describes its own work, so a
new sweep is a new manifest: a `python_script` to run, a `python_code` snippet,
a `call` to any importable function (its return value is captured), or a
`shell` command.

```bash
python scripts/dispatch/queue.py expand configs/dispatch/my_sweep.json --dry-run
python scripts/dispatch/queue.py expand configs/dispatch/my_sweep.json
python scripts/gpu_dispatch.py run .venv/bin/python scripts/dispatch/worker.py \
    --root shared/dispatch/my_sweep          # one per node
python scripts/dispatch/queue.py status --root shared/dispatch/my_sweep
```

Cells resume (`output_check` present → skipped), retry (`max_attempts`), time
out, and survive node death (stale claims are re-queued). See
[.agent-work/EMPIRE_AI_SETUP.md](.agent-work/EMPIRE_AI_SETUP.md) for the full
workflow.

## Tasks

Task modules are local to this repo and follow one contract (see
`tasks/__init__.py`): `load_<task>(split)`, `format_prompt(question)`,
`is_correct(generation, answer)`.

| Task | Source dataset | Split | Role |
|------|----------------|-------|------|
| `gsm8k` | `openai/gsm8k` (main) | test | Primary — probe train/val/test |
| `lsat` | `hails/agieval-lsat-ar` | test (230 rows) | Transfer — zero-shot eval only |

Neither dataset ships a difficulty field, so each module derives a bucket
(`easy`/`medium`/`hard`) used only for stratified evaluation: reasoning-step
count for GSM8K, constraint-sentence count for LSAT.

## References

- **Label Schema:** Binary "thinking_helped" with graded collection for future work
- **Experiment Design:** 60/20/20 split, 5-fold CV, cross-task transfer to LSAT for confound removal
- **Cluster dispatch:** [.agent-work/EMPIRE_AI_SETUP.md](.agent-work/EMPIRE_AI_SETUP.md)
- **Agent instructions:** [agent.md](agent.md) (symlinked as `claude.md`)

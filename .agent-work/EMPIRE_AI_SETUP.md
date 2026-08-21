# Empire AI Setup & Dispatch Conventions

This project reuses Empire AI infrastructure from HalluLens. Quick setup:

## Pre-Dispatch Setup (One-Time)

### 1. Symlink HalluLens Utilities
```bash
cd /Users/hong/Documents/code-projects/thinking-gating

# Task modules (don't copy; symlink to avoid duplication)
ln -s ../HalluLens/tasks tasks

# Utility modules
cp -r ../HalluLens/utils .

# Dispatch scripts
cp ../HalluLens/scripts/gpu_dispatch.py scripts/
cp ../HalluLens/scripts/launch_jupyter.py scripts/
```

### 2. Verify Empire AI Checkout
```bash
ssh empire-ai 'ls -la ~/LLM_research/thinking-gating/scripts/capture_inference_thinking.py'

# If missing, clone/pull:
ssh empire-ai 'cd ~/LLM_research && git clone <url> thinking-gating || \
  (cd thinking-gating && git fetch && git pull origin main)'
```

### 3. Install Dependencies on Empire AI
```bash
ssh empire-ai 'cd ~/LLM_research/thinking-gating && \
  pip install -r requirements.txt --user'
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

Then: `ssh empire-ai 'cmd'` reuses connections.

## Standard Dispatch Pattern

### Capture (Example)
```bash
# From macBook, dispatch to Empire AI GPU node
ssh empire-ai 'cd ~/LLM_research/thinking-gating && python scripts/gpu_dispatch.py run \
    --min-vram 20 \
    python scripts/capture_inference_thinking.py \
        --task gsm8k \
        --model Qwen/Qwen3-8B \
        --max-samples 500 \
        --out-dir /raid0/think-gating/gsm8k_thinking_qwen3 \
        --chat-template'

# Monitor:
ssh empire-ai 'python scripts/gpu_dispatch.py jobs --all'
```

### Training (Example)
```bash
ssh empire-ai 'cd ~/LLM_research/thinking-gating && python scripts/gpu_dispatch.py run \
    python scripts/run_experiment.py \
        --activations /raid0/think-gating/gsm8k_thinking_qwen3/activations_thinking_off.npz \
        --labels /raid0/think-gating/gsm8k_thinking_labels.jsonl \
        --method mlp \
        --seeds 42 1 2 3 4 \
        --out-dir /raid0/think-gating/probes/gsm8k_mlp'
```

## Data Paths

| Context | Path | Notes |
|---------|------|-------|
| Local (macBook) | `shared/icr_capture/` | Relative to repo root; can be symlink to `/Volumes/...` if large |
| Empire AI | `/raid0/think-gating/` | Fast SSD; sync back after capture with `scp -r` |
| Checkout | `~/LLM_research/thinking-gating/` | On login node; code only, not data |

## Jupyter (Interactive)

```bash
# Launch a single Jupyter node on Empire AI (guarded launcher, no approval needed)
ssh empire-ai 'cd ~/LLM_research/thinking-gating && python scripts/launch_jupyter.py 8882'

# Dry-run to preview sbatch line
ssh empire-ai 'cd ~/LLM_research/thinking-gating && python scripts/launch_jupyter.py 8882 --dry-run'

# Tunnel from macBook to Jupyter node
ssh -f -N -L 18882:alphagpuXX:8882 empire-ai

# Use jupyter_exec.py to run code against it
GPUNODE=localhost GPUNODEPORT=18882 python utils/jupyter_exec.py \
    "import torch; print(torch.cuda.get_device_name(0))"
```

## Rules

### ❌ Forbidden
- Train/infer on login node (alpha1) — use `gpu_dispatch.py`
- Direct edits or `scp` of code to Empire AI — use git
- Job submission without approval (except Jupyter via guarded launcher)
- Force-push to git

### ✅ Required
- All code commits before dispatch
- Explicit user approval for SLURM job submission (except guarded Jupyter)
- Multi-seed runs + CI reporting in results
- Sync data back with `scp` after long-running jobs

## Troubleshooting

### Job Stuck in PENDING
```bash
# Check why
ssh empire-ai 'squeue -j JOB_ID --format="%.10i %.9P %.8j %.8u %.2t %.10M %.6D %R"'

# Common: waiting for GPU. Check available nodes:
ssh empire-ai 'sinfo --Node --state=idle'
```

### OOM During Capture
- Reduce `--max-prompt-len` or `--max-response-len`
- Reduce batch size (default: auto-tuned, but can override)
- Check node VRAM: `ssh empire-ai 'gpustat'`

### Stale Checkout
```bash
ssh empire-ai 'cd ~/LLM_research/thinking-gating && git status'

# If dirty, stash and pull
ssh empire-ai 'cd ~/LLM_research/thinking-gating && git stash && git pull origin main'
```

## References

- **HalluLens CLAUDE.md:** `/Users/hong/Documents/code-projects/HalluLens/CLAUDE.md` (full docs)
- **GPU dispatch docs:** `HalluLens/docs/reference/GPU_DISPATCH_STATUS.md`
- **Jupyter guarded launcher:** See HalluLens CLAUDE.md section "Autonomous exception"

# Agent Instructions — Thinking-Mode Gating Experiment

This document guides agents (Claude Code, subagents) working on the thinking-gating repo. For human handoff context, see `.agent-work/HANDOFF.md`.

## Project Overview

**Goal:** Build and evaluate a probe that predicts whether thinking mode (extended reasoning) will improve query outcome, using only the prefill hidden state (last-prompt-token activation).

**Paper context:** Pathway 1 from HalluLens `docs/planning/PREFILL_APPLICATIONS.md`. Named gap in HRBench: no evaluated method uses target-model prefill state for thinking-mode routing.

## Current State (2026-08-20)

✅ **Phase 1 complete:**
- Repo bootstrapped at `/Users/hong/Documents/code-projects/thinking-gating/`
- `scripts/capture_inference_thinking.py` ported from HalluLens, adapted for:
  - Paired inference (thinking-off + thinking-on)
  - Prefill-only extraction (last-prompt-token hidden state)
  - Metadata storage for downstream label generation
- Configs locked: GSM8K (primary) + LSAT Logic Games (secondary transfer test)
- Requirements + .gitignore in place

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

### Reusable Code from HalluLens
- Task modules: `tasks/llmsknow/{gsm8k,lsat,mmlu,nq,popqa,sciq,searchqa}.py`
- Activation utilities: `activation_logging/generate_capture.py` (stitching, logprob extraction)
- Probe training skeleton: `activation_research/training.py` (can port ProgressiveCompressor if needed, but MLP is simpler)
- Dataset loading: reuse HalluLens loaders directly (import from there or copy)

## Environment & Dispatch

### Local / Interactive
- `python scripts/capture_inference_thinking.py --help` for full options
- Requires GPU for inference; CPU-only for label generation + training is possible but slow

### Empire AI Cluster
- **Login node:** `ssh empire-ai 'cmd'` (orchestration only)
- **GPU nodes:** Dispatch via `gpu_dispatch.py run` (from HalluLens, should work identically here)
- **Launch Jupyter:** `scripts/launch_jupyter.py <port>` (guarded launcher, see HalluLens CLAUDE.md)
- **Worker queue:** Can use cell-based worker queue from HalluLens dispatch system if running multiple experiments in parallel

### Data Paths
- Relative paths: `shared/icr_capture/`, `shared/`, `output/` (relative to repo root on local or `/home/vscode/.../thinking-gating/` on Empire AI)
- Absolute paths: `/mnt/large_ssd/` or `/raid0/` on Empire AI for large artifacts (sync via git after)

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
4. Fetch difficulty from original dataset (load via task module)
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
- Leak test labels during training (stratified eval must happen on held-out test set)
- Train a single probe on mixed GSM8K + LSAT data (defeats transfer test purpose)
- Ignore difficulty stratification (hard queries naturally benefit from thinking more)
- Use thinking-on activations for training the "thinking helps" predictor (logical circularity — train on thinking-off prefill only)

### ✅ Do
- Always report confidence intervals (5-fold × 5 seeds = 25 runs)
- Stratify evaluation by difficulty even if not training on it
- Save probe checkpoints + hyperparams for reproducibility
- Log base rates (% "helped" in training data) to contextualize AUROCs
- Test on held-out test split first, then transfer to LSAT

## Testing & Validation

### Smoke Test
```bash
# Minimal run: 100 GSM8K examples, single seed
python scripts/capture_inference_thinking.py \
    --task gsm8k --max-samples 100 \
    --model Qwen/Qwen3-8B --out-dir /tmp/gsm8k_smoke

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

---

**Last updated:** 2026-08-20 (Hong Yang)  
**Questions/blockers?** See `.agent-work/HANDOFF.md` for contact info and next steps.

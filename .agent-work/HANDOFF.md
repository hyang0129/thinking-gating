# Thinking-Mode Gating Experiment — Agent Handoff

**Date:** 2026-08-21  
**Status:** Bootstrap complete and self-contained; ready for Phase 2 (label generation & training)  
**Owner:** Hong Yang

---

## Context

Repo for Pathway 1 (thinking-mode gating). Goal: **train a small probe on prefill hidden states (last-prompt-token activation) to predict whether extended reasoning (thinking mode) will improve query outcome.**

**The repo is self-contained.** Task modules, dispatch tooling, and the Python environment all live here; nothing is symlinked or imported from a sibling checkout, on the laptop or on Empire AI. Setup is `bash scripts/setup_env.sh` (builds `./.venv`) in both places — see `.agent-work/EMPIRE_AI_SETUP.md`.

### Experiment Summary
- **Tasks:** GSM8K (primary, train/val/test split) + LSAT Logic Games (secondary, transfer test)
- **Labels:** Binary ("thinking_helped" = wrong→correct with thinking; "thinking_no_help" = everything else)
- **Probe:** MLP or contrastive on (N, num_layers, hidden_dim) prefill activations
- **Design:** 60/20/20 splits, 5-fold CV, multi-seed reporting, cross-task transfer for confound removal

### Phase 1 (Complete)
- ✅ Discovery: reasoning tasks survey (Agent A), label schema design (Agent B), experiment structure (Agent C)
- ✅ Bootstrap: repo initialized, `capture_inference_thinking.py` adapted, configs + requirements locked
- ✅ Self-containment pass (2026-08-21):
  - `tasks/gsm8k.py` (`openai/gsm8k`) and `tasks/lsat.py` (`hails/agieval-lsat-ar`, 230 test rows) written in-repo, with prompt formatting, answer extraction, graders, and heuristic difficulty buckets
  - `scripts/gpu_dispatch.py`, `scripts/launch_jupyter.py`, `utils/jupyter_exec.py` vendored in
  - `scripts/setup_env.sh` builds the repo's own `.venv`; `configs/nodes.example.json` templates the gitignored `configs/nodes.json`
  - `requirements.txt` re-pinned: the old `transformers==4.40` pin could not load Qwen3 at all, so `>= 4.51` is now a hard floor (`enable_thinking` lives there too); unused `pytorch-lightning`/`torchvision`/`json5` dropped, `accelerate`/`requests`/`websocket-client` added
  - Capture script's task registry trimmed to the two tasks that actually have modules here; `difficulty` now flows into `meta.jsonl`
- ✅ Cell + worker dispatch (2026-08-21): `scripts/dispatch/` — a generic worker plus a manifest-driven cell queue, so fan-out work (multi-seed, multi-method, per-dataset) never needs a new worker script. Four cell kinds (`python_script`, `python_code`, `call` any importable function, `shell`), resume via `output_check`, retries, timeouts, stale-claim recovery. 23 tests in `tests/test_dispatch.py`, all passing (stdlib only, no GPU).
- **Output:** `/Users/hong/Documents/code-projects/thinking-gating/` with capture script ready to run

---

## Phase 2: Label Generation & Training (Next Steps)

### 2a. Label Generation (`generate_labels.py`)
**Input:** `shared/icr_capture/{task}_thinking_qwen3/meta.jsonl` + activations NPZs  
**Output:** `shared/{task}_thinking_labels.jsonl` with schema:
```json
{
  "idx": 0,
  "prompt_hash": "...",
  "label": "helped",          // "helped" | "not_helped" | graded: "hurt"
  "correct_off": true,
  "correct_on": false,
  "difficulty": "hard",       // carried through from the task module via meta.jsonl
  "confidence": 0.8
}
```

**Tasks:**
1. Parse meta.jsonl and extract correctness pairs (correct_off, correct_on)
2. Generate binary labels: "helped" if wrong→correct, else "not_helped"
3. Optionally log graded labels (hurt) for analysis
4. Stratify by difficulty (read the `difficulty` field the capture script writes into meta.jsonl)
5. Write to JSONL with per-sample metadata
6. Report label distribution (base rates: expect ~20-25% "helped" on GSM8K)

**Key decisions locked:**
- Binary for pilot (graded collection is optional future work)
- Difficulty stratification is critical (prevents "hardness detector" probe)
- Same-dataset labels (no cross-task contamination in label generation)

---

### 2b. Probe Training & Evaluation (`run_experiment.py`)
**Input:** prefill activations (NPZ) + labels (JSONL)  
**Output:** trained probes + evaluation results

**Tasks:**
1. Load prefill activations from `activations_thinking_off.npz` (N, num_layers, hidden_dim)
2. Split into 60/20/20 train/val/test (5 independent seeds)
3. Train probe on each split:
   - MLP baseline: (hidden_dim) → [256, 64] → (1)
   - Optional contrastive: embed, then binary classification
   - Adam + early stopping on validation loss
4. Evaluate on test set (per-seed AUROC, mean ± 95% CI across 5 seeds)
5. Stratify eval by difficulty (catch if AUC on easy ≈ 0.5 → overfitting to difficulty)
6. Compare vs. baselines:
   - "Always think" accuracy
   - "Never think" accuracy
   - Oracle (ground-truth)
7. Save checkpoint per seed + aggregate metrics

**Output format:**
```json
{
  "method": "mlp",
  "task": "gsm8k",
  "seeds": [42, 1, 2, 3, 4],
  "train_auroc": [0.78, 0.80, 0.75, 0.81, 0.77],
  "val_auroc": [0.76, 0.79, 0.74, 0.80, 0.76],
  "test_auroc": [0.75, 0.78, 0.72, 0.79, 0.75],
  "test_auroc_mean": 0.758,
  "test_auroc_ci": [0.725, 0.791],
  "auc_by_difficulty": {
    "easy": 0.52,
    "medium": 0.71,
    "hard": 0.85
  }
}
```

---

### 2c. Transfer Evaluation (`eval_transfer.py`)
**Input:** trained probe from GSM8K + LSAT prefill activations + labels  
**Output:** cross-task transfer AUROC

**Tasks:**
1. Load trained probe from GSM8K (best-seed checkpoint)
2. Load LSAT prefill activations + labels (no retraining)
3. Evaluate probe zero-shot on LSAT test set
4. Report AUROC + compare to GSM8K test AUROC
5. Interpret:
   - <5pp drop: strong transfer ✓
   - 5–15pp drop: moderate (label distribution shift)
   - >15pp drop: transfer failure (format overfitting) ✗

**Expected:** ~0.75 GSM8K → ~0.70 LSAT (−5pp) if signal is reasoning-general

---

### 2d. Template Ablation (Optional, `template_ablation.py`)
**Input:** ~500 queries, render under 2 different prompt templates  
**Output:** probe drift vs. label drift alignment

**Tasks:**
1. Load subset of GSM8K (500 examples)
2. Render each query under 2 templates: (a) raw, (b) reformatted
3. Run thinking-off + thinking-on inference for both templates
4. Extract prefill activations for both
5. Compute probe predictions on both templates
6. Measure: drift in predictions vs. drift in correctness labels
7. If aligned → format-safe; if probe drift >> label drift → risk

---

## Implementation Notes

### Available in-repo
- Task modules with loader + grader + difficulty: `tasks/gsm8k.py`, `tasks/lsat.py`
- Capture with paired thinking-off/on inference and prefill extraction: `scripts/capture_inference_thinking.py`
- Cluster dispatch: `scripts/gpu_dispatch.py`, `scripts/launch_jupyter.py`, `utils/jupyter_exec.py`
- Fan-out dispatch: `scripts/dispatch/` (cell queue + generic worker) — write a manifest, not a worker
- Probe training is written fresh here (plain torch MLP + scikit-learn metrics) — no external skeleton to port

### New code (Phase 2) — written 2026-08-24
- `scripts/generate_labels.py` ✅
- `scripts/run_experiment.py` ✅ (torch MLP + logreg baseline, 5 seeds, stratified AUROC, routed accuracy)
- `scripts/eval_transfer.py` ✅
- `utils/capture_io.py` ✅ (shard-aware capture loading)
- `tests/test_pipeline.py` ✅ (synthetic end-to-end, incl. signal-free negative control)
- `scripts/template_ablation.py` — still optional, not written

### Dispatch & Execution
- Environment: `bash scripts/setup_env.sh` once per machine (~5 min, mostly the torch wheel)
- Label generation: CPU-only, quick (~5 min for 1k examples)
- Training: Single GPU (A100), ~1 hour for 5 seeds on 500-1k examples — or fan the seeds out as cells across nodes
- Transfer eval: Single GPU, ~10 min
- **Total:** ~2-3 hours on Empire AI (1 GPU node via `gpu_dispatch.py`, dispatching `.venv/bin/python`)

---

## Checkpoints & Success Criteria

| Phase | Done | Success Criterion | Next Action |
|-------|------|-------------------|-------------|
| **Phase 1** | ✅ | Repo bootstrap + capture script ready | Run capture on GSM8K pilot (user decision) |
| **Phase 2a** | ⏳ | Labels generated, base rate ~20% | Spot-check label distribution |
| **Phase 2b** | ⏳ | GSM8K probe train/val/test AUROC >0.70 | Proceed to transfer test |
| **Phase 2c** | ⏳ | LSAT transfer AUROC >0.65 (< 10pp drop) | Attempt template ablation |
| **Phase 2d** | ⏳ | Probe drift ≈ label drift on templates | Write up findings, start paper |

---

## Files & Paths

**Repo root:** `/Users/hong/Documents/code-projects/thinking-gating/`

**Key files:**
- `scripts/setup_env.sh` — builds `./.venv` (run first, on every machine)
- `scripts/capture_inference_thinking.py` — paired capture (ready)
- `tasks/gsm8k.py`, `tasks/lsat.py` — in-repo loaders/graders (ready)
- `scripts/gpu_dispatch.py`, `scripts/launch_jupyter.py` — cluster dispatch (ready)
- `configs/nodes.example.json` — template for the gitignored `configs/nodes.json`
- `scripts/generate_labels.py` — needs writing
- `scripts/run_experiment.py` — needs writing
- `scripts/eval_transfer.py` — needs writing
- `configs/datasets/{gsm8k,lsat}_thinking.json` — locked
- `configs/methods/{mlp,contrastive}.json` — locked
- `.agent-work/HANDOFF.md` — this file

**Data paths (relative to repo root or absolute on Empire AI):**
- `shared/icr_capture/gsm8k_thinking_qwen3/` — GSM8K thinking-mode capture (not yet generated)
- `shared/icr_capture/lsat_thinking_qwen3/` — LSAT thinking-mode capture (not yet generated)
- `shared/gsm8k_thinking_labels.jsonl` — labels (Phase 2a output)
- `shared/lsat_thinking_labels.jsonl` — labels (Phase 2a output)
- `output/gsm8k_probe/` — trained probes (Phase 2b output)

---

## Questions for Next Agent

1. **Capture priority:** Should we run the full GSM8K capture first, or do a smoke test (100 examples) to verify the script works?
2. **Label distribution:** Once labels are generated, what's the acceptable "thinking_helped" base rate? (Agent B estimated 20–25%, but pilot data might differ.)
3. **Probe complexity:** Start with simple MLP, or also train contrastive probe in parallel?
4. **LSAT sample size:** `hails/agieval-lsat-ar` ships a single 230-row test split. Is 230 enough for the transfer eval, or should a second reasoning task be added alongside it?

---

## Contacts & Context

- **User:** Hong Yang (hooong.yang@gmail.com)
- **Paper:** Paper draft will live in `thinking-gating/paper/`
- **Dispatch:** Empire AI cluster via this repo's `scripts/gpu_dispatch.py` — full setup in `.agent-work/EMPIRE_AI_SETUP.md`

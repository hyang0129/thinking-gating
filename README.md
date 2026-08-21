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

```bash
conda create -n thinking python=3.12
conda activate thinking
pip install -r requirements.txt
```

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
│   ├── capture_inference_thinking.py   # Thinking on/off toggle + prefill-only
│   ├── generate_labels.py              # Paired runs → thinking_helped labels
│   ├── run_experiment.py               # Probe training (MLP / contrastive)
│   ├── eval_transfer.py                # Cross-task transfer eval
│   └── template_ablation.py            # Minimal-pair confound test
├── configs/
│   ├── datasets/
│   │   ├── gsm8k_thinking.json
│   │   └── lsat_thinking.json
│   └── methods/
│       ├── mlp.json
│       └── contrastive.json
├── data/
│   └── (icr_capture format: activations, labels, metadata)
└── paper/
    └── (analysis notebooks, figures, draft sections)
```

## References

- **Pathway 1 Design:** `docs/planning/PREFILL_APPLICATIONS.md` (HalluLens repo)
- **Label Schema:** Binary "thinking_helped" with graded collection for future work
- **Experiment Design:** 60/20/20 split, 5-fold CV, cross-task transfer to LSAT for confound removal

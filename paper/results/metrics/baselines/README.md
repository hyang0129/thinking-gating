# What does the prefill state buy over reading the question?

The probe costs a forward pass through an 8B model. That is only justified if
it beats predictors that read the raw question text and nothing else. These
baselines run on identical splits, seeds, target and metric as
run_experiment.py (they import its split and AUROC code), so the numbers are
directly comparable.

## Results (Qwen3-8B, 5 seeds, bootstrap CIs over test examples)

### needs_thinking — prefill wins

| setting | prefill | best text baseline | gap |
|---|---|---|---|
| MATH-500 | 0.879 [0.802, 0.945] | tfidf_char 0.767 [0.664, 0.859] | +0.112 |
| pooled 3-task | 0.902 [0.876, 0.926] | tfidf_word 0.821 [0.785, 0.855] | +0.081 |

### rescued — mixed, and the single-task result does not survive

| setting | prefill | best text baseline | gap |
|---|---|---|---|
| MATH-500 | 0.703 [0.567, 0.824] | tfidf_word 0.680 [0.544, 0.813] | **+0.023** |
| pooled 3-task | 0.692 [0.618, 0.760] | tfidf_char 0.595 [0.517, 0.672] | +0.097 |

## Read this before quoting any of it

**Single-task MATH-500 `rescued` is a tie.** 0.703 vs 0.680 with almost
completely overlapping intervals. At n=369 an 8B forward pass is not
distinguishable from word n-grams. Any claim resting on that number alone is
not supported.

**Pooled `needs_thinking` is confounded by task identity.** Pooled base rates
are gsm8k 0.144, math500 0.738, mmlu_pro 0.615, so recognising which task a
question came from predicts the label. TF-IDF's 0.821 there is vocabulary
matching, not difficulty estimation -- the same failure mode as the BBH
subtask detector. Do not present pooled needs_thinking as evidence of
query-level signal.

**Pooled `rescued` is the cleanest comparison in the project.** Base rates are
0.721 / 0.713 / 0.569, close enough that task identity buys little, which is
why TF-IDF manages only 0.595. The prefill probe beats it by 0.097 with
barely-overlapping intervals, and TF-IDF *degrades* when pooling (0.680 ->
0.595) while the probe improves. That contrast -- lexical features are
task-specific, the prefill representation is not -- is the strongest claim
the evidence currently supports.

## Not yet run

- The model's own thinking-off confidence / answer entropy. This is the
  strongest remaining competitor and it is cheap; it should exist before any
  writeup.
- A small fine-tuned text encoder (MiniLM/DeBERTa) on the same labels.

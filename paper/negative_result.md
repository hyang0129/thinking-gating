# Prefill-state probes do not predict when thinking helps: a negative result, and the confound that made it look positive

Hong Yang — 2026-09-12. Every number traces to a file under `paper/results/`;
the path is given in the table captions.

## Abstract

We asked whether a small probe on a language model's prefill hidden state
(the last-prompt-token activation, before any generation) can predict
whether extended reasoning ("thinking mode") will change the answer, so that
thinking compute can be routed per query. Across four benchmarks on Qwen3-8B
and two on Nemotron-Nano-8B, with truncation-clean labels, it cannot do so
better than baselines that never touch the model. Correctness-without-
thinking (`needs_thinking`) is predicted at AUROC 0.66–0.70 on
gsm8k / math500 / mmlu_pro, but question length or TF-IDF lands inside every
interval and the model's own thinking-off confidence beats the probe on
math500 (0.814 vs 0.699). The objective that would have been novel —
whether thinking *rescues* a query the model otherwise fails — is at chance
(0.49–0.60, n = 89–376) and transfers at chance between tasks. The one
benchmark where the probe looked strong, BBH, is a subtask detector:
within-subtask AUROC is 0.49 / 0.51.

An earlier version of this experiment reported AUROC 0.88 on math500 and
replicated across three model families. That result was an artifact of a
320-token cap on the non-thinking pass: the probe was predicting answer
length, and the cap was shared by every model. We describe the confound,
the test that exposed it, and the controls that should have caught it
earlier, because those are the transferable findings.

## 1. Question and setup

**Claim under test.** The prefill representation of a query encodes the
marginal value of reasoning on it — not merely its difficulty — well enough
to route thinking compute before generating anything.

**Data.** For each query we ran the model twice with greedy decoding,
thinking off and thinking on, graded both, and saved the prefill hidden
state at every layer from a dedicated forward pass on the thinking-off
prompt. Labels:

| target | definition | what above-chance AUROC would mean |
|---|---|---|
| `needs_thinking` | `~correct_off` | prefill predicts the model fails without thinking (correctness prediction; well studied) |
| `helped` | `~correct_off & correct_on` | thinking flipped the answer (mostly driven by the first term) |
| `rescued` | `correct_on` on rows with `correct_off == False` | prefill predicts the *marginal value* of thinking, with difficulty held fixed by construction. **The load-bearing objective.** |

**Probe.** Logistic regression on the layer-18 activation (the a-priori
middle layer of 36; never swept). 60/20/20 stratified splits, 5 seeds.
Intervals are percentile bootstraps over test examples, averaged across
seeds; the seed-spread interval is 1.6–8.8× too narrow and is never quoted.

**Baselines on identical splits.** Question length in characters and
words; TF-IDF (word 1–2-grams, char 3–5-grams) → logistic regression; the
model's thinking-off mean/min token log-prob, mean predictive entropy,
answer length in tokens, and a logistic regression over those four. The
confidence baselines read the thinking-off *generation*, so they are
post-hoc routers where the probe is pre-hoc. That is the right comparison:
a pre-hoc probe that loses to a post-hoc scalar must justify itself on
latency alone.

**Models and tasks.** Qwen3-8B on gsm8k (1319), math500 (500), mmlu_pro
(1000, a fixed subset), bbh (540, 27 subtasks × 20). Nemotron-Nano-8B on
bbh (540) and lsat (230); its other three tasks are being re-captured and
are not in this writeup. Captures: `configs/dispatch/capture_qwen3v3.json`,
`capture_nemotronv3.json`, run 2026-09-04/05.

## 2. The confound, first

The first full analysis (2026-08-26) used `--max-response-len 320` for the
thinking-off pass. Qwen3's non-thinking mode writes chain-of-thought
anyway, so the off-pass was cut off mid-answer on 75% of math500, 49% of
mmlu_pro and 12% of gsm8k rows, and each cut-off row was graded wrong.
`correct_off` became almost a function of truncation (math500: 0.037
correct when capped, 0.944 when not), so `needs_thinking` became "did the
answer exceed 320 tokens", and `rescued` conditioned on that same subset
(98% of math500's "wrong without thinking" rows were truncated rows).

The decisive test was to train the identical probe to predict
`truncated_off` instead of the label
(`paper/results/metrics/truncation/README.md`):

| task | probe → truncation | probe → needs_thinking | off-truncation rate |
|---|---|---|---|
| math500 | **0.922** | 0.879 | 75.2% |
| mmlu_pro | **0.872** | 0.782 | 49.3% |
| gsm8k | **0.811** | 0.702 | 11.6% |

The probe predicted truncation better than it predicted the label on every
task, and the reported AUROCs ranked the tasks exactly by truncation rate.
The three-family replication (Qwen3, Nemotron, Granite at 70–85%
off-truncation each) reproduced the artifact rather than confirming the
finding: a confound in the design is invariant to the model.

The capture script warned on thinking-*on* truncation and was silent on
thinking-*off*, the one that defines both objectives. It now quarantines
any shard above 20% off-truncation and exits non-zero.

## 3. Corrected results

Thinking-off budgets were raised to 1024 (2048 for math500); thinking-on
budgets were left at v2 values for comparability. Residual off-truncation:
0.2 / 8.4 / 8.4 / 3.1% on gsm8k / math500 / mmlu_pro / bbh. Thinking-off
accuracy on math500 went from 0.25 to 0.76.

### 3.1 `needs_thinking`: real, small, and matched by baselines

Qwen3-8B. Probe vs best baseline of each kind, bootstrap 95% CIs.
(`paper/results/metrics/qwen3v3/baseline_comparison.txt`)

| task | n | base rate | prefill probe | best text baseline | best confidence baseline |
|---|---|---|---|---|---|
| gsm8k | 1319 | 0.067 | 0.687 [0.561, 0.800] | tfidf_word 0.634 [0.516, 0.750] | n_tokens_off 0.690 [0.543, 0.823] |
| math500 | 500 | 0.236 | 0.699 [0.561, 0.818] | length_chars 0.715 [0.590, 0.827] | confidence_lr **0.814** [0.700, 0.914] |
| mmlu_pro | 1000 | 0.376 | 0.659 [0.582, 0.736] | tfidf_word 0.617 [0.535, 0.697] | confidence_lr 0.646 [0.564, 0.724] |
| bbh | 540 | 0.324 | 0.820 [0.732, 0.898] | tfidf_char **0.846** [0.761, 0.920] | n_tokens_off 0.560 [0.445, 0.675] |

Against the confounded run these fell from 0.702 / 0.879 / 0.782 / 0.838.
On every task a predictor that never sees the model sits inside the
probe's interval. On math500 the thinking-off confidence regression is
clearly better. On bbh, character n-grams beat the probe — see §3.3 for
why.

### 3.2 `rescued`: at chance

Qwen3-8B (`paper/results/metrics/qwen3v3/{gsm8k,math500,mmlu_pro}__rescued.json`).

| task | n (rows wrong without thinking) | positive rate | prefill probe | best text | best confidence |
|---|---|---|---|---|---|
| gsm8k | 89 | 0.539 | 0.597 [0.308, 0.861] | length_chars 0.583 | min_logprob 0.578 |
| math500 | 118 | 0.331 | 0.559 [0.318, 0.787] | tfidf_char 0.642 | n_tokens_off 0.524 |
| mmlu_pro | 376 | 0.348 | 0.489 [0.349, 0.634] | length_chars 0.506 | n_tokens_off 0.607 |

The mmlu_pro interval, at the largest n, excludes everything above 0.63.
The earlier pooled `rescued` of 0.692 [0.618, 0.760] — the number the
project's strongest claim rested on — does not survive.

Zero-shot transfer between tasks for `rescued`
(`paper/results/metrics/qwen3v3/transfer__*__rescued.json`): every one of
the twelve ordered pairs lands in 0.42–0.59. The pairs the results table
labels "strong transfer" are a source at 0.56–0.60 landing on a target at
0.58: a small drop between two near-chance numbers.

### 3.3 BBH is a subtask detector

BBH looks like the probe's best benchmark (0.820 / 0.738). Seven of its 27
subtasks are settled by subtask identity alone — thinking-off is all-right
or all-wrong within them — and per-subtask AUROCs on ~4 test rows each are
uninformative. The statistic that settles it is one AUROC over all
(positive, negative) test pairs drawn from the *same* subtask, where
category identity cannot contribute
(`paper/results/metrics/qwen3v3/within_group__bbh__*.json`):

| target | overall AUROC | within-subtask AUROC |
|---|---|---|
| needs_thinking | 0.820 [0.732, 0.898] | 0.494 [0.230, 0.765] |
| rescued | 0.738 [0.557, 0.890] | 0.506 [0.009, 1.000] |

Char TF-IDF beating the probe on bbh is the same effect from the other
side: vocabulary identifies the subtask. This is the second time BBH has
produced this artifact in the project; the first was caught by per-subtask
stratification (`scripts/stratify_check.py`) and transfer at chance.

### 3.4 Second model family

Nemotron-Nano-8B, bbh and lsat only
(`paper/results/metrics/nemotronv3/baseline_comparison.txt`).

| task | target | n | probe | best text | best confidence |
|---|---|---|---|---|---|
| bbh | needs_thinking | 540 | 0.755 [0.650, 0.848] | tfidf_word 0.794 | confidence_lr 0.666 |
| bbh | rescued | 361 | 0.756 [0.547, 0.932] | tfidf_char 0.802 | confidence_lr 0.538 |
| lsat | needs_thinking | 230 | 0.562 [0.353, 0.761] | length_words 0.574 | n_tokens_off 0.608 |
| lsat | rescued | 174 | 0.689 [0.497, 0.861] | tfidf_word 0.640 | mean_entropy 0.619 |

BBH again loses to TF-IDF. Nemotron's LSAT thinking-off accuracy is 0.243,
below the 0.25 guess floor for five-way multiple choice, at 5.7% truncation
— so the v2 "degenerate labels" verdict on LSAT was correct and not a
truncation artifact. LSAT `rescued` at 0.689 is the only above-chance
number on a non-category task in the whole corrected set. It beats its
baselines by less than any of the intervals, on 174 rows of which a fifth
have a truncated thinking-on response. We record it as a lead and do not
claim it.

## 4. What bounds these numbers

- **Thinking-on truncation** is 16–17% on math500 and mmlu_pro and 20% on
  Nemotron LSAT. Those rows are graded wrong-with-thinking for running out
  of budget, which pushes `correct_on` down and contaminates `rescued`'s
  positive class in a direction that could hide signal as easily as create
  it.
- **Thinking-off truncation** of 8.4% on math500 and mmlu_pro is under the
  gate but not zero. `n_tokens_off` still predicts `needs_thinking` at 0.69
  on gsm8k, so answer length carries signal in both the label and the
  baselines.
- **`rescued` has 89–376 rows.** The intervals say so; a 0.6 at n = 89 is
  not distinguishable from 0.5 or from 0.75.
- **Last-token only.** Only the final prompt token's activation was ever
  saved; mean-pooling over prompt tokens is usually a material gain in
  probing work and was not tested.

## 5. What we conclude

1. **Negative, at this scale and design.** On truncation-clean labels the
   prefill probe on Qwen3-8B does not beat question-length, TF-IDF or the
   model's own thinking-off confidence on any objective, and shows no
   query-level `rescued` signal on gsm8k, math500 or mmlu_pro. We cannot
   support the claim that the prefill state encodes the marginal value of
   reasoning.
2. **The positive result was a length confound.** A response cap on the
   non-thinking pass turned "wrong without thinking" into "long without
   thinking", the probe learned length, and a cross-family replication
   confirmed only that every family shared the cap.
3. **The controls that matter**, in the order they would have saved time:
   (a) check the truncation rate of the pass that *defines the label*, not
   just the expensive one; (b) train the probe on the nuisance variable and
   compare; (c) run text and confidence baselines on the same splits before
   believing any AUROC; (d) for multi-category benchmarks, compute the
   pooled within-category AUROC — per-category AUROCs are underpowered and
   stratifying by difficulty does not catch category identity; (e) quote
   bootstrap intervals over examples, not seed spread.

## 6. What would change the conclusion

One capture, changing three things together, is the only experiment left
that could: thinking-on budgets large enough to stop truncating a sixth of
the rows (≥16k on math500 / mmlu_pro / lsat), mean-pooled prompt-token
activations saved beside the last token, and ~3k problems from the full
MATH corpus to take `rescued` from 118 rows to roughly 2000. A different
probe architecture on the present data is not worth running: every tuning
gain in this project came from more rows, and the one layer sweep that was
tried lowered test AUROC while raising validation AUROC.

The Nemotron redo captures (gsm8k, math500, mmlu_pro at larger off-budgets)
and the Qwen3 LSAT capture at a 4096 off-budget are queued on the cluster.
When they land they complete §3.4 and test whether the LSAT `rescued` lead
replicates across families. They do not change the conclusion above unless
Qwen3 LSAT `rescued` clears its text and confidence baselines with
non-overlapping intervals.

## Provenance

| section | files |
|---|---|
| §2 | `paper/results/metrics/truncation/README.md`; retracted run in `paper/results/retracted-2026-08-26/` |
| §3.1–3.2 | `paper/results/metrics/qwen3v3/{task}__{target}.json`, `baselines/{text,confidence}__{task}__{target}.json`, `baseline_comparison.{txt,csv}` |
| §3.2 transfer | `paper/results/metrics/qwen3v3/transfer__{src}_to_{tgt}__rescued.json` |
| §3.3 | `paper/results/metrics/qwen3v3/within_group__bbh__{needs_thinking,rescued}.json` |
| §3.4 | `paper/results/metrics/nemotronv3/` |
| §6 tuning claims | `paper/results/metrics/tuning/README.md` (confounded data, but the estimator findings are about the estimator) |

Produced by `scripts/run_full_analysis.sh` (`CAPTURE_SLUG=qwen3v3`,
`CAPTURE_SLUG=nemotronv3 TASKS="bbh lsat"`), `scripts/within_group_auroc.py`
and `scripts/compare_baselines.py` at commit 7942a36. Captures at
`configs/dispatch/capture_{qwen3,nemotron}v3.json`, Empire AI, 2026-09-04/05.

# Prefill-state probes do not predict when thinking helps: a negative result, and the confound that made it look positive

Hong Yang — 2026-09-12. Every number traces to a file under `paper/results/`;
the path is given in the table captions.

## Errata (2026-10-04; Qwen3 corrected results 2026-10-06)

A cleanup audit on 2026-10-04 found bugs in the capture, the graders and the
analysis code. They are fixed on branch `cleanup/system-one-carryover`
(commits 8404347, 7e224b8, f2eabef, 9e24091, d0fa2de, 52fa811, 8de124f; the
confidence re-score is 951d2b3). The text below the errata is the 2026-09-12
writeup, unchanged; read its numbers through this section.

**Where the corrected numbers come from.**

- "Published" numbers are in `paper/results/metrics/{qwen3v3,nemotronv3}/`.
- **Qwen3-8B corrected numbers are promoted** in
  `paper/results/metrics/qwen3v3_corrected/` (abbreviated `corrected/`
  below; see its README). They use regraded labels, re-scored (v2)
  confidence and the fixed analysis code, at commit b368bd7.
- Everything else, including all Nemotron numbers, is a **working number**
  from the 2026-10-04 audit, kept with its scripts in
  `paper/results/errata-2026-10-04/` (abbreviated `errata/`).

**What has been re-run.**

- **Fixed code on the published labels.** The analysis was first re-run on
  the **published (pre-regrade) labels** (`errata/rerun_metrics/`, diff in
  `errata/rerun_metrics/diff_published.out`). At the published layer, every
  probe AUROC, bootstrap interval, baseline test AUROC and within-group
  AUROC reproduces exactly.
- **Qwen3 on corrected inputs.** Qwen3 was then re-run end to end on the
  **regraded** labels with **v2** confidence (`corrected/`, and
  "Corrected Qwen3 results" below).
- **Nemotron** has not been re-run on regraded labels, and its confidence
  has not been re-scored. Its thinking-on pass mostly did not think (B7b),
  so it is out of scope for the re-score.

### B1 — confidence baselines invalid (critical)

The capture re-scored each thinking-off answer from a left-padded batch with
no attention mask, so the model attended to pad tokens. About 93% of Qwen3 v3
rows are padded. Spearman(pad count, mean log-prob) is −0.72 to −0.88 across
the four tasks. On unpadded rows only, mean-log-prob AUROC for
`needs_thinking` is 0.70 / 0.76 / 0.70 / 0.80 (n = 86 / 65 / 64 / 39), against
0.58 / 0.65 / 0.56 / 0.53 on all rows (`errata/b1_confidence_padding.md`).

- **Invalid:** every `mean_logprob`, `min_logprob`, `mean_entropy` and
  `confidence_lr` number, in both models' `baselines/confidence__*.json` and
  in §3.1, §3.2 and §3.4. That includes the abstract's
  "0.814 vs 0.699" on math500, mmlu_pro's 0.646, gsm8k `rescued`'s 0.578, and
  Nemotron's 0.666 / 0.538 / 0.619.
- **Clean:** `n_tokens_off`. Even so, a column chosen as "best confidence
  baseline" was chosen against corrupted competitors.
- **Likely direction:** the bug *understates* the confidence baseline.
  Confirmed for `needs_thinking`; see below.

**Re-scored 2026-10-06 (Qwen3 only).** `scripts/rescore_confidence.py`
re-scored every stored thinking-off answer one unpadded row at a time, with
no regeneration. It ran on Empire AI (alphagpu51, RTX PRO 6000 Blackwell;
`corrected/rescore/`).

- **Coverage.** All 3,359 rows were scored, and every prompt was rebuilt
  by its `prompt_hash`. The v3 BBH capture used the old BBH prompt,
  `PROMPT_TEMPLATE_V1`.
- **Exactness.** On the 254 rows that had no padding in their capture
  batch, v2 equals the stored value exactly.
- **Pad correlation.** Spearman(pad count, mean log-prob) moves from
  −0.72…−0.88 to −0.02…+0.19.
- **Tokenization.** 7 rows re-tokenize to a slightly different answer
  length, and 2 BBH rows (Δ = −3) are flagged.

`confidence_lr` test AUROC, v1 → v2, on the same regraded labels and
splits (`corrected/extra/conf_before_after.json`):

| task | `needs_thinking` | `rescued` |
|---|---|---|
| gsm8k | 0.677 → 0.810 [0.696, 0.907] | 0.649 → 0.527 [0.240, 0.805] |
| math500 | 0.887 → 0.919 [0.850, 0.975] | 0.561 → 0.739 [0.447, 0.978] |
| mmlu_pro | 0.670 → 0.786 [0.716, 0.851] | 0.442 → 0.635 [0.499, 0.760] |
| bbh | 0.556 → 0.715 [0.603, 0.817] | 0.523 → 0.438 [0.210, 0.671] |

Nemotron's confidence numbers remain B1-invalid.

### Graders (B2, B16, B17, B18) and unfinished reasoning (B7)

Each capture was regraded from its stored responses with the fixed graders
(`errata/regraded/TABLE.md`). An independent audit grader agrees to within
0–2 rows per capture.

- **B2, BBH:** the prompt asked for "(letter)" on non-lettered subtasks, and
  non-letter golds were exact-matched.
- **B16, MATH-500:** gold shorthand, units, prefixes and lists were not
  normalised.
- **B17, MMLU-Pro:** `\boxed{J}` was missed, and letters were read from prose.
- **B7:** a thinking-on response whose `<think>` never closed was graded on the
  unfinished trace. It is now graded wrong.
- **B18 (GSM8K)** fires on no current row.

| capture | acc_off | acc_on | `needs_thinking` rate | `rescued` n (pos.) | hurt | driven by |
|---|---|---|---|---|---|---|
| Qwen3 gsm8k | 0.933 → 0.933 | 0.940 → 0.923 | 0.067 → 0.067 | 89 (48) → 89 (42) | 38 → 54 | B7 (22 on-pass rows) |
| Qwen3 math500 | 0.764 → 0.824 | 0.782 → 0.820 | 0.236 → 0.176 | 118 (39) → 88 (33) | 30 → 35 | B16 (30 off, 23 on), B7 (4) |
| Qwen3 mmlu_pro | 0.624 → 0.617 | 0.693 → 0.648 | 0.376 → 0.383 | 376 (131) → 383 (102) | 62 → 71 | B7 (53), B17 (7 off, 8 on) |
| Qwen3 bbh | 0.676 → 0.761 | 0.822 → 0.843 | 0.324 → 0.239 | 175 (100) → 129 (69) | 21 → 25 | B2 (46 off, 16 on), B7 (5) |
| Nemotron bbh | 0.331 → 0.431 | 0.317 → 0.406 | 0.669 → 0.569 | 361 (36) → 307 (36) | 44 → 50 | B2 (55/−1 off, 48 on) |
| Nemotron lsat | 0.243 → 0.243 | 0.500 → 0.435 | 0.757 → 0.757 | 174 (68) → 174 (59) | 9 → 15 | B7 (15) |

Under the fixed graders, always-think is no better than never-think on gsm8k,
math500 and Nemotron BBH. Every target, base rate and n in §3 changes for
math500, mmlu_pro and BBH. The `rescued` positives fall on every task except
Nemotron BBH.

### B7b — Nemotron's "thinking on" mostly did not think

On BBH, 7 of 540 thinking-on responses contain `<think>`, 118 are
byte-identical to the thinking-off response, and the median thinking-on length
is 20 tokens. On LSAT, 77 of 230 have no `<think>`
(`errata/b7b/nemotron_think.out`). §3.4's BBH rows therefore compare two
non-thinking runs, and a third of its LSAT rows do too. The capture now logs a
reasoning-trace rate and warns below 80%.

### Nemotron layer, and the LSAT "lead"

§1's "layer 18, the a-priori middle layer of 36" is right for Qwen3-8B. For
Nemotron-Nano-8B, which has 33 hidden states, the a-priori middle layer is 16.
At layer 16, on the published labels (`errata/rerun_metrics/nemotronv3/metrics/`):

| task | target | published (layer 18) | layer 16 |
|---|---|---|---|
| bbh | needs_thinking | 0.755 [0.650, 0.848] | 0.744 [0.639, 0.839] |
| bbh | rescued | 0.756 [0.547, 0.932] | 0.776 [0.568, 0.950] |
| lsat | needs_thinking | 0.562 [0.353, 0.761] | 0.553 [0.344, 0.753] |
| lsat | rescued | 0.689 [0.497, 0.861] | **0.620 [0.420, 0.809]** |

The one lead in §3.4 moves 0.07 across two layers. At the a-priori layer it
sits below its text baseline: paired difference vs tfidf_word is −0.020
[−0.254, +0.217] (`errata/rerun_metrics/nemotronv3/metrics/baseline_comparison.txt`).
Its labels also change under B7 (68 → 59 positives), and on those labels it
has not been re-run. **It is not a lead.**

### Transfer verdicts (B9)

The verdict's "above chance" test used the seed-spread interval. Re-applying
the fixed rule (the bootstrap interval over target rows) to the per-probe
intervals stored in the published transfer files turns 7 of 42 verdicts into
"no transfer — target at chance". The AUROCs are unchanged
(`errata/transfer_verdicts/reverdict.out`; no retraining). The seven:

- **Qwen3:**
  - gsm8k→bbh `rescued` (was "strong")
  - math500→bbh `rescued` (was "strong")
  - bbh→mmlu_pro `helped` (was "weak")
  - math500→gsm8k `helped` and `needs_thinking` (were "partial")
- **Nemotron:**
  - lsat→bbh `needs_thinking` (was "strong")
  - lsat→bbh `rescued` (was "partial")

No `rescued` pair is labelled as transferring any more. §3.2's remark about
the "strong transfer" pairs is moot.

### Baseline selection (B10) and paired comparisons

The "best baseline" columns were picked by **test** AUROC. Picked on
validation (`errata/rerun_metrics/diff_published.out`), these change:

- §3.1 gsm8k text: tfidf_word 0.634 → length_words 0.604 [0.452, 0.749].
- §3.2 math500 confidence: n_tokens_off 0.524 → min_logprob 0.505 (B1-invalid).
- Qwen3 bbh `rescued` text: tfidf_word 0.723 → tfidf_char 0.717.
- §3.4 Nemotron bbh `rescued` confidence: confidence_lr 0.538 → n_tokens_off 0.521.
- §3.4 Nemotron lsat `rescued` confidence: mean_entropy 0.619 → min_logprob 0.550 (B1-invalid).

The tables also compared overlapping intervals. The fixed
`compare_baselines.py` reports a paired-bootstrap probe − best-baseline
difference on identical test rows. On the published labels, **no paired
difference excludes zero** for any Qwen3 or Nemotron (layer 16) task and
target. On the corrected Qwen3 inputs this changes for the confidence
baseline; see the next section.

### Corrected Qwen3 results (2026-10-06)

Regraded labels, v2 confidence, fixed code; probe − best baseline of each
kind (selected on validation), paired bootstrap on identical test rows
(`corrected/baseline_comparison.txt`, `corrected/extra/paired_by_kind.json`):

| task | target (n) | probe | probe − best text | probe − best confidence |
|---|---|---|---|---|
| gsm8k | needs_thinking (1319) | 0.687 [0.561, 0.800] | +0.082 [−0.111, +0.279] | −0.123 [−0.259, +0.005] |
| math500 | needs_thinking (500) | 0.775 [0.656, 0.879] | +0.078 [−0.072, +0.226] | **−0.144 [−0.253, −0.047]** |
| mmlu_pro | needs_thinking (1000) | 0.627 [0.549, 0.705] | +0.035 [−0.067, +0.134] | **−0.159 [−0.249, −0.069]** |
| bbh | needs_thinking (540) | 0.761 [0.653, 0.859] | −0.041 [−0.126, +0.042] | +0.046 [−0.098, +0.190] |
| gsm8k | rescued (89) | 0.631 [0.352, 0.885] | +0.093 [−0.273, +0.453] | −0.007 [−0.347, +0.317] |
| math500 | rescued (88) | 0.652 [0.351, 0.922] | +0.067 [−0.288, +0.399] | −0.155 [−0.517, +0.203] |
| mmlu_pro | rescued (383) | 0.471 [0.325, 0.621] | +0.035 [−0.147, +0.213] | −0.164 [−0.355, +0.029] |
| bbh | rescued (129) | 0.656 [0.424, 0.857] | +0.011 [−0.161, +0.189] | +0.154 [−0.157, +0.448] |

**Probe vs baselines.**
- **Text.** No probe − text interval excludes zero.
- **Confidence.** Thinking-off confidence **beats the probe** on math500
  and mmlu_pro `needs_thinking`, with paired intervals excluding zero.
  On gsm8k the difference is borderline. This holds with confidence's
  post-hoc advantage: it reads the thinking-off generation.

**`rescued`.**
- **Probe.** At chance on gsm8k, math500 and mmlu_pro.
- **Transfer.** All 12 ordered pairs are "no transfer" or "undefined".
- **Confidence.** The one `rescued` interval that clears 0.5 is the
  math500 confidence `mean_logprob`: 0.806 [0.554, 0.977], n = 88. It was
  validation-selected from five baselines and is unreplicated.

**BBH.** Still a subtask detector. Within-subtask AUROC is 0.512
[0.254, 0.763] for `needs_thinking` (21 of 27 subtasks informative) and
0.667 [0.25, 1.00] for `rescued` (12 of 21).

**Headroom proxy.** `corrected/extra/headroom_*.json`;
`scripts/headroom_proxy.py`; 1000 paired resamples. This measures routing
between Qwen3's two **generative** modes, the old project's setting. It
does not measure the proposal's single-pass decision-readout setting.

On gsm8k + math500 + mmlu_pro (n = 2,819):

- **Accuracy.** Thinking-off 0.801 vs thinking-on 0.807.
- **Per-task on − off gaps.** gsm8k −0.009 [−0.024, +0.006], math500 −0.004
  [−0.038, +0.027], mmlu_pro +0.031 [+0.003, +0.055]. BBH is +0.081
  [+0.050, +0.117].
- **nAUC over escalation 0–50%** (0 = random, 1 = oracle):
  - −mean log-prob: 0.137 [0.046, 0.221]
  - cross-fitted confidence: 0.101 [0.019, 0.182]
  - confidence + true task family: 0.056 [−0.030, 0.141]
  - With v1 confidence: −0.010 to 0.057, every interval spanning 0.
- **Family label.** It adds nothing: −0.045 [−0.091, −0.000] paired.

### Routed accuracy and the "waste" column (B11, B12, B14)

`min_routed_for_always_think_accuracy` was read off a 21-point grid with
unstable tie-breaking. On the exact curve it is lower or equal everywhere, so the `waste` column of
`results_table.txt` (1 − that fraction) rises. Qwen3 changes by 0–5 points:

- bbh 0.15 / 0.29 / 0.11 → 0.17 / 0.33 / 0.13 (helped / needs_thinking / rescued)
- gsm8k helped 0.84 → 0.88
- math500 helped 0.60 → 0.62
- mmlu_pro helped 0.04 → 0.07
- mmlu_pro needs_thinking 0.44 → 0.49

For Nemotron at layer 18 the change is at most 2.2 points (`diff_published.out`).

`results_table.txt` also labelled the seed-spread interval "AUROC [95% CI]"
(B14). The intervals quoted in this writeup were the bootstrap ones and are
unaffected.

### Within-group AUROC (B8)

`within_group_auroc.py` silently collapsed non-BBH tasks into a single group.
The published within-group numbers are BBH only and reproduce exactly:
0.494 [0.230, 0.765] with 20 of 27 subtasks informative, and 0.506
[0.009, 1.000] with 13 of 22. They are on pre-regrade labels.

### Novelty framing

The project's motivation stated that no evaluated method uses target-model
prefill state for thinking-mode routing (a "named gap in HRBench"). The
abstract calls `rescued` "the objective that would have been novel". The
first claim is false: Self-Route (arXiv 2505.20664, May 2025) routes
think/no-think from hidden states, and pre-generation correctness probes exist
(see `paper/proposal_system_one_gating.md` §8; those two citations are
unverified). Read §1 and the abstract without any novelty claim.

### What the errata change, and what they do not

**They do not change the conclusion in §5.** It holds on the published labels
and, for Qwen3, on the corrected labels and confidence (previous section).
On the published labels:

- The prefill probe still never beats a **text** baseline. Text baselines are
  untouched by B1.
- No paired probe − baseline difference excludes zero.
- `rescued` is at chance on gsm8k, math500 and mmlu_pro.
- BBH is a subtask detector.
- Transfer is now uniformly at chance for `rescued`.
- The only above-chance non-category lead, Nemotron LSAT `rescued`, does not
  survive the a-priori layer.

**They do change:**

1. **Confidence beats the probe.** The abstract's figures (math500 0.814 vs
   0.699) are superseded. On corrected Qwen3 inputs, thinking-off
   confidence beats the probe on math500 (0.919 vs 0.775) and mmlu_pro
   (0.786 vs 0.627) `needs_thinking`. The paired intervals exclude zero,
   so the claim now holds more strongly than the original writeup stated.
2. **Targets and probe AUROCs.** Every §3 target shifts under the fixed
   graders. The corrected Qwen3 probe AUROCs are in the table above:
   `needs_thinking` 0.687 / 0.775 / 0.627 / 0.761 and `rescued`
   0.631 / 0.652 / 0.471 / 0.656 (gsm8k / math500 / mmlu_pro / bbh).
3. §3.4's Nemotron BBH comparison does not test thinking at all.
4. The novelty framing is gone.

The confound story in §2 and the controls in §5 (point 3) are unaffected.

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

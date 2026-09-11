# Qwen3-8B, v3 captures (thinking-off budget fixed) — 2026-09-11

First results on data that does not carry the truncation confound. Every
file here was produced by `CAPTURE_SLUG=qwen3v3 bash scripts/run_full_analysis.sh`
on the captures from `configs/dispatch/capture_qwen3v3.json` (run on Empire
AI 2026-09-04/05), plus `scripts/within_group_auroc.py` for the two
`within_group__bbh__*` files. Logreg, layer 18 (a-priori middle, never
swept), 5 seeds, 60/20/20 splits.

`baseline_comparison.txt` is the table to read. It puts the probe beside the
best text baseline (`baselines/text__*`) and the best thinking-off confidence
baseline (`baselines/confidence__*`) on identical splits, seeds and target.
Every interval in it is the percentile bootstrap over test examples
(`test_auroc_bootstrap.ci`), never the seed-spread interval.

## Which captures

| task | rows | off-trunc | on-trunc | acc off → on | rescued rows |
|---|---|---|---|---|---|
| gsm8k | 1319 | 0.2% | 5.5% | 0.933 → 0.940 | 89 |
| math500 | 500 | 8.4% | 17.0% | 0.764 → 0.782 | 118 |
| mmlu_pro | 1000 | 8.4% | 16.1% | 0.624 → 0.693 | 376 |
| bbh | 540 | 3.1% | 5.6% | 0.676 → 0.822 | 175 |

LSAT is absent: all four shards were quarantined at 76–84% off-truncation
even at a 1024 budget and are being re-captured at 4096
(`capture_qwen3v3_redo`).

Compare the accuracies with the pre-fix numbers: MATH-500 thinking-off went
from 0.25 (75% truncated) to 0.76. Most of what looked like "needs thinking"
was "did not finish in 320 tokens".

## What the numbers say

**`needs_thinking` is real but no longer impressive, and text or length
matches it.** Probe 0.687 / 0.699 / 0.659 / 0.820 on gsm8k / math500 /
mmlu_pro / bbh, down from 0.702 / 0.879 / 0.782 / 0.838 on the confounded
data. On every task a baseline that never sees the model sits inside the
probe's interval: TF-IDF on gsm8k (0.634) and mmlu_pro (0.617), question
length on math500 (0.715 vs the probe's 0.699), char TF-IDF on bbh (0.846 vs
0.820). On math500 the thinking-off confidence regression is *better* than
the probe: 0.814 [0.700, 0.914] vs 0.699 [0.561, 0.818].

**`rescued` — the load-bearing objective — is at chance on the three
non-BBH tasks.** 0.597 [0.31, 0.86] on gsm8k (n=89), 0.559 [0.32, 0.79] on
math500 (n=118), 0.489 [0.35, 0.63] on mmlu_pro (n=376). The mmlu_pro
interval excludes nothing above 0.63 at a sample size where the earlier
pooled result claimed 0.692. The BBH 0.738 is a subtask detector, below.

**BBH is a subtask detector, again.** `within_group__bbh__*.json` scores the
probe only on (positive, negative) pairs from the same BBH subtask, so
category identity cannot contribute. Overall vs within-subtask:

| target | overall | within-subtask |
|---|---|---|
| needs_thinking | 0.820 [0.732, 0.898] | 0.494 [0.230, 0.765] |
| rescued | 0.738 [0.557, 0.890] | 0.506 [0.009, 1.000] |

Seven of 27 subtasks are settled by category alone (thinking-off all-right
or all-wrong). Char TF-IDF beating the probe on BBH is the same effect from
the other side: vocabulary identifies the subtask.

**Transfer is chance or near it for `rescued`** (`transfer__*__rescued.json`).
Every ordered pair lands in 0.42–0.59. The two pairs `results_table.txt`
labels "strong transfer" (gsm8k→bbh, math500→bbh) are a source at 0.56–0.60
landing on a target at 0.58: a small drop between two near-chance numbers,
not transfer. For `needs_thinking` the only pairs above 0.6 involve BBH as
source, which is the subtask detector transferring its "is this hard"
component.

**`helped` follows `needs_thinking`** with base rates of 3.6–18.5%, and is
not analysed further.

## Caveats that bound these numbers

- **Thinking-ON truncation is 16–17% on math500 and mmlu_pro** (5–6% on
  gsm8k and bbh). Those rows are graded wrong-with-thinking for running out
  of budget, which pushes `correct_on` down and contaminates `rescued`'s
  positive class. The on-budgets were left unchanged from v2 for
  comparability; a v4 with larger on-budgets is the obvious next capture if
  `rescued` is to be pursued.
- **Thinking-OFF truncation is 8.4% on math500 and mmlu_pro**, under the 20%
  gate but not the "near zero" the validation checklist asks for.
  `n_tokens_off` predicts `needs_thinking` at 0.69 on gsm8k and 0.61 on
  mmlu_pro `rescued`, so answer length still carries signal.
- **`rescued` sample sizes are 89–376.** The intervals say so.
- The confidence baseline reads the thinking-off *generation* (post-hoc
  router); the probe reads only the prompt (pre-hoc). The comparison is
  still the right one: a pre-hoc probe that loses to a post-hoc scalar has
  to justify itself on latency alone.

## What this rules out, and what it does not

It rules out the pre-08-30 story: at the truncation-corrected labels, the
prefill probe on Qwen3-8B does not beat cheap baselines on any objective,
and shows no query-level `rescued` signal on gsm8k, math500 or mmlu_pro.

It does not rule out the mechanism at larger n, with mean-pooled prompt
tokens instead of the last token, or with thinking-on budgets that do not
truncate a sixth of the rows. Those are the three changes worth one more
capture; another probe architecture on this data is not.

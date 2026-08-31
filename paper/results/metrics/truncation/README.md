# The truncation confound

**Every needs_thinking and rescued number produced before 2026-08-30 is
confounded by response truncation and should not be quoted.**

## What happened

Captures ran with `--max-response-len 320` for the thinking-OFF pass. Qwen3's
non-thinking mode still writes chain-of-thought, and a competition-math
solution does not fit in 320 tokens. So the off-pass was cut off mid-answer
and graded wrong.

| task | thinking-off truncated | correct_off when truncated | when not |
|---|---|---|---|
| MATH-500 | 75.2% | 0.037 | 0.944 |
| MMLU-Pro | 49.3% | 0.061 | 0.700 |
| GSM8K | 11.6% | 0.196 | 0.943 |

`correct_off` is therefore close to a pure function of truncation, and
`needs_thinking` (= `~correct_off`) is close to "did the answer exceed 320
tokens". `rescued` conditions on that same subset -- 98.1% of MATH-500's
"wrong without thinking" rows are truncated rows.

## The decisive test

Train the identical probe to predict `truncated_off` instead of the label:

| task | probe -> truncation | probe -> needs_thinking | truncation rate |
|---|---|---|---|
| MATH-500 | **0.922** | 0.879 | 75.2% |
| MMLU-Pro | **0.872** | 0.782 | 49.3% |
| GSM8K | **0.811** | 0.702 | 11.6% |

The probe predicts truncation *better* than it predicts the label, on every
task, and the reported AUROC ranks the three tasks in exactly the order of
their truncation rates. The probe is substantially a response-length
predictor.

## Cross-model replication does not rescue it

MATH-500 off-truncation: qwen3 75.2%, nemotron8b 70.2%, granite33 84.8%, with
92-98% of each "wrong" subset truncated. All models shared the same cap, so
the three-family replication reproduced the artifact rather than confirming
the finding. A confound in the experimental design is invariant to the model.

## Why it went unnoticed

The capture script warned on thinking-ON truncation and said nothing about
thinking-OFF -- the more damaging of the two, since `correct_off` defines both
objectives. Now reported: the script logs the off-truncation rate at ERROR level above
20%. It still exits 0, so a dispatched cell will not fail on it -- the log
has to be read.

## No salvage from existing data

Dropping truncated rows leaves MATH-500 with 124 rows of which 7 are wrong.
Re-capture with a larger off-budget is required.

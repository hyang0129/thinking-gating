# Retracted: the 2026-08-26 full analysis

**Nothing in this directory may be quoted.** It is kept as evidence, not as
results.

## What it is

The complete output of `run_full_analysis.sh` on 2026-08-26, and the widest
picture the project ever produced in one run:

- `metrics/<task>__<objective>.json` — 5 tasks × 2 objectives
- `metrics/transfer__<src>_to_<tgt>__<objective>.json` — all 20 ordered task
  pairs × 2 objectives, no retraining
- `labels/<task>.json` — label summaries for the five captures
- `results_table.csv` / `.txt` — the rendered table

It also carried a naming migration: `gsm8k_full` → `gsm8k` and `lsat_long` →
`lsat`. The old-named files are still in `../metrics/` and are equally
confounded.

## Why it is retracted

Every capture behind it ran the thinking-OFF pass at `--max-response-len 320`.
`correct_off` therefore measures whether the answer fit in 320 tokens rather
than whether the model could produce it, and **both** objectives derive from
`correct_off` — `needs_thinking` is `~correct_off`, and `helped` is
`~correct_off & correct_on`. So all 50 metrics files are confounded, not a
subset. See [../metrics/truncation/README.md](../metrics/truncation/README.md).

The table is where the confound is easiest to see. Reported AUROC ranks the
tasks in exactly the order of their thinking-OFF truncation rates:

| task | `needs_thinking` AUROC | off-truncation |
|---|---|---|
| MATH-500 | 0.879 | 75.2% |
| MMLU-Pro | 0.782 | 49.3% |
| GSM8K | 0.702 | 11.6% |

Those are the same three numbers the truncation writeup compares against a
probe trained to predict `truncated_off` (0.922 / 0.872 / 0.811) — which
scores *higher* on every task. This directory is the other half of that
comparison, which is the reason to keep it rather than delete it.

The LSAT rows are worth a second look for the same reason. `needs_thinking`
comes out at 0.464 with a base rate of 0.84 — read at the time as degenerate
labels, but a thinking-off accuracy of 0.152 sits below the 20% guess floor
for a 5-way multiple choice, which is a truncation signature rather than a
task property. LSAT is back in the v3 capture set to settle it.

## What is not affected

The transfer *machinery* and the verdict logic are fine; only the labels
underneath were wrong. Re-running `run_full_analysis.sh` on the v3 captures
regenerates this whole directory structure with trustworthy labels, and that
is the intended replacement.

## Provenance

Qwen3-8B, greedy, bf16, prefill at layer 18, logreg, 5 seeds. Produced
2026-08-26 on Empire AI; committed 2026-08-31 from the cluster checkout, where
it had been sitting uncommitted since.

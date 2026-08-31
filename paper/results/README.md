# Results

Metrics JSON copied verbatim from cluster runs. **Numbers quoted in the paper
come from these files, never retyped from a log.** Each file is the exact
output of the script named below, so a claim can always be traced to the run
that produced it.

## Retracted results

`retracted-2026-08-26/` holds the full 2026-08-26 analysis — 5 tasks × 2
objectives plus all 20 transfer pairs. **None of it may be quoted.** Every
capture behind it truncated the thinking-OFF pass, so `correct_off` measures
response length rather than capability and both objectives inherit that. It is
kept because it is one half of the comparison that diagnosed the confound; see
`metrics/truncation/README.md`. Note that the pre-2026-08-30 files still in
`metrics/` are confounded on the same grounds.

## Layout

    metrics/<capture>__<target>.json          run_experiment.py aggregate_metrics.json
    metrics/transfer__<src>_to_<tgt>__<target>.json   eval_transfer.py
    labels/<capture>.json                     generate_labels.py summary

`target` is what the probe was trained to predict:

- `helped` — thinking flipped the answer wrong → right (the original framing)
- `needs_thinking` — the model is wrong *without* thinking. This is what a
  router actually decides, it is better balanced, and unlike `helped` it does
  not depend on the thinking token budget, so truncated generations cannot
  corrupt it.

## Reading the numbers

- `aggregate.test_auroc` — mean ± 95% CI over 5 seeds, on held-out test splits.
- `aggregate.test_auroc_by_difficulty` — **the confound check**. The helped
  rate rises steeply with difficulty, so a probe that only detects hard
  questions scores well overall. Signal is only credible where the
  within-stratum CI excludes 0.5.
- `aggregate.min_routed_for_always_think_accuracy` — the smallest fraction of
  queries that must be routed to thinking to match always-think accuracy.
  1 − that is the share of thinking compute that is simply wasted.
- `baseline_never_think` / `baseline_always_think` / `oracle` — a probe is only
  interesting strictly between max(never, always) and oracle. On these tasks
  always-think already lands within a couple of points of oracle, which is why
  the compute framing matters more than the accuracy framing.

## Provenance

Model: Qwen/Qwen3-8B, greedy decoding, bf16, prefill state taken at layer 18
(chosen a priori as the middle layer, not swept, to avoid selection effects).
Captures were run on Empire AI H100/H200 nodes via `scripts/dispatch/`.

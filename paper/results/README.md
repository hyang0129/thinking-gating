# Results

**Writeup: `../negative_result.md`.** The project's conclusion is negative;
that document is the one place the argument is made end to end, and every
number in it points back to a file here.

## Which directories are current

- `metrics/qwen3v3/`, `metrics/nemotronv3/` — the **v3 (truncation-corrected)
  runs**, 2026-09-11. These are the only probe/transfer/baseline numbers
  that may be quoted. Each has a README with the caveats.
- `metrics/truncation/` — the diagnosis of the confound; quotable *as the
  diagnosis*.
- `metrics/tuning/`, `metrics/decomposition/`, `metrics/baselines/` — run on
  confounded data. The **estimator** findings in their READMEs (bootstrap vs
  seed-spread CI, layer-sweep selection effect, sample size) carry forward;
  the AUROCs do not.
- `metrics/*.json` at the top level, `labels/`, `retracted-2026-08-26/` —
  pre-v3, confounded, not quotable.

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
  router actually decides and it is better balanced than `helped`. It does
  not depend on the thinking-ON budget — but it depends entirely on the
  thinking-OFF budget, which is how the confound got in (see
  `metrics/truncation/`).
- `rescued` — `correct_on` restricted to rows with `correct_off == False`.
  Difficulty is held fixed by construction; this is the objective the novel
  claim rested on, and it is at chance on the corrected data.

## Reading the numbers

- `aggregate.test_auroc.mean` — mean over 5 seeds on held-out test splits.
  **Quote `aggregate.test_auroc_bootstrap.ci` for the interval**, never
  `test_auroc.ci` — the latter is seed-to-seed spread on a fixed sample and
  is 1.6–8.8× too narrow.
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

Models: Qwen/Qwen3-8B and nvidia/Llama-3.1-Nemotron-Nano-8B-v1, greedy
decoding, bf16, prefill state taken at layer 18 (chosen a priori as the
middle layer, not swept, to avoid selection effects). Captures were run on
Empire AI via `scripts/dispatch/`; v3 manifests are
`configs/dispatch/capture_{qwen3,nemotron}v3.json`.

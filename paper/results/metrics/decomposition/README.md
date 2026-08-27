# The decomposition run (Qwen3-8B, layer 18, logreg, 5 seeds)

Every number quoted about the decomposition comes from these files. Nothing
here is retyped from a log.

## What the three objectives are

| target | label | what an above-chance AUROC means |
|---|---|---|
| `needs_thinking` | `~correct_off` | the prefill state predicts the model will be **wrong** without thinking. This is correctness prediction, a well-studied problem (Kadavath 2022, Azaria & Mitchell 2023). |
| `helped` | `~correct_off & correct_on` | thinking flipped the answer. Confounded: mostly driven by the `~correct_off` term. |
| `rescued` | `correct_on`, **restricted to rows where `correct_off == False`** | **the load-bearing one.** On this subset `needs_thinking` is constant by construction, so difficulty cannot explain any signal. Above chance here means the prefill state encodes the *marginal value of reasoning*, which is the only claim in this work that is not already in the literature. |

## Which interval to quote

`aggregate.test_auroc.ci` is a normal approximation over 5 seeds. The seeds
re-split a **fixed** sample, so that interval measures split-to-split spread
and **not** uncertainty about the population. On these sample sizes it is
1.6x to 8.8x too narrow.

**Quote `aggregate.test_auroc_bootstrap.ci`** — a percentile bootstrap over
test examples, averaged across seeds.

## Reading the stratified files (`strat__*`)

`test_auroc_by_difficulty_bootstrap` holds the within-stratum intervals. Note
that a stratum's test split is often only 15-25 examples, so these intervals
are very wide and **failing to clear 0.5 in a stratum is not evidence the
signal is absent there** — it is mostly a statement about power. Counting
"how many strata clear chance" is therefore a weak instrument; a pooled
within-stratum AUROC is the statistic that should replace it.

An earlier draft reported "12/14 MMLU-Pro categories above chance" using the
seed-spread interval. With the bootstrap interval the same run gives 5/14.
The earlier figure was an artifact of the interval, not a finding.

## Reproduce

    python scripts/dispatch/queue.py expand configs/dispatch/probe_decomposition.json
    python scripts/dispatch/queue.py expand configs/dispatch/stratify_rescued.json

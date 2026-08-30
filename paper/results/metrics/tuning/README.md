# Can the probe be made more accurate?

Short answer: **only by adding data, not by changing the model.**

All numbers below are MATH-500 / `rescued` (or the 3-task pool), Qwen3-8B
unless noted, 5 seeds, bootstrap CIs over test examples.

## What worked

| change | pooled AUROC | vs baseline |
|---|---|---|
| single layer, logreg (baseline) | 0.624 [0.548, 0.698] | — |
| single layer, MLP | 0.659 [0.584, 0.731] | +0.035 |
| every 8th layer concatenated, logreg | 0.673 [0.599, 0.743] | +0.049 |
| every 4th layer concatenated, logreg | 0.684 [0.611, 0.753] | +0.060 |
| **every 8th layer concatenated, MLP** | **0.692 [0.618, 0.760]** | **+0.068** |

The winning config was chosen on **validation** AUROC, not test. Validation
and test happened to rank the five configs almost identically, which is the
sanity check that this is not selection noise.

## What did not work

**Layer sweep (37 layers, selected on validation).** Validation AUROC rose a
lot -- 0.695 -> 0.827 on Qwen3 -- and test AUROC *fell*: 0.703 -> 0.662, and
0.748 -> 0.713 on Granite. Mean across three models went 0.708 -> 0.692.
Picking the best of 37 layers on a ~74-example validation split selects luck,
not depth. The a-priori middle layer is as good as anything.

**The tuned config applied per-task.** On MATH-500 alone it *hurt*:
-0.008 / -0.018 / -0.003 across the three models (mean -0.010). Five
concatenated layers is 20480 features against ~221 training rows. The same
config helps on the pool only because the pool has 1174 rows.

## What this means

Sample size is the binding constraint, not model capacity. Every gain came
from having more rows (pooling three tasks) and every loss came from spending
capacity that the row count could not support.

The two untested levers both need new captures:

1. **More MATH data.** MATH-500 is a 500-row *evaluation* subset; the full
   MATH corpus has ~12.5k problems. Capturing 3k would take the `rescued`
   subset from 369 to roughly 2200 and is the cheapest large win available.
2. **Token pooling.** Only the last prompt token was ever saved. Mean-pooling
   over prompt tokens is usually a substantial gain in probing work, and
   testing it requires re-running every capture.

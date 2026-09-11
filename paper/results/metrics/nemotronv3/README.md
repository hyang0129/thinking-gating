# Nemotron-Nano-8B, v3 captures — partial, 2026-09-11

Produced by `CAPTURE_SLUG=nemotronv3 TASKS="bbh lsat" bash scripts/run_full_analysis.sh`.
Method and file layout as in `../qwen3v3/README.md`; read that first.

**Only bbh and lsat are here.** The other three Nemotron tasks did not
survive the first v3 pass — gsm8k quarantined at 24–28% off-truncation,
math500 one shard quarantined and the rest at 16.5%, mmlu_pro three shards
lost to a co-tenant CUDA OOM — and are being re-captured by
`configs/dispatch/capture_nemotronv3_redo.json` at larger budgets. Their
partial v3 captures were moved to `shared/icr_capture/_superseded/` and
nothing here was trained on them.

| task | rows | off-trunc | on-trunc | acc off → on | rescued rows |
|---|---|---|---|---|---|
| bbh | 540 | 1.5% | 5.6% | 0.331 → 0.317 | 361 |
| lsat | 230 | 5.7% | 20.0% | 0.243 → 0.500 | 174 |

## Read with these in mind

- **BBH thinking-off accuracy is 0.33 and thinking-on is 0.32.** Thinking
  does not help Nemotron on BBH on average; `helped` has 36 positives. The
  BBH `needs_thinking` (0.755) and `rescued` (0.756) probes both lose to
  TF-IDF (0.794, 0.802), and the Qwen3 within-subtask check applies here
  too: BBH numbers are subtask identity until shown otherwise. Not re-run
  for Nemotron because the conclusion does not change.
- **LSAT thinking-off accuracy is 0.243, below the 0.25 guess floor** for
  five-way multiple choice, at 5.7% off-truncation. The "degenerate labels"
  diagnosis from v2 was not a truncation artifact after all for this model.
  LSAT `needs_thinking` is at chance (0.562 [0.35, 0.76]) accordingly.
- **LSAT `rescued` 0.689 [0.497, 0.861]** is the one number above chance on a
  non-category task in the whole v3 set, on n=174 with a 20% thinking-on
  truncation rate that determines part of the positive class. It beats
  TF-IDF (0.640) and mean entropy (0.619) by less than any of the intervals.
  It is a lead, not a result; the Qwen3 LSAT redo is the replication.

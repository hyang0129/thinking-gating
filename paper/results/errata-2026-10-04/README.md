# Errata working files (2026-10-04)

These are the files behind the "Errata (2026-10-04)" section of
`paper/negative_result.md`. The section cites them as `errata/...`.

**These are working numbers, not promoted metrics.**
- They were produced during the cleanup audit, by the fixed code on branch
  `cleanup/system-one-carryover`, from the local v3 captures.
- The canonical corrected tables come from re-running
  `scripts/run_full_analysis.sh` (which now regrades labels) with
  `PROMOTE=1`, after the confidence re-score (B1).
- Scripts here may reference the audit's original scratch paths. Read them as
  a record of what was run, not as runnable tools.

| path | what |
|---|---|
| `b1_confidence_padding.md` | B1: evidence that thinking-off confidence attended to left pads |
| `regraded/TABLE.md` | Label before/after per capture under the fixed graders (B2, B16, B17, B7) |
| `regraded/*.summary.json` | `generate_labels.py --regrade` summaries; full label files are not kept here, so regenerate them with `--regrade` |
| `rerun_metrics/rerun.py`, `diff_published.py`, `diff_published.out` | Fixed analysis code re-run on the **published (pre-regrade)** labels, diffed against `paper/results/metrics/` |
| `rerun_metrics/nemotronv3/metrics/` | Nemotron re-run at the a-priori middle layer 16; per-row predictions omitted for size |
| `transfer_verdicts/` | B9: transfer verdicts recomputed from stored bootstrap intervals, without retraining |
| `b7b/` | B7b: how often Nemotron's thinking-on pass actually reasons |

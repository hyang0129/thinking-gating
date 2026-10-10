# Qwen3-8B v3 captures, corrected (2026-10-06)

This is the corrected counterpart of `../qwen3v3/`. That directory is the
provenance for the 2026-09-12 writeup and has not been changed. This one
is the same Qwen3-8B v3 captures, analysed again with two corrections and
the fixed analysis code. `paper/negative_result.md` (Errata) cites it.

| what | `../qwen3v3/` (published) | here |
|---|---|---|
| labels | stored grades at capture time | **regraded** from the stored responses with the fixed graders (B2, B16, B17, B18, B7 unclosed `<think>` graded wrong); `labels/*.json` records `regraded: true` and `grader_version` |
| thinking-off confidence | `confidence_off` as captured (**v1**, scored with left pads attended, B1) | **v2**: re-scored from the stored responses, one unpadded row at a time (`scripts/rescore_confidence.py`); every `baselines/confidence__*.json` records `confidence_source: confidence_off_v2 sidecar` |
| analysis code | 7942a36 | fixed code: B8–B14, validation-selected baselines, paired probe − baseline differences |
| tasks | gsm8k, math500, mmlu_pro, bbh | same (LSAT's Qwen3 capture is still quarantined) |

Produced at commit **b368bd7** (branch `cleanup/system-one-carryover`) by:

```
CAPTURE_SLUG=qwen3v3 TASKS="gsm8k math500 mmlu_pro bbh" \
  PROMOTE=1 PROMOTE_DIR=paper/results/metrics/qwen3v3_corrected \
  bash scripts/run_full_analysis.sh
```

The settings were logreg, layer 18 (the a-priori middle layer, never
swept), 5 seeds and 60/20/20 splits. Every interval quoted here is the
percentile bootstrap over test rows.

## The re-score (B1)

- **Job.** Run on Empire AI, alphagpu51 (NVIDIA RTX PRO 6000 Blackwell,
  bf16, SDPA), 2026-10-06. Dispatch job `93af9f173276`, cell queue
  `shared/dispatch/rescore_confidence_qwen3v3`, manifest
  `configs/dispatch/rescore_confidence_qwen3v3.json`, commit b368bd7.
  It took about 6 minutes for 3,359 rows.
- **Per-capture summaries.** `rescore/{task}.summary.json`: rows, flags and
  token-count deltas per shard. The sidecars themselves
  (`confidence_off_v2.shardNN.jsonl`) are data and live beside the meta
  shards in `shared/icr_capture/{task}_thinking_qwen3v3/` (gitignored).
- **Coverage.** Every row was scored: 1319 / 500 / 1000 / 540 rows, equal
  to the meta row counts.
- **Prompt rebuild.** Every prompt was rebuilt by `prompt_hash`. gsm8k,
  math500 and mmlu_pro match the current `format_prompt`; all 540 BBH rows
  match `tasks.bbh.PROMPT_TEMPLATE_V1`, the prompt the v3 capture used.
  Each rebuilt prompt token count equals the stored `prompt_len_off`.
- **Integrity.** 7 of 3,359 rows re-tokenize to a different answer length
  than `n_tokens_off`, because the model emitted a non-canonical
  tokenization: mmlu_pro has two rows at ±1, BBH has five at −3, −3, −2,
  −1 and +1. The two BBH rows at −3 carry the `token_count_mismatch` flag
  and are kept.
- **Validation.**
  - On the 254 rows that had **no** padding in their capture batch,
    v2 equals the stored v1 value exactly (max |Δ mean_logprob| = 0.000).
    The re-score therefore reproduces the capture's scorer, prompt and
    tokens.
  - On padded rows the median |Δ| is 1.3–2.3 nats per token.
  - Spearman(pad count, mean_logprob) moves from −0.72…−0.88 (v1) to
    −0.02…+0.19 (v2). Shorter prompts get more padding, so a small residual
    is expected and is a real feature.

## Confidence baselines: v1 → v2 (same regraded labels and splits)

`extra/conf_before_after.json` (tool: `tools/conf_before_after.py`; the
"before" files are the same run with stored v1 confidence). Test AUROC of
the regression over all four scalars (`confidence_lr`):

| task | needs_thinking v1 → v2 | rescued v1 → v2 |
|---|---|---|
| gsm8k | 0.677 → **0.810** [0.696, 0.907] | 0.649 → 0.527 [0.240, 0.805] |
| math500 | 0.887 → **0.919** [0.850, 0.975] | 0.561 → 0.739 [0.447, 0.978] |
| mmlu_pro | 0.670 → **0.786** [0.716, 0.851] | 0.442 → 0.635 [0.499, 0.760] |
| bbh | 0.556 → **0.715** [0.603, 0.817] | 0.523 → 0.438 [0.210, 0.671] |

- B1 understated confidence on `needs_thinking` on every task, by 0.03 to
  0.16.
- On `rescued`, the changes move both ways inside intervals of ±0.2–0.3.
- `n_tokens_off` is unchanged, which is the check that only the three
  log-prob scalars moved.

## Probe vs baselines (paired, identical test rows)

`baseline_comparison.txt` pairs the probe with the single best baseline,
chosen on validation across text and confidence. `extra/paired_by_kind.json`
(tool: `tools/paired_by_kind.py`) pairs it with the best of **each** kind:

| task | target | probe | probe − best text | probe − best confidence (v2) |
|---|---|---|---|---|
| gsm8k | needs_thinking | 0.687 [0.561, 0.800] | +0.082 [−0.111, +0.279] | −0.123 [−0.259, +0.005] vs confidence_lr |
| math500 | needs_thinking | 0.775 [0.656, 0.879] | +0.078 [−0.072, +0.226] | **−0.144 [−0.253, −0.047]** vs confidence_lr |
| mmlu_pro | needs_thinking | 0.627 [0.549, 0.705] | +0.035 [−0.067, +0.134] | **−0.159 [−0.249, −0.069]** vs confidence_lr |
| bbh | needs_thinking | 0.761 [0.653, 0.859] | −0.041 [−0.126, +0.042] | +0.046 [−0.098, +0.190] |
| gsm8k | rescued (n=89) | 0.631 [0.352, 0.885] | +0.093 [−0.273, +0.453] | −0.007 [−0.347, +0.317] vs n_tokens_off |
| math500 | rescued (n=88) | 0.652 [0.351, 0.922] | +0.067 [−0.288, +0.399] | −0.155 [−0.517, +0.203] vs mean_logprob |
| mmlu_pro | rescued (n=383) | 0.471 [0.325, 0.621] | +0.035 [−0.147, +0.213] | −0.164 [−0.355, +0.029] vs confidence_lr |
| bbh | rescued (n=129) | 0.656 [0.424, 0.857] | +0.011 [−0.161, +0.189] | +0.154 [−0.157, +0.448] |

**Reading.**

- **Probe vs text.** The prefill probe beats no text baseline: no
  probe − text interval excludes zero.
- **Probe vs confidence.** Thinking-off confidence (v2) **beats the probe**
  on math500 and mmlu_pro `needs_thinking`; the paired intervals exclude
  zero. gsm8k is borderline. Confidence is a post-hoc router: it reads the
  thinking-off generation, which the probe does not.
- **BBH.** It remains a subtask detector. The within-subtask AUROC is
  0.512 [0.254, 0.763] for `needs_thinking` (21 of 27 subtasks
  informative) and 0.667 [0.25, 1.00] for `rescued` (12 of 21).
- **`rescued`.**
  - The probe is at chance on gsm8k, math500 and mmlu_pro, and every
    `rescued` transfer pair is "no transfer" or "undefined".
  - The confidence baselines sit near chance on gsm8k and bbh.
  - math500 `mean_logprob` 0.806 [0.554, 0.977] (n = 88, 33 positives)
    is the one `rescued` number whose interval clears 0.5. It was chosen
    on validation from five baselines and is unreplicated. Treat it as a
    lead for confidence-based routing, not a result.

## Headroom proxy (issue #1 §0), corrected

`extra/headroom_*.json`, from `scripts/headroom_proxy.py` with 1000
paired bootstrap resamples.

**What this measures.** Routing between Qwen3's two **generative** modes:
thinking-off (which still writes a visible chain of thought) and
thinking-on. That is the old project's setting. It does not measure the
proposal-v3 S1-vs-S2 gap, where S1 is a single-pass decision readout, and
it does not settle v3's G2.

On gsm8k + math500 + mmlu_pro (n = 2,819), off 0.801 vs on 0.807. The
per-task on − off gaps (bootstrap 95%) differ in sign:

- gsm8k: −0.009 [−0.024, +0.006]
- math500: −0.004 [−0.038, +0.027]
- mmlu_pro: +0.031 [+0.003, +0.055]
- bbh: +0.081 [+0.050, +0.117] (secondary file)

On this mix "within 1 pp of always-think" is met with **no** escalation,
so that operating point is uninformative here. nAUC over escalation
0–50% (0 = random, 1 = oracle):

| router | v2 confidence | v1 confidence (same labels) |
|---|---|---|
| conf_raw (−mean_logprob) | 0.137 [0.046, 0.221] | 0.057 [−0.017, 0.131] |
| conf_fit (cross-fitted, 4 scalars) | 0.101 [0.019, 0.182] | −0.010 [−0.096, 0.063] |
| conf + true task family | 0.056 [−0.030, 0.141] | 0.047 [−0.037, 0.123] |
| oracle | 1 | 1 |

- **Confidence vs oracle.** Paired conf_fit − oracle is −0.899
  [−0.981, −0.818].
- **Adding the task family.** Paired conf+family − conf_fit is −0.045
  [−0.091, −0.000]: the privileged family label adds nothing and
  slightly overfits.
- **With BBH** (n = 3,359, `extra/headroom_with_bbh.json`): conf_fit
  0.124 [0.052, 0.194], conf+family − conf_fit +0.011 [−0.036, +0.060].
- **Router scores** are cross-fitted once and held fixed across
  resamples.

## Caveats (unchanged by these corrections)

- **Thinking-on truncation.**
  - math500: 85 of 500 rows (17%).
  - mmlu_pro and lsat: 16–20%.
  - Truncated rows count as wrong-with-thinking. Unclosed `<think>` is
    now graded wrong explicitly (65 rows on math500), which pushes
    `correct_on` down.
- **Thinking-off truncation** is 8.4% on math500 and mmlu_pro.
- **`rescued` sample sizes** are 88 to 383 rows. The intervals say so.

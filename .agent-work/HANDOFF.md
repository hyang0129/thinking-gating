# Thinking-Mode Gating Experiment — Agent Handoff

**Date:** 2026-09-11
**Status:** V3 (truncation-corrected) captures ran 2026-09-04/05; 8 of 10
published. The corrected result on Qwen3-8B is **negative**: the prefill probe
beats no text or confidence baseline, `rescued` is at chance off BBH, and BBH
is a subtask detector. Redo captures for the 5 failed task×model pairs are
queued with a watcher; two Jupyter allocations pending.
**Owner:** Hong Yang

---

## Context

Pathway 1 of the prefill-applications line of work: **train a small probe on the
prefill hidden state (last-prompt-token activation) to predict whether extended
reasoning will improve a query's outcome** — i.e. a router that decides where to
spend thinking compute. Named gap in HRBench: no evaluated method uses target-
model prefill state for thinking-mode routing.

**The repo is self-contained.** Task modules, dispatch tooling, and the venv all
live here. `bash scripts/setup_env.sh` on any machine. See
`.agent-work/EMPIRE_AI_SETUP.md` for cluster workflow, `agent.md` for the
operating rules (which machine may do what, dispatch hygiene, pitfalls).

### The three objectives

| target | label | meaning |
|---|---|---|
| `needs_thinking` | `~correct_off` | model is wrong *without* thinking. Correctness prediction — already well studied. Better balanced than `helped`, and does not depend on the thinking budget. |
| `helped` | `~correct_off & correct_on` | thinking flipped the answer. Confounded — mostly driven by the `~correct_off` term. The original framing; largely superseded. |
| `rescued` | `correct_on`, restricted to `correct_off == False` | **the load-bearing one.** On this subset `needs_thinking` is constant by construction, so difficulty cannot explain the signal. Above chance here means the prefill state encodes the *marginal value of reasoning* — the only claim here that is not already in the literature. |

---

## What is built (done, working, tested)

- **Capture** (`scripts/capture_inference_thinking.py`): paired thinking-off/on
  inference, prefill from a dedicated forward pass, batching + sharding,
  per-task budgets. Thinking toggle is detected from the chat template, not a
  name whitelist — covers `enable_thinking`, system-prompt toggles, graded
  reasoning levels, harmony-format extraction, and eager-attention fallback for
  architectures that reject SDPA.
- **Pipeline:** `generate_labels.py` → `run_experiment.py` (MLP/logreg, 5 seeds,
  AUROC ± bootstrap CI, by-difficulty, routed accuracy vs never/always/oracle)
  → `eval_transfer.py`. `utils/capture_io.py` loads shards; `run_full_analysis.sh`
  goes captures-in → results-table-out.
- **Controls:** `stratify_check.py`, `validate_bench.py`, `tests/test_pipeline.py`
  (synthetic end-to-end + signal-free negative control), `tests/test_dispatch.py`.
- **Dispatch:** `scripts/dispatch/` — manifest-driven cell queue, one generic
  worker, atomic claims, resume/retry/timeout/stale recovery.
- **Tasks:** gsm8k, lsat, math500, mmlu_pro, bbh.
- **Models run:** Qwen3-8B (anchor), Qwen3-14B, Nemotron-Nano-8B, Granite-3.3,
  gpt-oss-20b.

Still unwritten: `scripts/template_ablation.py` (optional, low priority).

---

## Where things stand (2026-09-11)

Full numbers and caveats: `paper/results/metrics/qwen3v3/README.md` and
`paper/results/metrics/nemotronv3/README.md`. The short version is in
`CLAUDE.md` → "Current State". Do not quote anything from
`paper/results/metrics/*.json` at the top level or from `truncation/`,
`baselines/`, `decomposition/`, `tuning/` — those are the retracted pre-v3
runs, kept as the record of the confound.

### What the v3 pass produced

| capture | rows | off-trunc | status |
|---|---|---|---|
| qwen3v3 gsm8k / math500 / mmlu_pro / bbh | 1319 / 500 / 1000 / 540 | 0.2 / 8.4 / 8.4 / 3.1% | analysed |
| nemotronv3 bbh / lsat | 540 / 230 | 1.5 / 5.7% | analysed |
| qwen3v3 lsat | — | 76–84% at 1024 | quarantined → redo @4096 |
| nemotronv3 gsm8k | — | 24–28% at 1024 | quarantined → redo @2048 |
| nemotronv3 math500 | 375/500 | 16.5–21.6% at 2048 | one shard quarantined → whole task redo @4096 |
| nemotronv3 mmlu_pro | 250/1000 | 6.8% | 3 shards co-tenant OOM → whole task redo @2048, batch 8 |

Local copies of the eight published captures are in `shared/icr_capture/`;
the superseded partials are under `shared/icr_capture/_superseded/` here and
on the cluster.

### The corrected result

- `needs_thinking`: probe 0.69 / 0.70 / 0.66 / 0.82 (gsm8k / math500 /
  mmlu_pro / bbh). A no-model baseline sits inside every interval; on
  math500 the thinking-off confidence regression is better (0.814 vs 0.699).
- `rescued`: 0.60 / 0.56 / 0.49 on gsm8k / math500 / mmlu_pro, n = 89 / 118
  / 376. Chance. BBH's 0.74 is 0.51 within-subtask.
- Transfer for `rescued`: 0.42–0.59 on every ordered pair.
- Nemotron LSAT `rescued` 0.689 [0.50, 0.86] is the only above-chance number
  on a non-category task; unreplicated, 20% thinking-on truncation.
- Nemotron LSAT thinking-off accuracy is 0.243 at 5.7% truncation — the v2
  "degenerate labels" verdict was correct, not a truncation artifact.

### Cluster state

- Queues `shared/dispatch/capture_nemotronv3_redo` (12 pending) and
  `capture_qwen3v3_redo` (4 pending), expanded from the `*_redo.json`
  manifests at commit 9e9692c or later.
- SLURM 81777 / 81778 = `jupyter_empire_8882` / `8883`, PENDING (Priority).
- `scripts/watch_and_dispatch.py` running detached (log:
  `shared/logs/watch_dispatch.log`, stop: `touch shared/dispatch/STOP_WATCH`).
  It dispatches one worker per root as allocations land. It will **not**
  dispatch while the cluster checkout is behind `origin/main`, so `git pull`
  there after every push.
- `shared/gpu_jobs.json`: 17 finished, 8 unknown (workers whose nodes are
  gone). Nothing running.
- Original v3 queues: failed cells moved to `<root>/retired/` with a
  `RETIRED.md`; re-expanding the original v3 manifests would recreate them.

### To finish the re-capture

1. Confirm the watcher dispatched (`queue.py status --root shared/dispatch/capture_*_redo`),
   then grep each cell log for the thinking-OFF truncation line; the gate
   quarantines above 20% but 8% is still not "near zero".
2. `scp`/tar the five new capture dirs back into `shared/icr_capture/`.
3. `CAPTURE_SLUG=nemotronv3 bash scripts/run_full_analysis.sh` and
   `CAPTURE_SLUG=qwen3v3 TASKS=lsat bash scripts/run_full_analysis.sh`
   (idempotent; finished steps are skipped). Then
   `scripts/compare_baselines.py --metrics-dir paper/results/metrics/<slug>`
   and update the two READMEs.
4. The decision point: does Qwen3 LSAT `rescued` replicate Nemotron's 0.689
   above its text and confidence baselines? If not, write the negative
   result. If yes, the next capture changes thinking-on budgets, adds
   mean-pooled prompt tokens, and adds ~3k MATH rows — together, once.

## Findings that survive the confound

These are about the estimator and the method, not the labels, so they carry
forward:

- **Quote the bootstrap CI.** `test_auroc.ci` is a normal approximation over 5
  seeds that re-split a *fixed* sample — it measures split-to-split spread, not
  population uncertainty, and is 1.6–8.8× too narrow. An earlier "12/14 MMLU-Pro
  categories above chance" became 5/14 under `test_auroc_bootstrap.ci`.
- **Sample size is the binding constraint, not capacity.** Every tuning gain
  came from more rows (pooling 3 tasks: 0.624 → 0.692); the same config applied
  per-task *hurt* (−0.010 mean). A 37-layer sweep selected on validation raised
  val AUROC 0.695 → 0.827 and *lowered* test 0.703 → 0.662. The a-priori middle
  layer is as good as anything.
- **Text baselines are mandatory** — an 8B forward pass has to beat TF-IDF on
  the raw question, on identical splits/seeds/target.
- **Pooled `needs_thinking` is confounded by task identity** (base rates 0.144 /
  0.738 / 0.615), so it is not evidence of query-level signal. Pooled `rescued`
  (0.721 / 0.713 / 0.569) is the clean comparison.
- **Stratification catches hardness detectors** — the BBH result was a subtask
  detector; `stratify_check.py` exists because of it.

---

## Open work beyond the re-capture

1. ~~Thinking-off confidence baseline~~ — `scripts/baseline_confidence.py`,
   run as part of `run_full_analysis.sh`. On v3 it beats the probe on
   math500 `needs_thinking`.
2. A small fine-tuned text encoder (MiniLM/DeBERTa) on the same labels —
   only worth it if some probe number survives that TF-IDF does not match.
3. `template_ablation.py` — minimal-pair format-robustness check, still optional.
4. Mean-pooled prompt-token activations and larger thinking-on budgets — a
   capture-script change, folded into whatever capture comes after the redo.

---

## Files & paths

**Repo:** `/Users/hong/Documents/code-projects/thinking-gating/`
**Cluster:** `~/LLM_research/thinking-gating/` on Empire AI; large artifacts in
`/raid0/think-gating/`.

- `agent.md` (symlinked `claude.md`) — operating rules, dispatch, pitfalls
- `.agent-work/EMPIRE_AI_SETUP.md` — cluster setup and dispatch workflow
- `paper/results/` — provenance record; every quoted number traces to a file
  here. Read the per-group READMEs (`truncation/`, `baselines/`,
  `decomposition/`, `tuning/`) before quoting anything — the caveats are the
  load-bearing part.
- `output/` — working metrics, promoted to `paper/results/` when citable
- `shared/icr_capture/` — captures; v3 published dirs are `{task}_thinking_{qwen3v3,nemotronv3}`, superseded partials under `_superseded/`

**Never edit a `.bib` file directly.** Cite in prose with enough context for a
human to verify, and get explicit approval before any bibliography insertion.

---

## Contacts

- **User:** Hong Yang (hooong.yang@gmail.com)
- **Paper draft:** `thinking-gating/paper/`

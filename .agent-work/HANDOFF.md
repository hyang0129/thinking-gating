# Thinking-Mode Gating Experiment — Agent Handoff

**Date:** 2026-08-31
**Status:** Pipeline complete and exercised across 5 models / 5 benchmarks. All
pre-08-30 probe results retracted (truncation confound). V3 re-captures queued
on Empire AI, blocked on GPU allocation.
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

## The blocker: the truncation confound (2026-08-30)

**Every `needs_thinking` and `rescued` number produced before 2026-08-30 is
confounded and must not be quoted.** Full writeup:
`paper/results/metrics/truncation/README.md`.

The thinking-OFF pass ran at `--max-response-len 320`. Qwen3 writes chain-of-
thought even with thinking off, so long answers were cut off and graded wrong
for running long rather than for being unable.

| task | off truncated | correct_off when truncated | when not |
|---|---|---|---|
| MATH-500 | 75.2% | 0.037 | 0.944 |
| MMLU-Pro | 49.3% | 0.061 | 0.700 |
| GSM8K | 11.6% | 0.196 | 0.943 |

Decisive test — train the identical probe to predict `truncated_off` instead of
the label:

| task | probe → truncation | probe → needs_thinking |
|---|---|---|
| MATH-500 | **0.922** | 0.879 |
| MMLU-Pro | **0.872** | 0.782 |
| GSM8K | **0.811** | 0.702 |

It predicts truncation better than the label on every task, and reported AUROC
ranks the tasks exactly by truncation rate. All three model families shared the
cap, so the cross-family replication reproduced the artifact — a design confound
is invariant to the model.

No salvage: dropping truncated rows leaves MATH-500 with 124 rows of which 7 are
wrong. Re-capture is required. The capture script warned on thinking-ON
truncation and was silent on thinking-OFF (the more damaging of the two, since
`correct_off` defines both objectives); it now logs at ERROR level above 20%.
It still exits 0, so a dispatched cell will not fail on it — the log must be
read.

---

## Next: the v3 re-capture

`configs/dispatch/capture_qwen3v3.json`, `configs/dispatch/capture_nemotronv3.json`
— 2 models × 5 benchmarks × 4 shards. Off-budgets 320 → 1024 (2048 for MATH-500),
sized from v2 data (complete off-responses were censored at the old cap, max
observed 311/320/319; MATH-500 answers reach ~720 words at p99). Thinking-on
budgets unchanged so the on-side stays comparable. LSAT is back in — its
"degenerate labels" diagnosis (off-accuracy below the guess floor) is itself a
plausible truncation artifact.

### Cluster state, checked 2026-08-31

- `shared/dispatch/capture_qwen3v3` and `capture_nemotronv3` are **already
  expanded**: 20 cells each, **all pending, none ever claimed, 0% done**.
- `squeue --me`: six `jupyter_empire_*` allocations, **all PENDING on
  `(Priority)`**. Nothing is running; no GPU node is held.
- `gpu_jobs.json`: 17 `finished`, 5 stale `unknown` (Aug 27–28 workers whose
  nodes went away without the manifest updating).
- Cluster checkout is at `f252f87`, same as local, with
  `paper/results/metrics/gsm8k_full__helped.json` showing as deleted in the
  working tree there.

### To restart

1. Cancel the redundant pending Jupyter jobs so they stop competing, keep one.
2. When a node lands, dispatch one worker per node:
   `python scripts/gpu_dispatch.py run .venv/bin/python scripts/dispatch/worker.py --root shared/dispatch/capture_qwen3v3`
   (re-expanding is idempotent; finished cells are skipped).
3. Watch: `python scripts/dispatch/queue.py status --root shared/dispatch/capture_qwen3v3`
4. `scp` the captures back (data is gitignored; the no-`scp` rule is about code
   going the *other* way).
5. Re-run labels → probes → decomposition → baselines.

**Consider folding into the same capture, since it requires re-running anyway:**
mean-pooled prompt-token activations (only the last token was ever saved;
token pooling is usually a large gain in probing work), and ~3k rows from the
full MATH corpus (would take `rescued` from 369 rows to ~2200 — the cheapest
large win available).

---

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

1. **Thinking-off confidence / answer entropy baseline** — the strongest
   remaining competitor to the probe, and cheap. Should exist before any
   writeup.
2. A small fine-tuned text encoder (MiniLM/DeBERTa) on the same labels.
3. `template_ablation.py` — minimal-pair format-robustness check, still optional.
4. Transfer eval re-run once v3 labels exist (the LSAT verdict is currently
   untrustworthy for the same truncation reason).

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
- `shared/icr_capture/` — captures (v1/v2 only; no v3 yet)

**Never edit a `.bib` file directly.** Cite in prose with enough context for a
human to verify, and get explicit approval before any bibliography insertion.

---

## Contacts

- **User:** Hong Yang (hooong.yang@gmail.com)
- **Paper draft:** `thinking-gating/paper/`

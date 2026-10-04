# thinking-gating

Can a model tell, before it spends the compute, whether thinking will help?

> **Status (2026-10-04).** The prefill-probe line is **closed as a negative
> result**: [paper/negative_result.md](paper/negative_result.md), with an
> errata section at the top from the 2026-10-04 cleanup audit. The repo is
> pivoting to **System One decision models** — whether a single-pass decision
> model can estimate the value of computation better than its own confidence:
> GitHub issue #1 and
> [paper/proposal_system_one_gating.md](paper/proposal_system_one_gating.md).
> The capture, grading, baseline, statistics and dispatch tooling carry over
> (fixed in the cleanup); the probe becomes one baseline among several.
> Operating rules for agents: [AGENTS.md](AGENTS.md).

## What the old line found

A probe on the prefill hidden state (last prompt token) predicted
`needs_thinking` (wrong without thinking) at AUROC 0.66–0.70 on Qwen3-8B, but a
question-length or TF-IDF baseline landed inside every interval; `rescued`
(does thinking fix a query the model gets wrong) was at chance; BBH was a
subtask detector. An earlier positive result was a truncation confound (below).
See the writeup and its errata for the numbers and what the audit changed.

The three objectives (`--target` on `run_experiment.py`):

| target | label | what above-chance means |
|---|---|---|
| `needs_thinking` | `~correct_off` | the model will be wrong without thinking — correctness prediction, well studied |
| `helped` | `~correct_off & correct_on` | thinking flipped wrong → right; mostly driven by the first term |
| `rescued` | `correct_on` on rows with `correct_off == False` | the marginal value of reasoning with difficulty held fixed by construction |

## Setup

The repo is self-contained: its own venv, task modules, and dispatch tooling.

```bash
bash scripts/setup_env.sh      # creates ./.venv and installs requirements.txt
source .venv/bin/activate
```

The same works on Empire AI; see
[.agent-work/EMPIRE_AI_SETUP.md](.agent-work/EMPIRE_AI_SETUP.md).
`transformers >= 4.51, < 5` is pinned (Qwen3 and the `enable_thinking`
template flag). The System One work needs Qwen3.5 / vLLM / peft and will get a
separate environment rather than an in-place upgrade.

## Pipeline

```bash
# 1. capture paired thinking-off/on runs + prefill states (GPU node only)
python scripts/capture_inference_thinking.py --task math500 --model Qwen/Qwen3-8B \
    --chat-template --max-response-len 2048 --max-response-len-thinking 4096 \
    --capture-logprobs --out-dir shared/icr_capture/math500_thinking_qwen3v3

# 2. labels (CPU); --regrade re-applies the current grader to stored responses
python scripts/generate_labels.py --regrade \
    --capture-dir shared/icr_capture/math500_thinking_qwen3v3 \
    --out-file shared/labels/qwen3v3/math500_labels.jsonl

# 3. probe + baselines + controls + transfer, for every task and objective
CAPTURE_SLUG=qwen3v3 TASKS="math500" bash scripts/run_full_analysis.sh
```

`--max-response-len` is required: its old default (320) produced the
truncation confound. Above 20% thinking-OFF truncation the capture
quarantines the shard (`meta.shardNN.jsonl.quarantined` +
`TRUNCATION_FAILURE.shardNN.json`) and exits 1. Known-good budgets are in
`configs/dispatch/capture_qwen3v3.json` and the `*_redo.json` manifests.

`run_full_analysis.sh` is CPU-only and incremental (a step re-runs when an
input is newer than its output), regrades labels by default (`REGRADE=0` keeps
stored grades), writes only to `output/<slug>/`, and copies into
`paper/results/metrics/<slug>/` only with `PROMOTE=1` (refusing to change an
existing file unless `FORCE=1`). Individual steps: `run_experiment.py`,
`baseline_text.py`, `baseline_confidence.py`, `compare_baselines.py`,
`within_group_auroc.py --group-key ...`, `eval_transfer.py`,
`results_table.py`; real invocations are in AGENTS.md.

## How to read a result

- **Quote `aggregate.test_auroc_bootstrap.ci`, never `test_auroc.ci`.** The
  latter is seed-to-seed spread on a fixed sample and runs 1.6–8.8× too
  narrow; one finding ("12/14 MMLU-Pro categories above chance") became 5/14
  under the correct interval.
- **Beat the text and confidence baselines** on identical rows.
  `compare_baselines.py` selects each baseline family's best on validation and
  reports a paired-bootstrap probe − baseline difference.
- **Watch for category identity.** On multi-category data use the pooled
  within-group AUROC (`within_group_auroc.py --group-key`); BBH looked like
  0.82 and was 0.49 within subtask.
- **A router is only interesting between max(never, always) and oracle.**
  `run_experiment.py` reports the exact routed-accuracy curve, its nAUC, and
  random and oracle routers at the matched escalation rate.

## The truncation confound

Captures before 2026-08-30 ran the thinking-OFF pass at 320 tokens. Qwen3
writes chain-of-thought even with thinking off, so long answers were cut off
and graded wrong for running long.

| task | off-pass truncated | `correct_off` when truncated | when not |
|---|---|---|---|
| MATH-500 | 75.2% | 0.037 | 0.944 |
| MMLU-Pro | 49.3% | 0.061 | 0.700 |
| GSM8K | 11.6% | 0.196 | 0.943 |

The identical probe trained to predict `truncated_off` scored 0.922 / 0.872 /
0.811 — higher than the same probe on `needs_thinking` (0.879 / 0.782 /
0.702). Three model families shared the cap, so the cross-family replication
reproduced the artifact. Full writeup:
[paper/results/metrics/truncation/README.md](paper/results/metrics/truncation/README.md).

## Repository layout

```
thinking-gating/
├── AGENTS.md / CLAUDE.md               # agent rules (CLAUDE.md imports AGENTS.md)
├── scripts/
│   ├── setup_env.sh                    # creates ./.venv, installs requirements
│   ├── capture_inference_thinking.py   # paired thinking-off/on + prefill extraction
│   ├── generate_labels.py              # captures → labels (--regrade, --drop-truncated)
│   ├── run_experiment.py               # prefill probe (logreg / MLP), 5 seeds
│   ├── baseline_text.py                # length / TF-IDF baselines
│   ├── baseline_confidence.py          # thinking-off log-prob / entropy / length baselines
│   ├── compare_baselines.py            # probe vs val-selected baselines, paired bootstrap
│   ├── within_group_auroc.py           # pooled within-group AUROC (category control)
│   ├── stratify_check.py               # per-group AUROC (archival; decomposition/ reproduces with it)
│   ├── eval_transfer.py                # cross-task transfer, no retraining
│   ├── results_table.py                # metrics dir → one table
│   ├── run_full_analysis.sh            # captures in, results out
│   ├── gpu_dispatch.py                 # GPU job dispatch through Jupyter kernels (Empire AI)
│   ├── launch_jupyter.py               # guarded Jupyter/SLURM launcher
│   ├── watch_and_dispatch.py           # puts workers on queued roots as allocations land
│   └── dispatch/                       # cell + worker queue (all fan-out work)
├── tasks/                              # gsm8k, lsat, math500, mmlu_pro, bbh (loaders + graders)
├── utils/                              # capture_io.py (shard-aware), metrics.py, jupyter_exec.py
├── tests/                              # dispatch, pipeline, graders, metrics, capture, confidence
├── configs/
│   ├── dispatch/                       # one manifest per sweep or capture batch
│   └── nodes.example.json              # template for gitignored configs/nodes.json
├── shared/                             # captures + labels (gitignored)
├── output/                             # working metrics (gitignored)
└── paper/                              # writeups; paper/results/ is the provenance record
```

## Tests

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m pytest tests -q   # needs numpy, scikit-learn, torch, transformers
python3 tests/test_dispatch.py                          # stdlib only, no GPU
```

No test needs a GPU. The capture and confidence tests run a tiny random
Llama on CPU; the MATH-500 grader cases are also checked against
`math_verify` when it is installed (skipped otherwise).

## Tasks

Contract (`tasks/__init__.py`): `load_<task>(split)`, `format_prompt(question)`,
`is_correct(generation, answer[, question=])`.

| Task | Source dataset | Rows | Role |
|------|----------------|------|------|
| `gsm8k` | `openai/gsm8k` (main) | 1319 test | Grade-school math |
| `math500` | `HuggingFaceH4/MATH-500` | 500 test | Competition math |
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro` | 1000 sampled | Multi-domain MC |
| `bbh` | `lukaemon/bbh` | 27 subtasks × 20 | Reasoning suite |
| `lsat` | `hails/agieval-lsat-ar` | 230 test | Analytical reasoning |

Only MATH-500 ships a difficulty field (its 1–5 level, mapped onto three
buckets); the others derive one heuristically, used only for stratified
evaluation.

## Running sweeps

Anything that fans out goes through the cell queue in
[scripts/dispatch/](scripts/dispatch/): a manifest expands into cells, and a
generic worker on each node claims them by atomic `rename(2)`.

```bash
python scripts/dispatch/queue.py expand configs/dispatch/example_probe_sweep.json --dry-run
python scripts/dispatch/queue.py expand configs/dispatch/example_probe_sweep.json \
    --root shared/dispatch/my_sweep
python scripts/gpu_dispatch.py run \
    ".venv/bin/python scripts/dispatch/worker.py --root shared/dispatch/my_sweep"   # one per node
python scripts/dispatch/queue.py status --root shared/dispatch/my_sweep
```

Expanding is idempotent; re-expanding an edited manifest is refused unless it
sets `"cell_id_hash": true` (new work should); retired cells are not
recreated (`--allow-resurrect` overrides). The old v3 capture manifests
(`capture_qwen3v3.json`, `capture_nemotronv3.json`) are archival — do not
re-expand them. A failure that leaves a capture `TRUNCATION_FAILURE` marker is
terminal and not retried.

## Where the numbers live

`paper/results/` is the provenance record: metrics JSON copied verbatim from
runs, one file per run, with a README per group. A number in a writeup traces
to a file there, never to a log scroll. Read the group READMEs before quoting
anything — the caveats are the load-bearing part.

Handoff and cluster state: [.agent-work/HANDOFF.md](.agent-work/HANDOFF.md).

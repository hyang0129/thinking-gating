# thinking-gating

Can a model tell, before it starts writing, whether thinking will help?

This repo trains a small probe on **prefill activations** — the last-prompt-token
hidden state, available before a single output token is generated — to predict
whether extended reasoning will improve the answer. If it works, it is a router:
spend thinking compute only where it buys something.

> **⚠️ Status (2026-08-31): all probe results before 2026-08-30 are retracted.**
> The thinking-OFF pass was truncated, so `correct_off` measured response length
> rather than capability and both objectives inherited the confound. See
> [The truncation confound](#the-truncation-confound). V3 re-captures are queued
> on Empire AI and waiting on a GPU allocation. The pipeline and tooling are
> unaffected; the numbers need re-running.

## The three objectives

`--target` on `run_experiment.py`. They are not interchangeable — the choice is
the experiment.

| target | label | what above-chance means |
|---|---|---|
| `needs_thinking` | `~correct_off` | the model will be **wrong without thinking**. This is correctness prediction — well-studied territory (Kadavath 2022, Azaria & Mitchell 2023). Better balanced than `helped`, and independent of the thinking budget. |
| `helped` | `~correct_off & correct_on` | thinking **flipped** the answer wrong→right. The original framing. Confounded: mostly driven by the `~correct_off` term. |
| `rescued` | `correct_on`, restricted to rows where `correct_off == False` | **the load-bearing one.** On this subset `needs_thinking` is constant by construction, so difficulty cannot explain any signal. Above chance here means the prefill state encodes the *marginal value of reasoning* — the only claim here that is not already in the literature. |

## Setup

The repo is self-contained: its own venv, task modules, and dispatch tooling.
Nothing is imported from a sibling checkout.

```bash
bash scripts/setup_env.sh      # creates ./.venv and installs requirements.txt
source .venv/bin/activate
```

The same two commands work on Empire AI. See
[.agent-work/EMPIRE_AI_SETUP.md](.agent-work/EMPIRE_AI_SETUP.md) for cluster
dispatch, and [agent.md](agent.md) for the operating rules — which machine may
run what, and the pitfalls that have already cost a round of results.

`transformers >= 4.51` is a hard floor: Qwen3 support and the `enable_thinking`
chat-template flag both landed there.

## Quick start

### 1. Capture thinking-mode pairs (GPU node)

```bash
python scripts/capture_inference_thinking.py \
    --task math500 \
    --model Qwen/Qwen3-8B \
    --out-dir shared/icr_capture/math500_thinking_qwen3v3 \
    --max-samples 500 \
    --max-response-len 2048 \
    --chat-template
```

**Set `--max-response-len` deliberately.** Its default (320) is what produced
the truncation confound; non-thinking modes still write chain-of-thought, and a
competition-math solution does not fit. The script logs at ERROR level when
thinking-OFF truncation exceeds 20% — but it still exits 0, so **check the log,
it will not fail the run for you.** Budgets that are known to work are in
`configs/dispatch/capture_qwen3v3.json`.

### 2. Generate labels (CPU)

```bash
python scripts/generate_labels.py \
    --capture-dir shared/icr_capture/math500_thinking_qwen3v3 \
    --out-file shared/math500_labels.jsonl
```

### 3. Train the probe (CPU is fine)

```bash
python scripts/run_experiment.py \
    --capture-dir shared/icr_capture/math500_thinking_qwen3v3 \
    --labels shared/math500_labels.jsonl \
    --target rescued --method mlp --layer 18 --seeds 42 1 2 3 4 \
    --out-dir output/math500_rescued
```

Layer 18 is the a-priori middle layer, chosen once and never swept — a
`--layer-sweep` selected on a ~74-example validation split *lowered* test AUROC.
`--layer-stride 8` concatenates every 8th layer, which is the one tuning change
that reliably helped, and only on pooled data.

### 4. Check it before believing it

```bash
python scripts/baseline_text.py  --capture-dir ... --labels ... --target rescued   # beat TF-IDF or it is not a result
python scripts/stratify_check.py --capture-dir ... --labels ... --group-from ...   # is it a difficulty/subtask detector?
python scripts/validate_bench.py                                                   # trained on a partial capture?
python scripts/eval_transfer.py  --probe output/.../checkpoint.json --capture-dir ...
```

Or run the whole thing: `bash scripts/run_full_analysis.sh` labels every
capture, trains a probe per (task × objective), evaluates every ordered
cross-task pair, and renders one table. Idempotent; CPU only; `FORCE=1` to
recompute.

## How to read a result

- **Quote `aggregate.test_auroc_bootstrap.ci`, never `test_auroc.ci`.** The
  latter is a normal approximation over 5 seeds that re-split a *fixed* sample —
  it measures split-to-split spread, not population uncertainty, and runs
  1.6–8.8× too narrow. One finding ("12/14 MMLU-Pro categories above chance")
  became 5/14 under the correct interval.
- **A probe is only interesting strictly between max(never-think, always-think)
  and oracle.** On these tasks always-think already lands within a couple of
  points of oracle, which is why `min_routed_for_always_think_accuracy` — the
  smallest fraction of queries that must be routed to thinking to match
  always-think accuracy — matters more than the accuracy framing. 1 − that is
  wasted thinking compute.
- **Beat the text baselines.** An 8B forward pass has to outperform TF-IDF on
  the raw question, on identical splits, seeds, and target.
- **Watch for task identity.** Pooled `needs_thinking` has base rates 0.144 /
  0.738 / 0.615 across tasks, so recognising *which task a question came from*
  predicts the label. That is vocabulary matching, not difficulty estimation.

## The truncation confound

Captures before 2026-08-30 ran the thinking-OFF pass at `--max-response-len 320`.
Qwen3 writes chain-of-thought even with thinking off, so long answers were cut
off mid-solution and graded wrong for running long rather than for being unable.

| task | off-pass truncated | `correct_off` when truncated | when not |
|---|---|---|---|
| MATH-500 | 75.2% | 0.037 | 0.944 |
| MMLU-Pro | 49.3% | 0.061 | 0.700 |
| GSM8K | 11.6% | 0.196 | 0.943 |

The decisive test — train the identical probe to predict `truncated_off` instead
of the label — gives **0.922 / 0.872 / 0.811**, higher than the same probe
predicting `needs_thinking` (0.879 / 0.782 / 0.702) on every task, and the
reported AUROC ranks the three tasks in exactly the order of their truncation
rates. The probe was substantially a response-length predictor.

Three model families shared the cap, so the cross-family replication reproduced
the artifact rather than confirming the result: a confound in the design is
invariant to the model. There is no salvage from the existing data — dropping
truncated rows leaves MATH-500 with 124 rows of which 7 are wrong.

Full writeup: [paper/results/metrics/truncation/README.md](paper/results/metrics/truncation/README.md).

## Repository layout

```
thinking-gating/
├── scripts/
│   ├── setup_env.sh                    # creates ./.venv, installs requirements
│   ├── capture_inference_thinking.py   # paired thinking-off/on + prefill extraction
│   ├── generate_labels.py              # paired runs → labels (--drop-truncated)
│   ├── run_experiment.py               # probe training (MLP / logreg), 5 seeds
│   ├── eval_transfer.py                # cross-task transfer, no retraining
│   ├── baseline_text.py                # TF-IDF / text-feature baselines
│   ├── stratify_check.py               # the control that caught the BBH artifact
│   ├── validate_bench.py               # flags results trained on partial captures
│   ├── results_table.py                # metrics dir → one table
│   ├── run_full_analysis.sh            # captures in, results table out
│   ├── gpu_dispatch.py                 # multi-node GPU job dispatch (Empire AI)
│   ├── launch_jupyter.py               # guarded Jupyter/SLURM launcher
│   └── dispatch/                       # cell + worker queue (all fan-out work)
├── tasks/                              # gsm8k, lsat, math500, mmlu_pro, bbh
├── utils/                              # capture_io.py (shard-aware), jupyter_exec.py
├── tests/                              # test_dispatch.py, test_pipeline.py
├── configs/
│   ├── datasets/  methods/             # dataset + probe configs
│   ├── dispatch/                       # one manifest per sweep or capture batch
│   └── nodes.example.json              # template for gitignored configs/nodes.json
├── shared/                             # captures + labels (gitignored)
├── output/                             # working metrics
└── paper/results/                      # promoted metrics — provenance record
```

Tests are stdlib-only and need no GPU: `python3 tests/test_dispatch.py`,
`python3 tests/test_pipeline.py`.

## Tasks

Task modules are local to this repo and follow one contract (`tasks/__init__.py`):
`load_<task>(split)`, `format_prompt(question)`, `is_correct(generation, answer)`,
`difficulty(row)`.

| Task | Source dataset | Rows | Role |
|------|----------------|------|------|
| `gsm8k` | `openai/gsm8k` (main) | 1319 test | Grade-school math |
| `math500` | `HuggingFaceH4/MATH-500` | 500 test | Competition math — primary for `rescued` |
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro` | 1000 sampled | Multi-domain MC |
| `bbh` | `lukaemon/bbh` | multi-subtask | Reasoning suite |
| `lsat` | `hails/agieval-lsat-ar` | 230 test | Analytical reasoning / transfer |

Only MATH-500 ships a difficulty field (its 1–5 level, mapped onto the shared
three buckets). The others derive one heuristically — reasoning-step count for
GSM8K, constraint-sentence count for LSAT, prompt length for MMLU-Pro and BBH —
used **only** for stratified evaluation, never for training.

## Running sweeps

Anything that fans out — multi-seed, multi-method, per-dataset batches, whole
capture campaigns — goes through the cell queue in
[scripts/dispatch/](scripts/dispatch/). Workers on any number of nodes claim
cells from a shared directory via atomic `rename(2)`.

**The worker is generic and never changes.** A cell describes its own work, so a
new sweep is a new manifest: a `python_script` to run, a `python_code` snippet,
a `call` to any importable function, or a `shell` command.

```bash
python scripts/dispatch/queue.py expand configs/dispatch/capture_qwen3v3.json --dry-run
python scripts/dispatch/queue.py expand configs/dispatch/capture_qwen3v3.json
python scripts/gpu_dispatch.py run .venv/bin/python scripts/dispatch/worker.py \
    --root shared/dispatch/capture_qwen3v3          # one per node
python scripts/dispatch/queue.py status --root shared/dispatch/capture_qwen3v3
```

Expanding is idempotent — finished cells are skipped, so re-expanding after
adding a task never repeats work. Cells retry (`max_attempts`), time out, and
survive node death (stale claims are re-queued).

## Models exercised

Qwen3-8B (anchor), Qwen3-14B, Nemotron-Nano-8B, Granite-3.3, gpt-oss-20b. The
thinking toggle is detected from the chat template rather than a name whitelist,
covering `enable_thinking`, system-prompt toggles, and graded reasoning levels.

## Where the numbers live

`paper/results/` is the provenance record: metrics JSON copied verbatim from
cluster runs, one file per run. **A number in the paper traces to a file there,
never to a log scroll.** Read the group READMEs before quoting anything — the
caveats are the load-bearing part:

- [`truncation/`](paper/results/metrics/truncation/) — what invalidated the pre-08-30 results
- [`baselines/`](paper/results/metrics/baselines/) — what prefill buys over reading the question
- [`decomposition/`](paper/results/metrics/decomposition/) — the three objectives, and which interval to quote
- [`tuning/`](paper/results/metrics/tuning/) — sample size is the binding constraint, not capacity

Current state and next steps: [.agent-work/HANDOFF.md](.agent-work/HANDOFF.md).

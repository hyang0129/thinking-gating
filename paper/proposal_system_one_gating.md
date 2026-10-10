# Proposal: Single-pass models can estimate the value of computation they cannot perform

> **Superseded 2026-10-04 by [`proposal_bolt_on_system1.md`](proposal_bolt_on_system1.md)** (v3: bolt-on System 1 via activated-adapter
> decision heads). Kept because the negative-result errata cite §8, and v3 reuses its gates (§3).

**Status:** research plan, 2026-10-04 (v2, revised after an internal
adversarial review). Supersedes the prefill-probe line
(`paper/negative_result.md`). ✓ = checked against the primary source;
everything else is from scout reports and must be re-verified before citing.

## 0. What "JEV" means

**Jev** is TypeSafe AI's "System One model" (early access 2026-09-15 ✓). It takes
text plus *typed* questions and returns structured values with calibrated
probabilities — Choice, Score, and Noul (yes/no). It writes no prose, claims
70–500 ms latency ✓, and its architecture is closed ("new model architecture,
parallel sampler, RLCD" ✓). Open models built the same way followed:

- **Kev** (0.5B–27B ✓, Qwen3.5-Base + LoRA + pointer head)
- **Jeeves-9B** (✓, thinks before deciding, confidence-gated)
- **Laya** (ModernBERT)
- **OpenJev-RLCD** (2609.38850 ✓)
- **LLM2Jev** (2610.02076 ✓), which shows an ordinary LLM's option-token
  probabilities already act as a decision model.

We study the **class**: one forward pass, then a distribution read off a
decide/option position. Jev is a pinned (`jev-1.13.0`), reported comparator.
No claim rests on it alone.

## 1. Literature position

| Hypothesis | Status | Key evidence |
|---|---|---|
| H1: decision models fail on problems that need thinking | Largely settled; becomes a **gate**, not a claim | Theory: 2310.07923, 2305.15408, 2402.12875. Jev: JEV-as-a-Judge 2609.26550 ✓ ("falls behind where the verdict must be derived, as in math, code, and logic"); Kev-4B model card names date arithmetic as its weakest family. |
| H2: predict whether thinking is required | Default everywhere is **own-confidence gating** (Jeeves `nothink_threshold` ✓, JEV-as-a-Judge, COREA 2603.03752, GLiDE ✓ closed). No one tests a learned signal against it for decision models. | Adjacent: Self-Route 2505.20664 ✓ (hidden-state router), Think When Needed 2601.18146 ✓ (ranking domain, no confidence baseline), 2608.20256 (GRPO mode choice, no confidence baseline) |
| H3: System 1 speed + System 2 reliability | Shipped as products (GLiDE; Jeeves gated: 0.806 @ 2.0 s vs 0.825 @ 3.3 s ✓). No matched, wall-clock, reasoning-heavy evaluation exists. | Router plateaus: 2606.07587, 2601.07206 |

**The risk inherited from our last project.** Thinking-off confidence beat our
learned predictor (math500 0.814 vs 0.699). "Thinking rescues it" was at chance
when predicted from the question alone. Decision models ship *calibrated*
confidence, so that baseline is now stronger. The whole design is built around
it.

## 2. The thesis: value of computation, not "need to think"

A calibrated decision model's max-probability estimates
**P(decision wrong | x)**. Escalation should rank by **expected gain**:

  E[gain | x] = P(reasoner right | x) − P(decision right | x) − λ·cost.

Calibration says nothing about P(reasoner right | x). A p ≈ 0.5 output pools
two kinds of uncertainty that thinking treats differently:

- **Irreducible uncertainty** — ambiguous items, unknown facts, label noise.
  Thinking does not fix it.
- **Computational uncertainty** — the answer follows from the input but needs
  serial depth beyond one pass. Thinking fixes it.

**Proposition (to state and prove).** Confidence gating is optimal iff
P(reasoner right | x) is constant on confidence level sets. The available
headroom is I(gain; x | conf).

**Why a single pass could carry the signal.** *Estimating* serial depth
(counting hops or nesting) is within reach of a fixed-depth pass even when
*computing* the answer is not. So a decision model may be able to say "this
needs computation" about items it cannot solve.

Precedent: Russell & Wefald's value of computation; rational metareasoning for
LLMs (2410.05563, verify).

Either outcome is reportable.

- **If no headroom:** "confidence suffices for decision models, and here is
  why" — the proposition plus measurement.
- **If headroom:** a value-of-computation flag that beats confidence.

The previous project could not tell these apart; this design separates them
by construction.

## 3. Pre-registered hypotheses and metrics

Commit this section, with its hash, **before any run**.

**G1 (precondition, H1).** Accuracy at matched length falls with serial depth
*d* for the single-pass readout, and is flat or much flatter with thinking on
the same weights.

*Contribution beyond replication:* **breakpoint scaling.** Does the *d\** where
accuracy reaches chance grow with model depth across Kev 0.5B → 27B?

*Architecture prediction:* Qwen3.5 is a hybrid of Gated DeltaNet and attention.
Linear-RNN layers can track some state, so families on either side of the
TC⁰/NC¹ line should break differently: counting/nesting vs permutation-style
state tracking. Treat this as a falsifiable prediction.

**H1-cal: does shipped calibration hold as depth grows?** Use shipped
calibration only. A depth-blind refit is secondary.

- *Primary metric — confident-error leakage:*
  - Fix τ on held-out *d* = 1 items for 95% accepted-precision.
  - CEL(*d*) = P(conf ≥ τ | wrong, *d*).
- *Secondary metric — signed overconfidence:* E[conf] − acc, per cell and per
  gold label.
- **Outcomes:**
  - *Collapse:* CEL(*d*max) ≤ 0.10 and the upper CI bound of overconfidence is
    ≤ 0.05.
  - *Confidently wrong:* CEL(*d*max) ≥ 0.25, or the lower CI bound of
    overconfidence is ≥ 0.10.
  - Otherwise: inconclusive, and reported as such.
- **Coherence checks:**
  - p(A) + p(¬A) ≈ 1 on negation pairs.
  - Yes-bias, which LLM2Jev documents on binary verification.

**G2: headroom test (decides whether a flag is worth building).**
- Fit a *privileged* router on confidence + true *d* + family label.
- Compare it with confidence alone on a fixed workload:
  - 50/50 shallow/deep;
  - plus 30% irreducible-uncertainty items.
- Metric: ΔnAUC, the normalized area under the routed-accuracy curve over 0–50%
  escalation, with a **paired** bootstrap.
- **Headroom exists iff** the lower CI bound of ΔnAUC is ≥ 0.03 **and**, within
  1 pp of always-think accuracy, escalations drop by ≥ 15% relative.
- If even privileged *d* cannot beat confidence, no learned flag will. Stop and
  write the analysis paper.

**G3: learned flag.**
- *What it is:* a second typed Noul question ("would working this out step by
  step change the answer?"), trained with Kev's native typed-question LoRA.
- *Targets:*
  - (a) `dec_wrong` — a **control**; it should tie confidence by construction;
  - (b) `gain` = `reasoner_right − dec_right`;
  - (c) a proper-scoring / RLCD-style reward on the joint (answer, flag) with an
    explicit λ.
- *Success:* the paired-bootstrap ΔnAUC (flag vs confidence) has a lower CI
  bound > 0
  - within family × depth;
  - and leave-one-family-out;
  - and on depth-annotated natural data.

**Headline experiment (2×2 at matched confidence).** Cross uncertainty source
(irreducible vs computational) with depth (shallow vs deep). Confidence cannot
separate the cells by construction; the flag must.

## 4. Design

### Pairing
- **Primary: one set of weights.** An LLM2Jev readout of a Qwen3.5 hybrid model
  with thinking off, vs the *same weights* with thinking on. This makes "gain"
  a property of compute alone.
- **Secondary: Kev.** Kev is the *Base* model plus LoRA, so its reasoner is a
  different post-trained model. Report it, but don't call it same-backbone.
- **Jev:** black-box comparator. Check API cost and rate limits first.

### Data

**Natural, depth-annotated — carries the claim:**
- StrategyQA, by decomposition length (perturbed variants for contamination);
- MuSiQue 2–4 hop;
- GSM8K-Verify, by solution step count;
- FOLIO, by proof depth;
- MATH500-verify;
- BBH binary subtasks, **after the grader fix**.

**Synthetic — controlled depth:**
- Families: boolean nesting, ProntoQA/ProofWriter hops, web-of-lies chains,
  arithmetic-verify. Generators remove the sample-size ceiling that bound the
  last project (`rescued` n = 89–376).
- **Length controls:** distractor clauses/rules so token count does not reveal
  *d*, plus minimal pairs that differ in *d* at equal length.
- **Answer perturbations:** in "Is the answer X?" items, control the
  perturbation distribution so magnitude or parity cues can't refute X without
  computing.

**Irreducible-uncertainty items:**
- ambiguous or underspecified questions;
- obscure-fact True/False;
- label-noise subsets.

**Contamination:**
- BoolQ is in Kev's training set ✓, so drop it as a control.
- Use fresh and perturbed items for every H1/G1 claim.

### Labels
- Decision answer and probability.
- Reasoner correctness at a budget with **<2% truncation** — measured, not
  assumed. The capture default `--max-response-len 320` is still a trap.
- `dec_wrong`, `gain`, `hurt`, *d*, family, and uncertainty source.

### Baselines
1. **Bounds:** always-S1, always-S2, random at matched rate, oracle.
2. **Own max-probability / entropy** — the incumbent.
3. **DART two-sample agreement** (2606.23181).
4. **Text routers:** length + TF-IDF LR; kNN on embeddings (2505.12601);
   ModernBERT.
5. **Regex depth counter**, and the privileged-*d* router.
6. **Prefill hidden-state probe** (Self-Route-style) — the existing pipeline.
7. **Task-family oracle router** — the BBH lesson.
8. **Matched autoregressive cascade:** same size, confidence deferral
   (COREA/AutoMix).

### Metrics
- Routed-accuracy curves and nAUC (paired bootstrap).
- Tokens **and** wall-clock on one H100 / vLLM.
- **Pooled within-family-within-depth AUROC.**
- Leave-one-family-out transfer.
- Reliability diagrams by *d*.

**KV reuse on escalation:** a stretch goal only. Gated DeltaNet recurrent
state complicates prefix caching, so verify vLLM support before claiming it.

## 5. Plan, gates, timeline

ICML 2027 is likely late January (**verify**), about 16 weeks away.

| Weeks | Work | Exit |
|---|---|---|
| 1 | **Pre-register:** commit §3 with hash. Fix BBH grader. Pin models, Jev API cost check. | — |
| 2–4 | **Phase 0:** generators with length controls; natural depth annotations; readout and thinking captures for the primary pairing plus Kev 0.5B–27B (~500 items per family×d cell); Jev on a subset. | **G1** fails → premise wrong, stop. Then H1-cal and **G2**. |
| 4 | **Decision point.** | **G2** fails → analysis paper: proposition + H1-cal + breakpoint scaling; target ICML or ARR. **G2** passes → continue. |
| 5–8 | **Phase 1:** flag training (targets a/b/c), the 2×2, leave-one-family-out, natural-data transfer. | **G3**. |
| 9–11 | **Phase 2:** mixed workloads (hard fraction 10–70%), wall-clock Pareto vs all baselines. | — |
| 12–15 | Write; arXiv. | — |

**Preprint policy.** Post early only if the preprint contains the
value-of-computation framing and the 2×2. An "H1 confirmed" preprint gives away
the generators and stakes nothing. The field is three weeks old and has about
ten papers, so speed matters, but not at the cost of the core idea.

Allow slack for the cluster queue: allocations sat PENDING for days during the
last project.

## 6. What carries over from this repo

**Reused:**
- paired off/on capture;
- the truncation quarantine;
- task-module contract;
- text + confidence baselines;
- `within_group_auroc.py`;
- routed-accuracy curve (`run_experiment.py:216-262`);
- dispatch queue;
- bootstrap-CI discipline.

**Fixed in cleanup (2026-10-04, branch `cleanup/system-one-carryover`):**
- `tasks/bbh.py` exact-matched without stripping the letter or parentheses
  (`(B) No` vs gold `No` and `(Yes)` vs gold `Yes` graded wrong; the prompt
  asked for `(X)` letters on non-lettered subtasks). Fixed with the MATH-500,
  MMLU-Pro and GSM8K grader bugs in 7e224b8; bbh graders receive the question
  (f2eabef); `generate_labels.py --regrade` re-applies the current grader to
  stored responses (9e24091) and `run_full_analysis.sh` regrades by default
  (8de124f). On Qwen3 BBH, thinking-off accuracy moves 0.676 -> 0.761.
- Capture: confidence scored with left padding attended to (B1, invalidates
  every published confidence baseline), required `--max-response-len`,
  prompt truncation, per-shard config, unclosed `<think>` graded wrong,
  reasoning-trace rate logged (8404347).
- Analysis: statistics hoisted into `utils/metrics.py` with paired bootstrap,
  exact routed curve and nAUC, explicit within-group keys, bootstrap transfer
  verdicts, val-selected baselines (d0fa2de); `run_full_analysis.sh` fails
  fast and promotes into `paper/results/` only on request (52fa811); tests
  (2134d8b).
- Dispatch: continuous watcher, heartbeat liveness, no resurrection of
  retired cells, terminal truncation failures, commit recorded per attempt
  (a437072).
- The corrections to `negative_result.md` are in its "Errata (2026-10-04)"
  section.

**New code:**
- generators with length controls;
- decision-model readout (LLM2Jev/Kev);
- Jev client;
- typed-question LoRA flag training;
- wall-clock harness.

## 7. Risks

- **G2 fails** (confidence suffices). Plausible given our prior result. It is
  pre-registered as a publishable outcome, not a failure.
- **The flag is a depth or length counter.** Regex-*d* and length-matched
  controls catch this. Natural data carries the claim.
- **Scoop.** Watch arXiv weekly for Jev + calibration + depth, and Jev +
  learned escalation.
- **Source quality.** The Jev literature is weeks-old preprints and vendor
  blogs. Re-verify every citation, and add none to a `.bib` without human
  approval.

## 8. Correction for the closed project

"No evaluated method uses target-model prefill state for thinking-mode
routing" is false. Self-Route (2505.20664 ✓, May 2025) routes think/no-think
from hidden states, and pre-generation correctness probes exist (2602.09924,
2603.20895; unverified). Drop that framing from `negative_result.md` and
`CLAUDE.md`. (Done 2026-10-04: errata in `negative_result.md`; the agent
instructions now live in `AGENTS.md` and omit it.)

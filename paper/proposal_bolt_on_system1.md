# Proposal: Bolt-on System 1 — activated-adapter decision heads for reasoning models

**Status:** research plan v4, 2026-10-10. v4 makes C3 the headline after the
2026-10-10 scoop check, and replaces §7 Phase 0 with the decision-head plan in
issue #3. v3 (2026-10-04) replaced v2 ("value of computation"). Tracked in
issue #1. ✓ = checked against the primary source; everything else
is from scout reports and must be re-verified before citing.

## 1. Pitch

We want **Jev speed with System 2 reliability**, from one model.

**Jev-style decision models** answer typed questions with calibrated
probabilities in a single forward pass. They are fast (Jev claims 70–500 ms ✓),
but they cannot do chain-of-thought (CoT). Their own vendor and independent
work show they fall behind where a verdict must be *derived* (math, code,
logic) ✓.

**Reasoning models** can derive the answer, but at 10–100× the latency.

**Our model:** take an existing hybrid-thinking reasoning model and add a
decision head with **Activated LoRA (aLoRA)**: an adapter that is active only
on an appended typed question.

| Step | What runs | Cost |
|---|---|---|
| 1. Prompt | The frozen base model processes the prompt **once**. | Prefill only |
| 2. Decide (System 1) | Append the typed question ("True or False?" / "Would working this out change your answer?"). The adapter turns on for those tokens only. Read the answer and an escalate flag at the decide position. | One prefill-length pass, Jev speed |
| 3. Escalate (System 2) | If escalating: drop the adapter, switch thinking on, continue from the **same cached prompt computation**. | Thinking tokens only; the reasoner is untouched |

## 2. Contributions

**Order changed in v4.** C1's mechanism has shipped: DoccyHealth Solomon (HF,
2026-09-21 ✓) and AutoTrust GEV-26B-Decide (HF, Oct 2026 ✓) both bolt a LoRA
decision head onto a reasoner and fall back to the unmodified base model's
thinking. **C3 is the headline**; C1 and C2 are the platform, and cite both as
prior art for the mechanism. Every system found gates on its own confidence.

**Headline — C3. Learned think-flag vs the model's own confidence.** Item 3
below; pre-registered endpoint in issue #3 §8.

**Platform:**

1. **Method (C1): a bolt-on System 1 for any reasoning model.** What remains
   ours: exact, measured cache reuse on a hybrid-recurrent model (Qwen3.5
   Gated DeltaNet, snapshot and restore of the recurrent state), and an aLoRA
   vs full-LoRA quality ablation for decision heads.
   - One set of base weights serves both modes.
   - Escalation is free of re-prefill, because the cache is reused exactly
     (aLoRA leaves prompt states at base values; on Qwen3.5 the GDN state
     must also be snapshotted and restored).
   - The System 2 fallback **is** the original reasoner, with no reasoning
     degradation from decision tuning.
2. **Evaluation (C2): an open, reasoning-heavy, wall-clock comparison of
   decision models with CoT fallback.** Partly done by GEV (GPQA-D, knowledge
   mix) and InnerJev / JEVal (2610.03935).
   - Systems compared: Jev (closed comparator), Kev, Jeeves gated, a
     conventional cascade, always-think, and ours.
   - Workloads: math, logic and agent mixes.
3. **Headline (C3): when to escalate.**
   - Our own model's confidence vs a learned think-flag, across the design
     ladder (§4), with the old prefill-probe result as the control condition.
   - A null ("confidence suffices, and here is why") is still a finding; 1 and 2
     stand without it.

## 3. Prior art and how we differ

| Work | What it is | Our difference |
|---|---|---|
| **DoccyHealth Solomon** (HF, 2026-09-21 ✓) | LoRA on Qwen3.8-27B, off on the document and on from the question; escalates when reliability < threshold to `<think>` generated with the adapter disabled from a deep-copied cached prefix | **Takes C1's mechanism.** Gates on confidence; model card and code only, no paper, no Pareto |
| **AutoTrust GEV-26B-Decide** (HF, Oct 2026 ✓) | LoRA + 24-slot head on Gemma-4-26B-A4B; below 0.8 escalates to the unmodified base model's thinking, p = ½p1 + ½p2 | Fallback taken; gates on confidence; no KV reuse documented. Its gate is our baseline |
| Qualcomm "Efficient Reasoning on the Edge" (2603.16867) | Masked LoRA off on the prompt; a prefill MLP decides whether to think | Not compared against confidence |
| InnerJev / JEVal (2610.03935) | Accuracy-latency Pareto of open decision models + Jev | Distils reasoning instead of escalating |
| Jev (TypeSafe, 2026-09-15 ✓) | Closed System One model; no CoT | We add CoT fallback; open |
| Kev (✓ HF) | Open Jev-style: Qwen3.5-Base + LoRA + pointer head | No fallback; Base weights cannot think |
| **Jeeves-9B** (PostHog ✓) | Open Jev-style that can think before deciding; `nothink_threshold` 0.9 → 0.806 @ 2.0 s vs 0.825 @ 3.3 s full think, on 325 dev items from MMLU/SciQ/PAWS ✓ | Its reasoning runs through the decision-tuned LoRA (per README); ours runs on the untouched reasoner with exact cache reuse. Its evaluation is not reasoning-heavy. |
| **GLiDE** (Fastino, 2026-09-30 ✓, closed) | "First thinking decision model": fast distribution, reasons when uncertain | Closed, mechanism undisclosed, evaluated on its own vendor index. We are open, the mechanism is specified, and the evaluation is independent. |
| LLM2Jev (2610.02076 ✓) | Read option-token probabilities from any LLM | Training-free baseline (D0) |
| aLoRA (2504.12397 ✓ NeurIPS 2025); serving 2512.17910; tradeoffs 2609.17109 | Adapter active only after an invocation sequence → base KV reuse | We apply it to decision heads plus thinking fallback (verify nobody has) |
| Self-Route (2505.20664 ✓), our negative result | Frozen prefill-state routers | Design D1 in our ladder, i.e. the control |
| JEV-as-a-Judge, COREA, REFLEX | Confidence-gated cascades to a *different* model | Same weights; free escalation |

**Novelty check (done 2026-10-10, `.agent-work/research-2026-10-10-decision-head.md`):**
C1's mechanism is taken (Solomon, GEV); C2 is partial; C3 is open. Keep
watching arXiv weekly.

## 4. Design ladder

All four designs decide from the prompt alone, before thinking. They differ in
how the decision is computed.

| ID | Design | Cache reuse | Role |
|---|---|---|---|
| D0 | LLM2Jev readout, no training | exact | baseline |
| D1 | Frozen model + linear/MLP heads on the decide-token state | exact | control (≈ the old probe, plus an answer head) |
| **D2** | **aLoRA on the appended typed question + pointer head** | **exact** | **main model** |
| D3 | Full LoRA (Kev-style) + pointer head | no (escalation re-prefills) | capacity upper bound |

**Outputs:**
- `answer`: a typed Choice or True/False.
- `think`: P(the thinking answer differs from the fast answer and is correct).
- `ask`: missing information. Phase 2, agents only.

**Base model (decided v4):** Qwen3.5-4B to develop, Qwen3.5-9B to confirm
(size-matched to Jeeves), Qwen3-8B as fallback (attention-only, exact
`crop()`, size-matched to Kev-8B). PEFT ≥ 0.18 has aLoRA; vLLM refuses aLoRA
adapters, so serving is a custom HF loop. Qwen3.5's GDN recurrent state is
updated in place, so the loop snapshots and restores it (issue #3 §2).

**Typed-question format (v4 default):** LLM-as-Jev bracketed IDs, `[1] … [K]`
with `Best answer: [` prefilled and renormalised over legal suffixes, shared by
D0–D3. The think flag is a second typed question on a separate flag aLoRA.

## 5. Training

**1. System 2 labels (fixed, computed once).**
- Run the base model with thinking on and the adapter off, at a budget giving
  **<2% truncation** (measured).
- Record `s2_correct`. This is the expensive step, and the existing capture
  pipeline does it.

**2. Answer head.**
- SFT on gold answers, using a Jev-style mix (Kev/Jeeves public data) plus the
  task mix.
- Calibrate with temperature scaling, then a proper-scoring objective
  (OpenJev-RLCD style).

**3. Flag labels.**
- Use **cross-fitted** System 1 predictions: k-fold, so an item's S1
  correctness comes from a model that never trained on it. The flag target
  depends on S1's own errors, which shift as S1 trains.
- `gain = s2_correct − s1_correct`. Keep "hurt" (gain < 0) distinct.

**4. Flag head.**
- SFT on `gain > 0`.
- Optionally a joint utility reward, `correct − λ·escalated`, with λ swept.

## 6. Evaluation: what each claim needs

**C1 — System 2 fallback is unharmed.**
- *Measure:* accuracy of the escalated path on hard items vs (a) the untouched
  base model thinking and (b) Jeeves's think mode, at equal size where possible.
- *Pass:* we match (a) within CI, by construction, and beat or match (b).

**C1b — escalation is free.**
- *Measure:* wall-clock of escalated items vs (prefill + thinking) and vs
  re-prefill. One GPU, custom HF loop (vLLM refuses aLoRA), fast GDN kernels.
- *Pass:* overhead ≈ decide-suffix only.

**C2 — Pareto.**
- *Measure:* accuracy vs mean wall-clock, and vs tokens, on mixed workloads
  with the hard fraction swept 10–70%.
- *Systems:* always-S1 (ours), always-think, Jeeves gated / no-think / think,
  Kev + same-weights thinking cascade, a conventional small-model cascade,
  Jev on a subset.
- *Pass:* ours dominates Jeeves-gated somewhere material.

**C3 — flag vs confidence.**
- *Measure (primary, pre-registered in issue #3 §8):* paired-bootstrap ΔnAUC
  (routed accuracy over 0–50% escalation) of the D2 flag vs the best
  own-confidence router (p_max, margin, entropy; selected on validation), on
  identical eval rows, stratified by family.
- *Pass:* lower CI bound > 0 **and** the flag beats per-family τ (pooled
  within-family ΔnAUC lower bound > 0). *Null:* CI covers 0 with half-width
  ≤ 0.05. *Underpowered:* half-width > 0.05.
- Secondary: leave-one-family-out, fixed-budget accuracy at 10/20/30%, the S2
  label-noise ceiling (4 thinking samples per eval item), GEV-style gate, the
  same comparison at D1 and D3.
- Pre-registered gates G2/G3 from v2 still apply.

**Benchmarks.**

| Phase | Families |
|---|---|
| Phase 0 | MMLU-Pro (3k eval), all BBH subtasks as bracket IDs, GSM8K / MATH500 verify (controlled perturbations + own wrong answers), LSAT, decision-v7 test (Kev comparability) |
| Phase 1 | ProntoQA/boolean nesting with depth control (the 2×2) |
| Phase 2 | When2Call, BFCL `miss_*`, τ²-bench, for answer / think / ask |

**The gap that matters is System 1 (decision readout) vs System 2.** Measured
on Qwen3-8B D0 (issue #1): think − S1 = +17.8 pp on MMLU-Pro, +30.9 pp on BBH
lettered; S1 ECE 0.39–0.45; confidence routing needs 74–86% escalation to get
within 1 pp of think, the oracle 17–30%. The original framing follows.

Our existing captures compare thinking-off *generation* with thinking-on
generation: 0.801 vs 0.807 on regraded Qwen3-8B gsm8k + math500 + mmlu_pro.
Qwen3 thinking-off still writes visible chain-of-thought, so both are
generative, and that 0.6 pp is not the S1-vs-S2 gap. A single-pass readout
should sit well below both on derived answers (Jev 68.4% vs GPT-6 95.9% on
JudgeBench reasoning ✓). The larger that gap, the more the routing decision
is the contribution.

**Phase 0 step 1 was to measure it (done, D0, 2026-10).** Run a training-free D0 readout on the
items that already have thinking labels:
- mmlu_pro, BBH and LSAT natively, since they are already typed choices;
- gsm8k and math500 via a verify format ("is the answer X?").

This costs one forward pass per item and gives the per-family S1 / no-think /
think accuracy table that the rest of the evaluation is built on.

**Three tiers, one model.** From the same KV cache, escalation can go to:
1. non-thinking generation (visible CoT, hundreds of tokens);
2. full thinking (thousands of tokens).

So the router chooses among decision pass → no-think generation → thinking.
Our existing captures already label the no-think vs think pair.

**Controls carried over from v2:**
- length-matched depth;
- regex-depth and privileged-depth routers;
- task-family oracle;
- within-family AUROC;
- contamination (BoolQ is in Kev's training data ✓).

## 7. Plan

**Phase 0 (weeks 1–4): decision head + the C3 test.** Full plan, tests and
cost table: issue #3. In brief:
- **0a, CPU now:** C3 pre-check on the existing Qwen3-8B D0 readout: a
  cross-fitted frozen flag probe (D1) and a TF-IDF flag vs D0 p_max and
  per-family τ.
- **Platform:** custom HF loop with aLoRA, snapshot and restore of the GDN
  state, equality tests T1–T6 (CPU on tiny configs, then once on real
  checkpoints).
- **Data:** Kev decision-v7 plus our task mix (MMLU-Pro, BBH as bracket IDs,
  GSM8K/MATH verify, LSAT); eval pool ≥ 4k rows.
- **Labels:** S2 = base thinking, adapter off, < 2% truncation;
  cross-fitted S1 (5 folds); flag target `1[gain = +1]`, hurt kept.
- **Baselines:** best own-confidence router, GEV-style gate, per-family τ,
  TF-IDF, D0/D1, Kev-4B/8B, Jeeves-9B gated/no-think/think, D3 full LoRA.
- **Exit:** E1 cache reuse exact and escalation token-identical to vanilla
  thinking; E2 D2 ≥ Kev-4B and within 2 pp of D3; E3 the pre-registered C3
  endpoint reported (pass, null, or underpowered).
- **Deliverable:** C3 result on a working platform, open weights.

**Phase 1 (weeks 5–8): the ladder and the 2×2.**
- Flag at every rung D0–D3; three tiers (S1 → no-think → think); proper-score
  calibration (OpenJev-RLCD).
- The 2×2: computational vs irreducible uncertainty × shallow vs deep.
- G2/G3.

**Phase 2 (weeks 9–11):** agents (answer / think / ask) and the full C2
evaluation.

**Phase 3 (weeks 12–14):** write. Target ICML 2027 or ARR; **verify the
deadlines.**

## 8. Risks

- **Scoop.** Jeeves, GLiDE, Solomon and GEV shipped within a month; C1's
  mechanism is taken. C3 is the hedge. Watch arXiv weekly.
- **aLoRA capacity.** Adapting only the suffix may underfit compared with full
  LoRA. D3 bounds this; quantify the gap.
- **Unfair Jeeves comparison.** Different size or base would confound it.
  Match size and family where possible, and report Jeeves at its native
  settings.
- **Flag never beats confidence.** Likely, given our prior result. With C1's
  mechanism taken, a null C3 is the paper: "confidence suffices", with the
  ladder, the per-family τ baseline and the S2 label-noise ceiling as evidence.
  The endpoint is sized (eval ≥ 4k rows) so a null is a tight null.
- **Serving support.** vLLM refuses aLoRA, so wall-clock comes from a custom HF
  loop with the fast GDN kernels; numbers from the torch fallback are refused.

## 9. Repo carry-over

Merged in PR #2:
- the paired off/on capture, which gives S2 labels;
- graders, with `--regrade`;
- `utils/metrics.py` (paired bootstrap, nAUC, CEL);
- baselines;
- the dispatch queue.

The confidence re-score and corrected headroom proxy are done
(`paper/results/metrics/qwen3v3_corrected/`), as is the D0 readout
(`scripts/capture_readout.py`, `scripts/s1_gap_table.py`).

**New code needed:**
- the aLoRA decision-head trainer;
- the typed-question formatter;
- the decide + escalate inference loop with cache reuse;
- a Jeeves/Kev evaluation harness;
- task generators;
- the wall-clock harness;
- a separate environment (`requirements-s1.txt`: transformers 5.x, peft
  ≥ 0.18, flash-linear-attention, causal-conv1d, vLLM for label capture only).

# Proposal: Bolt-on System 1 — activated-adapter decision heads for reasoning models

**Status:** research plan v3, 2026-10-04. Replaces v2 ("value of computation").
Tracked in issue #1. ✓ = checked against the primary source; everything else
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

1. **Method: a bolt-on System 1 for any reasoning model.**
   - One set of base weights serves both modes.
   - Escalation is free of re-prefill, because the cache is reused exactly
     (aLoRA guarantees it).
   - The System 2 fallback **is** the original reasoner, with no reasoning
     degradation from decision tuning.
2. **Evaluation: the first open, reasoning-heavy, wall-clock comparison of
   decision models with CoT fallback.**
   - Systems compared: Jev (closed comparator), Kev, Jeeves gated, a
     conventional cascade, always-think, and ours.
   - Workloads: math, logic and agent mixes.
3. **Analysis: when to escalate.**
   - Our own model's confidence vs a learned think-flag, across the design
     ladder (§4), with the old prefill-probe result as the control condition.
   - A null ("confidence suffices, and here is why") is still a finding; 1 and 2
     stand without it.

## 3. Prior art and how we differ

| Work | What it is | Our difference |
|---|---|---|
| Jev (TypeSafe, 2026-09-15 ✓) | Closed System One model; no CoT | We add CoT fallback; open |
| Kev (✓ HF) | Open Jev-style: Qwen3.5-Base + LoRA + pointer head | No fallback; Base weights cannot think |
| **Jeeves-9B** (PostHog ✓) | Open Jev-style that can think before deciding; `nothink_threshold` 0.9 → 0.806 @ 2.0 s vs 0.825 @ 3.3 s full think, on 325 dev items from MMLU/SciQ/PAWS ✓ | Its reasoning runs through the decision-tuned LoRA (per README); ours runs on the untouched reasoner with exact cache reuse. Its evaluation is not reasoning-heavy. |
| **GLiDE** (Fastino, 2026-09-30 ✓, closed) | "First thinking decision model": fast distribution, reasons when uncertain | Closed, mechanism undisclosed, evaluated on its own vendor index. We are open, the mechanism is specified, and the evaluation is independent. |
| LLM2Jev (2610.02076 ✓) | Read option-token probabilities from any LLM | Training-free baseline (D0) |
| aLoRA (2504.12397 ✓ NeurIPS 2025); serving 2512.17910; tradeoffs 2609.17109 | Adapter active only after an invocation sequence → base KV reuse | We apply it to decision heads plus thinking fallback (verify nobody has) |
| Self-Route (2505.20664 ✓), our negative result | Frozen prefill-state routers | Design D1 in our ladder, i.e. the control |
| JEV-as-a-Judge, COREA, REFLEX | Confidence-gated cascades to a *different* model | Same weights; free escalation |

**Novelty check before building:** search for "activated LoRA" + "decision"
or "classification head" + "reasoning fallback", and for any Jeeves/GLiDE
technical report. If someone has done exactly C1, the evaluation (C2) and
analysis (C3) become the paper.

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

**Base model:** a hybrid-thinking model with a thinking toggle. Start at about
4B and confirm at about 8B. Use the same family as Kev and Jeeves (Qwen3.5) if
it has a thinking toggle; otherwise use Qwen3, which our capture code already
supports. **Check:** PEFT and vLLM support for aLoRA on the chosen
architecture. Qwen3.5's Gated-DeltaNet layers may complicate cache reuse.

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
  re-prefill. One H100; vLLM, or HF if vLLM lacks aLoRA support.
- *Pass:* overhead ≈ decide-suffix only.

**C2 — Pareto.**
- *Measure:* accuracy vs mean wall-clock, and vs tokens, on mixed workloads
  with the hard fraction swept 10–70%.
- *Systems:* always-S1 (ours), always-think, Jeeves gated / no-think / think,
  Kev + same-weights thinking cascade, a conventional small-model cascade,
  Jev on a subset.
- *Pass:* ours dominates Jeeves-gated somewhere material.

**C3 — flag vs confidence.**
- *Measure:* paired-bootstrap ΔnAUC (routed accuracy over 0–50% escalation),
  flag vs the same design's own confidence, for D1–D3.
- Report within family × depth and leave-one-family-out.
- Pre-registered gates G2/G3 from v2 still apply.

**Benchmarks.**

| Phase | Families |
|---|---|
| Phase 0 | True/False math verify (GSM8K-Verify, MATH500-verify, with controlled perturbations); BBH binary subtasks (graders fixed); ProntoQA/boolean nesting with depth control |
| Phase 2 | When2Call, BFCL `miss_*`, τ²-bench, for answer / think / ask |

**The gap that matters is System 1 (decision readout) vs System 2, and it is
unmeasured.**

Our existing captures compare thinking-off *generation* with thinking-on
generation: 0.801 vs 0.807 on regraded Qwen3-8B gsm8k + math500 + mmlu_pro.
Qwen3 thinking-off still writes visible chain-of-thought, so both are
generative, and that 0.6 pp is not the S1-vs-S2 gap. A single-pass readout
should sit well below both on derived answers (Jev 68.4% vs GPT-6 95.9% on
JudgeBench reasoning ✓). The larger that gap, the more the routing decision
is the contribution.

**Phase 0 step 1 is to measure it.** Run a training-free D0 readout on the
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

**Phase 0 (weeks 1–3): minimal working system, the scoop hedge.**
- Build D2 on a ~4B hybrid model, answer head only, gated on its own
  confidence.
- Two benchmarks: math-verify and BBH-binary.
- Compare against Jeeves gated / no-think / think and against always-think on
  the same items.
- Measure wall-clock and cache reuse.
- **Exit:** C1 and C1b hold, and fast-path accuracy is ≥ Jeeves no-think on
  easy items.
- **If D2's fast path is much worse than Jeeves no-think:** try D3, and check
  whether aLoRA capacity is the bottleneck (2609.17109).
- **Deliverable:** a short preprint and open weights. The claim is C1 + C1b +
  a first Pareto.

**Phase 1 (weeks 4–7): the flag and the ladder.**
- D0–D3, cross-fitted flag labels.
- The 2×2: computational vs irreducible uncertainty × shallow vs deep.
- G2/G3.

**Phase 2 (weeks 8–10):** agents (answer / think / ask) and the full C2
evaluation.

**Phase 3 (weeks 11–14):** write. Target ICML 2027 or ARR; **verify the
deadlines.**

## 8. Risks

- **Scoop.** Jeeves and GLiDE shipped in three weeks. Phase 0 exists to stake
  C1 fast. Watch arXiv weekly.
- **aLoRA capacity.** Adapting only the suffix may underfit compared with full
  LoRA. D3 bounds this; quantify the gap.
- **Unfair Jeeves comparison.** Different size or base would confound it.
  Match size and family where possible, and report Jeeves at its native
  settings.
- **Flag never beats confidence.** Likely, given our prior result. C1 and C2
  carry the paper; C3 reports the null with the ladder as evidence.
- **Serving support.** If vLLM lacks aLoRA plus a thinking-mode switch, report
  HF wall-clock and isolate cache reuse analytically and empirically.

## 9. Repo carry-over

These are already fixed on `cleanup/system-one-carryover`:
- the paired off/on capture, which gives S2 labels;
- graders, with `--regrade`;
- `utils/metrics.py` (paired bootstrap, nAUC, CEL);
- baselines;
- the dispatch queue.

The re-score of the old confidence values and the corrected headroom proxy
are in progress (issue #1).

**New code needed:**
- the aLoRA decision-head trainer;
- the typed-question formatter;
- the decide + escalate inference loop with cache reuse;
- a Jeeves/Kev evaluation harness;
- task generators;
- the wall-clock harness;
- a separate environment (`requirements-s1.txt`: peft with aLoRA, possibly
  vLLM).

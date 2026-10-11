# Research notes for the decision head (2026-10-10)

These are inputs to the Phase 0 decision-head plan. ✓ = the PI checked it
against the primary source. Everything else comes from agent reports and needs
re-verification before it is cited.

## 1. Scoop status (verified 2026-10-10)

Our contributions: C1 = bolt-on decision head with fallback to the untouched
base model and cache reuse; C2 = Pareto comparison on reasoning-heavy tasks;
C3 = learned think-flag versus the model's own confidence.

| Work | What it already does | Overlap |
|---|---|---|
| **DoccyHealth Solomon** (HF, 2026-09-21) ✓ | LoRA on Qwen3.8-27B. "The adapter is off while the document is read, and on from the question onward." `src/solomon/engine_reasoning.py` deep-copies the cached prefix and generates `<think>` with `'adapter_enabled': False`, reporting `reused_prefix_tokens`. `src/solomon/routing.py` escalates to that reasoning when reliability < threshold. Model card plus code only; no paper and no Pareto. | **C1 mechanism taken** |
| **AutoTrust GEV-26B-Decide** (HF, Oct 2026) ✓ | LoRA + 24-slot linear head on Gemma-4-26B-A4B. Escalates to the **unmodified** base model's thinking when the leading option is < 0.8, then sets p = ½p1 + ½p2. Results: GPQA-D S1 42.9% → adaptive 78.6% (87% of items thinking); knowledge mix adaptive 83.4% vs always-think 83.8%. Its documentation does not mention KV reuse. | C1 (fallback), C2 partial |
| **PostHog Jeeves-9B** | Qwen3.5-9B, LoRA r16 + pointer head. `nothink_threshold`: thinking 0.825 @ 3.3 s, gated (0.9) 0.806 @ 2.0 s, no-think 0.775 @ 0.3 s. Per a scout, `inference/engine.py` reuses the prompt KV but thinks *through the LoRA* (UNVERIFIED by PI). | C1 partial; C3 baseline |
| **InnerJev / JEVal** (arXiv 2610.03935) | Accuracy-latency Pareto across open decision models plus Jev. Distils reasoning into the readout instead of escalating. | C2 partial |
| **Qualcomm "Efficient Reasoning on the Edge"** (2603.16867) | Masked LoRA, off on the prompt, so prefill is shared with the base model. A prefill MLP decides whether to think. Not compared against confidence. | C1-adjacent (roles inverted) |
| GLiDE (Fastino), JEV-as-a-Judge, REFLEX | All escalate on confidence. | C3 baseline |

**Verdict.** C1 is not novel; frame it as the platform, cite Solomon and GEV.
What is left of C1 is (a) exact, measured cache reuse on **hybrid-recurrent**
Qwen3.5, which needs the recurrent state snapshotted and restored (see §2),
and (b) an aLoRA vs full-LoRA quality ablation for decision heads. C2 is
partially done. **C3 is open, and it is now the headline:** every system found
gates on its own confidence.

## 2. Base model and aLoRA support

- **Qwen3.5** (Apache-2.0, small sizes 0.8B/2B/4B/9B released 2026-02-27/28).
  - Toggle: `enable_thinking` via `chat_template_kwargs`. Off inserts
    `<think>\n\n</think>\n\n`; on inserts `<think>\n`. The `/think` soft
    switches are not supported.
  - Layout for 4B and 9B: 32 layers in `8 × (3 × Gated DeltaNet → 1 × attention)`.
  - The checkpoints load as `Qwen3_5ForConditionalGeneration` (multimodal).
- **Cache reuse on GDN.** aLoRA leaves every prompt-token state at its exact
  base value. In practice the GDN recurrent state is overwritten in place, and
  transformers' `is_croppable` is False when a recurrent state is present. So:
  1. Clone `recurrent_states` and `conv_states` at the prompt end (~50 MB on 4B).
  2. Crop the attention layers.
  3. Test with `torch.equal` against a fresh base prefill.
  4. Also test that the adapter actually changed the suffix logits.
- **PEFT.** aLoRA has been merged since v0.18.0 (current v0.21.2).
  - Usage: `LoraConfig(alora_invocation_tokens=…)`. It activates from the
    *last* occurrence of the invocation tokens.
  - Silent failure modes: invocation tokens merged by the tokenizer, or
    loading via `model.load_adapter()` skipping the offset. Either way the
    base model runs with no warning.
  - Docs recommend r ≥ 32. Merge and beam search are not supported.
  - One report puts `generate()` at about 50% slower (UNVERIFIED).
- **vLLM: not usable.** The aLoRA PR #19710 was declined. #60223 / PR #60225
  (closed 2026-10-09) make vLLM refuse aLoRA adapters. The 2512.17910 fork
  predates Qwen3.5, excludes Mamba-style models, and does cross-request reuse
  only. Hybrid prefix caching is experimental (`--mamba-cache-mode align`), and
  NaN bug #55766 is open.
- **Recommendation.**
  - Use a custom Hugging Face loop: transformers v5.x plus `flash-linear-attention`
    and `causal-conv1d` (pinned; the slow torch fallback makes wall-clock numbers
    meaningless).
  - Develop on **Qwen3.5-4B**, confirm on **Qwen3.5-9B** (same family as Jeeves),
    and keep Qwen3-8B as the fallback (attention-only, exact `crop()`).
  - Batch size 1 first.
- **aLoRA quality.** The aLoRA paper (Granite 3.2 8B, r32) shows it roughly
  matches LoRA. 2609.17109's aLoRA number is not usable. Run our own ablation.

## 3. Kev recipe (github.com/jaredpalmer/kev, Apache-2.0)

- **Pointer head.**
  - Five reused Qwen special tokens act as delimiters:
    `<|fim_prefix|> <|fim_middle|> <|box_start|> <|box_end|> <|fim_suffix|>`,
    standing for `<state> <q> <opt> </opt> <decide>`.
  - Layout: `[<state> …] + per question [<q> instr <opt> o </opt> … <decide>]`.
  - Score: a bilinear dot product (d_p = 256) between the `</opt>` hidden
    states and the `<decide>` hidden state, then a softmax over K.
  - Options are option *text*, not letters. This avoids the "(B)" failure that
    hit our training-free readout.
- **Calibration and inference.**
  - A single temperature fitted after training on a 0.25–4 grid; shipped
    values are about 1.3–2.4.
  - Each question is block-causal on top of a cached state prefix, but the
    LoRA is also active on state tokens, so this is not base-cache reuse.
- **Typed questions.** noul = ["no", "yes"]; choice has 1–255 options; score
  is ordinal. There is no abstain or think output.
- **Training.**
  - Data: decision-v7, about 12.6k records from 10 public sets plus synthetic
    policy and rule data.
  - LoRA: r16/α32, targets q/k/v/o/gate/up/down plus the DeltaNet
    `in_proj_*` / `out_proj`.
  - Compute: Kev-4B takes about 56 min on one H100.
  - **kev-8b is on Qwen3-8B-Base**, so it is a size-matched comparison for our
    Qwen3 data.
- **Benchmarks.**
  - ekzhang MMLU-Pro (1000): Kev-4B 0.472, Kev-8B 0.492, Jev 0.835.
  - Our training-free D0 readout on Qwen3-8B, on the v3 MMLU-Pro 1000-row
    subset: 0.469. The sets may differ, so check.
  - Kev evaluates no GSM8K, MATH, BBH or LSAT.
- **Reusable:**
  - `kev/model.py` (encode, PointerHead, branch mask, Qwen3 packed path);
  - `kev/train.py:question_loss`;
  - `kev/metrics.py` (ECE, Brier, Cov@5%, AURC, paired bootstrap, `fit_temperature`);
  - the `/v1/systemone` schema;
  - the eval harness `kev.benchmark --remote …` and its suites (JSONL with
    pinned revisions).

## 4. Other formats and recipes

- **LLM-as-Jev / LLM2Jev** (2610.02076).
  - Prompt: options listed as `[1] … [K]`, with the assistant turn prefilled as
    `Best answer: [`.
  - Scoring: each legal suffix `k]` is scored (prefix-free) and the scores are
    renormalised over the legal suffixes only. Off-list mass, such as "(B)",
    therefore cannot leak.
  - Yes/No is posed as `[1]`/`[2]` acceptance criteria.
  - Results: Qwen3.5-4B training-free scores 81.4% on public JevBench; LoRA
    with KL anchors reaches 84.0%.
- **OpenJev-RLCD** (2609.38850, code public). A Brier proper-score reward on
  the option readout after sampled rationales, trained "calibrate, then
  reinforce", with a KL anchor. Without the anchor it suffers "System-One
  collapse".

## 5. Our Phase 0 measurements so far (Qwen3-8B, training-free D0 readout)

Source: issue #1 comments 6017130639 and 6092002786; `output/qwen3v3/s1_gap_v2_*`.

| | S1 | no-think | think | think − S1 |
|---|---|---|---|---|
| MMLU-Pro (1000) | 47.0 | 61.7 | 64.8 | +17.8 |
| BBH lettered (340) | 55.6 | 80.3 | 86.5 | +30.9 |

- S1 is badly overconfident: ECE 0.41–0.45, mean p_max 0.91 at ~50% accuracy.
- Routing on raw S1 confidence needs 74–86% escalation to get within 1 pp of
  think, against 17–30% for the oracle. A family router beats it on BBH.
- The BBH word-answer subtasks are unreliable under this prompt format.

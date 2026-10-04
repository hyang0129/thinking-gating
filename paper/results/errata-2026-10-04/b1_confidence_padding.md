### B1 · confidence corrupted by left padding [pub][new] · CRITICAL
- **Where:** `capture_inference_thinking.py:354–356`. `sequence_confidence` slices `sequences[i:i+1, :prompt_len+n]` from a left-padded batch and runs the forward pass without an attention mask, so the model attends to the pad tokens.
- **Evidence (Qwen3 v3):**
  - About 93% of rows are padded.
  - Spearman(pad count, mean_logprob) is −0.72 to −0.88 across the four tasks.
  - Median mean_logprob is −0.08 on unpadded rows and −1.4 to −2.0 on padded rows.
  - On unpadded rows only, mean_logprob AUROC for needs_thinking is 0.70 / 0.76 / 0.70 / 0.80 (n = 86 / 65 / 64 / 39). On all rows it is 0.58 / 0.65 / 0.56 / 0.53.
- **Fix:** score `sequences[i, pad_i : prompt_len+n]`, with offsets shifted by `pad_i = width − mask.sum()`, or pass `attention_mask` plus `position_ids`. Add a batched test.
- **Recovery:** existing captures can be re-scored offline from the stored text, which needs a GPU but no regeneration.
- **Effect on the writeup:** every column built on mean_logprob, min_logprob, mean_entropy, or confidence_lr changes. That covers §3.1 math500 and mmlu_pro, §3.2 gsm8k, the §3.4 Nemotron entries, and the abstract's "0.814". `n_tokens_off` is clean.


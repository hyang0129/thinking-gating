#!/usr/bin/env python3
"""
s2_capture.py — thinking-on (S2) answers with vLLM, base model, no adapter
(issue #3, step 4; Q2: vLLM for labels only). Runs in `.venv-vllm`.

For each task item: the vanilla enable_thinking=True prompt (the same tokens
s1_cache.prompt_split + escalate produces), --n-samples sampled continuations
up to --max-tokens, each graded by the task's own grader. One row per item:

    {sample_id, task, family, samples: [{correct, n_tokens, truncated,
     has_reasoning, unclosed, answer}], mean_correct}

Budget sweep (step 4a): run once at a large --max-tokens; the summary reports
the truncation rate every smaller budget would have had (a sample is truncated
at budget b iff it needed more than b tokens), so one pass picks the budget.

    .venv-vllm/bin/python scripts/s2_capture.py --model Qwen/Qwen3.5-4B \\
        --task mmlu_pro --n-items 100 --max-tokens 16384 \\
        --out-dir shared/s2/sweep_qwen3.5-4b/mmlu_pro
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from capture_inference_thinking import (  # noqa: E402
    _TASK_REGISTRY, _correct_str, _prompt_default, assess_reasoning,
    git_provenance, strip_thinking)

SWEEP_BUDGETS = (2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 40960, 81920)


def load_items(task: str, n: int | None, seed: int) -> list[dict]:
    import random
    loader, _, split = _TASK_REGISTRY[task]
    items = getattr(importlib.import_module(f"tasks.{task}"), loader)(split)
    if n is not None and n < len(items):
        # Stratified by family where the task has one, so a 100-item sweep
        # spans BBH subtasks / MMLU-Pro categories instead of a prefix.
        rng = random.Random(seed)
        fam = lambda it: it.get("family") or it.get("subtask") or it.get("category") or it.get("subject") or ""  # noqa: E731
        groups: dict[str, list] = {}
        for it in items:
            groups.setdefault(fam(it), []).append(it)
        for g in groups.values():
            rng.shuffle(g)
        out, i = [], 0
        while len(out) < n:
            for g in groups.values():
                if i < len(g) and len(out) < n:
                    out.append(g[i])
            i += 1
        items = out
    return items


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True, choices=sorted(_TASK_REGISTRY))
    p.add_argument("--n-items", type=int, default=None,
                   help="family-stratified sample of the task's test split (sweeps)")
    p.add_argument("--pool", choices=("eval", "train"), default=None,
                   help="use the Phase 0 pool (utils/s1_pools.py) instead")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--max-tokens", type=int, required=True)
    # Defaults: the Qwen3.5 model card's "thinking mode for general tasks".
    # Qwen3's 0.6 / no penalty (the card's "precise coding" setting) looped on
    # Qwen3.5-4B: 12% of GSM8K samples repeated the answer line to 32k tokens
    # (sweep v1, 2026-10-18). Pass --temperature 0.6 --presence-penalty 0 for Qwen3.
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--presence-penalty", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_mod = importlib.import_module(f"tasks.{args.task}")
    if args.pool:
        if args.n_items is not None:
            raise SystemExit("--pool and --n-items are exclusive")
        from utils.s1_pools import POOL_VERSION, load_pool
        items = load_pool(args.pool, args.task)
        args.pool_version = POOL_VERSION
    else:
        items = load_items(args.task, args.n_items, args.seed)
    items = items[args.shard_index::args.shard_count]
    if not items:
        raise SystemExit("empty shard")
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": _prompt_default(task_mod, it)}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True) for it in items]
    plen = [len(tok(x, add_special_tokens=False)["input_ids"]) for x in prompts]

    llm = LLM(model=args.model, seed=args.seed, enable_prefix_caching=False,
              max_model_len=args.max_model_len or max(plen) + args.max_tokens + 16,
              gpu_memory_utilization=args.gpu_memory_utilization)
    sp = SamplingParams(n=args.n_samples, max_tokens=args.max_tokens,
                        temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, min_p=args.min_p,
                        presence_penalty=args.presence_penalty, seed=args.seed)
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": tok(x, add_special_tokens=False)["input_ids"]}
                         for x in prompts], sp)
    secs = time.time() - t0

    rows, lengths = [], []
    for it, prompt, o in zip(items, prompts, outs):
        samples = []
        for c in o.outputs:
            text = c.text
            r = assess_reasoning(text, prompt)
            n_tok = len(c.token_ids)
            trunc = c.finish_reason == "length"
            samples.append({
                "correct": bool(not trunc and not r["unclosed"]
                                and _correct_str(task_mod, strip_thinking(text), it)),
                "n_tokens": n_tok, "truncated": trunc,
                "has_reasoning": r["has_reasoning"], "unclosed": r["unclosed"],
                "answer": strip_thinking(text)[-500:]})
            lengths.append((n_tok, trunc))
        rows.append({"sample_id": it.get("key") or it.get("sample_id"), "task": args.task,
                     "pool": args.pool,
                     "family": it.get("family") or it.get("subtask") or it.get("category") or it.get("subject"),
                     "prompt_tokens": len(o.prompt_token_ids),
                     "mean_correct": sum(s["correct"] for s in samples) / len(samples),
                     "samples": samples})

    with open(out_dir / "s2.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    allS = [s for r in rows for s in r["samples"]]
    # A truncated sample needed > max_tokens, so it is truncated at every budget.
    sweep = {b: sum(t or n > b for n, t in lengths) / len(lengths)
             for b in SWEEP_BUDGETS if b <= args.max_tokens}
    import vllm
    summary = {
        "args": vars(args), "n_items": len(rows), "n_samples": len(allS),
        "accuracy": sum(s["correct"] for s in allS) / len(allS),
        "truncated_rate": sum(s["truncated"] for s in allS) / len(allS),
        "has_reasoning_rate": sum(s["has_reasoning"] for s in allS) / len(allS),
        "unclosed_rate": sum(s["unclosed"] for s in allS) / len(allS),
        "truncation_at_budget": sweep,
        "n_tokens_quantiles": {q: sorted(n for n, _ in lengths)[
            min(len(lengths) - 1, int(q * len(lengths)))] for q in (0.5, 0.9, 0.95, 0.99)},
        "seconds": round(secs, 1), "vllm": vllm.__version__, **git_provenance(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

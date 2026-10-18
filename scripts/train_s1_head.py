#!/usr/bin/env python3
"""
train_s1_head.py — the D2 answer head: an aLoRA on the base model trained to
answer typed questions (issue #3, step 5). Runs in `.venv-s1`, GPU node.

Each training sequence is exactly what serving sees:

    [chat prefix: user = record.context, up to "<|im_start|>assistant\\n"]
    [decide suffix: invocation + question + [1]..[K] + "Best answer: ["]
    [gold "k]" tokens]                                   <- loss here only

The aLoRA switches on at the invocation, so the prefix states stay the base
model's and the escalation reuses them (utils/s1_cache.py).

Cross-fitting (§5): records of our train pool are split into --n-folds folds
by hash of their id; --fold k trains on every fold but k (decision-v7 is
always in), --fold -1 on everything (the final head). Phase 0 objective is
plain SFT on the gold continuation; no option shuffling.

    .venv-s1/bin/python scripts/train_s1_head.py --model Qwen/Qwen3.5-4B \\
        --records shared/s1/records/train.jsonl shared/s1/records/dv7_train.jsonl \\
        --fold 0 --out-dir shared/s1/heads/qwen3.5-4b/answer_fold0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch  # noqa: E402

from utils import s1_cache, typed_question  # noqa: E402

INVOCATION = "<|im_start|>decide\n"


def fold_of(rec_id: str, n_folds: int) -> int:
    return int(hashlib.sha256(f"s1fold:{rec_id}".encode()).hexdigest()[:8], 16) % n_folds


def load_records(paths, fold: int, n_folds: int) -> list[dict]:
    out = []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if fold >= 0 and r["task"] != "decision_v7" and fold_of(r["id"], n_folds) == fold:
                continue
            out.append(r)
    return out


def encode(tok, rec: dict, max_len: int) -> dict | None:
    """Token ids and loss mask for one record; context trimmed from the front."""
    q = typed_question.build(tok, INVOCATION, rec["question"], rec["options"])
    gold = q["candidate_ids"][rec["gold"]]
    budget = max_len - len(q["suffix_ids"]) - len(gold)
    ctx = rec["context"]
    for _ in range(8):
        prefix, _ = s1_cache.prompt_split(tok, [{"role": "user", "content": ctx}])
        if len(prefix) <= budget:
            break
        ctx = ctx[-int(len(ctx) * budget / len(prefix) * 0.95):]   # keep the question end
    else:
        return None
    ids = prefix + q["suffix_ids"] + gold
    labels = [-100] * (len(ids) - len(gold)) + gold
    return {"input_ids": ids, "labels": labels}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--records", nargs="+", required=True)
    p.add_argument("--fold", type=int, required=True, help="-1 = final head")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-tokens", type=int, default=16384,
                   help="tokens per optimizer step (padded), via accumulation")
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-steps", type=int, default=None, help="smoke runs only")
    args = p.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("GPU node only")

    import peft
    import transformers
    torch.manual_seed(args.seed)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    recs = load_records(args.records, args.fold, args.n_folds)
    data = [e for e in (encode(tok, r, args.max_len) for r in recs) if e]
    dropped = len(recs) - len(data)
    random.Random(args.seed).shuffle(data)

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    base, info = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=args.device, output_loading_info=True)
    if info["missing_keys"]:
        raise SystemExit(f"missing weights, e.g. {sorted(info['missing_keys'])[:5]}")
    inv = tok(INVOCATION, add_special_tokens=False)["input_ids"]
    model = peft.get_peft_model(base, peft.LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, target_modules="all-linear",
        alora_invocation_tokens=inv, task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    mean_len = sum(len(d["input_ids"]) for d in data) / len(data)
    accum = max(1, round(args.batch_tokens / (mean_len * args.micro_batch)))
    n_micro = math.ceil(len(data) * args.epochs / args.micro_batch)
    total = math.ceil(n_micro / accum) if args.max_steps is None else args.max_steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(total, 2),
                                                pct_start=0.05, anneal_strategy="cos")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "w")
    model.train()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    step, t0, run_loss, run_n = 0, time.time(), 0.0, 0
    for m in range(n_micro):
        batch = [data[(m * args.micro_batch + i) % len(data)] for i in range(args.micro_batch)]
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.tensor([b["input_ids"] + [pad] * (L - len(b["input_ids"])) for b in batch])
        lab = torch.tensor([b["labels"] + [-100] * (L - len(b["labels"])) for b in batch])
        att = (torch.arange(L)[None, :] < torch.tensor([len(b["input_ids"]) for b in batch])[:, None]).long()
        loss = model(input_ids=ids.to(args.device), attention_mask=att.to(args.device),
                     labels=lab.to(args.device)).loss
        (loss / accum).backward()
        run_loss += float(loss)
        run_n += 1
        if (m + 1) % accum == 0 or m == n_micro - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total:
                log.write(json.dumps({"step": step, "loss": run_loss / run_n,
                                      "lr": sched.get_last_lr()[0],
                                      "secs": round(time.time() - t0, 1)}) + "\n")
                log.flush()
                run_loss, run_n = 0.0, 0
            if step >= total:
                break

    model.save_pretrained(out / "adapter")
    meta = {"args": vars(args), "n_records": len(recs), "n_dropped_too_long": dropped,
            "n_train": len(data), "mean_len": mean_len, "accum": accum, "steps": step,
            "seconds": round(time.time() - t0, 1), "invocation": INVOCATION,
            "invocation_ids": inv, "transformers": transformers.__version__,
            "peft": peft.__version__,
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                         text=True, cwd=_PROJECT_ROOT).stdout.strip()}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "args"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

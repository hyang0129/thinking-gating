#!/usr/bin/env python3
"""
s1_predict.py — run a decision head over typed-question records (issue #3).
Runs in `.venv-s1`, GPU node.

Per record: base prefill of the context (the escalation's cache), then the
decide suffix and every candidate "k]" scored with utils/typed_question.score.

  --adapter DIR   D2: the trained aLoRA. Without it: D0, the untouched model
                  reading the same format (training-free).
  --hidden-layers D1 features: the base model's (adapter off) hidden state at
                  the decide position for these layers, saved to hidden.npz.
  --fold k        only records of fold k (out-of-fold S1 labels for the flag);
                  -1 = all.

Writes predictions.jsonl: {id, task, family, kind, gold, probs, argmax,
p_max, correct, legal_mass, candidate_is_gold}.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from train_s1_head import INVOCATION, fold_of  # noqa: E402
from utils import s1_cache, typed_question  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--records", nargs="+", required=True)
    p.add_argument("--fold", type=int, default=-1)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--hidden-layers", type=int, nargs="*", default=[])
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--limit", type=int, default=None, help="smoke runs only")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("GPU node only")

    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model, info = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=args.device, output_loading_info=True)
    if info["missing_keys"]:
        raise SystemExit(f"missing weights, e.g. {sorted(info['missing_keys'])[:5]}")
    if args.adapter:
        import peft
        model = peft.PeftModel.from_pretrained(model, args.adapter)
        if model.peft_config["default"].alora_invocation_tokens != \
                tok(INVOCATION, add_special_tokens=False)["input_ids"]:
            raise SystemExit("adapter invocation tokens differ from INVOCATION")
    model.eval()

    recs = [json.loads(l) for f in args.records for l in open(f)]
    if args.fold >= 0:
        recs = [r for r in recs if fold_of(r["id"], args.n_folds) == args.fold]
    recs = recs[:args.limit]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hidden, n_trim = [], 0
    t0 = time.time()
    # Resumable when no hidden states are kept (QOS burst preempts about
    # hourly): rows go to predictions.jsonl.partial as they finish, a restart
    # skips them, and predictions.jsonl is written in record order at the end.
    part = out / "predictions.jsonl.partial"
    done = {}
    if part.exists() and not args.hidden_layers:
        for line in open(part):
            try:
                x = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[x["id"]] = x
        part.write_text("".join(json.dumps(x) + "\n" for x in done.values()))
    with open(part, "a" if done else "w") as f, torch.no_grad():
        for r in recs:
            if r["id"] in done:
                continue
            q = typed_question.build(tok, INVOCATION, r["question"], r["options"])
            prefix, _ = s1_cache.prompt_split(tok, [{"role": "user", "content": r["context"]}])
            if len(prefix) > args.max_len:
                n_trim += 1
                prefix = prefix[-args.max_len:]
            cache, _ = s1_cache.prefill(model, prefix)
            res = typed_question.score(model, cache, q)
            if args.hidden_layers:
                snap = s1_cache.snapshot(cache)
                with s1_cache.adapter_off(model):
                    o = model(input_ids=torch.tensor([q["suffix_ids"]], device=model.device),
                              past_key_values=cache, use_cache=True, output_hidden_states=True)
                s1_cache.restore(cache, snap)
                hidden.append(np.stack([o.hidden_states[L][0, -1].float().cpu().numpy()
                                        for L in args.hidden_layers]).astype(np.float16))
            am = res["argmax"] - 1
            f.write(json.dumps({"id": r["id"], "task": r["task"], "family": r["family"],
                                "kind": r["kind"], "gold": r["gold"], "probs": res["probs"],
                                "argmax": am, "p_max": res["p_max"],
                                "correct": am == r["gold"], "legal_mass": res["legal_mass"],
                                "candidate_is_gold": r.get("candidate_is_gold")}) + "\n")
            f.flush()
    if hidden:
        np.savez_compressed(out / "hidden.npz", hidden=np.stack(hidden),
                            layers=np.array(args.hidden_layers),
                            id=np.array([r["id"] for r in recs]))
    rows = {}
    for line in open(part):
        x = json.loads(line)
        rows[x["id"]] = x
    preds = [rows[r["id"]] for r in recs]
    (out / "predictions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in preds))
    part.unlink()
    by_task: dict[str, list] = {}
    for x in preds:
        by_task.setdefault(x["task"], []).append(x["correct"])
    summary = {"args": vars(args), "n": len(preds), "n_prefix_trimmed": n_trim,
               "accuracy": sum(x["correct"] for x in preds) / max(len(preds), 1),
               "accuracy_by_task": {t: sum(v) / len(v) for t, v in by_task.items()},
               "seconds": round(time.time() - t0, 1),
               "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                            text=True, cwd=_PROJECT_ROOT).stdout.strip()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

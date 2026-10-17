#!/usr/bin/env python3
"""
s1_cache_check.py — Phase 0 step 3 (issue #3, gate E1): the serving-loop tests
on a real checkpoint, GPU node only.

  T1  snapshot/restore: after a decide pass (random aLoRA on) and a restore,
      base logits on a continuation equal a fresh prefill's. torch.equal is
      recorded; the gate is max-abs <= --atol plus identical greedy tokens over
      --t1-steps, since GPU kernels may not be bitwise deterministic.
  T2  no leak: PEFT logits on the prompt equal the base model's.
  T3  adapter activates: decide logits move by > 1e-3 with random LoRA B.
  T5  load path: save, PeftModel.from_pretrained, offsets still set, T3 holds.
  T6  escalation: on --n-items MMLU-Pro prompts, after a decide pass, greedy
      decoding from the restored prefix cache with the adapter off is
      token-identical to vanilla generate() on the enable_thinking=True prompt.

The aLoRA is random (no training happened yet), which is what these checks
need: an adapter that visibly changes the decide logits.

    .venv-s1/bin/python scripts/s1_cache_check.py --model Qwen/Qwen3.5-4B \\
        --out output/s1/cache_check_qwen3.5-4b.json
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

import torch  # noqa: E402

from utils import s1_cache, typed_question  # noqa: E402

INVOCATION = "<|im_start|>decide\n"


def random_alora(model, inv_ids, seed=0):
    import peft
    cfg = peft.LoraConfig(r=8, lora_alpha=16, target_modules="all-linear",
                          alora_invocation_tokens=inv_ids, task_type="CAUSAL_LM")
    pm = peft.get_peft_model(model, cfg)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.copy_((torch.randn(p.shape, generator=g) * 0.02).to(p.dtype))
    return pm.eval()


def step(model, cache, toks):
    return model(input_ids=torch.tensor([toks], device=model.device),
                 past_key_values=cache, use_cache=True).logits[0].float()


def greedy(model, cache, last, n):
    out = []
    logits = last
    for _ in range(n):
        t = int(logits.argmax())
        out.append(t)
        logits = step(model, cache, [t])[-1]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-items", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--t1-steps", type=int, default=256)
    p.add_argument("--atol", type=float, default=1e-2)
    p.add_argument("--device", default="cuda",
                   help="cpu only for the smoke run on a tiny random checkpoint")
    args = p.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("GPU node only")

    import transformers
    from tasks.mmlu_pro import format_prompt, load_mmlu_pro
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    base, info = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        device_map=args.device, output_loading_info=True)
    base.eval()
    # A Qwen3.5 repo stores the text model as model.language_model.*; a missed
    # remap would leave random weights, and every check below would still pass.
    if info["missing_keys"]:
        raise SystemExit(f"missing weights, e.g. {sorted(info['missing_keys'])[:5]}")
    inv_ids = tok(INVOCATION, add_special_tokens=False)["input_ids"]
    pm = random_alora(base, inv_ids)
    items = load_mmlu_pro("test", max_rows=args.n_items)
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None}

    def prompt(it):
        return s1_cache.prompt_split(tok, [{"role": "user",
                                            "content": format_prompt(it["question"])}])

    res = {"model": args.model, "transformers": transformers.__version__,
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True, cwd=_PROJECT_ROOT).stdout.strip(),
           "invocation_ids": inv_ids,
           "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"}

    with torch.no_grad():
        # T1 / T2 / T3 on the first item
        prefix, esc = prompt(items[0])
        q = typed_question.build(tok, INVOCATION, "Pick one.",
                                 [f"option {k}" for k in range(10)])
        cont = esc + tok("Let me think", add_special_tokens=False)["input_ids"]
        cache, last = s1_cache.prefill(pm, prefix)
        snap = s1_cache.snapshot(cache)
        dec_on = step(pm, cache, q["suffix_ids"])[-1]
        s1_cache.restore(cache, snap)
        with s1_cache.adapter_off(pm):
            got = step(pm, cache, cont)
            g_got = greedy(pm, cache, got[-1], args.t1_steps)
            fresh, _ = s1_cache.prefill(pm, prefix)
            want = step(pm, fresh, cont)
            g_want = greedy(pm, fresh, want[-1], args.t1_steps)
            ref_cache, _ = s1_cache.prefill(pm, prefix)
            dec_off = step(pm, ref_cache, q["suffix_ids"])[-1]
        d1 = float((got - want).abs().max())
        res["T1"] = {"torch_equal": bool(torch.equal(got, want)), "max_abs": d1,
                     "greedy_identical": g_got == g_want,
                     "pass": d1 <= args.atol and g_got == g_want}
        ids = torch.tensor([prefix], device=pm.device)
        lp = pm(input_ids=ids).logits.float()
        with s1_cache.adapter_off(pm):
            lb = pm(input_ids=ids).logits.float()
        d2 = float((lp - lb).abs().max())
        res["T2"] = {"torch_equal": bool(torch.equal(lp, lb)), "max_abs": d2,
                     "pass": d2 == 0.0}
        d3 = float((dec_on - dec_off).abs().max())
        res["T3"] = {"max_abs_decide_shift": d3, "pass": d3 > 1e-3}

        # T5: load path
        save = Path(args.out).with_suffix(".adapter")
        pm.save_pretrained(save)
        import peft
        base = pm.unload()                   # back to the plain base weights
        loaded = peft.PeftModel.from_pretrained(base, save).eval()
        cache, _ = s1_cache.prefill(loaded, prefix)
        d5 = float((step(loaded, cache, q["suffix_ids"])[-1] - dec_on).abs().max())
        res["T5"] = {"alora_invocation_tokens":
                     loaded.peft_config["default"].alora_invocation_tokens,
                     "max_abs_vs_saved": d5,
                     "pass": (loaded.peft_config["default"].alora_invocation_tokens == inv_ids
                              and d5 <= args.atol)}
        pm = loaded

        # T6: escalation == vanilla thinking generate()
        rows, t0 = [], time.time()
        for it in items:
            prefix, esc = prompt(it)
            cache, last = s1_cache.prefill(pm, prefix)
            snap = s1_cache.snapshot(cache)
            typed_question.score(pm, cache, q)      # a decide pass, restores itself
            ours = s1_cache.escalate(pm, cache, esc, last.float(),
                                     args.max_new_tokens, eos)
            with s1_cache.adapter_off(pm):
                full = torch.tensor([prefix + esc], device=pm.device)
                van = pm.generate(full, max_new_tokens=args.max_new_tokens,
                                  do_sample=False, eos_token_id=sorted(eos))
            van = van[0, full.shape[1]:].tolist()
            first = next((i for i, (a, b) in enumerate(zip(ours, van)) if a != b), None)
            rows.append({"sample_id": it.get("key"), "identical": ours == van,
                         "first_diff": first, "n_ours": len(ours), "n_vanilla": len(van)})
        n_id = sum(r["identical"] for r in rows)
        res["T6"] = {"n": len(rows), "n_identical": n_id, "rows": rows,
                     "seconds": round(time.time() - t0, 1), "pass": n_id == len(rows)}

    res["E1_pass"] = all(res[k]["pass"] for k in ("T1", "T2", "T3", "T5", "T6"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {kk: vv for kk, vv in v.items() if kk != "rows"})
                      for k, v in res.items()}, indent=1))
    return 0 if res["E1_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

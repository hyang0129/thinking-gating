#!/usr/bin/env python3
"""
test_confidence.py — sequence_confidence must actually measure confidence.

Runs against real gpt2 (~500MB, downloaded once) rather than a random-weight
stub: an untrained model assigns near-uniform probability to everything, so it
cannot distinguish a correct implementation from one that returns noise. On
trained gpt2 the model's own greedy continuation scores about -1.4 nats/token
against about -14 for random tokens, which a broken implementation will not
reproduce.

    python tests/test_confidence.py
"""
import sys, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.capture_inference_thinking import sequence_confidence

name = "gpt2"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name); model.eval()
if tok.pad_token_id is None: tok.pad_token = tok.eos_token

prompt = tok("The capital of France is", return_tensors="pt")
P = prompt.input_ids.shape[1]
with torch.no_grad():
    gen = model.generate(**prompt, max_new_tokens=8, do_sample=False,
                         pad_token_id=tok.pad_token_id)
n_new = gen.shape[1] - P

greedy = sequence_confidence(model, gen, P, [n_new], tok.pad_token_id)[0]

# same prompt, but random continuation tokens
torch.manual_seed(0)
rand = gen.clone()
rand[0, P:] = torch.randint(0, model.config.vocab_size, (n_new,))
random_ = sequence_confidence(model, rand, P, [n_new], tok.pad_token_id)[0]

print("greedy :", {k: round(v, 4) for k, v in greedy.items()})
print("random :", {k: round(v, 4) for k, v in random_.items()})

ok = True
if not (greedy["mean_logprob"] > random_["mean_logprob"]):
    print("FAIL: greedy continuation should score higher than random"); ok = False
if greedy["mean_logprob"] > 0:
    print("FAIL: a log-probability must be <= 0"); ok = False
if not (0 <= greedy["mean_entropy"] <= torch.log(torch.tensor(float(model.config.vocab_size)))):
    print("FAIL: entropy outside [0, log V]"); ok = False
if greedy["min_logprob"] > greedy["mean_logprob"]:
    print("FAIL: min must be <= mean"); ok = False
# empty generation must not crash
empty = sequence_confidence(model, gen, P, [0], tok.pad_token_id)[0]
if empty["mean_logprob"] is not None:
    print("FAIL: empty generation should yield None"); ok = False
print("ALL PASS" if ok else "FAILURES ABOVE")
raise SystemExit(0 if ok else 1)

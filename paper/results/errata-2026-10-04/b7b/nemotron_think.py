import json, glob, statistics
ROOT = "/Users/hong/Documents/code-projects/thinking-gating/shared/icr_capture/"
for t in ("bbh", "lsat"):
    d = ROOT + f"{t}_thinking_nemotronv3"
    rows = [json.loads(l) for f in sorted(glob.glob(d + "/meta.shard*.jsonl")) for l in open(f) if l.strip()]
    has = sum("<think>" in r["response_on"] for r in rows)
    ident = sum(r["response_on"] == r["response_off"] for r in rows)
    ntok = [r.get("n_tokens_on") for r in rows if r.get("n_tokens_on") is not None]
    print(t, "n", len(rows), "with <think>", has, "without", len(rows) - has,
          "byte-identical on==off", ident, "median n_tokens_on", statistics.median(ntok) if ntok else None)

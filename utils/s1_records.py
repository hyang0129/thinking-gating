"""
s1_records.py — typed-question records for the S1 decision head (issue #3, §3-4).

A record is one decision:

    {"id", "task", "family", "kind", "context", "question", "options", "gold",
     "candidate", "candidate_is_gold"}

  context   the user message of the cached prefix. For our tasks it is the
            task's own prompt, i.e. exactly what the thinking-on escalation
            sees, so the decide suffix reuses the escalation's cache.
  question  the typed question put in the decide suffix
  options   option texts, shown as [1] .. [K]
  gold      0-based index into options

kinds: "choice" (MMLU-Pro, BBH lettered, LSAT, decision-v7 choice/score),
"binary" (BBH yes/no-style, decision-v7 noul), "verify" ("Is the final answer
X?" for GSM8K, MATH and BBH's free-form subtasks; X is the gold answer or a
negative, see verify_candidate).

render / option_text are ported from Kev (github.com/jaredpalmer/kev,
kev/api.py, Apache-2.0) so decision-v7 records read as Kev trained on them.
"""

from __future__ import annotations

import hashlib
import importlib
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

CHOICE_Q = "Which option is correct?"
YES_NO = ["Yes", "No"]


# --- Kev rendering (Apache-2.0) -------------------------------------------------

def render(v, indent: int = 0) -> str:
    """Flatten str | object | array into text. Field names are kept as labels."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list))
                     else f"{pad}{k}: {render(x)}" for k, x in v.items())


def option_text(name: str, desc) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


# --- decision-v7 ------------------------------------------------------------------

def from_decision_v7(rec: dict) -> list[dict]:
    """One record per question of a Kev decision-v7 record."""
    out = []
    base = rec["_meta"]["id"]
    for qid, q in rec["questions"].items():
        instr = q["instructions"]
        instr = render(instr) if not isinstance(instr, dict) else \
            " ".join(str(instr[k]) for k in ("question", "focus") if instr.get(k))
        c = q.get("criteria")
        if q["type"] == "noul":
            c = c or {}
            opts = [option_text("Yes", c.get("true")), option_text("No", c.get("false"))]
            gold, kind = (0 if q["label"] else 1), "binary"
        elif q["type"] == "choice":
            names = list(c)
            opts = [option_text(k, v) for k, v in c.items()]
            gold, kind = names.index(q["label"]), "choice"
        else:                                   # score: ordinal levels
            opts = [render(x) for x in c]
            gold, kind = int(q["label"]), "choice"
        out.append({"id": f"dv7:{base}:{qid}", "task": "decision_v7", "family": q["src"],
                    "kind": kind, "context": render(rec["state"]), "question": instr,
                    "options": opts, "gold": gold, "candidate": None,
                    "candidate_is_gold": None})
    return out


# --- our tasks ----------------------------------------------------------------------

def _letter_texts(prompt: str, letters: list[str]) -> list[str] | None:
    texts = []
    for L in letters:
        m = re.search(rf"^\s*\({L}\)\s*(.+?)\s*$", prompt, re.MULTILINE)
        if not m:
            return None
        texts.append(m.group(1))
    return texts


def _hash01(s: str) -> float:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) / 16 ** 8


def perturb(gold: str, key: str) -> str | None:
    """A deterministic wrong numeric answer (±1, ×10, sign), or None."""
    try:
        x = float(gold.replace(",", ""))
    except ValueError:
        return None
    opts = [x + 1, x - 1, x * 10, -x if x else x + 2]
    y = opts[int(_hash01(key + ":p") * len(opts))]
    return str(int(y)) if float(y).is_integer() else str(y)


def verify_candidate(item: dict, gold: str, nothink_answer: str | None,
                     nothink_correct: bool | None) -> tuple[str, bool] | None:
    """(candidate, is_gold) for a verify item: half gold, half negative by hash.

    The negative is the model's own wrong no-think answer when there is one
    (a hard negative), else a numeric perturbation of the gold; None if
    neither exists (non-numeric gold, no wrong no-think answer): such items
    keep the gold candidate.
    """
    want_neg = _hash01(item["key"] + ":neg") < 0.5
    if want_neg:
        if nothink_answer and nothink_correct is False:
            return nothink_answer.strip(), False
        p = perturb(gold, item["key"])
        if p is not None and p != gold:
            return p, False
    return gold, True


def from_task(task: str, item: dict, nothink: dict | None = None) -> dict | None:
    """Typed record for one pool item, or None if it has no S1 form.

    ``nothink`` is the item's no-think row ({"answer", "correct"}), used only
    for verify negatives.
    """
    from capture_inference_thinking import _prompt_default
    mod = importlib.import_module(f"tasks.{task}")
    context = _prompt_default(mod, item)
    base = {"id": f"{task}:{item['key']}", "task": task, "context": context,
            "candidate": None, "candidate_is_gold": None}
    fam = item.get("category") or item.get("subtask") or item.get("subject") or task

    if task in ("mmlu_pro", "bbh"):
        from capture_readout import answer_spec
        spec = answer_spec(task, item)
        if spec["covered"]:
            if spec["kind"] == "letter":
                texts = _letter_texts(item["question"], spec["options"])
                if texts is None:
                    return None
                return {**base, "family": fam, "kind": "choice", "question": CHOICE_Q,
                        "options": texts, "gold": spec["options"].index(spec["gold"])}
            return {**base, "family": fam, "kind": "binary", "question": CHOICE_Q,
                    "options": list(spec["options"]),
                    "gold": spec["options"].index(spec["gold"])}
        if task == "mmlu_pro" or spec["kind"] not in ("number", "dyck", "word_list"):
            return None                             # unparseable options
        gold = str(item["answer"]).strip()          # BBH free-form: verify
    elif task == "lsat":
        import ast
        ch = item["choices"]
        ch = ast.literal_eval(ch) if isinstance(ch, str) else list(ch)
        texts = [re.sub(r"^\([A-E]\)\s*", "", c).strip() for c in ch]
        gold = "ABCDE".index(str(item["answer"]).strip())
        return {**base, "family": "lsat", "kind": "choice", "question": CHOICE_Q,
                "options": texts, "gold": gold}
    elif task == "gsm8k":
        gold = str(item.get("gold") or item["answer"]).strip()
    elif task == "math500":
        gold = str(item["answer"]).strip()
    else:
        raise ValueError(task)

    nt = nothink or {}
    # The no-think row keeps the response tail; the hard negative is the
    # answer the task's own extractor reads from it.
    nt_ans = mod.extract_prediction(nt["answer"]) if nt.get("answer") else None
    cand = verify_candidate(item, gold, nt_ans, nt.get("correct"))
    x, is_gold = cand
    return {**base, "family": fam, "kind": "verify",
            "question": f"Is the final answer {x}?", "options": list(YES_NO),
            "gold": 0 if is_gold else 1, "candidate": x, "candidate_is_gold": is_gold}

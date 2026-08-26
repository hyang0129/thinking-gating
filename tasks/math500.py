"""MATH-500 — competition mathematics (mid-difficulty reasoning task).

Self-contained loader for `HuggingFaceH4/MATH-500`, the 500-problem evaluation
subset of Hendrycks MATH.

Chosen to sit between GSM8K (which Qwen3-8B already solves ~86% of the time
without thinking, leaving almost nothing for a router to skip) and LSAT (which
it solves ~16% of the time, leaving almost nothing to route away). A task the
model gets partly right unaided is where thinking-mode routing is actually a
live decision.

It is also the only task here that ships a **real difficulty label** — `level`
1-5, assigned by the dataset authors — so the difficulty-stratified confound
check no longer rests on a heuristic proxy.

Upstream schema (single 500-row "test" split):
    problem   str   the question
    solution  str   worked solution, final answer inside \\boxed{}
    answer    str   the bare answer expression, e.g. "\\left( 3, \\frac{\\pi}{2} \\right)"
    subject   str   Precalculus, Algebra, ...
    level     int   1 (easiest) - 5 (hardest)
"""

from __future__ import annotations

import re

DATASET_ID = "HuggingFaceH4/MATH-500"
DATASET_CONFIG = "default"

PROMPT_TEMPLATE = (
    "Solve the following mathematics problem. "
    "Put your final answer inside \\boxed{{}}.\n\n"
    "Problem: {question}"
)

_ANSWER_MARKER = re.compile(r"(?:final answer|answer)\s*(?:is)?\s*[:=]?\s*", re.IGNORECASE)


def format_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip())


def extract_boxed(text: str) -> str | None:
    r"""Return the content of the LAST \boxed{...}, brace-matched.

    A regex cannot do this correctly: answers routinely nest braces, as in
    \boxed{\frac{1}{2}}, and a greedy or lazy pattern gets one of them wrong.
    Scan for the final \boxed and walk the braces.
    """
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    brace = text.find("{", idx)
    if brace == -1:
        # \boxed12 style — take the rest of the token
        rest = text[idx + len("\\boxed"):].strip()
        return rest.split()[0] if rest else None
    depth, out = 0, []
    for ch in text[brace:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(ch)
    return "".join(out) or None  # unbalanced (truncated mid-answer)


def normalize_answer(expr: str | None) -> str | None:
    r"""Canonicalize a LaTeX answer so cosmetic differences stop mattering.

    Handles the variation that shows up constantly in practice: \left/\right
    decorations, \dfrac vs \frac, spacing macros, $ delimiters, \text{}
    wrappers, trailing punctuation, and "0.50" vs ".5".
    """
    if expr is None:
        return None
    s = expr.strip()
    for token in (r"\left", r"\right", r"\!", r"\,", r"\;", r"\:", r"$", " ", "\n"):
        s = s.replace(token, "")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{([^}]*)\}", r"\1", s)
    s = s.replace(r"\%", "").replace("%", "")
    s = s.replace(r"^{\circ}", "").replace(r"^\circ", "")
    s = s.rstrip(".")
    s = re.sub(r"\\+$", "", s)
    if s.endswith("}") and s.count("{") < s.count("}"):
        s = s[:-1]
    # Numeric forms: compare by value so 0.5, .5 and 1/2-as-decimal agree.
    try:
        value = float(s.replace(",", ""))
        return str(int(value)) if value == int(value) else str(value)
    except ValueError:
        pass
    return s


def extract_prediction(generation: str) -> str | None:
    """Pull the model's final answer out of a generation."""
    if not generation:
        return None
    boxed = extract_boxed(generation)
    if boxed is not None:
        return normalize_answer(boxed)
    hits = list(_ANSWER_MARKER.finditer(generation))
    if hits:
        tail = generation[hits[-1].end():].strip().splitlines()
        if tail:
            return normalize_answer(tail[0])
    last = [ln for ln in generation.strip().splitlines() if ln.strip()]
    return normalize_answer(last[-1]) if last else None


_FRAC_RE = re.compile(r"^-?\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}$")


def as_float(expr: str | None) -> float | None:
    """Best-effort numeric value of a simple answer, else None.

    Only plain numbers and single fractions — enough to reconcile 0.5 with
    \\frac{1}{2}, which is the common cosmetic mismatch, without pulling in a
    symbolic evaluator whose failure modes would be far harder to audit.
    """
    if not expr:
        return None
    s = expr.strip()
    try:
        return float(s.replace(",", ""))
    except ValueError:
        pass
    m = _FRAC_RE.match(s)
    if m:
        try:
            num, den = float(m.group(1)), float(m.group(2))
            value = num / den if den else None
            return -value if value is not None and s.startswith("-") else value
        except (ValueError, ZeroDivisionError):
            return None
    if s.count("/") == 1:
        left, right = s.split("/")
        try:
            return float(left) / float(right)
        except (ValueError, ZeroDivisionError):
            return None
    return None


def is_correct(generation: str, answer: str) -> bool:
    """Exact match on the normalized answer expression.

    Deliberately strict: no symbolic algebra, no equivalence solving. A false
    negative costs a mislabeled row; a false positive from loose matching would
    quietly inflate every accuracy number downstream.
    """
    gold = normalize_answer(answer if "\\boxed" not in str(answer)
                            else extract_boxed(str(answer)))
    if gold is None or gold == "":
        return False
    pred = extract_prediction(generation)
    if pred is None:
        return False
    if pred == gold:
        return True
    pred_val, gold_val = as_float(pred), as_float(gold)
    if pred_val is not None and gold_val is not None:
        return abs(pred_val - gold_val) < 1e-6
    return False


def difficulty(row: dict) -> str:
    """Map the dataset's own 1-5 level onto the shared three-bucket scheme."""
    level = row.get("level")
    if level is None:
        return "medium"
    level = int(level)
    if level <= 2:
        return "easy"
    if level == 3:
        return "medium"
    return "hard"


def load_math500(split: str = "test") -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=split)
    rows = []
    for idx, row in enumerate(ds):
        rows.append({
            "key": f"math500-{split}-{idx}",
            "question": row["problem"],
            "answer": row["answer"],
            "difficulty": difficulty(row),
            "level": int(row["level"]),
            "subject": row["subject"],
        })
    return rows

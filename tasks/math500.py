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


_ROW_BREAK = "#ROW#"
_SPACING = (r"\left", r"\right", r"\displaystyle", r"\!", r"\,", r"\;", r"\:", r"\ ", "~")
_TEXT_WRAPPER = re.compile(r"\\(?:text|textbf|textit|textrm|mbox|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}")
# A unit written in a text wrapper after a quantity: "864 \mbox{ inches}^2",
# "5.4 \text{ cents}", "\frac{270}7\text{ degrees}", "12^{th} \text{ grade}".
_TRAILING_UNIT = re.compile(
    r"(?<=[\d}])(?:\s|\\[,;:! ])*\\(?:text|mbox|mathrm|textrm)\s*\{\s*[A-Za-z][A-Za-z ]*\}(?:\^\{?\d\}?)?\s*$")
_ORDINAL = re.compile(r"(\d)\^\{?\s*(?:\\(?:text|mathrm|rm)\s*\{?\s*)?(?:st|nd|rd|th)\s*\}?\}?")
_DEGREES = re.compile(r"\^\s*\{?\s*\\circ\s*\}?|°|\\degree")
# Shorthand arguments: "\frac65", "\frac 34", "\frac9{19}", "\frac{270}7", "\sqrt2".
_FRAC_ARG = r"(\{[^{}]*\}|\\[A-Za-z]+|[0-9A-Za-z])"
_FRAC_SHORTHAND = re.compile(r"\\frac" + _FRAC_ARG + _FRAC_ARG)
_SQRT_SHORTHAND = re.compile(r"\\sqrt(\\[A-Za-z]+|[0-9A-Za-z])")
_SLASH_FRACTION = re.compile(r"(?<![\w.])(\d+)/(\d+)(?![\w{.])")
_THOUSANDS = re.compile(r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
# "x=", "(a,b,c,d)=", "\square=", "x\in" — a bare variable naming the answer.
_TUPLE_PREFIX = re.compile(r"^\((?:[A-Za-z],)*[A-Za-z]\)=(?=.)")
_VAR_PREFIX = re.compile(r"^(?:[A-Za-z](?:_\{?\w+\}?)?|\\square)(?:=|\\in)(?=.)")
_BASE_SUBSCRIPT = re.compile(r"^([0-9A-Fa-f]+)_\{?\d+\}?$")
_PAREN_LETTER = re.compile(r"^\(([A-Z])\)$")


def _brace(arg: str) -> str:
    return arg if arg.startswith("{") else "{" + arg + "}"


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    """Split on `sep` outside any (), [], {} nesting."""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _canonical_number(s: str) -> str | None:
    """'0.50' -> '0.5', '24.0' -> '24', '32,348' -> '32348'; None if not a number."""
    t = s.replace(",", "") if _THOUSANDS.match(s) else s
    try:
        value = float(t)
    except ValueError:
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return str(int(value)) if value == int(value) else str(value)


def _normalize_item(s: str) -> str:
    """Canonicalize one element of a (possibly comma-separated) answer."""
    s = _VAR_PREFIX.sub("", s)
    s = s.rstrip(".")
    m = _BASE_SUBSCRIPT.match(s)
    if m:
        s = m.group(1)
    m = _PAREN_LETTER.match(s)
    if m:
        s = m.group(1)
    number = _canonical_number(s)
    return number if number is not None else s


def normalize_answer(expr: str | None) -> str | None:
    r"""Canonicalize a LaTeX answer so cosmetic differences stop mattering.

    Applied identically to the gold answer and the prediction. Handles:
    \left/\right and spacing macros, $ and \$ delimiters, \dfrac/\tfrac,
    shorthand \frac65 / \frac 34 / \frac9{19} / \sqrt2, simple a/b fractions,
    \text{}/\mbox{}/\textbf{} wrappers, a trailing unit in a text wrapper
    ("\text{ cm}^2"), degree signs and ordinals, percent signs, "x=" / "x \in" /
    "(a,b)=" prefixes, base subscripts ("4210_{5}"), "(C)" option letters, and
    numeric forms ("0.50" vs ".5", "32,348").
    """
    if expr is None:
        return None
    s = expr.strip()
    # Matrix/array row breaks first, so "\\ " is not eaten as the "\ " space.
    s = s.replace("\\\\", _ROW_BREAK)
    s = s.replace(r"\$", "").replace("$", "")
    s = _DEGREES.sub("", s)
    s = _ORDINAL.sub(r"\1", s)
    while True:
        stripped = _TRAILING_UNIT.sub("", s)
        if stripped == s:
            break
        s = stripped
    for token in _SPACING:
        s = s.replace(token, "")
    s = re.sub(r"\s+", "", s)
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    while True:
        unwrapped = _TEXT_WRAPPER.sub(r"\1", s)
        if unwrapped == s:
            break
        s = unwrapped
    s = _FRAC_SHORTHAND.sub(lambda m: r"\frac" + _brace(m.group(1)) + _brace(m.group(2)), s)
    s = _SQRT_SHORTHAND.sub(lambda m: r"\sqrt{" + m.group(1) + "}", s)
    s = _SLASH_FRACTION.sub(r"\\frac{\1}{\2}", s)
    s = s.replace(r"\%", "").replace("%", "")
    s = s.rstrip(".")
    s = re.sub(r"\\+$", "", s)
    if s.endswith("}") and s.count("{") < s.count("}"):
        s = s[:-1]
    s = _TUPLE_PREFIX.sub("", s)
    if _THOUSANDS.match(s):
        return _canonical_number(s)
    return ",".join(_normalize_item(item) for item in _split_top_level(s))


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


_FRAC_RE = re.compile(r"^(-?)\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}$")


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
        return float(s.replace(",", "") if _THOUSANDS.match(s) else s)
    except ValueError:
        pass
    m = _FRAC_RE.match(s)
    if m:
        try:
            value = float(m.group(2)) / float(m.group(3))
        except (ValueError, ZeroDivisionError):
            return None
        return -value if m.group(1) else value
    if s.count("/") == 1:
        left, right = s.split("/")
        try:
            return float(left) / float(right)
        except (ValueError, ZeroDivisionError):
            return None
    return None


def _expand_pm(items: list[str]) -> list[str]:
    r"""'3\pm2\sqrt{2}' -> ['3+2\sqrt{2}', '3-2\sqrt{2}'] (one \pm per item)."""
    out = []
    for item in items:
        if item.count(r"\pm") == 1:
            left, right = item.split(r"\pm")
            out += [f"{left}+{right}", f"{left}-{right}"]
        else:
            out.append(item)
    return out


def _scalar_match(pred: str, gold: str) -> bool:
    if pred == gold:
        return True
    pred_val, gold_val = as_float(pred), as_float(gold)
    if pred_val is not None and gold_val is not None:
        return abs(pred_val - gold_val) < 1e-6
    # Word answers ("\text{East}" vs "\text{east}") compare case-insensitively.
    if re.fullmatch(r"[A-Za-z]{2,}", pred) and re.fullmatch(r"[A-Za-z]{2,}", gold):
        return pred.lower() == gold.lower()
    return False


def answers_match(pred: str, gold: str) -> bool:
    """Compare two normalized answers; comma lists match as unordered multisets."""
    if _scalar_match(pred, gold):
        return True
    pred_items = _expand_pm(_split_top_level(pred))
    gold_items = _expand_pm(_split_top_level(gold))
    if len(gold_items) < 2 and len(pred_items) < 2:
        return False
    if len(pred_items) != len(gold_items):
        return False
    remaining = list(gold_items)
    for item in pred_items:
        hit = next((i for i, g in enumerate(remaining) if _scalar_match(item, g)), None)
        if hit is None:
            return False
        remaining.pop(hit)
    return True


def is_correct(generation: str, answer: str) -> bool:
    """Match of the normalized final answer against the normalized gold.

    Still no symbolic algebra: "\\frac{9a+11}{20}" does not match
    "\\frac{11+9a}{20}". A false negative costs a mislabeled row; a false
    positive from loose matching would quietly inflate every accuracy number
    downstream. The normalization is purely notational and applied to both
    sides, and comma-separated answers compare as unordered lists.
    """
    gold = normalize_answer(answer if "\\boxed" not in str(answer)
                            else extract_boxed(str(answer)))
    if gold is None or gold == "":
        return False
    pred = extract_prediction(generation)
    if pred is None or pred == "":
        return False
    return answers_match(pred, gold)


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

"""MMLU-Pro — 10-option multiple choice across 14 academic domains.

Self-contained loader for `TIGER-Lab/MMLU-Pro`. Harder than MMLU, and with ten
options rather than four the guess floor drops from 25% to 10%, so accuracy
differences between thinking-off and thinking-on are much less likely to be
chance.

It also contributes the knowledge-heavy end of the task spectrum. GSM8K, MATH,
and LSAT all reward step-by-step derivation; if a prefill probe is detecting
"this needs reasoning" rather than "I happen to know this", the two kinds of
task should behave differently.

Upstream schema (test split, ~12k rows):
    question      str        the stem
    options       list[str]  up to 10 options, unlabeled
    answer        str        the correct letter, "A"-"J"
    answer_index  int        index into options
    category      str        business, physics, law, ...
"""

from __future__ import annotations

import re

DATASET_ID = "TIGER-Lab/MMLU-Pro"
DATASET_CONFIG = "default"

LETTERS = [chr(ord("A") + i) for i in range(10)]

PROMPT_TEMPLATE = (
    "Answer the following multiple choice question. "
    "Finish your reply with the letter of the correct option on its own line, "
    "written as 'Answer: <letter>'.\n\n"
    "{question}"
)

# A letter after a word-bounded "answer" marker. The marker is case-insensitive;
# the letter is NOT, so prose ("the answer is a ...", "the answer I got") does
# not read as an option. An optional "option"/"choice" word, $\boxed{...},
# \text{...} and "**" wrappers are allowed between marker and letter (a bare
# "$" is not: "the answer is $ A i $" is math, not option A).
_ANSWER_MARKER = re.compile(
    r"(?<![<\w])(?i:answer)\b\s*(?:\**\s*(?i:is)\b)?\s*\**\s*(?P<sep>[:=])?")
_LETTER_AFTER_MARKER = re.compile(
    r"\s*\**\s*(?:(?i:option|choice)\s+)?"
    r"(?:\$*\s*(?P<boxed>\\boxed\s*\{\s*))?(?:\\text(?:bf)?\s*\{\s*)?"
    r"(?P<open>\()?\s*(?P<letter>[A-J])\s*(?P<close>\))?(?![A-Za-z0-9])")
# \boxed{J} / \boxed{\text{(J)}} anywhere — Qwen3 closes with "$$\boxed{J}$$".
_BOXED_LETTER = re.compile(r"\\boxed\s*\{\s*(?:\\text(?:bf)?\s*\{\s*)?\(?\s*([A-J])\s*\)?\s*\}")
# Last resort: a final line that is nothing but the letter ("(C)", "C.", "**C**",
# "Option C"). The old fallback took the last "(X)" / "X." anywhere, which read
# hydrogen "(H)" or initials "I.F." out of prose as an answer.
_FINAL_LINE_LETTER = re.compile(
    r"^\W*(?:(?i:the\s+)?(?:(?i:correct)\s+)?(?i:option|choice)\s*(?:(?i:is)\s*)?:?\s*)?"
    r"\(?([A-J])\)?\W*$")


def render_question(question: str, options: list[str]) -> str:
    """Inline the options as a lettered list — the model never sees raw indices."""
    lines = [question.strip(), ""]
    lines += [f"({LETTERS[i]}) {opt}" for i, opt in enumerate(options) if i < len(LETTERS)]
    return "\n".join(lines)


def format_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip())


def extract_prediction(generation: str) -> str | None:
    """The answer letter: last marked letter or \\boxed letter, else a bare final line."""
    if not generation:
        return None
    candidates: list[tuple[int, str]] = []
    for hit in _ANSWER_MARKER.finditer(generation):
        m = _LETTER_AFTER_MARKER.match(generation, hit.end())
        if not m:
            continue
        # "answer: B" / "answer is B" need the separator; without one the
        # letter must be wrapped ("answer (B)", "answer \boxed{B}"), so
        # "the answer I got" is not read as option I.
        explicit = hit.group("sep") or re.search(r"(?i:\bis)\s*\**\s*$", hit.group(0))
        wrapped = m.group("boxed") or (m.group("open") and m.group("close"))
        if explicit or wrapped:
            candidates.append((hit.start(), m.group("letter")))
    for m in _BOXED_LETTER.finditer(generation):
        candidates.append((m.start(), m.group(1)))
    if candidates:
        return max(candidates)[1]
    lines = [ln for ln in generation.strip().splitlines() if ln.strip()]
    if lines:
        m = _FINAL_LINE_LETTER.match(lines[-1].strip())
        if m:
            return m.group(1)
    return None


def is_correct(generation: str, answer: str) -> bool:
    gold = str(answer).strip().upper()
    if gold not in LETTERS:
        return False
    return extract_prediction(generation) == gold


def difficulty(row: dict, thresholds: tuple[int, int] = (900, 1400)) -> str:
    """Heuristic bucket from prompt length.

    MMLU-Pro ships no difficulty field. Total stem+options length is a crude
    proxy for how much a question demands; it is reported as a stratifier only,
    never trained on, and the thresholds come from the corpus length terciles.
    """
    size = len(row.get("question", "")) + sum(len(o) for o in row.get("options", []))
    if size < thresholds[0]:
        return "easy"
    if size < thresholds[1]:
        return "medium"
    return "hard"


def load_mmlu_pro(split: str = "test", max_rows: int | None = None,
                  shuffle_seed: int = 0) -> list[dict]:
    """Load MMLU-Pro, deterministically shuffled.

    The upstream split is ordered by category, so a --max-samples prefix would
    be several domains rather than a sample of the benchmark. Shuffling with a
    fixed seed makes any prefix a representative draw and keeps it reproducible.
    """
    import random

    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=split)
    order = list(range(len(ds)))
    random.Random(shuffle_seed).shuffle(order)
    ds = ds.select(order)
    rows = []
    for idx, row in enumerate(ds):
        if max_rows is not None and len(rows) >= max_rows:
            break
        gold_idx = int(row["answer_index"])
        if gold_idx >= len(LETTERS) or gold_idx >= len(row["options"]):
            continue
        rows.append({
            "key": f"mmlupro-{split}-{row['question_id']}",
            "question": render_question(row["question"], list(row["options"])),
            "answer": LETTERS[gold_idx],
            "difficulty": difficulty(row),
            "category": row["category"],
        })
    return rows

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

_MARKED_LETTER = re.compile(
    r"(?:answer|choice|option)\s*(?:is)?\s*[:=]?\s*\**\s*\(?([A-J])\)?\b", re.IGNORECASE)
_BARE_LETTER = re.compile(r"\(([A-J])\)|(?<![A-Za-z])([A-J])[).]")


def render_question(question: str, options: list[str]) -> str:
    """Inline the options as a lettered list — the model never sees raw indices."""
    lines = [question.strip(), ""]
    lines += [f"({LETTERS[i]}) {opt}" for i, opt in enumerate(options) if i < len(LETTERS)]
    return "\n".join(lines)


def format_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip())


def extract_prediction(generation: str) -> str | None:
    if not generation:
        return None
    marked = list(_MARKED_LETTER.finditer(generation))
    if marked:
        return marked[-1].group(1).upper()
    bare = list(_BARE_LETTER.finditer(generation))
    if bare:
        return (bare[-1].group(1) or bare[-1].group(2)).upper()
    stripped = generation.strip().rstrip(".").upper()
    return stripped if stripped in LETTERS else None


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

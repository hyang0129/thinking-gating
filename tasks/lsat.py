"""LSAT Analytical Reasoning ("logic games") — secondary / transfer task.

Self-contained loader for `hails/agieval-lsat-ar` (the AGIEval LSAT-AR split).
Chosen because it is pure deduction with no world knowledge, which is where
thinking mode buys the most, and because it is far enough from GSM8K to be a
real cross-task transfer test.

Upstream schema (single "test" split, 230 rows):
    query   str        passage + question + inlined "Answer Choices: (A)... (B)..."
    choices list[str]  each entry prefixed with its own "(A)" label
    gold    list[int]  index/indices into `choices`
"""

from __future__ import annotations

import re

DATASET_ID = "hails/agieval-lsat-ar"
DATASET_CONFIG = "default"

LETTERS = ["A", "B", "C", "D", "E"]

PROMPT_TEMPLATE = (
    "Solve the following logical reasoning problem. "
    "Finish your reply with the letter of the best answer on its own line, "
    "written as 'Answer: <letter>'.\n\n"
    "{question}"
)

# "Answer: B", "answer is (B)", "**Answer:** B." — capture the letter after the marker.
_MARKED_LETTER = re.compile(
    r"(?:answer|choice|option)\s*(?:is)?\s*[:=]?\s*\**\s*\(?([A-E])\)?\b",
    re.IGNORECASE,
)
# Bare "(C)" / "C)" fallback for replies that skip the marker.
_BARE_LETTER = re.compile(r"\(([A-E])\)|(?<![A-Za-z])([A-E])\)")


def format_prompt(question: str) -> str:
    """Render the raw (pre-chat-template) prompt for one question."""
    return PROMPT_TEMPLATE.format(question=question.strip())


def extract_prediction(generation: str) -> str | None:
    """Pull the predicted answer letter out of a model generation."""
    if not generation:
        return None

    marked = list(_MARKED_LETTER.finditer(generation))
    if marked:
        return marked[-1].group(1).upper()

    bare = list(_BARE_LETTER.finditer(generation))
    if bare:
        letter = bare[-1].group(1) or bare[-1].group(2)
        return letter.upper()

    # Last resort: a reply that is nothing but the letter.
    stripped = generation.strip().rstrip(".").upper()
    return stripped if stripped in LETTERS else None


def is_correct(generation: str, answer: str) -> bool:
    """Letter match against the gold choice."""
    gold = str(answer).strip().upper()
    if gold not in LETTERS:
        return False
    return extract_prediction(generation) == gold


def difficulty(row: dict) -> str:
    """Heuristic difficulty bucket from constraint count in the setup.

    LSAT-AR ships no difficulty field. Sentence count in the passage before the
    question tracks the number of constraints a solver has to juggle, which is
    the closest available proxy for how much reasoning the item demands.
    """
    query = row.get("query", "")
    setup = query.split("Q:")[0] if "Q:" in query else query
    constraints = len([s for s in re.split(r"(?<=[.!?])\s+", setup) if s.strip()])
    if constraints <= 4:
        return "easy"
    if constraints <= 7:
        return "medium"
    return "hard"


def load_lsat_logic(split: str = "test") -> list[dict]:
    """Load LSAT-AR and return rows matching the task contract.

    Note: upstream ships only a "test" split; any other split name will raise
    from `datasets`.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=split)
    rows = []
    for idx, row in enumerate(ds):
        gold_indices = row["gold"]
        if not gold_indices:
            continue
        gold_idx = int(gold_indices[0])
        if gold_idx >= len(LETTERS):
            continue
        rows.append(
            {
                "key": f"lsat-{split}-{idx}",
                "question": row["query"],
                "answer": LETTERS[gold_idx],
                "choices": list(row["choices"]),
                "difficulty": difficulty(row),
            }
        )
    return rows

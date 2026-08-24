"""GSM8K — grade-school math word problems (primary task).

Self-contained loader: pulls `openai/gsm8k` (config "main") from HuggingFace and
normalizes it to the row contract in tasks/__init__.py.

Gold answers in GSM8K look like:

    Natalia sold 48/2 = <<48/2=24>>24 clips in May.
    #### 72

so the answer is the number after the "####" marker.
"""

from __future__ import annotations

import re

DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"

PROMPT_TEMPLATE = (
    "Solve the following grade school math problem. "
    "Finish your reply with the final numeric answer on its own line, "
    "written as 'Answer: <number>'.\n\n"
    "Problem: {question}"
)

# Answers may carry thousands separators, a currency symbol, or a trailing period.
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
_FINAL_MARKER = re.compile(r"(?:answer|result)\s*(?:is)?\s*[:=]?\s*", re.IGNORECASE)


def format_prompt(question: str) -> str:
    """Render the raw (pre-chat-template) prompt for one question."""
    return PROMPT_TEMPLATE.format(question=question.strip())


def normalize_number(text: str) -> str | None:
    """Strip formatting from a numeric string; return None if it isn't numeric."""
    if text is None:
        return None
    cleaned = text.strip().rstrip(".").replace(",", "").replace("$", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    # Render 24.0 and 24 identically so string comparison is safe.
    return str(int(value)) if value == int(value) else str(value)


def extract_gold(answer_field: str) -> str | None:
    """Pull the gold number out of a GSM8K answer field (text after '####')."""
    if "####" in answer_field:
        return normalize_number(answer_field.rsplit("####", 1)[1])
    matches = _NUMBER.findall(answer_field)
    return normalize_number(matches[-1]) if matches else None


def extract_prediction(generation: str) -> str | None:
    """Pull the predicted number out of a model generation.

    Prefers a number following an explicit 'Answer:'-style marker; otherwise
    falls back to the last number in the text, which is where a chain-of-thought
    reply lands its result.
    """
    if not generation:
        return None

    tail = generation
    marker_hits = list(_FINAL_MARKER.finditer(generation))
    if marker_hits:
        tail = generation[marker_hits[-1].end():]
        after_marker = _NUMBER.search(tail)
        if after_marker:
            return normalize_number(after_marker.group(0))

    matches = _NUMBER.findall(generation)
    return normalize_number(matches[-1]) if matches else None


def is_correct(generation: str, answer: str) -> bool:
    """Exact match on the normalized final number."""
    gold = extract_gold(answer) if "####" in str(answer) else normalize_number(str(answer))
    if gold is None:
        return False
    pred = extract_prediction(generation)
    return pred is not None and pred == gold


def difficulty(row: dict) -> str:
    """Heuristic difficulty bucket from reasoning-chain length.

    GSM8K ships no difficulty field, so we stratify on the number of reasoning
    steps in the gold solution — the standard proxy in the literature.
    """
    solution = row.get("answer", "")
    steps = len([ln for ln in solution.split("\n") if ln.strip() and not ln.startswith("####")])
    if steps <= 2:
        return "easy"
    if steps <= 4:
        return "medium"
    return "hard"


def load_gsm8k(split: str = "test") -> list[dict]:
    """Load GSM8K and return rows matching the task contract."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=split)
    rows = []
    for idx, row in enumerate(ds):
        rows.append(
            {
                "key": f"gsm8k-{split}-{idx}",
                "question": row["question"],
                "answer": row["answer"],
                "gold": extract_gold(row["answer"]),
                "difficulty": difficulty(row),
            }
        )
    return rows

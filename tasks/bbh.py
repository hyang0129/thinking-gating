"""BIG-Bench Hard — the tasks BIG-Bench models failed at without reasoning.

Self-contained loader for `lukaemon/bbh`. BBH was assembled precisely
because chain-of-thought turns these tasks around, which makes it the sharpest
available test of a "will thinking help" probe: the helped population should be
large and it should not be explainable by surface difficulty alone.

The suite is 27 subtasks with different answer shapes — "(A)" style multiple
choice, Yes/No, True/False, and short free-form strings — so `is_correct`
normalizes then exact-matches, with a letter path for the multiple-choice ones.
By default a balanced sample is drawn across all subtasks rather than reading
them in order, so a capture shard is never one subtask's quirks.

Upstream schema (one config per subtask, "test" split, 6511 rows total):
    input   str
    target  str

Note the source: `maveriq/bigbenchhard` carries a loading script, and datasets
4.x refuses those outright ("Dataset scripts are no longer supported"), so this
uses the parquet-native `lukaemon/bbh` mirror instead.
"""

from __future__ import annotations

import re

DATASET_ID = "lukaemon/bbh"

# The 27 BBH subtasks, in the order the paper lists them.
SUBTASKS = (
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate",
    "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting",
)

PROMPT_TEMPLATE = (
    "{question}\n\n"
    "Finish your reply with the final answer on its own line, "
    "written as 'Answer: <answer>'. If the question offers lettered options, "
    "give the letter in parentheses, e.g. 'Answer: (A)'."
)

_ANSWER_MARKER = re.compile(r"answer\s*(?:is)?\s*[:=]?\s*\**\s*", re.IGNORECASE)
_PAREN_LETTER = re.compile(r"\(([A-Z])\)")


def format_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip())


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles/markup — then compare exactly."""
    s = (text or "").strip().lower()
    s = s.replace("**", "").replace("$", "")
    s = re.sub(r"^(the answer is|answer)\s*[:=]?\s*", "", s)
    s = s.strip().strip(".").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_prediction(generation: str) -> str | None:
    """Take the text after the last answer marker, else the last non-empty line."""
    if not generation:
        return None
    hits = list(_ANSWER_MARKER.finditer(generation))
    if hits:
        tail = generation[hits[-1].end():].strip().splitlines()
        if tail and tail[0].strip():
            return tail[0].strip()
    lines = [ln for ln in generation.strip().splitlines() if ln.strip()]
    return lines[-1].strip() if lines else None


def is_correct(generation: str, answer: str) -> bool:
    """Normalized exact match, with a letter path for multiple-choice targets."""
    gold_raw = str(answer).strip()
    pred_raw = extract_prediction(generation)
    if pred_raw is None:
        return False

    gold_letter = _PAREN_LETTER.fullmatch(gold_raw)
    if gold_letter:
        # Multiple choice: compare letters, accepting "(A)", "A", or "A." forms.
        hits = _PAREN_LETTER.findall(pred_raw)
        if hits:
            return hits[-1].upper() == gold_letter.group(1).upper()
        bare = normalize(pred_raw).upper().strip(".")
        return bare == gold_letter.group(1).upper()

    return normalize(pred_raw) == normalize(gold_raw)


def difficulty(row: dict, thresholds: tuple[int, int] = (250, 700)) -> str:
    """Heuristic bucket from input length — BBH ships no difficulty field."""
    size = len(row.get("input", ""))
    if size < thresholds[0]:
        return "easy"
    if size < thresholds[1]:
        return "medium"
    return "hard"


def load_bbh(split: str = "test", per_subtask: int = 20,
             subtasks: tuple[str, ...] = SUBTASKS) -> list[dict]:
    """Load a balanced sample: `per_subtask` items from each of the 27 configs.

    Balanced rather than sequential so that any capture shard — and any
    train/test split downstream — sees the whole suite instead of over-weighting
    whichever subtasks happen to sort first.
    """
    from datasets import load_dataset

    rows = []
    for name in subtasks:
        ds = load_dataset(DATASET_ID, name, split=split)
        for idx, row in enumerate(ds):
            if idx >= per_subtask:
                break
            rows.append({
                "key": f"bbh-{name}-{idx}",
                "question": row["input"],
                "answer": row["target"],
                "difficulty": difficulty(row),
                "subtask": name,
            })
    return rows

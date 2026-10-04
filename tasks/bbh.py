"""BIG-Bench Hard — the tasks BIG-Bench models failed at without reasoning.

Self-contained loader for `lukaemon/bbh`. BBH was assembled precisely
because chain-of-thought turns these tasks around, which makes it the sharpest
available test of a "will thinking help" probe: the helped population should be
large and it should not be explainable by surface difficulty alone.

The suite is 27 subtasks with different answer shapes — "(A)" style multiple
choice, Yes/No, True/False, valid/invalid, integers, bracket sequences, and
sorted word lists — so both the prompt and the grader are answer-kind aware:

  * `format_prompt` asks for a lettered option only when the question actually
    offers lettered options. The old one-size prompt ("if the question offers
    lettered options, give the letter in parentheses") was applied by models to
    the unlettered "- Yes / - No" lists, producing "(B) No", "(B)", "(Yes)".
  * `is_correct` keeps the letter path for "(A)" golds and grades every other
    gold by its kind (yes/no-style word, integer, dyck closing sequence, word
    list), tolerating the decorations models actually emit.

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

# Answer kind per subtask. Every gold in a "letter" subtask is "(X)"; every
# other subtask's gold is the bare word/number/sequence.
_LETTER_SUBTASKS = (
    "date_understanding", "disambiguation_qa", "geometric_shapes", "hyperbaton",
    "logical_deduction_five_objects", "logical_deduction_seven_objects",
    "logical_deduction_three_objects", "movie_recommendation",
    "penguins_in_a_table", "reasoning_about_colored_objects", "ruin_names",
    "salient_translation_error_detection", "snarks", "temporal_sequences",
    "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
)
SUBTASK_KIND = {
    **{name: "letter" for name in _LETTER_SUBTASKS},
    "causal_judgement": "yes_no", "navigate": "yes_no",
    "sports_understanding": "yes_no", "web_of_lies": "yes_no",
    "boolean_expressions": "true_false",
    "formal_fallacies": "valid_invalid",
    "multistep_arithmetic_two": "number", "object_counting": "number",
    "dyck_languages": "dyck",
    "word_sorting": "word_list",
}

_PROMPT_HEAD = "{question}\n\nFinish your reply with the final answer on its own line, "
# No angle-bracket placeholders: models parrot them back ("Answer: <answer>",
# "<9>", "<sorted list>"), and a parroted placeholder used to be graded.
_PROMPT_TAIL = {
    "letter": "written as 'Answer: (X)', where X is the letter of the correct option.",
    "yes_no": "written as either 'Answer: Yes' or 'Answer: No'.",
    "true_false": "written as either 'Answer: True' or 'Answer: False'.",
    "valid_invalid": "written as either 'Answer: valid' or 'Answer: invalid'.",
    "number": "written as 'Answer: ' followed by the number alone, in digits.",
    "dyck": ("written as 'Answer: ' followed by only the closing brackets that "
             "complete the sequence, separated by spaces."),
    "word_list": ("written as 'Answer: ' followed by the sorted words, "
                  "separated by single spaces."),
    None: "written as 'Answer: ' followed by your answer.",
}
# Kept for reference / backward compatibility: the pre-2026-10 prompt that
# every v3 capture was generated with.
PROMPT_TEMPLATE_V1 = (
    "{question}\n\n"
    "Finish your reply with the final answer on its own line, "
    "written as 'Answer: <answer>'. If the question offers lettered options, "
    "give the letter in parentheses, e.g. 'Answer: (A)'."
)

# Word-bounded, and never the "answer" inside a parroted "<answer>" placeholder.
_ANSWER_MARKER = re.compile(r"(?<![<\w])answer\b\s*(?:is\b)?\s*[:=]?\s*\**\s*", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"^\W*<\s*answer\s*>\W*$", re.IGNORECASE)
_PAREN_LETTER = re.compile(r"\(([A-Z])\)")
_LETTERED_OPTION = re.compile(r"^\s*\(A\)", re.MULTILINE)
_DASH_OPTIONS = re.compile(r"Options:\s*\n((?:[ \t]*-[^\n]*(?:\n|$))+)")
_BOOLEAN_EXPR = re.compile(r"^[\s()]*(?:(?:not|True|False|and|or)[\s()]*)+is\s*$")
_CATEGORICAL = {"yes", "no", "true", "false", "valid", "invalid"}
# A leading option letter: "(B) No", "(B))", "(B)". Bare "B." is not taken —
# it is indistinguishable from prose.
_LEAD_LETTER = re.compile(r"^\(\s*([A-Z])\s*\)\)?[\s.:]*(.*)$", re.DOTALL)
_DYCK_GOLD = re.compile(r"^[\s()\[\]{}<>]+$")


def dash_options(question: str | None) -> list[str] | None:
    """The unlettered '- option' list of a question, in order, or None."""
    if not question:
        return None
    m = _DASH_OPTIONS.search(question)
    if not m:
        return None
    opts = [ln.strip()[1:].strip() for ln in m.group(1).splitlines() if ln.strip().startswith("-")]
    return opts or None


def infer_answer_kind(question: str) -> str | None:
    """Answer kind from the question text alone (no subtask name needed).

    Lets `format_prompt(question)` stay a one-argument call: the capture script
    passes only the question, and the subtask's shape is visible in it.
    """
    q = question or ""
    if _LETTERED_OPTION.search(q):
        return "letter"
    opts = dash_options(q)
    if opts:
        low = {o.lower() for o in opts}
        if low == {"valid", "invalid"}:
            return "valid_invalid"
        if low == {"yes", "no"}:
            return "yes_no"
    if "parentheses are closed properly" in q and "Input:" in q:
        return "dyck"
    if q.lstrip().startswith("Sort the following words"):
        return "word_list"
    if _BOOLEAN_EXPR.match(q):
        return "true_false"
    if "tell the truth?" in q or "Is the following sentence plausible?" in q:
        return "yes_no"
    if q.rstrip().endswith("=") or re.search(r"\bHow many\b", q):
        return "number"
    return None


def format_prompt(question: str, subtask: str | None = None) -> str:
    """Render the raw prompt, asking for the answer in the subtask's own shape.

    `subtask` is optional; without it the kind is inferred from the question
    (which agrees with `SUBTASK_KIND` on every BBH item we have captured).
    """
    kind = SUBTASK_KIND.get(subtask) if subtask else None
    if kind is None:
        kind = infer_answer_kind(question)
    return (_PROMPT_HEAD + _PROMPT_TAIL[kind]).format(question=question.strip())


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles/markup — then compare exactly."""
    s = (text or "").strip().lower()
    s = s.replace("**", "").replace("$", "")
    s = re.sub(r"^(the answer is|answer)\s*[:=]?\s*", "", s)
    s = s.strip().strip(".").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_prediction(generation: str) -> str | None:
    """Text after the last real answer marker, else the last non-empty line.

    A marker whose line is only a parroted "<answer>" placeholder is skipped in
    favour of the one before it ("Answer: (C)\\n\\nAnswer: <answer>" -> "(C)").
    """
    if not generation:
        return None
    for hit in reversed(list(_ANSWER_MARKER.finditer(generation))):
        tail = generation[hit.end():].strip().splitlines()
        if tail and tail[0].strip() and not _PLACEHOLDER.match(tail[0]):
            return tail[0].strip()
    lines = [ln for ln in generation.strip().splitlines() if ln.strip()]
    return lines[-1].strip() if lines else None


def _clean(pred: str) -> str:
    s = pred.replace("**", "").replace("`", "").replace("$", "").replace("−", "-")
    s = s.strip()
    s = re.sub(r"^<\s*(.*?)\s*>", r"\1", s)  # "<9>" -> "9", "<a, b>" -> "a, b"
    return s.strip().strip(".").strip()


def _categorical_correct(pred: str, gold: str, question: str | None) -> bool:
    s = _clean(pred)
    lead = _LEAD_LETTER.match(s)
    if lead:
        rest = lead.group(2).strip()
        if re.search(r"[A-Za-z]", rest):
            s = rest
        else:
            # A bare "(B)": the B-th item of the unlettered "- option" list.
            opts = dash_options(question)
            idx = ord(lead.group(1)) - ord("A")
            if not opts or idx >= len(opts):
                return False
            s = opts[idx]
    s = re.sub(r"^\(\s*([A-Za-z]+)\s*\)", r"\1", s)  # "(No) because ..." -> "No because ..."
    low = s.lower()
    # "True and not not ( not False ) is True": the claim is the final "is X".
    final = re.search(r"\bis\s+(true|false)\W*$", low)
    if final and gold.lower() in ("true", "false"):
        return final.group(1) == gold.lower()
    words = re.findall(r"[a-z]+", low)
    return bool(words) and words[0] == gold.lower()


def _number_correct(pred: str, gold: str) -> bool:
    s = _clean(pred)
    s = s.rsplit("=", 1)[-1]
    s = re.sub(r"^\(\s*[A-Z]\s*\)\s*", "", s.strip())  # "(B) 29" -> "29"
    m = re.search(r"-?\d[\d,]*", s)
    if not m:
        return False
    try:
        return int(m.group(0).replace(",", "")) == int(gold)
    except ValueError:
        return False


def _dyck_correct(pred: str, gold: str, question: str | None) -> bool:
    def squash(t: str) -> str:
        return re.sub(r"[\s`*]", "", t)
    p, g = squash(_clean(pred)), squash(gold)
    if p == g:
        return True
    # Models often return the whole completed sequence rather than the suffix.
    if question and "Input:" in question:
        return p == squash(question.split("Input:")[-1]) + g
    return False


def _word_list_correct(pred: str, gold: str) -> bool:
    s = _clean(pred).lower()
    return [t for t in re.split(r"[\s,]+", s) if t] == gold.lower().split()


def is_correct(generation: str, answer: str, question: str | None = None) -> bool:
    """Grade one BBH reply against its target.

    Letter golds keep the original letter path. Other golds are graded by kind;
    `question` (optional) lets a bare "(B)" map onto the question's unlettered
    option list and lets a dyck reply that repeats the input be accepted.
    Without it those two cases are graded False, never guessed.
    """
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

    if normalize(pred_raw) == normalize(gold_raw):
        return True
    if gold_raw.lower() in _CATEGORICAL:
        return _categorical_correct(pred_raw, gold_raw, question)
    if re.fullmatch(r"-?\d+", gold_raw):
        return _number_correct(pred_raw, gold_raw)
    if _DYCK_GOLD.fullmatch(gold_raw):
        return _dyck_correct(pred_raw, gold_raw, question)
    return _word_list_correct(pred_raw, gold_raw)


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

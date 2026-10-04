"""Grader regression tests — one case per audited bug, plus cases the old graders got right.

The failing strings come from the 2026-10-04 grader audit of the v3 captures
(row numbers refer to those captures). No GPU, no network, no captured data:

    .venv/bin/python -m pytest tests/test_graders.py -q

If `math_verify` happens to be installed, the MATH-500 cases are also checked
against it as an oracle; it is never required.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tasks import bbh, gsm8k, math500, mmlu_pro  # noqa: E402

import generate_labels  # noqa: E402

# ---------------------------------------------------------------------------
# BBH (B2)
# ---------------------------------------------------------------------------

Q_CAUSAL = ("How would a typical person answer each of the following questions about causation?\n"
            "... Did the black wire cause the short circuit?\nOptions:\n- Yes\n- No")
Q_NAVIGATE = ("If you follow these instructions, do you return to the starting point? "
              "Always face forward. Take 1 step backward. Take 3 steps right.\nOptions:\n- Yes\n- No")
Q_FALLACY = ("\"Here comes a perfectly valid argument: ...\"\nIs the argument, given the explicitly "
             "stated premises, deductively valid or invalid?\nOptions:\n- valid \n- invalid")
Q_WEB_OF_LIES = ("Question: Sherrie tells the truth. Vernell says Sherrie tells the truth. "
                 "Does Vernell tell the truth?")
Q_SPORTS = 'Is the following sentence plausible? "Elias Lindholm beat the buzzer."'
Q_BOOLEAN = "not ( True ) and ( True ) is"
Q_DYCK = ("Complete the rest of the sequence, making sure that the parentheses are closed "
          "properly. Input: [ [")
Q_SORT = "Sort the following words alphabetically: List: syndrome therefrom"
Q_COUNT = ("I have a flute, a piano, a trombone, four stoves, a violin, an accordion, a clarinet, "
           "a drum, two lamps, and a trumpet. How many musical instruments do I have?")
Q_ARITH = "((-1 + 2 + 9 * 5) - (-2 + -4 + -4 * -7)) ="
Q_LETTER = ("Today is Christmas Eve of 1937. What is the date tomorrow in MM/DD/YYYY?\nOptions:\n"
            "(A) 12/11/1937\n(B) 12/25/1937\n(C) 01/04/1938")


@pytest.mark.parametrize("generation, gold, question", [
    ("Answer: (B) No", "No", Q_CAUSAL),
    ("Answer: (Yes)", "Yes", Q_CAUSAL),
    ("Answer: No.", "No", Q_CAUSAL),
    ("Answer: **No**", "No", Q_CAUSAL),
    ("Answer: False ()", "False", Q_BOOLEAN),
    ("Answer: (B)", "No", Q_NAVIGATE),
    ("Answer: (invalid)", "invalid", Q_FALLACY),
    ("Answer: No, Mookie Betts did not skate behind the net.", "no", Q_SPORTS),
    ("Answer: 5 objects.", "5", Q_COUNT),
    ("Answer: <9>", "9", Q_COUNT),
    ("Answer: (-16)", "-16", Q_ARITH),
    ("Answer: (B) 29", "29", Q_ARITH),
    ("Answer: (-7 * -9 + 8 * -3) * (5 + -7 - 4 * -5) = 702", "702", Q_ARITH),
    ("Answer: built, poland, swab, thunderclap", "built poland swab thunderclap", Q_SORT),
    ("Answer: <campfire contrast crowfoot purgatory scrupulous>",
     "campfire contrast crowfoot purgatory scrupulous", Q_SORT),
    # dyck: the whole completed sequence instead of the suffix
    ("Answer: [ [ ] ]", "] ]", Q_DYCK),
    ("Answer: ] ]", "] ]", Q_DYCK),
    # "<answer>" placeholder after the real answer (Nemotron rows 349/415/518)
    ("Answer: (C) \n\nAnswer: <answer>", "(C)", Q_LETTER),
])
def test_bbh_audit_false_negatives_now_correct(generation, gold, question):
    assert bbh.is_correct(generation, gold, question=question)


@pytest.mark.parametrize("generation, gold, question", [
    # Nemotron row 21 (dyck, gold ">"): the tail ">" of a parroted placeholder
    # used to be read as the answer — a false positive.
    ("Answer: <answer>", ">", Q_DYCK),
    ("Answer: (B) No", "Yes", Q_CAUSAL),
    ("Answer: (A)", "No", Q_NAVIGATE),
    ("Answer: 9 musical instruments.", "8", Q_COUNT),
    ("Answer: (B) not False", "False", Q_BOOLEAN),
    ("Answer: valid", "invalid", Q_FALLACY),
    ("Answer: [ [ < > ] ]", "] ]", Q_DYCK),
    ("Answer: arapaho, bacteria, bel, bock, burley", "arapaho bacteria bela bock burley", Q_SORT),
    # semantic maps are deliberately NOT applied (True for "tells the truth?")
    ("Answer: True", "Yes", Q_WEB_OF_LIES),
    ("Answer: <plausible>", "yes", Q_SPORTS),
])
def test_bbh_wrong_answers_stay_wrong(generation, gold, question):
    assert not bbh.is_correct(generation, gold, question=question)


def test_bbh_bare_letter_without_question_is_not_guessed():
    # The capture script calls is_correct(generation, answer) with no question;
    # a bare "(B)" for an unlettered gold then cannot be resolved, so it is wrong.
    assert not bbh.is_correct("Answer: (B)", "No")
    assert bbh.is_correct("Answer: (B) No", "No")


@pytest.mark.parametrize("generation, gold, expected", [
    # letter golds: original behaviour preserved
    ("Answer: (A)", "(A)", True),
    ("**Answer: (B)**", "(B)", True),
    ("Answer: B", "(B)", True),
    ("Answer: B.", "(B)", True),
    ("Answer: (D) Lola** — but that's not correct.", "(D)", True),
    ("Answer: (C)", "(B)", False),
    ("Therefore, the answer is B. But wait, looking at the options, option C is 10/02/2019.",
     "(B)", False),
    # non-letter golds the old grader already got right
    ("Answer: No", "No", True),
    ("Answer: True", "True", True),
    ("Answer: 24", "24", True),
    ("Answer: ) >", ") >", True),
    ("Answer: syndrome therefrom", "syndrome therefrom", True),
])
def test_bbh_regressions(generation, gold, expected):
    assert bbh.is_correct(generation, gold) is expected


def test_bbh_answer_marker_needs_word_boundary():
    assert bbh.extract_prediction("Answer: (C)\n\nAnswer: <answer>") == "(C)"
    assert bbh.extract_prediction("The answered question: (B)\n(C)") == "(C)"
    assert bbh.extract_prediction("Final answer: (A)") == "(A)"


@pytest.mark.parametrize("question, subtask, kind", [
    (Q_LETTER, "date_understanding", "letter"),
    (Q_CAUSAL, "causal_judgement", "yes_no"),
    (Q_NAVIGATE, "navigate", "yes_no"),
    (Q_FALLACY, "formal_fallacies", "valid_invalid"),
    (Q_WEB_OF_LIES, "web_of_lies", "yes_no"),
    (Q_SPORTS, "sports_understanding", "yes_no"),
    (Q_BOOLEAN, "boolean_expressions", "true_false"),
    (Q_DYCK, "dyck_languages", "dyck"),
    (Q_SORT, "word_sorting", "word_list"),
    (Q_COUNT, "object_counting", "number"),
    (Q_ARITH, "multistep_arithmetic_two", "number"),
])
def test_bbh_prompt_is_subtask_aware(question, subtask, kind):
    assert bbh.SUBTASK_KIND[subtask] == kind
    assert bbh.infer_answer_kind(question) == kind
    prompt = bbh.format_prompt(question)
    assert prompt == bbh.format_prompt(question, subtask=subtask)
    assert prompt.startswith(question)
    assert "<answer>" not in prompt and "<" not in prompt[len(question):]
    asks_for_letter = "letter" in prompt[len(question):]
    assert asks_for_letter == (kind == "letter")
    if kind == "yes_no":
        assert "'Answer: Yes'" in prompt and "'Answer: No'" in prompt
    if kind == "valid_invalid":
        assert "'Answer: valid'" in prompt and "'Answer: invalid'" in prompt
    if kind == "true_false":
        assert "'Answer: True'" in prompt and "'Answer: False'" in prompt


def test_bbh_every_subtask_has_a_kind():
    assert set(bbh.SUBTASK_KIND) == set(bbh.SUBTASKS)


# ---------------------------------------------------------------------------
# MATH-500 (B16)
# ---------------------------------------------------------------------------

MATH_NOW_CORRECT = [
    # shorthand fractions / roots in the GOLD (rows 164, 223, 201, 276, 340, 433, 382)
    (r"\boxed{\frac{6}{5}}", r"\frac65"),
    (r"\boxed{\dfrac{6}{5}}", r"\frac65"),
    (r"\boxed{\frac{3}{4}}", r"\frac 34"),
    (r"\boxed{\frac{9}{19}}", r"\frac9{19}"),
    (r"\boxed{\dfrac{4}{3}}", r"\frac43"),
    (r"\boxed{\frac{1}{4}}", r"\frac14"),
    (r"\boxed{\dfrac{5}{9}}", r"\frac 59"),
    (r"\boxed{11\sqrt{2}}", r"11\sqrt2"),
    # "\$" left a stray backslash (rows 122, 310, 489)
    (r"\boxed{36}", r"\$36"),
    (r"\boxed{32348}", r"\$32,\!348"),
    (r"\boxed{18.90}", r"\$18.90"),
    # units in the gold (rows 189, 200, 491, 291) — and in the prediction
    (r"\boxed{864} \text{ square inches}", r"864 \mbox{ inches}^2"),
    (r"\boxed{5.4} \text{ cents}", r"5.4 \text{ cents}"),
    (r"\boxed{15}", r"15\mbox{ cm}^2"),
    (r"\boxed{15 \, \text{cm}^2}", r"15\mbox{ cm}^2"),
    (r"\boxed{\frac{270}{7}}", r"\frac{270}7\text{ degrees}"),
    (r"\boxed{30^\circ}", "30"),
    (r"\boxed{12^{\text{th}} \text{ grade}}", "12"),
    (r"\boxed{12^{\text{th}}}", "12"),
    # variable prefixes / sets (rows 9, 91, 380, 470, 234)
    (r"\boxed{x = 3, \; x = 5, \; x = 7}", "3, 5, 7"),
    (r"\boxed{(a, b, c, d) = (1, -16, -4, 43)}", "(1,-16,-4,43)"),
    (r"\boxed{5}", "x=5"),
    (r"\boxed{[-2, 7]}", r"x \in [-2,7]"),
    (r"\boxed{\square = 3}", "3"),
    # unordered lists (rows 114, 131; 131's old prediction collapsed to "-21")
    (r"\boxed{1, -2}", "-2,1"),
    (r"\boxed{-2, 1}", "1,-2"),
    (r"\boxed{3 + 2\sqrt{2},\ 3 - 2\sqrt{2}}", r"3 \pm 2 \sqrt{2}"),
    # pmatrix with a/b in the gold (rows 102, 105, 399)
    (r"\boxed{\begin{pmatrix} \frac{1}{5} \\ -\frac{18}{5} \end{pmatrix}}",
     r"\begin{pmatrix} 1/5 \\ -18/5 \end{pmatrix}"),
    (r"\boxed{\begin{pmatrix} -\dfrac{1}{3} \\ \dfrac{2}{3} \\ \dfrac{5}{3} \end{pmatrix}}",
     r"\begin{pmatrix} -1/3 \\ 2/3 \\ 5/3 \end{pmatrix}"),
    # base subscripts (rows 406, 257, 267, 334)
    (r"\boxed{4210}", "4210_{5}"),
    (r"\boxed{4210_5}", "4210_{5}"),
    (r"\boxed{52}", "52_8"),
    # multiple-choice letters and word answers (rows 431, 438, 74, 149)
    (r"\boxed{C}", r"\text{(C)}"),
    (r"\boxed{\text{E}}", r"\text{(E)}"),
    (r"\boxed{\textbf{(B)}}", r"\text{(B)}"),
    (r"\boxed{\text{East}}", r"\text{east}"),
]

MATH_STILL_CORRECT = [  # the old grader got these right
    (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
    (r"\boxed{0.5}", r"\frac{1}{2}"),
    (r"\boxed{.5}", "0.50"),
    (r"\boxed{\left( 3, \frac{\pi}{2} \right)}", r"\left( 3, \frac{\pi}{2} \right)"),
    (r"\boxed{10\%}", "10"),
    (r"\boxed{1,000}", "1000"),
    (r"The final answer is 42.", "42"),
    (r"\boxed{\frac{\sqrt{3}}{2}}", r"\frac{\sqrt{3}}{2}"),
    (r"\boxed{x^2+7x+10}", "x^2+7x+10"),
]

MATH_WRONG = [
    (r"\boxed{-21}", "1,-2"),
    (r"\boxed{\frac{5}{6}}", r"\frac65"),
    (r"\boxed{3, 5}", "3, 5, 7"),
    (r"\boxed{3, 5, 5}", "3, 5, 7"),
    (r"\boxed{2}", r"\frac12"),
    (r"\boxed{(2, 1)}", "(1,2)"),           # ordered pairs stay ordered
    (r"\boxed{11\sqrt{3}}", r"11\sqrt2"),
    (r"\boxed{36}", r"\$3.60"),
    (r"\boxed{East}", r"\text{west}"),
    (r"\boxed{B}", r"\text{(C)}"),
    ("I could not finish", "7"),
    ("", "7"),
]


@pytest.mark.parametrize("generation, gold", MATH_NOW_CORRECT)
def test_math500_audit_false_negatives_now_correct(generation, gold):
    assert math500.is_correct(generation, gold)


@pytest.mark.parametrize("generation, gold", MATH_STILL_CORRECT)
def test_math500_regressions(generation, gold):
    assert math500.is_correct(generation, gold)


@pytest.mark.parametrize("generation, gold", MATH_WRONG)
def test_math500_wrong_answers_stay_wrong(generation, gold):
    assert not math500.is_correct(generation, gold)


@pytest.mark.xfail(strict=True, reason="no symbolic algebra: commuted terms do not match (row 329)")
def test_math500_commuted_expression_known_limitation():
    assert math500.is_correct(r"\boxed{\dfrac{9a + 11}{20}}", r"\frac{11+9a}{20}")


def test_math500_gold_and_prediction_normalize_identically():
    for gold in (r"\frac65", r"\$32,\!348", r"864 \mbox{ inches}^2", "4210_{5}", r"\text{(C)}"):
        assert math500.is_correct(r"\boxed{" + gold + "}", gold), gold


def test_math500_imports_no_third_party_dependency():
    import ast
    tree = ast.parse(Path(math500.__file__).read_text())
    top = {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    top |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert top <= {"__future__", "re", "datasets"}, top


def _math_verify():
    try:
        from math_verify import parse, verify
    except ImportError:
        return None
    return parse, verify


@pytest.mark.skipif(_math_verify() is None, reason="math_verify not installed (optional oracle)")
@pytest.mark.parametrize("generation, gold, expected",
                         [(g, a, True) for g, a in MATH_NOW_CORRECT + MATH_STILL_CORRECT
                          if r"\boxed" in g] +
                         [(g, a, False) for g, a in MATH_WRONG if r"\boxed" in g])
def test_math500_agrees_with_math_verify(generation, gold, expected):
    parse, verify = _math_verify()
    boxed = math500.extract_boxed(generation)
    oracle = bool(verify(parse("$" + gold + "$"), parse("$" + boxed + "$")))
    if oracle != expected:
        pytest.skip(f"math_verify itself says {oracle} for {boxed!r} vs {gold!r}")
    assert math500.is_correct(generation, gold) is expected


# ---------------------------------------------------------------------------
# MMLU-Pro (B17)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("generation, gold, expected", [
    # \boxed{J} not recognised (row 301 and 7 others): old pred came from prose
    ("Option C looks close but...\n\n### Final Answer\n\n$$\\boxed{J}$$", "J", True),
    ("... so (C) is out.\n\nThe answer is $\\boxed{\\text{(D)}}$", "D", True),
    # bare-letter fallback read letters out of prose on truncated off responses
    ("each side:\n\n- Left: 11 (Na) + 1 (H) = 12 electrons\n- Right: 12 (Mg) = 12", "H", False),
    ("the letter was signed by Mr. I.F.Sarbop, who", "I", False),
    ("Group A. has 2 pixels, Group B has 11", "A", False),
    ("- (F) Abducens nerve – Controls the lateral rectus muscle", "F", False),
    ("Wait — the question says the answer is $ A i $, and the options are", "A", False),
    # IGNORECASE / last-match latent bugs
    ("I think the answer is a bit unclear", "A", False),
    ("the answer I got does not match", "I", False),
    ("Answer: B. Option E is also tempting", "B", True),
    ("Answer: B. Option E is also tempting", "E", False),
])
def test_mmlu_pro_audit_cases(generation, gold, expected):
    assert mmlu_pro.is_correct(generation, gold) is expected


@pytest.mark.parametrize("generation, gold", [
    ("Answer: C", "C"),
    ("**Answer:** (D)", "D"),
    ("The answer is (E).", "E"),
    ("Final answer: J", "J"),
    ("answer is option C", "C"),
    ("Some reasoning.\n\nAnswer: **E**", "E"),
    ("Some reasoning.\n\n(C)", "C"),
    ("C", "C"),
    ("The correct option is (C).", "C"),
    ("Answer: (A) because ...\nOn reflection, Answer: (G)", "G"),
])
def test_mmlu_pro_regressions(generation, gold):
    assert mmlu_pro.is_correct(generation, gold)


# ---------------------------------------------------------------------------
# GSM8K (B18)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("generation, gold, expected", [
    ("Answer: 7\n\nThis result uses 3 steps.", "#### 7", True),
    ("Answer: 7\n\nThis result uses 3 steps.", "#### 3", False),
    ("Answer: 18\n(... 3 results in 6.)", "#### 18", True),
    ("Answer: −5", "#### -5", True),
    ("The temperature fell to −5 degrees.", "#### -5", True),
    ("Answer: 1/2", "#### 1", False),
    ("Answer: 3/2", "#### 1.5", True),
    ("See the <answer> tag below.\nAnswer: 12", "#### 12", True),
])
def test_gsm8k_audit_cases(generation, gold, expected):
    assert gsm8k.is_correct(generation, gold) is expected


def test_gsm8k_fraction_parses_as_value():
    assert gsm8k.normalize_number("1/2") == "0.5"
    assert gsm8k.normalize_number("4/2") == "2"
    assert gsm8k.normalize_number("1/0") is None
    assert gsm8k.normalize_number("−3") == "-3"


@pytest.mark.parametrize("generation, gold", [
    ("Natalia sold 48/2 = 24 clips in May.\nAnswer: 72", "#### 72"),
    ("Answer: $1,234.00", "#### 1234"),
    ("So the total is 72.", "#### 72"),
    ("### Final Answer\n\n$$\n\\boxed{48}\n$$", "#### 48"),
    ("The final result is 10 apples.", "#### 10"),
    ("Answer: 72.", "72"),
])
def test_gsm8k_regressions(generation, gold):
    assert gsm8k.is_correct(generation, gold)


# ---------------------------------------------------------------------------
# generate_labels --regrade (B7 + passthroughs)
# ---------------------------------------------------------------------------

def _identity(text: str) -> str:
    """Stand-in for capture_inference_thinking.strip_thinking in unit tests."""
    return text.rsplit("</think>", 1)[-1].strip()


@pytest.mark.parametrize("text, expected", [
    ("<think>So the answer is 2.00. Therefore, Answer: 2.00. But the user might", True),
    ("<think>reasoning</think>\nAnswer: 2", False),
    ("Answer: 2", False),
    ("<think>a</think> b <think>c", True),
    ("", False),
    (None, False),
])
def test_has_unclosed_think(text, expected):
    assert generate_labels.has_unclosed_think(text) is expected


def _meta_row(**kw):
    row = {"sample_id": "gsm8k-test-146", "prompt_hash": "h", "question": "q?",
           "answer": "blah\n#### 2", "difficulty": "easy",
           "response_off": "Answer: 3", "response_on": "<think>x</think>\nAnswer: 2",
           "correct_off": False, "correct_on": True,
           "truncated_off": False, "truncated_on": False,
           "n_tokens_off": 10, "n_tokens_on": 50}
    row.update(kw)
    return row


def test_regrade_unclosed_think_on_is_wrong_and_flagged():
    # gsm8k row 146: truncated inside <think>, stored correct_on=True.
    row = _meta_row(response_on="<think>... So the answer is 2.00. Therefore, Answer: 2.00. "
                                "But the user might have a different expectation",
                    truncated_on=True)
    out = generate_labels.regrade_row(row, gsm8k, _identity)
    assert out["correct_on"] is False and out["unclosed_think_on"] is True
    assert out["correct_on_stored"] is True and out["correct_off_stored"] is False
    lab = generate_labels.build_label_row(out, "gsm8k:abc")
    assert lab["truncated_on"] is True and lab["truncated_off"] is False
    assert lab["unclosed_think_on"] is True and lab["grader_version"] == "gsm8k:abc"
    assert lab["label"] == "not_helped"


def test_regrade_grades_post_think_text_with_current_grader():
    row = _meta_row(response_on="<think>Answer: 3</think>\nAnswer: 2",
                    response_off="Answer: 7\n\nThis result uses 2 steps.", correct_off=True)
    out = generate_labels.regrade_row(row, gsm8k, _identity)
    assert out["correct_on"] is True and out["unclosed_think_on"] is False
    assert out["correct_off"] is False and out["correct_off_stored"] is True


def test_regrade_passes_question_to_graders_that_take_it():
    row = _meta_row(sample_id="bbh-navigate-0", question=Q_NAVIGATE, answer="No",
                    response_off="Answer: (B)", response_on="<think>x</think>Answer: (A)")
    out = generate_labels.regrade_row(row, bbh, _identity)
    assert out["correct_off"] is True and out["correct_on"] is False


def test_regrade_refuses_rows_without_responses():
    row = _meta_row()
    del row["response_on"]
    with pytest.raises(KeyError):
        generate_labels.regrade_row(row, gsm8k, _identity)


def test_grader_version_tracks_module_source():
    v = generate_labels.grader_version(bbh)
    assert v.startswith("bbh:") and len(v) == len("bbh:") + 12
    assert v == generate_labels.grader_version(bbh)
    assert v != generate_labels.grader_version(gsm8k).replace("gsm8k", "bbh")


def test_label_row_schema_is_backward_compatible():
    lab = generate_labels.build_label_row(_meta_row())
    for key in ("sample_id", "prompt_hash", "label", "graded_label", "correct_off",
                "correct_on", "difficulty", "truncated_on", "n_tokens_on", "n_tokens_off"):
        assert key in lab
    assert lab["grader_version"] is None and "correct_off_stored" not in lab


def test_resolve_task():
    assert generate_labels.resolve_task({"task": "bbh"}, Path("x")) == "bbh"
    assert generate_labels.resolve_task({}, Path("mmlupro_thinking_qwen3")) == "mmlu_pro"
    assert generate_labels.resolve_task({}, Path("math500_thinking_qwen3v3")) == "math500"
    assert generate_labels.resolve_task({"task": "bbh"}, Path("x"), "lsat") == "lsat"


def _write_capture(cap: Path, rows: list[dict], task: str) -> None:
    np = pytest.importorskip("numpy")
    cap.mkdir(parents=True)
    with open(cap / "meta.shard00.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    for mode in ("off", "on"):
        np.savez_compressed(cap / f"activations_thinking_{mode}.shard00.npz",
                            activations=np.zeros((len(rows), 2, 4), dtype=np.float16))
    (cap / "config.json").write_text(json.dumps({"task": task, "model_name": "synthetic"}))


def _run_labels(*args: str) -> None:
    proc = subprocess.run([sys.executable, str(_PROJECT_ROOT / "scripts" / "generate_labels.py"),
                           *args], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_generate_labels_regrade_end_to_end(tmp_path):
    rows = [
        _meta_row(sample_id="gsm8k-test-0"),  # stored helped, regrade keeps it
        _meta_row(sample_id="gsm8k-test-1", truncated_on=True,  # B7: unclosed think
                  response_on="<think>Therefore, Answer: 2. But wait"),
        _meta_row(sample_id="gsm8k-test-2", correct_off=True,  # B18 marker fix flips off
                  response_off="Answer: 7\n\nThis result uses 2 steps."),
    ]
    cap = tmp_path / "gsm8k_thinking_synth"
    _write_capture(cap, rows, "gsm8k")

    stored, regraded = tmp_path / "stored.jsonl", tmp_path / "regraded.jsonl"
    _run_labels("--capture-dir", str(cap), "--out-file", str(stored))
    _run_labels("--regrade", "--capture-dir", str(cap), "--out-file", str(regraded))

    old = [json.loads(ln) for ln in stored.read_text().splitlines()]
    new = [json.loads(ln) for ln in regraded.read_text().splitlines()]
    assert [r["correct_on"] for r in old] == [True, True, True]
    assert [r["correct_on"] for r in new] == [True, False, True]
    assert [r["correct_off"] for r in new] == [False, False, False]
    assert [r["label"] for r in new] == ["helped", "not_helped", "helped"]
    assert new[1]["unclosed_think_on"] and new[1]["truncated_on"]
    assert all(r["grader_version"] == new[0]["grader_version"] for r in new)
    assert old[0]["grader_version"] is None

    summary = json.loads(regraded.with_suffix(".summary.json").read_text())
    assert summary["regraded"] and summary["grader_version"].startswith("gsm8k:")
    assert summary["unclosed_think_on"] == 1
    assert summary["regrade_flips"]["on_true_to_false"] == 1
    assert summary["regrade_flips"]["off_true_to_false"] == 1
    assert summary["regrade_flips"]["rows_changed"] == 2
    old_summary = json.loads(stored.with_suffix(".summary.json").read_text())
    assert old_summary["regraded"] is False and old_summary["grader_version"] is None

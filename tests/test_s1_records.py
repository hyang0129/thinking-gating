"""utils/s1_records on hand-made items (no dataset download)."""

from utils import s1_records as R


def test_decision_v7_kinds_and_gold():
    rec = {"_meta": {"id": "x/1"}, "state": {"a": 1},
           "questions": {
               "c": {"type": "choice", "instructions": {"question": "Q?", "focus": "F."},
                     "criteria": {"p": None, "q": "desc"}, "label": "q", "src": "s"},
               "n": {"type": "noul", "instructions": "Yes?", "label": False, "src": "s"},
               "o": {"type": "score", "instructions": "Rate", "criteria": ["lo", "hi"],
                     "label": 1, "src": "s"}}}
    c, n, o = R.from_decision_v7(rec)
    assert c["options"] == ["p", "q: desc"] and c["gold"] == 1 and c["question"] == "Q? F."
    assert n["options"] == ["Yes", "No"] and n["gold"] == 1 and n["kind"] == "binary"
    assert o["options"] == ["lo", "hi"] and o["gold"] == 1
    assert c["context"] == "a: 1"


def test_verify_candidate_prefers_hard_negative_and_is_deterministic():
    keys = [f"k{i}" for i in range(200)]
    res = [R.verify_candidate({"key": k}, "12", "13", False) for k in keys]
    assert res == [R.verify_candidate({"key": k}, "12", "13", False) for k in keys]
    negs = [x for x, g in res if not g]
    assert 60 < len(negs) < 140 and set(negs) == {"13"}
    # a correct no-think answer is never used as a negative
    assert all(x != "12" or g for x, g in (R.verify_candidate({"key": k}, "12", "12", True)
                                            for k in keys))
    # non-numeric gold, no hard negative: keep the gold
    assert R.verify_candidate({"key": "k"}, "\\frac{1}{2}", None, None) == ("\\frac{1}{2}", True)

"""test_interview.py — Unit tests for interview question bank and entry generator."""

from usual.interview import (
    list_questions,
    get_question,
    answer_to_judgment_entry,
)


def test_list_and_get_questions():
    all_q = list_questions()
    assert len(all_q) >= 10

    prod_q = list_questions(domain="product")
    assert len(prod_q) >= 4
    for q in prod_q:
        assert q["domain"] == "product"

    q_p1 = get_question("P1")
    assert q_p1 is not None
    assert q_p1["id"] == "P1"
    assert len(q_p1["options"]) == 2

    q_idx0 = get_question(0)
    assert q_idx0 is not None
    assert q_idx0["id"] == "P1"


def test_answer_to_judgment_entry():
    entry = answer_to_judgment_entry(
        question_id="P1",
        call="Ship in 2 days rough",
        reasons="Momentum beats polish early on",
        quote="I will always ship in 2 days rough because momentum beats polish."
    )

    assert entry["domain"] == "product"
    assert "unpolished feature" in entry["situation"]
    assert entry["call"] == "Ship in 2 days rough"
    assert entry["reasons"] == ["Momentum beats polish early on"]
    assert entry["provenance"]["file"] == "interview"
    assert entry["confidence"] == 1.0

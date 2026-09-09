"""Everyday onboarding. Prepares user statements without storing them on the server.

This offline prototype uses explicit answers and labelled examples, not inference.
Only deterministic redaction runs; no provider is called.
"""

import datetime

from .redact import assert_clean, redact_entries


QUESTIONS = [
    {
        "id": "writing",
        "label": "A message that sounds like you",
        "question": "When you ask for help writing a message, how should it sound?",
        "hint": "Think of a thank-you message to a friend.",
        "options": ["Warm and simple", "Short and to the point", "A little more thoughtful and detailed"],
        "situation": "When asking for help writing a personal message",
        "domain": "voice",
        "example_request": "Help me thank a friend for having us over for dinner.",
        "examples": [
            "Thank you for having us over! It was lovely to spend time together. We really enjoyed it.",
            "Thanks for dinner! We had a great time.",
            "Thank you for such a lovely evening. It meant a lot to spend time together, and we really appreciated you having us over. Let’s do it again soon.",
        ],
    },
    {
        "id": "dinner",
        "label": "Dinner on an ordinary evening",
        "question": "It’s a busy evening. What matters most when choosing something to cook?",
        "hint": "There’s no right answer. Choose what usually works for you.",
        "options": ["Quick to make, with little washing up", "Use ingredients I already have", "Try something a little different"],
        "situation": "When choosing what to cook on a busy evening",
        "domain": "other",
        "example_request": "Help me decide what to cook tonight.",
        "examples": [
            "Let’s keep dinner simple and use just one pan. What ingredients do you have?",
            "Let’s start with what’s already in your kitchen. What ingredients would you like to use up?",
            "Let’s try something different that fits your evening. What ingredients do you have, and how much time?",
        ],
    },
    {
        "id": "planning",
        "label": "A day out, your way",
        "question": "When planning a day out, what feels best to you?",
        "hint": "Imagine a free Saturday with someone you enjoy spending time with.",
        "options": ["One or two things, with time to relax", "A full day with plenty to see", "A few ideas, then decide on the day"],
        "situation": "When planning a free day out",
        "domain": "other",
        "example_request": "Help me plan a day out this Saturday.",
        "examples": [
            "Let’s choose one main stop and somewhere for lunch, with time to take it easy. Where would you like to go?",
            "Let’s make the most of the day with several stops. Where are you starting from?",
            "Let’s put together a few options you can choose between on Saturday. What area do you have in mind?",
        ],
    },
]


def prepare_answers(payload):
    """Validate and redact up to three answers; never touch the shared corpus."""
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), list):
        raise ValueError("Please send a list of answers.")
    answers = payload["answers"]
    if not 1 <= len(answers) <= len(QUESTIONS):
        raise ValueError("Choose between one and three answers.")
    questions = {q["id"]: q for q in QUESTIONS}
    entries, seen = [], set()
    for answer in answers:
        if not isinstance(answer, dict):
            raise ValueError("Each answer must include a question and your choice.")
        qid = answer.get("id")
        if not isinstance(qid, str) or qid not in questions or qid in seen:
            raise ValueError("Please choose each question only once.")
        seen.add(qid)
        call, reason = answer.get("call"), answer.get("reason", "")
        if not isinstance(call, str) or not call.strip() or len(call) > 500:
            raise ValueError("Keep each answer between 1 and 500 characters.")
        if not isinstance(reason, str) or len(reason) > 500:
            raise ValueError("Keep each explanation under 500 characters.")
        q = questions[qid]
        entries.append({
            "id": qid, "situation": q["situation"], "call": call.strip(),
            "reasons": [reason.strip()] if reason.strip() else [],
            "domain": q["domain"], "date": datetime.date.today().isoformat(),
            "provenance": {"file": "consumer_onboarding", "quote": call.strip()},
            "confidence": 1.0,
        })
    redact_entries(entries)
    assert_clean(entries)
    return entries

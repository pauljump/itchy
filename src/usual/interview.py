"""interview.py — Cold-start dilemma question bank and judgment entry generator.

Provides calibrated forced-choice dilemma questions across product, design,
factory, money, people, and voice, and converts user answers into vetted
judgment entries.
"""

from __future__ import annotations

import datetime
from typing import Any


INTERVIEW_QUESTIONS: list[dict[str, Any]] = [
    # Product
    {
        "id": "P1",
        "domain": "product",
        "title": "Speed vs Polish",
        "question": "You have two versions of a feature: one ships in two days and is rough around the edges, the other ships in two weeks and is clean. Which do you ship, and why?",
        "options": ["Ship in 2 days rough", "Ship in 2 weeks clean"],
        "context": "When deciding whether to ship an unpolished feature quickly or wait for a clean implementation."
    },
    {
        "id": "P2",
        "domain": "product",
        "title": "Core Love vs Broad Confusion",
        "question": "A small group of people love your product deeply, but a large group finds it confusing. Do you fix the confusion or protect the love? What's your reasoning?",
        "options": ["Fix the confusion for the majority", "Protect the love for the core fans"],
        "context": "When facing feedback where power users love a specialized feature but casual users find it confusing."
    },
    {
        "id": "P3",
        "domain": "product",
        "title": "Simplicity vs Technical Power",
        "question": "A feature is technically impressive but adds interface complexity. The simpler alternative is less impressive but understood immediately. Which do you build?",
        "options": ["Build the technically impressive feature", "Build the immediately understandable alternative"],
        "context": "When choosing between a technically powerful feature with UI complexity vs a simpler intuitive alternative."
    },
    {
        "id": "P4",
        "domain": "product",
        "title": "Two Features, One Winner",
        "question": "Your product does two things. Users only use one, but you built both. Do you cut the weak half or keep both?",
        "options": ["Cut the weak feature entirely", "Keep both features available"],
        "context": "When a product has two capabilities but users only engage with one."
    },
    {
        "id": "P5",
        "domain": "product",
        "title": "Unintended Emergent Use",
        "question": "Users are using your product in an unexpected way that is more compelling than your original intent. Do you pivot to follow them or stick to your vision?",
        "options": ["Follow the emergent user behavior", "Stick to the original product vision"],
        "context": "When users discover an unintended workflow that outperforms the intended design."
    },
    # Design
    {
        "id": "D1",
        "domain": "design",
        "title": "Clean vs Warm",
        "question": "A screen looks clean and minimalist, but feels cold and sterile. How do you make it warm without adding clutter?",
        "options": ["Add subtle personality/copy flourishes", "Keep it strictly austere and functional"],
        "context": "When balancing minimalist aesthetic cleanliness against emotional warmth."
    },
    {
        "id": "D2",
        "domain": "design",
        "title": "Self-Explaining vs Invisible",
        "question": "When does a user interface need an explicit explanatory label or tooltip versus letting the user discover it by feel?",
        "options": ["Explicitly label and explain everything", "Rely on intuitive spatial affordances"],
        "context": "When deciding whether an interface element requires explanatory text or implicit visual discovery."
    },
    {
        "id": "D3",
        "domain": "design",
        "title": "Cutting the Screen in Half",
        "question": "You are forced to cut 50% of the information currently displayed on the main screen. What is your iron rule for what stays versus what goes?",
        "options": ["Keep only the primary action and status", "Keep high-density information and compress it"],
        "context": "When ruthlessly pruning 50% of content from a primary screen."
    },
    # Factory / Architecture
    {
        "id": "F1",
        "domain": "factory",
        "title": "Technical Debt vs Momentum",
        "question": "You can build something quickly with technical debt today, or build it correctly and take 3x longer. How do you decide?",
        "options": ["Take on debt to validate immediately", "Build it properly with clean abstractions first"],
        "context": "When trading off speed to validation against architectural technical debt."
    },
    {
        "id": "F2",
        "domain": "factory",
        "title": "Zero-Dependency vs Convenience",
        "question": "You can implement a feature in 50 lines of stdlib code or pull in a popular 3rd-party library. Which do you choose?",
        "options": ["Write 50 lines of stdlib code", "Install the proven 3rd-party library"],
        "context": "When choosing between a small custom stdlib implementation vs adding a dependency."
    },
    {
        "id": "F3",
        "domain": "factory",
        "title": "Code Duplication vs Early Abstraction",
        "question": "Two separate projects share 80% of the same logic. Do you extract a shared library now or keep them independent?",
        "options": ["Extract a shared package now", "Keep them separate until 3+ callers exist"],
        "context": "When deciding whether to deduplicate shared logic into a shared package or tolerate duplication."
    },
    # Money / Commercial
    {
        "id": "M1",
        "domain": "money",
        "title": "Freemium vs Paid-Only",
        "question": "Do you offer a free tier to build a viral top of funnel, or charge upfront to filter for high-intent customers?",
        "options": ["Offer a generous free tier", "Charge upfront with zero free tier"],
        "context": "When deciding on early pricing strategy and whether to support free tier users."
    },
    {
        "id": "M2",
        "domain": "money",
        "title": "Flat Rate vs Usage-Based",
        "question": "Do you price with predictable flat-rate subscription tiers, or granular usage-based billing matching your COGS?",
        "options": ["Predictable flat subscription", "Metered usage-based billing"],
        "context": "When choosing between predictable pricing for users vs margin protection."
    },
    # Voice / Culture
    {
        "id": "V1",
        "domain": "voice",
        "title": "Hedging vs Directness",
        "question": "When an AI assistant or tool communicates advice, should it hedge with disclaimers or take a direct, opinionated stance?",
        "options": ["Direct and opinionated stance", "Nuanced and balanced with disclaimers"],
        "context": "When tuning assistant communication style for strategic or technical advice."
    },
]


def list_questions(domain: str | None = None) -> list[dict[str, Any]]:
    """Return all questions, optionally filtered by domain."""
    if not domain:
        return list(INTERVIEW_QUESTIONS)
    return [q for q in INTERVIEW_QUESTIONS if q.get("domain") == domain]


def get_question(question_id_or_index: str | int) -> dict[str, Any] | None:
    """Retrieve question by ID (e.g. 'P1') or 0-based index."""
    if isinstance(question_id_or_index, int):
        if 0 <= question_id_or_index < len(INTERVIEW_QUESTIONS):
            return INTERVIEW_QUESTIONS[question_id_or_index]
        return None

    qid = str(question_id_or_index).upper()
    for q in INTERVIEW_QUESTIONS:
        if q["id"] == qid:
            return q
    return None


def answer_to_judgment_entry(
    question_id: str,
    call: str,
    reasons: list[str] | str,
    quote: str = "",
    date_str: str | None = None,
) -> dict[str, Any]:
    """
    Convert a question answer into a valid judgment-entry.schema.json dict.
    """
    q = get_question(question_id)
    domain = q["domain"] if q else "other"
    situation = q["context"] if q else f"In response to interview question {question_id}"

    if isinstance(reasons, str):
        cleaned_reasons = [reasons.strip()] if reasons.strip() else []
    else:
        cleaned_reasons = [r.strip() for r in reasons if r.strip()]

    call_clean = call.strip()
    verbatim_quote = quote.strip() or f"{call_clean}. {' '.join(cleaned_reasons)}".strip()

    return {
        "situation": situation,
        "call": call_clean,
        "reasons": cleaned_reasons,
        "domain": domain,
        "date": date_str or datetime.date.today().isoformat(),
        "provenance": {
            "file": "interview",
            "quote": verbatim_quote
        },
        "confidence": 1.0  # Explicit user interview answers have maximum confidence
    }

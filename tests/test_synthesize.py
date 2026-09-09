"""test_synthesize.py — Unit tests for prompt synthesizer."""

import pytest
from usual.synthesize import (
    synthesize_chatgpt_instructions,
    synthesize_markdown_knowledge,
    synthesize_system_prompt,
)


@pytest.fixture
def sample_entries():
    return [
        {
            "situation": "When deciding whether to ship an unpolished feature in 2 days or wait 2 weeks",
            "call": "Ship in 2 days rough",
            "reasons": ["Momentum beats polish in early validation"],
            "domain": "product",
            "confidence": 0.9,
        },
        {
            "situation": "When choosing whether to use Redis or SQLite for lightweight state",
            "call": "Prefer SQLite and keep it zero-dependency",
            "reasons": ["Avoid external daemon management and network latency overhead"],
            "domain": "factory",
            "confidence": 0.85,
        },
        {
            "situation": "When users complain about missing advanced settings",
            "call": "Protect simplicity and reject settings bloat",
            "reasons": ["Settings are a failure of design conviction"],
            "domain": "design",
            "confidence": 0.8,
        },
    ]


def test_synthesize_chatgpt_instructions(sample_entries):
    instructions = synthesize_chatgpt_instructions(sample_entries, max_chars=1450)
    assert len(instructions) <= 1450
    assert "# How to Decide & Respond" in instructions
    assert "[Product]" in instructions
    assert "Ship in 2 days rough" in instructions
    assert "Momentum beats polish" in instructions
    assert "[Factory]" in instructions
    assert "SQLite" in instructions


def test_synthesize_markdown_knowledge(sample_entries):
    doc = synthesize_markdown_knowledge(sample_entries)
    assert "# Personal Judgment Corpus" in doc
    assert "## Domain: PRODUCT" in doc
    assert "Ship in 2 days rough" in doc
    assert "Momentum beats polish in early validation" in doc


def test_synthesize_system_prompt(sample_entries):
    xml = synthesize_system_prompt(sample_entries)
    assert "<judgment_corpus>" in xml
    assert "</judgment_corpus>" in xml
    assert '<judgment id="1" domain="product">' in xml
    assert "<decision>Ship in 2 days rough</decision>" in xml

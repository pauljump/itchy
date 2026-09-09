"""test_parsers.py — Unit tests for universal export parser and candidate mining."""

import pytest
from usual.parsers import (
    extract_linear_turns_chatgpt,
    extract_linear_turns_claude,
    parse_conversation_export,
    extract_candidate_judgments,
    classify_domain,
)


def test_extract_linear_turns_chatgpt():
    # Simulated ChatGPT mapping tree
    conv = {
        "title": "Database Architecture",
        "current_node": "node3",
        "mapping": {
            "node1": {
                "id": "node1",
                "parent": None,
                "children": ["node2"],
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": ["Should I use Postgres or SQLite?"]}
                }
            },
            "node2": {
                "id": "node2",
                "parent": "node1",
                "children": ["node3"],
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"parts": ["Postgres gives you full scale and replication."]}
                }
            },
            "node3": {
                "id": "node3",
                "parent": "node2",
                "children": [],
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": ["No, don't use Postgres. Prefer SQLite because zero maintenance is better for a solo dev."]}
                }
            }
        }
    }

    turns = extract_linear_turns_chatgpt(conv)
    assert len(turns) == 3
    assert turns[0]["role"] == "user"
    assert turns[1]["role"] == "assistant"
    assert turns[2]["role"] == "user"
    assert "Prefer SQLite" in turns[2]["text"]


def test_extract_linear_turns_claude():
    conv = {
        "name": "Design Review",
        "chat_messages": [
            {"sender": "human", "text": "What do you think of this modal?"},
            {"sender": "assistant", "text": "It's clean, but maybe add tabs."},
            {"sender": "human", "text": "Never add tabs. Cut the extra settings instead because simpler is better."}
        ]
    }

    turns = extract_linear_turns_claude(conv)
    assert len(turns) == 3
    assert turns[0]["role"] == "user"
    assert turns[1]["role"] == "assistant"
    assert turns[2]["role"] == "user"
    assert "Never add tabs" in turns[2]["text"]


def test_parse_conversation_export_chatgpt():
    payload = [{
        "title": "API Decisions",
        "create_time": 1700000000,
        "mapping": {
            "root": {
                "id": "root",
                "parent": None,
                "children": ["leaf"],
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": ["Drop Redis because in-memory dict is enough."]}
                }
            },
            "leaf": {
                "id": "leaf",
                "parent": "root",
                "children": [],
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"parts": ["Understood."]}
                }
            }
        },
        "current_node": "leaf"
    }]

    parsed = parse_conversation_export(payload)
    assert len(parsed) == 1
    assert parsed[0]["title"] == "API Decisions"
    assert len(parsed[0]["turns"]) == 2


def test_extract_candidate_judgments():
    conversations = [{
        "title": "Pricing Strategy",
        "date": "2026-08-15",
        "turns": [
            {
                "role": "assistant",
                "text": "We could offer a free tier with 5 seats to maximize adoption."
            },
            {
                "role": "user",
                "text": "Don't do a free tier. Charge $50/mo upfront instead because free users generate 90% of support tickets."
            }
        ]
    }]

    candidates = extract_candidate_judgments(conversations, source_label="chatgpt_test")
    assert len(candidates) == 1
    c = candidates[0]
    assert "Pricing Strategy" in c["situation"]
    assert "Don't do a free tier" in c["call"]
    assert len(c["reasons"]) > 0
    assert "free users generate 90% of support tickets" in c["reasons"][0]
    assert c["domain"] in ("product", "money")
    assert c["provenance"]["file"] == "chatgpt_test"
    assert c["confidence"] >= 0.7


def test_classify_domain():
    assert classify_domain("We need to adjust product pricing and features") in ("product", "money")
    assert classify_domain("Fix button color, visual hierarchy and screen layout") == "design"
    assert classify_domain("Refactor git repo, build script and db schema") == "factory"

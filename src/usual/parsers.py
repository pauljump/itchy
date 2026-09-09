"""parsers.py — Universal conversation export parser and decision miner.

Supports:
  1. OpenAI ChatGPT data export (conversations.json with mapping trees)
  2. Anthropic Claude web export (conversations.json with chat_messages arrays)
  3. Raw text or message turn lists

Extracts user judgment candidates where the user pushed back, made a decision,
or articulated a principle/reason.
"""

from __future__ import annotations

import datetime
import json
import os
import re
from typing import Any


# ---------------------------------------------------------------------------
# Format detection & turn extraction
# ---------------------------------------------------------------------------

def extract_linear_turns_chatgpt(conversation: dict[str, Any]) -> list[dict[str, str]]:
    """
    Reconstruct linear chronological turns from an OpenAI conversation node tree.
    Returns list of dicts: [{"role": "user"|"assistant"|"system", "text": "..."}]
    """
    mapping = conversation.get("mapping")
    if not mapping or not isinstance(mapping, dict):
        return []

    current_id = conversation.get("current_node")
    # If no current_node, find any leaf node
    if not current_id or current_id not in mapping:
        leaf_nodes = [
            nid for nid, node in mapping.items()
            if not node.get("children")
        ]
        current_id = leaf_nodes[-1] if leaf_nodes else None

    if not current_id or current_id not in mapping:
        return []

    # Walk upwards from leaf to root
    path: list[dict[str, Any]] = []
    visited: set[str] = set()
    node_id = current_id

    while node_id and node_id in mapping and node_id not in visited:
        visited.add(node_id)
        node = mapping[node_id]
        path.append(node)
        node_id = node.get("parent")

    path.reverse()

    turns: list[dict[str, str]] = []
    for node in path:
        msg = node.get("message")
        if not msg:
            continue

        author = msg.get("author", {})
        role = author.get("role")
        if role not in ("user", "assistant", "system"):
            continue

        content = msg.get("content", {})
        parts = content.get("parts", [])
        if not parts:
            continue

        # parts can be strings or dicts
        text_parts = []
        for p in parts:
            if isinstance(p, str):
                text_parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                text_parts.append(str(p["text"]))

        full_text = "\n".join(text_parts).strip()
        if full_text:
            turns.append({"role": role, "text": full_text})

    return turns


def extract_linear_turns_claude(conversation: dict[str, Any]) -> list[dict[str, str]]:
    """
    Extract linear turns from an Anthropic Claude web export conversation.
    """
    chat_messages = conversation.get("chat_messages", [])
    if not isinstance(chat_messages, list):
        return []

    turns: list[dict[str, str]] = []
    for msg in chat_messages:
        sender = msg.get("sender")
        if sender == "human":
            role = "user"
        elif sender == "assistant":
            role = "assistant"
        else:
            continue

        text = msg.get("text", "").strip()
        if text:
            turns.append({"role": role, "text": text})

    return turns


def parse_conversation_export(data: Any) -> list[dict[str, Any]]:
    """
    Parse an arbitrary conversation export JSON (list of conversations or single dict).
    Returns list of parsed conversations:
      [{"title": str, "date": str, "turns": [{"role": str, "text": str}]}]
    """
    if isinstance(data, dict):
        # Single conversation or wrapped
        if "conversations" in data and isinstance(data["conversations"], list):
            conv_list = data["conversations"]
        else:
            conv_list = [data]
    elif isinstance(data, list):
        conv_list = data
    else:
        return []

    results: list[dict[str, Any]] = []

    for conv in conv_list:
        if not isinstance(conv, dict):
            continue

        title = conv.get("title") or conv.get("name") or "Untitled Conversation"
        created_time = conv.get("create_time") or conv.get("created_at")

        date_str = datetime.date.today().isoformat()
        if isinstance(created_time, (int, float)):
            try:
                date_str = datetime.date.fromtimestamp(created_time).isoformat()
            except Exception:
                pass
        elif isinstance(created_time, str):
            # Try ISO 8601 YYYY-MM-DD
            m = re.search(r"(\d{4}-\d{2}-\d{2})", created_time)
            if m:
                date_str = m.group(1)

        if "mapping" in conv:
            turns = extract_linear_turns_chatgpt(conv)
        elif "chat_messages" in conv:
            turns = extract_linear_turns_claude(conv)
        elif "messages" in conv and isinstance(conv["messages"], list):
            # Generic [{role, content}] format
            turns = []
            for m in conv["messages"]:
                r = m.get("role") or m.get("sender")
                if r in ("human", "user"):
                    r = "user"
                elif r in ("assistant", "ai", "bot"):
                    r = "assistant"
                txt = m.get("content") or m.get("text") or ""
                if isinstance(txt, list):
                    txt = " ".join(str(x) for x in txt)
                txt = str(txt).strip()
                if txt and r:
                    turns.append({"role": r, "text": txt})
        else:
            turns = []

        if turns:
            results.append({
                "title": title,
                "date": date_str,
                "turns": turns
            })

    return results


# ---------------------------------------------------------------------------
# Decision & pushback mining heuristics
# ---------------------------------------------------------------------------

PUSHBACK_PATTERNS = [
    r"\b(?:don'?t|do not|never|stop|instead|rather|prefer|won'?t)\b",
    r"\b(?:cut|drop|kill|remove|skip|omit|leave out)\b",
    r"\b(?:keep|always|rule|must|requirement|mandate)\b",
    r"\b(?:too (?:complex|slow|heavy|expensive|confusing|much))\b",
    r"\b(?:simpler|cleaner|faster|leaner|lightweight)\b",
    r"\b(?:my rule is|my approach is|my philosophy is)\b",
]

REASON_PATTERNS = [
    r"\bbecause\b",
    r"\bsince\b",
    r"\bthe reason (?:is|being)\b",
    r"\bso that\b",
    r"\bin order to\b",
    r"\bwhy:\s*",
    r"\btradeoff\b",
    r"\boverhead\b",
    r"\blatency\b",
    r"\bmaintenance\b",
    r"\bcost\b",
]

def classify_domain(text: str) -> str:
    """Classify domain based on keyword distribution."""
    lower = text.lower()
    scores = {
        "product": len(re.findall(r"\b(?:product|feature|user|customer|market|launch|ship|mvp|roadmap|pricing)\b", lower)),
        "design": len(re.findall(r"\b(?:design|ui|ux|screen|button|visual|look|feel|layout|color|typography)\b", lower)),
        "factory": len(re.findall(r"\b(?:code|build|git|repo|infra|server|database|db|test|deploy|script|pipeline|api|architecture|cli)\b", lower)),
        "money": len(re.findall(r"\b(?:money|price|cost|fee|dollar|budget|revenue|charge|burn|margin|invest)\b", lower)),
        "people": len(re.findall(r"\b(?:hire|team|culture|manager|coworker|client|meeting|feedback|review|employee)\b", lower)),
        "voice": len(re.findall(r"\b(?:voice|tone|writing|word|post|tweet|email|copy|headline|sentence)\b", lower)),
    }
    best_domain, best_count = max(scores.items(), key=lambda item: item[1])
    return best_domain if best_count > 0 else "other"


def extract_candidate_judgments(
    conversations: list[dict[str, Any]],
    source_label: str = "chatgpt_export"
) -> list[dict[str, Any]]:
    """
    Mine candidate judgment entries from conversations where user asserted a choice.
    Returns entries conforming to `schema/judgment-entry.schema.json`.
    """
    candidates: list[dict[str, Any]] = []

    for conv in conversations:
        title = conv.get("title", "Conversation")
        date_str = conv.get("date", datetime.date.today().isoformat())
        turns = conv.get("turns", [])

        for i, turn in enumerate(turns):
            if turn.get("role") != "user":
                continue

            user_text = turn.get("text", "").strip()
            # Ignore trivial short replies or massive code dumps
            if len(user_text) < 15 or len(user_text) > 3000:
                continue

            # Check pushback / principle score
            has_pushback = any(re.search(p, user_text, re.IGNORECASE) for p in PUSHBACK_PATTERNS)
            has_reason = any(re.search(p, user_text, re.IGNORECASE) for p in REASON_PATTERNS)

            if not (has_pushback or has_reason):
                continue

            # Derive the situation from prior assistant turn or conversation title
            prior_assistant_text = ""
            if i > 0 and turns[i-1].get("role") == "assistant":
                prior_assistant_text = turns[i-1].get("text", "").strip()

            if prior_assistant_text:
                sentences = re.split(r"(?<=[.!?])\s+", prior_assistant_text)
                situation_context = " ".join(sentences[:2])
                if len(situation_context) > 200:
                    situation_context = situation_context[:197] + "..."
                situation = f"In discussion '{title}', the assistant proposed: {situation_context}"
            else:
                situation = f"In discussion '{title}', when deciding on the approach or direction."

            reasons: list[str] = []
            reason_matches = re.findall(r"\b(?:because|since)\s+([^.,;\n]{8,120})", user_text, re.IGNORECASE)
            for rm in reason_matches:
                clean_r = rm.strip()
                if clean_r and clean_r not in reasons:
                    reasons.append(clean_r)

            first_sentence = re.split(r"(?<=[.!?\n])\s+", user_text)[0].strip()
            call = first_sentence if len(first_sentence) <= 250 else first_sentence[:247] + "..."

            domain = classify_domain(user_text + " " + title)

            confidence = 0.5
            if has_pushback and has_reason:
                confidence = 0.8
            elif has_pushback or has_reason:
                confidence = 0.65

            candidates.append({
                "situation": situation,
                "call": call,
                "reasons": reasons,
                "domain": domain,
                "date": date_str,
                "provenance": {
                    "file": source_label,
                    "quote": user_text[:300]
                },
                "confidence": confidence
            })

    return candidates

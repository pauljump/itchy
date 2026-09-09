"""Bounded, speaker-aware import of local coding transcripts. No network calls."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .parsers import classify_domain, parse_conversation_export
from .redact import _redact_text, assert_clean

MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TURN_CHARS = 3000


def scrub(text: str) -> str:
    """Redact every persisted free-text field before it enters SQLite."""
    clean, _ = _redact_text(str(text), {})
    clean = re.sub(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{16,}", "[REDACTED-KEY]", clean)
    clean = re.sub(r"\bgithub_pat_[A-Za-z0-9_]{20,}", "[REDACTED-KEY]", clean)
    clean = re.sub(r"(?i)(https?://)[^\s/@:]+:[^\s/@]+@", r"\1[REDACTED-CREDENTIAL]@", clean)
    clean = re.sub(r"(?i)([?&](?:token|key|secret|password|api_key)=)[^\s&#]+", r"\1[REDACTED]", clean)
    try:
        assert_clean([{"call": clean}])
    except RuntimeError:
        raise ValueError("Residual secret detected; nothing was saved.") from None
    return clean


def content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(block.get("text", "") for block in content
                         if isinstance(block, dict) and block.get("type") in
                         ("text", "input_text", "output_text") and isinstance(block.get("text"), str))
    return ""


def human_text(text: str) -> str:
    # Provider-injected wrappers and pasted instruction packs aren't authored choices.
    for tag in ("environment_context", "system-reminder", "instructions", "INSTRUCTIONS",
                "recommended_plugins", "user_instructions", "permissions instructions"):
        text = re.sub(rf"<{re.escape(tag)}\b[^>]*>[\s\S]*?</{re.escape(tag)}>", "", text)
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Keep the authored introduction; don't learn a pasted specification as a preference.
    text = re.split(r"\b(?:here(?:'s| is) (?:the |a )?(?:job description|jd|article|transcript|document|specification))\b", text, maxsplit=1, flags=re.I)[0]
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(">"))
    if text.lstrip().startswith(("# AGENTS.md instructions", "<command", "<local-command", "<task-notification", "<subagent")):
        return ""
    return text.strip()


def read_transcript(path: Path) -> tuple[list[dict], dict]:
    """Read supported formats without treating tools, system text, or duplicates as users."""
    if path.is_symlink():
        raise ValueError("Transcript symlinks are not imported; choose the original file.")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Transcript exceeds the 32 MiB per-file import limit.")
    raw = path.read_text(encoding="utf-8")
    stats = {"malformed_lines": 0, "ignored_turns": 0, "duplicate_turns": 0}
    turns = []
    if path.suffix == ".json":
        for conv in parse_conversation_export(json.loads(raw)):
            for i, turn in enumerate(conv["turns"]):
                turns.append({**turn, "line": i + 1, "date": conv["date"], "project": ""})
    else:
        records = []
        for n, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError()
                records.append((n, item))
            except (ValueError, TypeError):
                stats["malformed_lines"] += 1
        # Codex often duplicates user messages as event_msg and response_item. Prefer the latter.
        has_codex_responses = any(d.get("type") == "response_item" and
            d.get("payload", {}).get("role") == "user" for _, d in records if isinstance(d.get("payload", {}), dict))
        project = ""
        for n, record in records:
            kind = record.get("type")
            payload = record.get("payload", {})
            payload = payload if isinstance(payload, dict) else {}
            if kind == "session_meta":
                project = str(payload.get("cwd", ""))
            role, text = None, ""
            if kind == "response_item" and payload.get("type") == "message":
                role = payload.get("role")
                text = content_text(payload.get("content"))
            elif kind == "event_msg" and not has_codex_responses and payload.get("type") == "user_message":
                role, text = "user", payload.get("message", "")
            elif kind in ("user", "assistant") and not record.get("isMeta") and not record.get("isSidechain"):
                message = record.get("message", {})
                if isinstance(message, dict):
                    role = message.get("role")
                    text = content_text(message.get("content"))
                    project = str(record.get("cwd", project))
            elif record.get("role") in ("user", "assistant"):
                role, text = record["role"], content_text(record.get("content", record.get("text")))
            if role in ("user", "assistant") and isinstance(text, str) and text.strip():
                turns.append({"role": role, "text": text, "line": n,
                              "date": str(record.get("timestamp", ""))[:10], "project": project})
    clean_turns, seen = [], set()
    for turn in turns:
        text = human_text(turn["text"]) if turn["role"] == "user" else turn["text"].strip()
        if not text or len(text) > MAX_TURN_CHARS:
            stats["ignored_turns"] += 1
            continue
        fingerprint = (turn["role"], text)
        if fingerprint in seen:
            stats["duplicate_turns"] += 1
            continue
        seen.add(fingerprint)
        clean_turns.append({**turn, "text": text})
    return clean_turns, stats


CHOICE = re.compile(r"\b(?:prefer|instead|rather|because|always|never|keep|choose|use|don.?t|do not|should|want|let.s|skip|avoid|make it|too complex|simpler)\b", re.I)


def mine_transcript(path: Path, scope: str) -> tuple[list[dict], dict]:
    turns, stats = read_transcript(path)
    entries, context = [], ""
    for turn in turns:
        if turn["role"] == "assistant":
            context = turn["text"][:1600]
            continue
        text = turn["text"]
        if len(text) < 20 or not CHOICE.search(text):
            stats["ignored_turns"] += 1
            context = ""
            continue
        quote = scrub(text)
        situation = scrub(context) if context else "User stated a coding preference or direction."
        call = quote  # Keep exact wording; interpretation belongs to the session model.
        fingerprint = hashlib.sha256((scope + "\0" + quote).encode()).hexdigest()
        entries.append({"id": "e_" + fingerprint[:20], "fingerprint": fingerprint,
                        "scope": scope, "situation": situation, "call": call, "quote": quote,
                        "domain": classify_domain(text), "source": scrub(path.name), "line": turn["line"],
                        "date": turn["date"], "origin": "observed", "status": "active"})
        context = ""
    stats["user_turns"] = sum(t["role"] == "user" for t in turns)
    stats["candidates"] = len(entries)
    return entries, stats


def discover(provider: str, limit: int = 20) -> list[Path]:
    roots = []
    if provider in ("codex", "both"):
        roots.append(Path.home() / ".codex" / "sessions")
    if provider in ("claude", "both"):
        roots.append(Path.home() / ".claude" / "projects")
    files = [p for root in roots if root.exists() for p in root.rglob("*.jsonl")
             if not p.is_symlink() and "subagents" not in p.parts]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]

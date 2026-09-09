"""redact.py — MANDATORY secrets redaction per spec §2.2.

Two passes:
  1. Regex pass (deterministic, always runs, implemented here).
  2. LLM scan pass (semantic, done by the skill — this module exposes the interface).

FAILURE SEMANTICS:
  - redact_entries() fires the regex pass and logs redacted field counts to stderr.
  - assert_clean() checks for residual high-entropy secrets and RAISES on any hit.
    build_exam.py MUST call assert_clean() before writing any file.
  - --no-llm-scan suppresses the LLM pass WARNING for offline/CI use.
  - Never silently drop an entry — only redact the matching text.

CLI usage (regex-only):
    python3 -m usual.redact [--no-llm-scan] < entries.jsonl > redacted.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_FIELDS = ("situation", "call", "provenance_quote")  # logical names

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # passwords — (?<!-) skips the PASSWORD in [REDACTED-PASSWORD] (preceded by hyphen).
    # (?!\[) lookahead skips cases where the value is already a [REDACTED-*] token,
    # preventing "password [REDACTED-PASSWORD]" from being eaten on a second pass.
    (re.compile(r"(?i)(?<!-)\bpassword\b\s*[:=]?\s*(?!\[REDACTED)\S+"), "[REDACTED-PASSWORD]"),
    (re.compile(r"(?i)(?<!-)\bpasswd\b\s*[:=]?\s*(?!\[REDACTED)\S+"), "[REDACTED-PASSWORD]"),
    # PINs — \W{0,6}? tolerates natural phrasing like "my PIN is 0411" (verify F3)
    (re.compile(r"(?i)\bPIN\b(?:\W+\w+){0,2}?\W{0,3}?(?<![\w.])(\d{3,})"), "[REDACTED-PIN]"),
    # OpenAI / Anthropic API keys
    (re.compile(r"sk-[A-Za-z0-9]{16,}"), "[REDACTED-KEY]"),
    # Generic API keys / secrets / tokens / bearers
    (
        re.compile(
            r"(?i)\b(api[_-]?key|secret|token|bearer)\b\s*[:=]?\s*[A-Za-z0-9._\-]{12,}"
        ),
        "[REDACTED-KEY]",
    ),
    # AWS access key IDs
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED-KEY]"),
    # GitHub personal access tokens
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "[REDACTED-KEY]"),
    # JWTs (eyJ header)
    (re.compile(r"eyJ[A-Za-z0-9._\-]{20,}"), "[REDACTED-KEY]"),
    # PEM private keys (multiline; treat as single-line via DOTALL on the block start)
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[^-]*-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
        "[REDACTED-PRIVATE-KEY]",
    ),
]

_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_LONG_NUMBER_PATTERN = re.compile(r"\b\d{12,}\b")

# Residual detection (post-redaction sanity check)
_RESIDUAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\bpassword\b\s*[:=]?\s*[^\s\[]{4,}"),  # actual value follows
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"eyJ[A-Za-z0-9._\-]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _redact_text(text: str, cfg: dict[str, Any]) -> tuple[str, int]:
    """Apply all regex patterns to *text*. Returns (redacted_text, n_subs)."""
    n = 0
    for pat, replacement in _PATTERNS:
        new, count = pat.subn(replacement, text)
        n += count
        text = new

    if cfg.get("redact_emails", True):
        new, count = _EMAIL_PATTERN.subn("[REDACTED-EMAIL]", text)
        n += count
        text = new

    if cfg.get("redact_long_numbers", True):
        new, count = _LONG_NUMBER_PATTERN.subn("[REDACTED-NUMBER]", text)
        n += count
        text = new

    return text, n


def _entry_texts(entry: dict[str, Any]) -> list[tuple[str, ...]]:
    """Return (path, value) pairs for all text fields in an entry that need scanning."""
    fields: list[tuple[str, ...]] = []
    for key in ("situation", "call"):
        if key in entry and isinstance(entry[key], str):
            fields.append((key,))
    if "reasons" in entry and isinstance(entry["reasons"], list):
        for i, r in enumerate(entry["reasons"]):
            if isinstance(r, str):
                fields.append(("reasons", i))
    if "provenance" in entry and isinstance(entry.get("provenance"), dict):
        if "quote" in entry["provenance"] and isinstance(entry["provenance"]["quote"], str):
            fields.append(("provenance", "quote"))
    return fields


def _get_nested(entry: dict, path: tuple) -> str:
    obj: Any = entry
    for key in path:
        obj = obj[key]
    return str(obj)


def _set_nested(entry: dict, path: tuple, value: str) -> None:
    obj: Any = entry
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def redact_entries(
    entries: list[dict[str, Any]],
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Run the regex redaction pass over *entries* (in place — returns the same list).
    Logs a summary to stderr.  Does NOT run the LLM pass (that is the skill's job).
    """
    if cfg is None:
        cfg = {"redact_emails": True, "redact_long_numbers": True}

    redact_cfg = cfg.get("redaction", cfg)  # accept either top-level or nested
    total_fields = 0
    total_entries = 0

    for entry in entries:
        entry_touched = False
        for path in _entry_texts(entry):
            original = _get_nested(entry, path)
            redacted, n = _redact_text(original, redact_cfg)
            if n:
                _set_nested(entry, path, redacted)
                total_fields += n
                entry_touched = True
        if entry_touched:
            total_entries += 1

    if total_fields:
        print(
            f"[redact] redacted {total_fields} field(s) across {total_entries} entry/entries",
            file=sys.stderr,
        )
    return entries


def assert_clean(entries: list[dict[str, Any]]) -> None:
    """
    Scan redacted entries for any residual high-entropy secret patterns.
    RAISES RuntimeError if any are found — the caller MUST abort the run.
    """
    violations: list[str] = []
    for entry in entries:
        eid = entry.get("id", "<no-id>")
        for path in _entry_texts(entry):
            text = _get_nested(entry, path)
            for pat in _RESIDUAL_PATTERNS:
                m = pat.search(text)
                if m:
                    violations.append(
                        f"  entry {eid} field {'.'.join(str(p) for p in path)!r}: "
                        f"residual secret matched by {pat.pattern!r} → {m.group()[:40]!r}"
                    )

    if violations:
        msg = (
            "ABORT: residual secrets detected after redaction. "
            "No corpus file will be written. Fix and re-run.\n"
            + "\n".join(violations)
        )
        raise RuntimeError(msg)


def redact_llm(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Interface for the LLM semantic scan pass.  The BASE implementation is a no-op
    (prints a WARNING); the skill overrides this by monkey-patching or subclassing.

    WARNING: the LLM scan is the skill's responsibility.  Corpus files written via
    the CLI tool alone have only had the REGEX pass applied.
    """
    print(
        "[redact] WARNING: LLM semantic scan pass NOT run. "
        "Run via the /usual skill to apply the full two-pass redaction.",
        file=sys.stderr,
    )
    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Regex-only redaction pass for usual entries (stdin → stdout JSONL)."
    )
    parser.add_argument(
        "--no-llm-scan",
        action="store_true",
        help="Suppress the LLM scan warning (for offline / CI use).",
    )
    parser.add_argument(
        "--no-emails",
        action="store_true",
        help="Skip email redaction.",
    )
    parser.add_argument(
        "--no-long-numbers",
        action="store_true",
        help="Skip long-number redaction.",
    )
    args = parser.parse_args()

    cfg = {
        "redact_emails": not args.no_emails,
        "redact_long_numbers": not args.no_long_numbers,
    }

    entries = []
    for line in sys.stdin:
        line = line.strip()
        if line:
            entries.append(json.loads(line))

    redact_entries(entries, cfg)

    if not args.no_llm_scan:
        redact_llm(entries)

    for entry in entries:
        print(json.dumps(entry))


if __name__ == "__main__":
    _main()

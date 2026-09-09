"""curate.py — curation rules per spec §2.3.

Encodes the rules proven in the original build-exam.py, generalized from
hardcoded id sets into predicates + config-overridable manifest.

Rules (in priority order):
  1. manifest_overrides.drops / .scrubs / .keeps — explicit overrides win.
  2. DROP approvals-of-assistant (bare assent + low confidence with no reasons).
  3. DROP mechanical/no-judgment entries (ritual/procedural heuristic).
  4. DROP duplicates (same normalized situation+call or same provenance.quote).
  5. SCRUB codifications that leak holdout answers (handled by caller after holdout split).

Key distinction (§2.3):
  - Same DECISION restated in different words → leak → scrub/drop.
  - Same PRINCIPLE newly applied to a DIFFERENT situation → keep (fair game).
  curate() uses situation-similarity to distinguish these (same situation = leak).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Short bare-assent tokens that signal an approval-of-assistant call.
_ASSENT_TOKENS: frozenset[str] = frozenset(
    {
        "yeah", "yes", "yep", "yup", "ok", "okay", "sure", "fine",
        "do it", "do 1-2", "do it.", "agreed", "agree",
        "sounds good", "looks good", "great", "perfect",
        "a", "b", "c",  # bare option picks
        "aligned with your recommendation",
        "go ahead",
    }
)

# Ritual/mechanical call patterns (heuristic).
# Deliberately narrow: only match clear procedural/ritual phrases, not ordinary use
# of the word (e.g. "handoff between parents" is a product decision, not a ritual).
_RITUAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\bclose[-\s]session\b"),
    re.compile(r"(?i)\bstart\s+the\s+session\b"),
    re.compile(r"(?i)\bend\s+of\s+(session|day)\b"),
    re.compile(r"(?i)\bsession[-\s]handoff\b"),  # "session handoff" ritual only, not "handoff between parents"
    re.compile(r"(?i)\brun\s+the\s+(session\s+)?handoff\b"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation/whitespace, normalize unicode."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_assent(entry: dict[str, Any], min_confidence: float) -> bool:
    """Return True if the entry is a bare approval-of-assistant."""
    call_norm = _normalize(entry.get("call", ""))
    quote_norm = _normalize(entry.get("provenance", {}).get("quote", ""))

    # Bare token match on call or quote.
    if call_norm in _ASSENT_TOKENS or quote_norm in _ASSENT_TOKENS:
        return True

    # Very short call (≤4 words) + low confidence + no reasons.
    words = call_norm.split()
    conf = float(entry.get("confidence", 1.0))
    reasons = entry.get("reasons", [])
    if len(words) <= 4 and conf < min_confidence and not reasons:
        return True

    return False


def _is_ritual(entry: dict[str, Any]) -> bool:
    """Return True if the entry is a mechanical/no-judgment ritual."""
    text = " ".join(
        [
            entry.get("situation", ""),
            entry.get("call", ""),
            entry.get("provenance", {}).get("quote", ""),
        ]
    )
    return any(pat.search(text) for pat in _RITUAL_PATTERNS)


def _signature(entry: dict[str, Any]) -> str:
    """Normalized (situation, call) string for duplicate detection."""
    return _normalize(entry.get("situation", "")) + "|||" + _normalize(entry.get("call", ""))


def _quote_key(entry: dict[str, Any]) -> str:
    """Normalized provenance.quote for duplicate detection."""
    return _normalize(entry.get("provenance", {}).get("quote", ""))


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

class CurationResult:
    """Returned by curate()."""

    def __init__(self) -> None:
        self.kept: list[dict[str, Any]] = []
        self.dropped: list[dict[str, str]] = []   # {id, reason}
        self.scrubbed: list[dict[str, str]] = []  # {id, reason}

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "drops": sorted(d["id"] for d in self.dropped),
            "scrubs": sorted(s["id"] for s in self.scrubbed),
            "drop_reasons": {d["id"]: d["reason"] for d in self.dropped},
            "scrub_reasons": {s["id"]: s["reason"] for s in self.scrubbed},
            "study_count": len(self.kept),
        }


def curate(
    entries: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    holdout_ids: set[str] | None = None,
) -> CurationResult:
    """
    Apply curation rules to *entries* (which must already have `id` fields set).

    config keys used:
      - min_confidence (default 0.5)
      - heuristic_curation (default True) — set False to skip auto-heuristics and
        rely entirely on manifest_overrides for drops/scrubs (useful for regression
        tests that must reproduce an exact manifest-driven split).
      - manifest_overrides.drops / .scrubs / .keeps

    holdout_ids: if provided, scrub-for-leak check is skipped (caller handles it
    by not passing study entries that would leak holdout; the original scrub set
    in manifest_overrides covers the known codification leak cases).

    Returns a CurationResult with .kept / .dropped / .scrubbed lists.
    """
    min_confidence: float = float(config.get("min_confidence", 0.5))
    use_heuristics: bool = bool(config.get("heuristic_curation", True))
    overrides: dict[str, Any] = config.get("manifest_overrides", {})
    force_drops: set[str] = set(overrides.get("drops", []))
    force_scrubs: set[str] = set(overrides.get("scrubs", []))
    force_keeps: set[str] = set(overrides.get("keeps", []))

    result = CurationResult()

    seen_sigs: dict[str, str] = {}    # sig → first id seen
    seen_quotes: dict[str, str] = {}  # quote → first id seen

    for entry in entries:
        eid: str = entry["id"]

        # --- Priority 0: force-keep (explicit override wins everything) ---
        if eid in force_keeps:
            result.kept.append(entry)
            continue

        # --- Priority 1: explicit manifest drops ---
        if eid in force_drops:
            result.dropped.append({"id": eid, "reason": "manifest_override: drop"})
            continue

        # --- Priority 2: explicit manifest scrubs ---
        if eid in force_scrubs:
            result.scrubbed.append({"id": eid, "reason": "manifest_override: scrub"})
            continue

        # --- Priority 3: approval-of-assistant (heuristic, skippable) ---
        if use_heuristics and _is_assent(entry, min_confidence):
            result.dropped.append({"id": eid, "reason": "approval_of_assistant"})
            continue

        # --- Priority 4: mechanical/ritual (heuristic, skippable) ---
        if use_heuristics and _is_ritual(entry):
            result.dropped.append({"id": eid, "reason": "mechanical_ritual"})
            continue

        # --- Priority 5: duplicate detection (heuristic, skippable) ---
        sig = _signature(entry)
        quote = _quote_key(entry)

        if use_heuristics:
            if sig and sig in seen_sigs:
                # Same situation+call = duplicate decision → DROP (leak risk).
                result.dropped.append(
                    {
                        "id": eid,
                        "reason": f"duplicate_of:{seen_sigs[sig]} (same situation+call)",
                    }
                )
                continue

            if quote and quote in seen_quotes:
                # Same provenance.quote = literal duplicate → DROP.
                result.dropped.append(
                    {
                        "id": eid,
                        "reason": f"duplicate_of:{seen_quotes[quote]} (same quote)",
                    }
                )
                continue

        # Register this entry's sig and quote (always, for future duplicate detection).
        if sig:
            seen_sigs[sig] = eid
        if quote:
            seen_quotes[quote] = eid

        # --- KEEP ---
        result.kept.append(entry)

    return result

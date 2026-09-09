"""Tests for usual.curate — 9 test classes, 20 tests."""
from __future__ import annotations

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from usual.curate import curate, CurationResult, _normalize, _is_assent, _is_ritual


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(eid, situation="decide something", call="do the thing", confidence=0.9, **kwargs):
    return {
        "id": eid,
        "situation": situation,
        "call": call,
        "domain": kwargs.get("domain", "product"),
        "confidence": confidence,
        "reasons": list(kwargs.get("reasons", ["a clear reason"])),
        "provenance": {"file": "t.jsonl", "quote": kwargs.get("quote", call)},
    }


def _cfg(**kwargs):
    base = {
        "min_confidence": 0.5,
        "heuristic_curation": True,
        "manifest_overrides": {"drops": [], "scrubs": [], "holdout": [], "keeps": []},
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# 1. Manifest overrides — drops win first
# ---------------------------------------------------------------------------

class TestManifestDrops:
    def test_explicit_drop_removed(self):
        e = _entry("t:1")
        cfg = _cfg(manifest_overrides={"drops": ["t:1"], "scrubs": [], "holdout": [], "keeps": []})
        r = curate([e], cfg)
        assert len(r.kept) == 0
        assert any(d["id"] == "t:1" for d in r.dropped)

    def test_explicit_drop_reason_is_manifest(self):
        e = _entry("t:2")
        cfg = _cfg(manifest_overrides={"drops": ["t:2"], "scrubs": [], "holdout": [], "keeps": []})
        r = curate([e], cfg)
        assert "manifest_override" in r.dropped[0]["reason"]

    def test_keep_overrides_drop(self):
        """keep wins over everything including manifest drop."""
        e = _entry("t:3")
        cfg = _cfg(manifest_overrides={"drops": ["t:3"], "scrubs": [], "holdout": [], "keeps": ["t:3"]})
        r = curate([e], cfg)
        assert len(r.kept) == 1
        assert len(r.dropped) == 0


# ---------------------------------------------------------------------------
# 2. Manifest overrides — scrubs
# ---------------------------------------------------------------------------

class TestManifestScrubs:
    def test_explicit_scrub(self):
        e = _entry("t:4")
        cfg = _cfg(manifest_overrides={"drops": [], "scrubs": ["t:4"], "holdout": [], "keeps": []})
        r = curate([e], cfg)
        assert len(r.scrubbed) == 1
        assert r.scrubbed[0]["id"] == "t:4"
        assert len(r.kept) == 0

    def test_scrub_reason_contains_manifest(self):
        e = _entry("t:5")
        cfg = _cfg(manifest_overrides={"drops": [], "scrubs": ["t:5"], "holdout": [], "keeps": []})
        r = curate([e], cfg)
        assert "manifest_override" in r.scrubbed[0]["reason"]


# ---------------------------------------------------------------------------
# 3. Approval-of-assistant heuristic
# ---------------------------------------------------------------------------

class TestAssent:
    def test_bare_yeah_dropped(self):
        e = _entry("t:10", call="yeah", confidence=0.3, reasons=[])
        r = curate([e], _cfg())
        assert len(r.dropped) == 1
        assert r.dropped[0]["reason"] == "approval_of_assistant"

    def test_option_pick_a_dropped(self):
        e = _entry("t:11", call="A", quote="A", confidence=0.3, reasons=[])
        r = curate([e], _cfg())
        assert len(r.dropped) == 1

    def test_substantive_call_kept(self):
        e = _entry("t:12", call="Ship the feature to TestFlight tonight", confidence=0.9)
        r = curate([e], _cfg())
        assert len(r.kept) == 1

    def test_heuristic_disabled_keeps_assent(self):
        e = _entry("t:13", call="yeah", confidence=0.3, reasons=[])
        cfg = _cfg(heuristic_curation=False)
        r = curate([e], cfg)
        assert len(r.kept) == 1


# ---------------------------------------------------------------------------
# 4. Ritual / mechanical heuristic
# ---------------------------------------------------------------------------

class TestRitual:
    def test_close_session_dropped(self):
        e = _entry("t:20", situation="close-session ritual", call="close-session now")
        r = curate([e], _cfg())
        assert len(r.dropped) == 1
        assert r.dropped[0]["reason"] == "mechanical_ritual"

    def test_normal_handoff_kept(self):
        """'handoff between parents' is a product decision, NOT a ritual."""
        e = _entry("t:21", situation="handoff between parents in the app")
        r = curate([e], _cfg())
        assert len(r.kept) == 1


# ---------------------------------------------------------------------------
# 5. Duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicates:
    def test_same_situation_and_call_second_dropped(self):
        e1 = _entry("t:30", situation="ship to prod", call="use cloud run")
        e2 = _entry("t:31", situation="ship to prod", call="use cloud run")
        r = curate([e1, e2], _cfg())
        assert len(r.kept) == 1
        assert len(r.dropped) == 1
        assert "duplicate_of:t:30" in r.dropped[0]["reason"]

    def test_same_quote_second_dropped(self):
        e1 = _entry("t:32", situation="first context", call="first call", quote="same exact quote")
        e2 = _entry("t:33", situation="second context", call="second call", quote="same exact quote")
        r = curate([e1, e2], _cfg())
        assert len(r.kept) == 1
        assert "duplicate_of:t:32" in r.dropped[0]["reason"]

    def test_different_situation_same_principle_kept(self):
        """Same principle, different situation = keep (not a duplicate)."""
        e1 = _entry("t:34", situation="choosing auth for app A", call="use JWT",
                    quote="use JWT for app A — consistent with our kit")
        e2 = _entry("t:35", situation="choosing auth for app B (different product)", call="use JWT",
                    quote="same rule applies to app B — JWT for auth")
        r = curate([e1, e2], _cfg())
        assert len(r.kept) == 2


# ---------------------------------------------------------------------------
# 6. CurationResult.manifest_dict
# ---------------------------------------------------------------------------

class TestManifestDict:
    def test_manifest_dict_shape(self):
        e1 = _entry("t:40")
        e2 = _entry("t:41", call="yeah", confidence=0.3, reasons=[])
        r = curate([e1, e2], _cfg())
        md = r.manifest_dict()
        assert "drops" in md
        assert "scrubs" in md
        assert "drop_reasons" in md
        assert "study_count" in md

    def test_manifest_dict_study_count(self):
        entries = [_entry(f"t:{i}") for i in range(5)]
        r = curate(entries, _cfg())
        md = r.manifest_dict()
        assert md["study_count"] == len(r.kept)


# ---------------------------------------------------------------------------
# 7. normalize helper
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lowercase(self):
        assert _normalize("HELLO") == "hello"

    def test_strips_punctuation(self):
        assert _normalize("hello, world!") == "hello world"


# ---------------------------------------------------------------------------
# 8. Empty / edge inputs
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_input(self):
        r = curate([], _cfg())
        assert r.kept == []
        assert r.dropped == []
        assert r.scrubbed == []

    def test_all_kept_when_clean(self):
        entries = [
            _entry(f"e:{i}", situation=f"unique situation {i}", call=f"unique call {i}", quote=f"unique quote {i}")
            for i in range(3)
        ]
        r = curate(entries, _cfg())
        assert len(r.kept) == 3


# ---------------------------------------------------------------------------
# 9. heuristic_curation=False disables all heuristics
# ---------------------------------------------------------------------------

class TestHeuristicDisabled:
    def test_ritual_kept_when_heuristic_off(self):
        e = _entry("t:50", call="close-session now", situation="close-session ritual")
        cfg = _cfg(heuristic_curation=False)
        r = curate([e], cfg)
        assert len(r.kept) == 1

    def test_duplicate_kept_when_heuristic_off(self):
        e1 = _entry("t:51", situation="same", call="same")
        e2 = _entry("t:52", situation="same", call="same")
        cfg = _cfg(heuristic_curation=False)
        r = curate([e1, e2], cfg)
        assert len(r.kept) == 2

"""tests/test_build_exam.py — exam engine tests per spec §5 / Acceptance criteria.

Tests:
  1. Items file is answer-free (no 'call', no 'reasons') — the anti-leak assert.
  2. Redaction ABORTS when residual secrets exist.
  3. Curation drops approvals-of-assistant and ritual entries.
  4. grade.py regression: K=75.0/66.67 and M=62.5/52.78 from day0-grades.json inputs.
  5. build_exam full pipeline on synthetic fixtures produces correct counts.
  6. Duplicate entries are dropped.
  7. ID assignment format (w<wave>b<batch>:<lineno>).
"""

import json
import os
import sys
import tempfile
import shutil
import pytest

# Make the package importable without installing.
SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
# The reference user's usual dir (used for regression tests only; may not exist in CI).
# tests/ ->  -> packages/ -> worktree_root/ (3 levels up), then brain/usual.
REAL_USUAL = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "brain", "usual"
))

from usual.build_exam import (
    load_mining_entries,
    assert_items_answer_free,
    build_exam,
)
from usual.redact import redact_entries, assert_clean
from usual.curate import curate
from usual.grade import compute_arm_scores, grade_arms
from usual.config import load_config, DEFAULTS


# ---------------------------------------------------------------------------
# Helper: minimal config for tests
# ---------------------------------------------------------------------------

def _base_config(**overrides):
    cfg = dict(DEFAULTS)
    cfg["redaction"] = {"regex": True, "llm_scan": False, "redact_emails": True, "redact_long_numbers": True}
    cfg["holdout_size"] = 3
    cfg["holdout_strategy"] = "random-seeded"
    cfg["seed"] = 42
    cfg["manifest_overrides"] = {"drops": [], "scrubs": [], "holdout": [], "keeps": []}
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# 1. assert_items_answer_free — the anti-leak gate
# ---------------------------------------------------------------------------

class TestItemsAnswerFree:
    def test_clean_items_pass(self):
        items = [
            {"id": "w1b1:1", "situation": "Some decision context.", "domain": "product", "date": "2026-01-01"},
        ]
        # Should not raise.
        assert_items_answer_free(items)

    def test_call_key_triggers_abort(self):
        items = [
            {"id": "w1b1:1", "situation": "Some context.", "domain": "product", "date": "2026-01-01",
             "call": "Ship it"},
        ]
        with pytest.raises(AssertionError, match="call"):
            assert_items_answer_free(items)

    def test_reasons_key_triggers_abort(self):
        items = [
            {"id": "w1b1:1", "situation": "Some context.", "domain": "product", "date": "2026-01-01",
             "reasons": ["because X"]},
        ]
        with pytest.raises(AssertionError, match="reasons"):
            assert_items_answer_free(items)

    def test_both_keys_triggers_abort(self):
        items = [
            {"id": "w1b1:1", "situation": "Some context.", "domain": "product", "date": "2026-01-01",
             "call": "do it", "reasons": []},
        ]
        with pytest.raises(AssertionError):
            assert_items_answer_free(items)


# ---------------------------------------------------------------------------
# 2. Redaction — abort on residual
# ---------------------------------------------------------------------------

class TestRedaction:
    def test_password_is_redacted(self):
        entries = [{"id": "t:1", "situation": "password: hunter2", "call": "change it",
                    "reasons": [], "provenance": {"file": "x.jsonl", "quote": "hunter2"}}]
        redact_entries(entries, {"redaction": {"redact_emails": True, "redact_long_numbers": True}})
        assert "[REDACTED-PASSWORD]" in entries[0]["situation"]

    def test_api_key_is_redacted(self):
        entries = [{"id": "t:1", "situation": "Set token=sk-abc123456789012345678", "call": "ok",
                    "reasons": [], "provenance": {"file": "x.jsonl", "quote": "sk-abc123456789012345678"}}]
        redact_entries(entries, {"redaction": {"redact_emails": True, "redact_long_numbers": True}})
        assert "sk-abc" not in entries[0]["situation"]
        assert "[REDACTED-KEY]" in entries[0]["situation"]

    def test_clean_entries_pass_assert_clean(self):
        entries = [{"id": "t:1", "situation": "Normal text here.", "call": "proceed",
                    "reasons": [], "provenance": {"file": "x.jsonl", "quote": "proceed"}}]
        # Should not raise.
        assert_clean(entries)

    def test_residual_triggers_abort(self):
        # Simulate an entry that slipped through redaction.
        entries = [{"id": "t:1", "situation": "password: sneaky_value_here", "call": "change it",
                    "reasons": [], "provenance": {"file": "x.jsonl", "quote": "sneaky_value_here"}}]
        with pytest.raises(RuntimeError, match="ABORT"):
            assert_clean(entries)

    def test_redacted_password_is_idempotent(self):
        """Re-running redaction on already-redacted text must not further mangle it."""
        entries = [{"id": "t:1", "situation": "[REDACTED-PASSWORD] is set", "call": "ok",
                    "reasons": [], "provenance": {"file": "x.jsonl", "quote": "[REDACTED-PASSWORD]"}}]
        redact_entries(entries, {"redaction": {"redact_emails": True, "redact_long_numbers": True}})
        # Should remain as-is (token itself is not a secret pattern).
        assert entries[0]["situation"] == "[REDACTED-PASSWORD] is set"


# ---------------------------------------------------------------------------
# 3. Curation
# ---------------------------------------------------------------------------

class TestCuration:
    def _entry(self, eid, call, quote="", reasons=None, confidence=0.9):
        return {
            "id": eid,
            "situation": f"A decision situation for {eid}.",
            "call": call,
            "reasons": reasons if reasons is not None else ["a real reason"],
            "domain": "product",
            "date": "2026-01-01",
            "provenance": {"file": "f.jsonl", "quote": quote or call},
            "confidence": confidence,
        }

    def test_assent_is_dropped(self):
        entries = [
            self._entry("t:1", "do it", "do it", reasons=[], confidence=0.2),
            self._entry("t:2", "Ship the prototype", reasons=["speed matters"]),
        ]
        cfg = _base_config()
        result = curate(entries, cfg)
        ids_kept = {e["id"] for e in result.kept}
        assert "t:1" not in ids_kept
        assert "t:2" in ids_kept

    def test_ritual_is_dropped(self):
        entries = [
            self._entry("t:1", "Run the close-session ritual."),
            self._entry("t:2", "Pick the grouped card layout.", reasons=["looks better"]),
        ]
        cfg = _base_config()
        result = curate(entries, cfg)
        ids_kept = {e["id"] for e in result.kept}
        assert "t:1" not in ids_kept
        assert "t:2" in ids_kept

    def test_manifest_override_drop_wins(self):
        entries = [
            self._entry("t:1", "Real substantive decision.", reasons=["good reason"]),
        ]
        cfg = _base_config()
        cfg["manifest_overrides"]["drops"] = ["t:1"]
        result = curate(entries, cfg)
        assert not result.kept

    def test_manifest_override_scrub(self):
        entries = [
            self._entry("t:1", "Codification that leaks a holdout answer."),
        ]
        cfg = _base_config()
        cfg["manifest_overrides"]["scrubs"] = ["t:1"]
        result = curate(entries, cfg)
        assert not result.kept
        assert result.scrubbed[0]["id"] == "t:1"

    def test_duplicate_is_dropped(self):
        """Two entries with the same situation+call: second is dropped."""
        entry1 = self._entry("t:1", "Ship it rough", "Ship it rough", reasons=["speed"])
        # Same situation text (id changes but same decision context since we use same template).
        entry2 = dict(entry1)
        entry2["id"] = "t:2"
        entries = [entry1, entry2]
        cfg = _base_config()
        result = curate(entries, cfg)
        ids_kept = {e["id"] for e in result.kept}
        assert "t:1" in ids_kept
        assert "t:2" not in ids_kept  # dropped as duplicate

    def test_manifest_keep_overrides_assent(self):
        """A force-keep survives even if it looks like an assent."""
        entries = [
            self._entry("t:1", "do it", "do it", reasons=[], confidence=0.1),
        ]
        cfg = _base_config()
        cfg["manifest_overrides"]["keeps"] = ["t:1"]
        result = curate(entries, cfg)
        assert result.kept[0]["id"] == "t:1"


# ---------------------------------------------------------------------------
# 4. grade.py regression — must reproduce K=75.0/66.67, M=62.5/52.78
# ---------------------------------------------------------------------------

class TestGradeRegression:
    """
    Feed the exact K and M columns from brain/usual/exam/day0-grades.json
    and verify the recomputed totals match the proven numbers.
    Skipped if the real usual dir is unavailable.
    """

    @pytest.fixture(autouse=True)
    def check_real_data(self):
        grades_path = os.path.join(REAL_USUAL, "exam", "day0-grades.json")
        if not os.path.exists(grades_path):
            pytest.skip("Real usual exam data not available (brain/usual/exam/day0-grades.json)")
        self.grades_path = grades_path

    def test_K_scores(self):
        with open(self.grades_path) as fh:
            grades = json.load(fh)
        scores = compute_arm_scores(grades["K"])
        assert scores["call_pct"] == 75.0, f"K call_pct: expected 75.0, got {scores['call_pct']}"
        assert scores["reason_pct"] == 66.67, f"K reason_pct: expected 66.67, got {scores['reason_pct']}"

    def test_M_scores(self):
        with open(self.grades_path) as fh:
            grades = json.load(fh)
        scores = compute_arm_scores(grades["M"])
        assert scores["call_pct"] == 62.5, f"M call_pct: expected 62.5, got {scores['call_pct']}"
        assert scores["reason_pct"] == 52.78, f"M reason_pct: expected 52.78, got {scores['reason_pct']}"

    def test_null_reason_excluded_from_denominator(self):
        """w1b2:7 and w1b2:12 have reason_match: null — they must not count in reason denominator."""
        with open(self.grades_path) as fh:
            grades = json.load(fh)
        k_items = grades["K"]
        null_ids = {item["id"] for item in k_items if item.get("reason_match") is None}
        assert "w1b2:7" in null_ids, "w1b2:7 should have null reason_match"
        assert "w1b2:12" in null_ids, "w1b2:12 should have null reason_match"
        # Verify our compute function excludes them.
        reason_items = [i for i in k_items if i.get("reason_match") is not None]
        manual_reason_pct = round(sum(i["reason_match"] for i in reason_items) / (len(reason_items) * 2) * 100, 2)
        scores = compute_arm_scores(k_items)
        assert scores["reason_pct"] == manual_reason_pct

    def test_grade_arms_full(self):
        with open(self.grades_path) as fh:
            grades = json.load(fh)
        result = grade_arms({"K": grades["K"], "M": grades["M"]})
        assert result["totals"]["K"]["call_pct"] == 75.0
        assert result["totals"]["K"]["reason_pct"] == 66.67
        assert result["totals"]["M"]["call_pct"] == 62.5
        assert result["totals"]["M"]["reason_pct"] == 52.78


# ---------------------------------------------------------------------------
# 5. Full pipeline on synthetic fixtures
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        # Copy fixtures to tmpdir/mining/
        mining_dir = os.path.join(self.tmpdir, "mining")
        shutil.copytree(FIXTURES, mining_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_pipeline_produces_correct_counts(self):
        """Synthetic fixtures: 10 raw entries, 2 should be dropped (assent + ritual)."""
        cfg = _base_config(holdout_size=3, seed=42)
        summary = build_exam(cfg, self.tmpdir, day=0)
        # 10 total entries.
        assert summary["mined"] == 10
        # Items file exists and is answer-free.
        items_path = os.path.join(self.tmpdir, "exam", "day0-items.json")
        assert os.path.exists(items_path)
        with open(items_path) as fh:
            items = json.load(fh)
        assert_items_answer_free(items)
        assert len(items) == 3

    def test_items_file_never_contains_call_or_reasons(self):
        """Core anti-leak invariant must hold after the pipeline runs."""
        cfg = _base_config(holdout_size=2, seed=7)
        build_exam(cfg, self.tmpdir, day=0)
        items_path = os.path.join(self.tmpdir, "exam", "day0-items.json")
        with open(items_path) as fh:
            items = json.load(fh)
        for item in items:
            assert "call" not in item, f"item {item['id']} has 'call' field — LEAK"
            assert "reasons" not in item, f"item {item['id']} has 'reasons' field — LEAK"

    def test_study_corpus_written(self):
        cfg = _base_config(holdout_size=2, seed=0)
        build_exam(cfg, self.tmpdir, day=0)
        corpus_path = os.path.join(self.tmpdir, "study-corpus.jsonl")
        assert os.path.exists(corpus_path)
        with open(corpus_path) as fh:
            lines = [l for l in fh if l.strip()]
        assert len(lines) > 0

    def test_curation_manifest_written(self):
        cfg = _base_config(holdout_size=2, seed=1)
        build_exam(cfg, self.tmpdir, day=0)
        manifest_path = os.path.join(self.tmpdir, "curation-manifest.json")
        assert os.path.exists(manifest_path)
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        assert "drops" in manifest
        assert "scrubs" in manifest
        assert "holdout" in manifest
        assert "study_count" in manifest


# ---------------------------------------------------------------------------
# 6. ID assignment format
# ---------------------------------------------------------------------------

class TestIdAssignment:
    def test_id_format(self):
        entries = load_mining_entries(FIXTURES)
        for eid in entries:
            # Must match w<wave>b<batch>:<lineno>
            assert eid.startswith("w"), f"id {eid!r} does not start with 'w'"
            assert "b" in eid, f"id {eid!r} missing 'b'"
            assert ":" in eid, f"id {eid!r} missing ':'"

    def test_ids_are_unique(self):
        entries = load_mining_entries(FIXTURES)
        assert len(entries) == len(set(entries.keys()))

    def test_lineno_is_1indexed(self):
        entries = load_mining_entries(FIXTURES)
        # batch-1 first entry should be w1b1:1
        assert "w1b1:1" in entries


# ---------------------------------------------------------------------------
# 7. Regression proof: the reference wave-1 mining reproduces original counts
# ---------------------------------------------------------------------------

class TestWave1Regression:
    """
    Feeds the reference user's actual mining dir + exact wave-1 manifest_overrides into build_exam
    and asserts mined=167 dropped=11 scrubbed=1 holdout=20 study=135.
    Skipped if the real usual dir is unavailable.
    """

    DROPS = [
        "w1b1:17", "w1b1:21", "w1b2:1", "w1b2:3", "w1b2:16",
        "w1b2:20", "w1b2:42", "w1b2:43", "w1b3:40", "w1b1:12", "w1b4:7",
    ]
    SCRUBS = ["w1b2:24"]
    HOLDOUT = [
        "w1b2:23", "w1b2:25", "w1b2:22", "w1b2:19", "w1b2:18",
        "w1b2:17", "w1b2:21", "w1b2:15", "w1b2:7",  "w1b2:12",
        "w1b4:1",  "w1b4:6",  "w1b3:45", "w1b3:46", "w1b4:31",
        "w1b4:34", "w1b3:29", "w1b4:23", "w1b3:44", "w1b4:29",
    ]

    @pytest.fixture(autouse=True)
    def check_real_data(self, tmp_path):
        mining_dir = os.path.join(REAL_USUAL, "mining")
        if not os.path.exists(mining_dir):
            pytest.skip("Real usual mining data not available (brain/usual/mining/)")
        self.real_mining = mining_dir
        self.tmpdir = str(tmp_path)

    def test_wave1_counts(self):
        """Reproduce mined=167 dropped=11 scrubbed=1 holdout=20 study=135."""
        # Copy mining dir into tmp space so we can write output there.
        mining_dst = os.path.join(self.tmpdir, "mining")
        shutil.copytree(self.real_mining, mining_dst)

        cfg = dict(DEFAULTS)
        cfg["redaction"] = {
            "regex": True, "llm_scan": False,
            "redact_emails": False, "redact_long_numbers": False,
        }
        cfg["holdout_size"] = 20
        cfg["holdout_strategy"] = "manual"
        # heuristic_curation=False: use only the explicit manifest, not auto-heuristics,
        # so the count exactly matches the original build-exam.py output.
        cfg["heuristic_curation"] = False
        cfg["manifest_overrides"] = {
            "drops": self.DROPS,
            "scrubs": self.SCRUBS,
            "holdout": self.HOLDOUT,
            "keeps": [],
        }

        summary = build_exam(cfg, self.tmpdir, day=0)

        assert summary["mined"] == 167, f"mined: expected 167, got {summary['mined']}"
        assert summary["dropped"] == 11, f"dropped: expected 11, got {summary['dropped']}"
        assert summary["scrubbed"] == 1, f"scrubbed: expected 1, got {summary['scrubbed']}"
        assert summary["holdout"] == 20, f"holdout: expected 20, got {summary['holdout']}"
        assert summary["study"] == 135, f"study: expected 135, got {summary['study']}"

        # Verify items file is answer-free.
        items_path = os.path.join(self.tmpdir, "exam", "day0-items.json")
        with open(items_path) as fh:
            items = json.load(fh)
        assert_items_answer_free(items)

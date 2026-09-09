"""Tests for usual.redact — 10 test classes, 32 tests."""
from __future__ import annotations

import copy
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from usual.redact import redact_entries, assert_clean, _redact_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(situation="normal situation", call="normal call", quote="normal quote", **kwargs):
    e = {
        "id": kwargs.get("id", "t:1"),
        "situation": situation,
        "call": call,
        "domain": "factory",
        "confidence": 0.9,
        "reasons": list(kwargs.get("reasons", [])),
        "provenance": {"file": "test.jsonl", "quote": quote},
    }
    return e


# ---------------------------------------------------------------------------
# 1. Password redaction
# ---------------------------------------------------------------------------

class TestPasswordRedaction:
    def test_plain_password_value(self):
        e = _entry(situation="use password hunter2 to log in")
        redact_entries([e], {})
        assert "[REDACTED-PASSWORD]" in e["situation"]
        assert "hunter2" not in e["situation"]

    def test_password_with_colon(self):
        e = _entry(call="password: s3cr3t")
        redact_entries([e], {})
        assert "[REDACTED-PASSWORD]" in e["call"]
        assert "s3cr3t" not in e["call"]

    def test_passwd_variant(self):
        e = _entry(situation="set passwd mysecret")
        redact_entries([e], {})
        assert "[REDACTED-PASSWORD]" in e["situation"]

    def test_idempotent_already_redacted_token(self):
        """password [REDACTED-PASSWORD] must NOT be consumed again."""
        text = "login using the shared password [REDACTED-PASSWORD]."
        e = _entry(call=text)
        before = copy.deepcopy(e)
        redact_entries([e], {})
        assert e["call"] == before["call"], (
            f"Non-idempotent: got {e['call']!r}"
        )

    def test_idempotent_standalone_token(self):
        e = _entry(situation="[REDACTED-PASSWORD] is already safe")
        before = copy.deepcopy(e)
        redact_entries([e], {})
        assert e["situation"] == before["situation"]

    def test_password_in_quote(self):
        e = _entry(quote="just use password abc123")
        redact_entries([e], {})
        assert "[REDACTED-PASSWORD]" in e["provenance"]["quote"]


# ---------------------------------------------------------------------------
# 2. API key redaction
# ---------------------------------------------------------------------------

class TestApiKeyRedaction:
    def test_sk_key(self):
        e = _entry(situation="key=sk-ABCDEFGH012345678")
        redact_entries([e], {})
        assert "[REDACTED-KEY]" in e["situation"]
        assert "sk-ABCDEFGH012345678" not in e["situation"]

    def test_short_sk_key_not_redacted(self):
        """sk- with fewer than 16 chars is NOT a key."""
        e = _entry(situation="sk-short is fine")
        before = copy.deepcopy(e)
        redact_entries([e], {})
        assert e["situation"] == before["situation"]

    def test_akia_key(self):
        e = _entry(call="AKIAIOSFODNN7EXAMPLE is an AWS key")
        redact_entries([e], {})
        assert "[REDACTED-KEY]" in e["call"]

    def test_github_pat(self):
        e = _entry(situation="token ghp_ABCDEFGHIJ1234567890AB here")
        redact_entries([e], {})
        assert "[REDACTED-KEY]" in e["situation"]

    def test_jwt(self):
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxw"
        e = _entry(call=f"bearer {jwt}")
        redact_entries([e], {})
        assert "[REDACTED-KEY]" in e["call"]


# ---------------------------------------------------------------------------
# 3. assert_clean
# ---------------------------------------------------------------------------

class TestAssertClean:
    def test_raises_on_residual_password(self):
        e = _entry(situation="password hunter2 is real")
        with pytest.raises(RuntimeError, match="residual secrets"):
            assert_clean([e])

    def test_raises_on_residual_sk_key(self):
        e = _entry(call="sk-ABCDEFGH012345678 is still here")
        with pytest.raises(RuntimeError, match="residual secrets"):
            assert_clean([e])

    def test_passes_on_already_redacted(self):
        e = _entry(situation="[REDACTED-PASSWORD] and [REDACTED-KEY] are fine")
        assert_clean([e])  # must not raise

    def test_passes_on_clean_entry(self):
        e = _entry()
        assert_clean([e])  # must not raise

    def test_violation_includes_entry_id(self):
        e = _entry(id="problem:42", situation="password exposed123")
        with pytest.raises(RuntimeError, match="problem:42"):
            assert_clean([e])


# ---------------------------------------------------------------------------
# 4. redact_text internal helper
# ---------------------------------------------------------------------------

class TestRedactText:
    def test_counts_substitutions(self):
        text = "password hunter2 and password secret2"
        result, n = _redact_text(text, {})
        assert n == 2
        assert result.count("[REDACTED-PASSWORD]") == 2

    def test_zero_subs_on_clean(self):
        _, n = _redact_text("nothing sensitive here", {})
        assert n == 0

    def test_email_redacted_by_default(self):
        text = "email me at user@example.com please"
        result, n = _redact_text(text, {"redact_emails": True})
        assert "[REDACTED-EMAIL]" in result
        assert n >= 1

    def test_email_skipped_when_disabled(self):
        text = "email user@example.com here"
        result, _ = _redact_text(text, {"redact_emails": False})
        assert "user@example.com" in result

    def test_long_number_redacted(self):
        text = "account 123456789012 is flagged"
        result, n = _redact_text(text, {"redact_long_numbers": True})
        assert "[REDACTED-NUMBER]" in result

    def test_short_number_not_redacted(self):
        text = "item 12345 is fine"
        result, _ = _redact_text(text, {"redact_long_numbers": True})
        assert "12345" in result


# ---------------------------------------------------------------------------
# 5. Multi-field redaction
# ---------------------------------------------------------------------------

class TestMultiFieldRedaction:
    def test_redacts_situation_and_call(self):
        e = _entry(
            situation="my password abc is here",
            call="change password xyz now",
        )
        redact_entries([e], {})
        assert "[REDACTED-PASSWORD]" in e["situation"]
        assert "[REDACTED-PASSWORD]" in e["call"]

    def test_redacts_reasons_list(self):
        e = _entry()
        e["reasons"] = ["use password mysecret here", "normal reason"]
        redact_entries([e], {})
        assert "[REDACTED-PASSWORD]" in e["reasons"][0]
        assert e["reasons"][1] == "normal reason"

    def test_entry_touched_flag(self, capsys):
        e = _entry(situation="password exposed123 here")
        redact_entries([e], {})
        captured = capsys.readouterr()
        assert "redacted" in captured.err


# ---------------------------------------------------------------------------
# 6. Idempotency on corpus
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_double_pass_stable(self):
        """Running redact twice must produce the same result as running once."""
        e = _entry(situation="password [REDACTED-PASSWORD] already cleaned")
        entries = [e]
        redact_entries(entries, {})
        first_pass = copy.deepcopy(entries)
        redact_entries(entries, {})
        assert entries == first_pass

    def test_no_op_on_clean_entry(self):
        e = _entry()
        before = copy.deepcopy(e)
        redact_entries([e], {})
        assert e == before


# ---------------------------------------------------------------------------
# 7. Logging / stderr output
# ---------------------------------------------------------------------------

class TestLogging:
    def test_no_output_when_nothing_redacted(self, capsys):
        e = _entry()
        redact_entries([e], {})
        assert capsys.readouterr().err == ""

    def test_output_when_redacted(self, capsys):
        e = _entry(situation="password abc123")
        redact_entries([e], {})
        err = capsys.readouterr().err
        assert "[redact]" in err
        assert "1 entry" in err


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_entry_list(self):
        result = redact_entries([], {})
        assert result == []

    def test_entry_missing_optional_fields(self):
        e = {"id": "t:x", "situation": "password abc123", "call": "ok",
             "domain": "other", "confidence": 0.5}
        redact_entries([e], {})
        assert "[REDACTED-PASSWORD]" in e["situation"]

    def test_none_config(self):
        e = _entry(situation="password abc123")
        redact_entries([e], None)
        assert "[REDACTED-PASSWORD]" in e["situation"]


# ---------------------------------------------------------------------------
# 9. PIN redaction
# ---------------------------------------------------------------------------

class TestPinRedaction:
    def test_pin_redacted(self):
        e = _entry(call="PIN 1234 is set")
        redact_entries([e], {})
        assert "[REDACTED-PIN]" in e["call"]

    def test_short_pin_not_redacted(self):
        e = _entry(call="PIN 12 is short")
        before_call = e["call"]
        redact_entries([e], {})
        # 2 digits < 3, should not be redacted
        assert "[REDACTED-PIN]" not in e["call"]


# ---------------------------------------------------------------------------
# 10. Fixture-driven (synthetic.jsonl)
# ---------------------------------------------------------------------------

class TestFixtureDriven:
    def test_fixture_contains_secrets(self):
        fixture = os.path.join(os.path.dirname(__file__), "fixtures", "synthetic.jsonl")
        import json
        entries = [json.loads(l) for l in open(fixture) if l.strip()]
        # At least one entry must have a password or key in raw form
        raw_texts = [
            e.get("situation", "") + e.get("call", "") +
            e.get("provenance", {}).get("quote", "")
            for e in entries
        ]
        combined = " ".join(raw_texts)
        assert "password" in combined.lower() or "sk-" in combined

    def test_fixture_redacts_cleanly(self):
        fixture = os.path.join(os.path.dirname(__file__), "fixtures", "synthetic.jsonl")
        import json
        entries = [json.loads(l) for l in open(fixture) if l.strip()]
        redact_entries(entries, {})
        # assert_clean must not raise after redact
        assert_clean(entries)

"""Consumer answers stay isolated and are scrubbed before browser persistence."""

import json
import pytest

from usual.onboarding import prepare_answers
from usual.server import GLOBAL_STORE


def test_preparation_preserves_context_without_writing_shared_corpus():
    before = GLOBAL_STORE.get_entries()
    result = prepare_answers({"answers": [{
        "id": "planning", "call": "Leave room to change plans",
        "reason": "Unless I am traveling with the children",
    }]})
    assert result[0]["call"] == "Leave room to change plans"
    assert result[0]["reasons"] == ["Unless I am traveling with the children"]
    assert result[0]["situation"] == "When planning a free day out"
    assert GLOBAL_STORE.get_entries() == before


def test_redaction_includes_answer_reason_and_provenance():
    result = prepare_answers({"answers": [{
        "id": "writing", "call": "Email me at example@example.com",
        "reason": "password: example-secret-123",
    }]})
    encoded = json.dumps(result)
    assert "example@example.com" not in encoded
    assert "example-secret-123" not in encoded
    assert "REDACTED-EMAIL" in encoded
    assert "REDACTED-PASSWORD" in encoded


@pytest.mark.parametrize("payload", [
    None, [], {}, {"answers": []}, {"answers": [None]},
    {"answers": [{"id": "unknown", "call": "hello"}]},
    {"answers": [{"id": [], "call": "hello"}]},
    {"answers": [{"id": "writing", "call": " "}]},
    {"answers": [{"id": "writing", "call": "x" * 501}]},
    {"answers": [{"id": "writing", "call": "hello", "reason": []}]},
    {"answers": [{"id": "writing", "call": "hello"}] * 2},
])
def test_rejects_invalid_answers(payload):
    with pytest.raises(ValueError):
        prepare_answers(payload)

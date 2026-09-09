import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from examples.demo import demo_rows
from itchy import Predictor, Task
from itchy.model import fingerprint, wilson_lower


@pytest.fixture
def task(tmp_path):
    return Task(tmp_path / "task", labels=["billing", "access", "bug"])


@pytest.fixture
def trained(task, tmp_path):
    source = tmp_path / "seed.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in demo_rows()))
    task.import_jsonl(source)
    report = task.fit(target_precision=0.9, min_support=30)
    assert report["ready"], report["reasons"]
    return task, report


def test_cold_start_fallback_is_not_training(task):
    calls = []
    def fallback(text):
        calls.append(text)
        return "billing"
    decision = task.decide("Refund please", fallback=fallback)
    assert calls == ["Refund please"]
    assert decision.route == "fallback"
    assert task.status()["examples"] == {}
    assert len(task.review_queue()) == 1
    task.correct(decision.id, "billing")
    assert sum(task.status()["examples"].values()) == 1
    assert task.review_queue() == []


def test_invalid_fallback_and_failure_are_not_labels(task):
    with pytest.raises(ValueError, match="taxonomy"):
        task.decide("hi", fallback=lambda _: "invented")
    def failed(_):
        raise RuntimeError("provider unavailable")
    with pytest.raises(RuntimeError, match="unavailable"):
        task.decide("hi", fallback=failed)
    assert task.status()["examples"] == {}
    assert task.status()["decisions"] == {}


def test_same_text_and_related_groups_cannot_leak(task):
    task.teach("Café   BILL", "billing", split="train", group="thread-1")
    with pytest.raises(ValueError, match="cross"):
        task.teach("Cafe\u0301 bill", "billing", split="test")
    with pytest.raises(ValueError, match="cross"):
        task.teach("different reply to that bill", "billing", split="test", group="thread-1")
    assert task.teach("third reply", "billing", group="thread-1") == "train"
    assert fingerprint("Café BILL") == fingerprint("Cafe\u0301 bill")


def test_corrections_keep_assignment_across_restarts(task):
    split = task.teach("forgot my password", "billing")
    restored = Task(task.path)
    assert restored.teach("forgot my password", "access") == split
    assert len(restored._examples()) == 1
    assert restored._examples()[0]["label"] == "access"
    with pytest.raises(ValueError, match="immutable"):
        Task(task.path, labels=["one", "two"])


def test_import_rolls_back_on_invalid_row(task, tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text":"refund","label":"billing"}\n{"text":"bad","label":"invented"}\n')
    with pytest.raises(ValueError, match="line 2"):
        task.import_jsonl(path)
    assert task.status()["examples"] == {}


def test_small_sample_does_not_claim_certainty(task):
    for split in ("train", "calibration", "test"):
        for label in task.labels:
            task.teach(f"{label} message {split}", label, split=split)
    report = task.fit()
    assert not report["ready"]
    with pytest.raises(ValueError, match="gate"):
        task.promote(report["model_id"])
    assert wilson_lower(3, 3) < 0.5
    assert wilson_lower(100, 100) < 0.97


def test_full_loop_and_portable_artifact(trained, tmp_path):
    task, report = trained
    assert task.status()["champion"] is None
    task.promote(report["model_id"])
    decision = task.decide("charged twice for my subscription", fallback=lambda _: pytest.fail("unnecessary fallback"))
    assert decision.route == "local" and decision.label == "billing"
    assert task.decide("紫の宇宙船 🪐 мир").route == "abstain"
    assert any(row["route"] == "local" for row in task.review_queue())
    assert all(row["route"] == "local" for row in task.review_queue(route="local"))
    assert len(task.review_queue(route="abstain")) == 1
    exported = Path(task.export(tmp_path / "portable"))
    assert {p.name for p in exported.iterdir()} == {"model.json", "weights.npz", "report.json"}
    prediction = Predictor.load(exported).predict("charged twice for my subscription")
    assert prediction.label == decision.label
    assert prediction.score == decision.score
    with pytest.raises(FileExistsError):
        task.export(exported)
    arrays = exported / "weights.npz"
    arrays.write_bytes(arrays.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        Predictor.load(exported)


def test_stale_candidate_cannot_be_promoted(trained):
    task, report = trained
    task.teach("another reviewed invoice", "billing")
    with pytest.raises(ValueError, match="changed"):
        task.promote(report["model_id"])


def test_test_labels_never_change_weights_or_thresholds(trained):
    task, first = trained
    original = Predictor.load(task._artifact(first["model_id"]))
    for row in task._examples():
        if row["split"] == "test":
            flipped = task.labels[(task.labels.index(row["label"]) + 1) % len(task.labels)]
            task.teach(row["text"], flipped)
    second = task.fit(target_precision=0.9, min_support=30)
    changed = Predictor.load(task._artifact(second["model_id"]))
    np.testing.assert_array_equal(original.weights, changed.weights)
    np.testing.assert_array_equal(original.bias, changed.bias)
    assert first["thresholds"] == second["thresholds"]
    assert not second["ready"]
    assert second["test"]["precision"] == 0
    with pytest.raises(ValueError, match="gate"):
        task.promote(second["model_id"])


def test_input_validation_and_abstention(trained):
    task, report = trained
    task.promote(report["model_id"])
    assert task.decide("a" * 9000).reason == "input_too_long"
    with pytest.raises(ValueError, match="nonempty"):
        task.decide("   ")
    with pytest.raises(ValueError, match="exceeds"):
        task.teach("a" * 9000, "billing")
    with pytest.raises(ValueError, match="model ID"):
        task.promote("../elsewhere")


def test_cli_errors_and_cold_start(tmp_path):
    prefix = [sys.executable, "-m", "itchy", "--task", str(tmp_path / "cli")]
    result = subprocess.run(prefix + ["init", "--labels", "yes", "no"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(prefix + ["predict", "hello"], capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)["route"] == "abstain"
    result = subprocess.run(prefix + ["fit"], capture_output=True, text=True)
    assert result.returncode == 1
    assert "train needs examples" in json.loads(result.stderr)["error"]


def test_non_object_jsonl_has_a_useful_error(task, tmp_path):
    path = tmp_path / "array.jsonl"
    path.write_text('[]\n')
    with pytest.raises(ValueError, match="line 1.*JSON object"):
        task.import_jsonl(path)
    assert task.status()["examples"] == {}

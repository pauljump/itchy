import json
import sqlite3

import numpy as np
import pytest

from itchy.preference import PreferenceModel, fit_whetstone, suggest
from itchy.whetstone import inspect_store, reviewed_choices, revision, split_choices


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "whetstone.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript('''
            PRAGMA user_version=1;
            CREATE TABLE runs (id TEXT PRIMARY KEY, scope TEXT);
            CREATE TABLE consultations (id TEXT PRIMARY KEY, run_id TEXT, question TEXT, options TEXT, kind TEXT, gate TEXT);
            CREATE TABLE decisions (id TEXT PRIMARY KEY, consultation_id TEXT, run_id TEXT, choice TEXT, basis TEXT, review TEXT, correction TEXT);
            CREATE TABLE reviews (decision_id TEXT, verdict TEXT, statement TEXT);
            CREATE TABLE evidence (source TEXT, origin TEXT, status TEXT);
        ''')
    return path


def insert(path, key, *, verdict="accepted", correction=None, kind="implementation", gate="advisory", scope="/demo", retired=False):
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO runs VALUES (?,?)", (key, scope))
        db.execute("INSERT INTO consultations VALUES (?,?,?,?,?,?)", (
            "ask-" + key, key, "Choose storage for a private prototype " + key,
            json.dumps(["Local SQLite", "Hosted database"]), kind, gate))
        db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?)", (
            key, "ask-" + key, key, "Local SQLite", "prediction", verdict, correction))
        if verdict != "pending":
            db.execute("INSERT INTO reviews VALUES (?,?,?)", (key, verdict, correction or ""))
            db.execute("INSERT INTO evidence VALUES (?,?,?)", (
                "review:" + key, "endorsed", "retired" if retired else "active"))


def test_only_explicit_unambiguous_active_reviews_become_labels(store):
    insert(store, "accepted")
    insert(store, "corrected", verdict="corrected", correction="Hosted database")
    insert(store, "freeform", verdict="corrected", correction="It depends on who needs access")
    insert(store, "pending", verdict="pending")
    insert(store, "rejected", verdict="rejected")
    insert(store, "retired", retired=True)
    insert(store, "guard", kind="permission", gate="requires_user")
    with sqlite3.connect(store) as db:
        db.execute("INSERT INTO evidence VALUES ('a transcript', 'observed', 'active')")
    before = store.read_bytes()
    rows, report = reviewed_choices(store, scope="/demo")
    assert {row["id"] for row in rows} == {"accepted", "corrected"}
    assert next(r for r in rows if r["id"] == "corrected")["chosen"] == "Hosted database"
    assert report["observed_quotes_not_labels"] == 1
    assert report["skipped"]["needs_choice_mapping"] == 1
    assert store.read_bytes() == before


def test_empty_store_refuses_training_without_writing_artifact(store, tmp_path):
    destination = tmp_path / "model"
    report = fit_whetstone(store, scope="/demo", destination=destination)
    assert report["reason"] == "insufficient_reviewed_choices"
    assert not destination.exists()
    assert inspect_store(store)["eligible_choices"] == 0


def test_transitive_run_and_question_groups_never_cross_splits():
    rows = []
    for identity, run, question in [("1", "a", "same"), ("2", "b", "same"), ("3", "b", "different")]:
        rows.append({"id": identity, "run_id": run, "question": question,
                     "options": ["yes", "no"], "kind": "design", "chosen": "yes"})
    split = split_choices(rows)
    nonempty = [r for r in split.values() if r]
    assert len(nonempty) == 1 and len(nonempty[0]) == 2  # exact duplicate isn't extra support
    for row in rows:
        row["options"].reverse()
        row["chosen"] = "no"
    assert {k: [r["id"] for r in v] for k, v in split_choices(rows).items()} == {
        k: [r["id"] for r in v] for k, v in split.items()}


def test_ranker_learns_context_and_does_not_prefer_option_position():
    rows = []
    for i in range(30):
        for question, chosen in [("private offline prototype", "Local SQLite"),
                                 ("shared collaborative team server", "Hosted database")]:
            options = ["Local SQLite", "Hosted database"]
            if i % 2:
                options.reverse()
            rows.append({"question": question, "options": options, "chosen": chosen, "kind": "implementation"})
    model = PreferenceModel.train(rows, scope="/demo", dataset_revision="test")
    for question, expected in [("private offline prototype", "Local SQLite"),
                               ("shared collaborative team server", "Hosted database")]:
        options = ["Local SQLite", "Hosted database"]
        forward, margin = model.rank(question, options, "implementation")
        reverse, reverse_margin = model.rank(question, options[::-1], "implementation")
        assert forward[0][0] == expected
        assert forward == reverse
        assert margin == reverse_margin and margin > 0.1


def test_guard_scope_and_retirement_block_shadow_suggestions(store, tmp_path):
    insert(store, "reviewed")
    insert(store, "new", verdict="pending")
    insert(store, "permission", kind="permission", gate="requires_user", verdict="pending")
    insert(store, "other", scope="/other", verdict="pending")
    rows, _ = reviewed_choices(store, scope="/demo")
    model = PreferenceModel.train(rows, scope="/demo", dataset_revision=revision(rows))
    model.threshold = 0.05
    model.qualified = True  # fixture for testing runtime eligibility, not an evaluation claim
    output = tmp_path / "artifact"
    model.save(output, {"fixture": True})
    result = suggest(store, artifact=output, consultation_id="ask-new")
    assert result["choice"] == "Local SQLite"
    assert result["mode"] == "shadow_only" and not result["writes_whetstone"]
    assert suggest(store, artifact="/does/not/exist", consultation_id="ask-permission")["reason"] == "host_gate"
    assert suggest(store, artifact=output, consultation_id="ask-other")["reason"] == "scope_mismatch"
    with sqlite3.connect(store) as db:
        db.execute("UPDATE evidence SET status='retired'")
    assert suggest(store, artifact=output, consultation_id="ask-new")["reason"] == "review_data_changed"
    restored = PreferenceModel.load(output)
    np.testing.assert_array_equal(model.weights, restored.weights)


def test_test_reviews_never_enter_weights_or_calibration(store, tmp_path):
    for i in range(160):
        insert(store, f"fixture-{i}")
    first_path, second_path = tmp_path / "first", tmp_path / "second"
    first = fit_whetstone(store, scope="/demo", destination=first_path, target=0.6, min_support=5)
    assert first["ready"]
    rows, _ = reviewed_choices(store, scope="/demo")
    with sqlite3.connect(store) as db:
        for row in split_choices(rows)["test"]:
            db.execute("UPDATE decisions SET choice='Hosted database' WHERE id=?", (row["id"],))
    second = fit_whetstone(store, scope="/demo", destination=second_path, target=0.6, min_support=5)
    np.testing.assert_array_equal(PreferenceModel.load(first_path).weights, PreferenceModel.load(second_path).weights)
    assert first["threshold"] == second["threshold"]
    assert not second["ready"] and second["test"]["accepted_accuracy"] == 0

"""Read-only Whetstone v2 bridge. Observed quotes are not preference labels."""
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3

from .model import normalize

LEARNABLE_KINDS = {"implementation", "design", "dependency", "testing"}


@contextmanager
def open_store(path):
    path = Path(path).expanduser().resolve(strict=True)
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        if db.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise ValueError("expected Whetstone autopilot schema 1")
        yield db
    finally:
        db.close()


def decision_key(row):
    # Option order, source IDs, and the user's label cannot determine the split.
    value = [normalize(row["question"]), sorted(normalize(o) for o in row["options"]), row["kind"]]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def reviewed_choices(path, *, scope=None):
    rows, skipped = [], Counter()
    with open_store(path) as db:
        observed = db.execute("SELECT count(*) FROM evidence WHERE origin='observed' AND status='active'").fetchone()[0]
        records = db.execute('''
            SELECT d.id, d.choice, d.basis, d.review, d.correction, d.run_id,
                   c.question, c.options, c.kind, c.gate, r.scope,
                   v.verdict, v.statement, e.status AS evidence_status
            FROM decisions d JOIN consultations c ON c.id=d.consultation_id
            JOIN runs r ON r.id=d.run_id
            LEFT JOIN reviews v ON v.decision_id=d.id
            LEFT JOIN evidence e ON e.source='review:' || d.id AND e.origin='endorsed'
            WHERE (? IS NULL OR r.scope=?) ORDER BY d.id
        ''', (scope, scope)).fetchall()
        for raw in records:
            row = dict(raw)
            if row["gate"] != "advisory" or row["kind"] not in LEARNABLE_KINDS or row["basis"] == "escalated":
                skipped["authority_or_scope_decision"] += 1
                continue
            if row["review"] not in ("accepted", "corrected"):
                skipped["pending_or_rejected"] += 1
                continue
            if row["verdict"] != row["review"] or row["evidence_status"] != "active":
                skipped["missing_or_retired_review_evidence"] += 1
                continue
            options = json.loads(row["options"])
            if (not isinstance(options, list) or not 2 <= len(options) <= 8
                    or any(not isinstance(o, str) or not o.strip() for o in options)
                    or len({normalize(o) for o in options}) != len(options)):
                skipped["invalid_options"] += 1
                continue
            chosen = row["choice"] if row["review"] == "accepted" else row["statement"]
            if row["review"] == "corrected" and row["correction"] != chosen:
                skipped["inconsistent_review"] += 1
                continue
            # Freeform corrections need a human mapping. Never manufacture a winning option.
            matches = [o for o in options if normalize(o) == normalize(chosen)]
            if len(matches) != 1:
                skipped["needs_choice_mapping"] += 1
                continue
            rows.append({"id": row["id"], "run_id": row["run_id"], "scope": row["scope"],
                         "question": row["question"], "options": options,
                         "kind": row["kind"], "chosen": matches[0]})
    # Reject ambiguous source joins rather than count one decision multiple times.
    counts = Counter(r["id"] for r in rows)
    duplicate_ids = {key for key, count in counts.items() if count != 1}
    skipped["ambiguous_review_rows"] = sum(counts[key] for key in duplicate_ids)
    rows = [r for r in rows if r["id"] not in duplicate_ids]
    return rows, {"eligible_choices": len(rows), "observed_quotes_not_labels": observed,
                  "skipped": dict(skipped), "scopes": dict(Counter(r["scope"] for r in rows))}


def revision(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def split_choices(rows):
    """Keep entire runs AND repeated questions/options together, transitively."""
    parent = {}
    def root(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def union(a, b):
        ra, rb = root(a), root(b)
        parent[max(ra, rb)] = min(ra, rb)
    for row in rows:
        union("run:" + row["run_id"], "question:" + decision_key(row))
    splits = {name: [] for name in ("train", "calibration", "test")}
    seen = set()
    for row in rows:
        # Exact repeats with the same answer are one observation, not extra support.
        identity = (decision_key(row), normalize(row["chosen"]))
        if identity in seen:
            continue
        seen.add(identity)
        bucket = int(hashlib.sha256(root("run:" + row["run_id"]).encode()).hexdigest()[:8], 16) % 10
        split = "train" if bucket < 6 else "calibration" if bucket < 8 else "test"
        splits[split].append(row)
    return splits


def inspect_store(path, scope=None):
    rows, report = reviewed_choices(path, scope=scope)
    splits = split_choices(rows)
    return {**report, "splits": {k: len(v) for k, v in splits.items()},
            "dataset_revision": revision(rows),
            "note": "Only explicit, active reviews with an unambiguous chosen option are labels. No source writes."}


def consultation(path, consultation_id):
    with open_store(path) as db:
        row = db.execute('''SELECT c.*, r.scope FROM consultations c
                            JOIN runs r ON r.id=c.run_id WHERE c.id=?''', (consultation_id,)).fetchone()
        if row is None:
            raise ValueError("unknown Whetstone consultation")
        result = dict(row)
        result["options"] = json.loads(result["options"])
        return result

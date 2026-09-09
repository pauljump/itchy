"""Persistent supervision, independent evaluation, and explicit model promotion."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
import uuid

from .model import Predictor, calibrate, features, fingerprint, measure, normalize


@dataclass(frozen=True)
class Decision:
    id: str
    label: str | None
    route: str
    proposed_label: str | None
    score: float
    reason: str
    model_id: str | None


class Task:
    """One finite-label decision and its local feedback loop.

    teach() is explicit trusted supervision. decide() never teaches, even when a
    fallback returns a label. Use correct(decision.id, label) after review.
    """
    def __init__(self, path, *, labels=None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.path / "task.sqlite3"
        self._cached = None
        with self._db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS examples (
                    fingerprint TEXT PRIMARY KEY, text TEXT NOT NULL, label TEXT NOT NULL,
                    group_id TEXT NOT NULL, split TEXT NOT NULL, source TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY, text TEXT NOT NULL, label TEXT, route TEXT NOT NULL,
                    proposed_label TEXT, score REAL NOT NULL, reason TEXT NOT NULL,
                    model_id TEXT, created REAL NOT NULL, reviewed INTEGER NOT NULL DEFAULT 0);
            ''')
            stored = db.execute("SELECT value FROM config WHERE key='labels'").fetchone()
            if stored:
                self.labels = tuple(json.loads(stored[0]))
                if labels is not None and tuple(labels) != self.labels:
                    raise ValueError("labels are immutable; create a new task for a new taxonomy")
            else:
                if (labels is None or isinstance(labels, str) or len(labels) < 2
                        or any(not isinstance(x, str) or not x.strip() for x in labels)
                        or len(set(labels)) != len(labels)):
                    raise ValueError("provide at least two unique nonempty labels")
                self.labels = tuple(labels)
                db.execute("INSERT INTO config VALUES ('labels', ?)", (json.dumps(self.labels),))

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def _teach(self, db, text, label, group=None, split=None, source="reviewed"):
        if label not in self.labels:
            raise ValueError(f"label must be one of {self.labels}")
        normalized = normalize(text)
        features(text, 8192, 8192)  # validate exactly the model's input limit
        key = fingerprint(text)
        existing = db.execute("SELECT * FROM examples WHERE fingerprint=?", (key,)).fetchone()
        if existing:
            if group is not None and group != existing["group_id"]:
                raise ValueError("duplicate text cannot move groups")
            if split is not None and split != existing["split"]:
                raise ValueError("duplicate text cannot cross data splits")
            db.execute("UPDATE examples SET label=?, source=? WHERE fingerprint=?", (label, source, key))
            return existing["split"]
        group = group or key
        if not isinstance(group, str) or not group:
            raise ValueError("group must be a nonempty string")
        grouped = db.execute("SELECT split FROM examples WHERE group_id=? LIMIT 1", (group,)).fetchone()
        if grouped:
            if split is not None and split != grouped[0]:
                raise ValueError("related examples cannot cross data splits")
            split = grouped[0]
        elif split is None:
            bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
            split = "train" if bucket < 6 else "calibration" if bucket < 8 else "test"
        if split not in ("train", "calibration", "test"):
            raise ValueError("split must be train, calibration, or test")
        db.execute("INSERT INTO examples VALUES (?, ?, ?, ?, ?, ?)",
                   (key, normalized, label, group, split, source))
        return split

    def teach(self, text, label, *, group=None, split=None, source="reviewed"):
        with self._db() as db:
            return self._teach(db, text, label, group, split, source)

    def import_jsonl(self, path):
        """All-or-nothing import; wrong rows cannot leave half a dataset behind."""
        count = 0
        with self._db() as db, open(path, encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("each row must be a JSON object")
                    self._teach(db, row["text"], row["label"], row.get("group"),
                                row.get("split"), row.get("source", "imported"))
                except (KeyError, ValueError, TypeError) as exc:
                    raise ValueError(f"line {number}: {exc}") from exc
                count += 1
        return count

    def _examples(self):
        with self._db() as db:
            return [dict(row) for row in db.execute("SELECT * FROM examples ORDER BY fingerprint")]

    @staticmethod
    def _revision(rows):
        return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()

    def fit(self, *, target_precision=0.95, min_support=50, min_coverage=0.1,
            dimensions=8192, epochs=40):
        if not 0.5 < target_precision < 1 or min_support < 1 or not 0 < min_coverage <= 1:
            raise ValueError("invalid precision, support, or coverage policy")
        rows = self._examples()
        splits = {name: [r for r in rows if r["split"] == name]
                  for name in ("train", "calibration", "test")}
        for name, subset in splits.items():
            missing = set(self.labels) - {r["label"] for r in subset}
            if missing:
                raise ValueError(f"{name} needs examples of {sorted(missing)}")
        started = time.perf_counter()
        model = Predictor.train(splits["train"], list(self.labels), dimensions=dimensions, epochs=epochs)
        calibrate(model, splits["calibration"], target_precision, min_support)
        calibration = measure(model, splits["calibration"])
        audit = measure(model, splits["test"])
        reasons = []
        if not model.thresholds:
            reasons.append("No label qualified on calibration data.")
        for label in model.thresholds:
            counts = audit["per_label"][label]
            if counts["accepted"] < min_support or counts["wilson_lower_95"] < target_precision:
                reasons.append(f"{label}: independent test evidence fails precision/support policy.")
        if audit["coverage"] < min_coverage:
            reasons.append("Independent test coverage is below policy.")
        model_id = uuid.uuid4().hex
        artifact = self.path / "models" / model_id
        model.save(artifact)
        report = {
            "model_id": model_id, "dataset_revision": self._revision(rows),
            "learner": {"name": "byte-ngram-linear-softmax", "dimensions": dimensions,
                        "epochs": epochs, "seed": 0, "version": "0.1.0"},
            "policy": {"target_precision": target_precision, "min_support": min_support,
                       "min_coverage": min_coverage, "interval": "two-sided 95% Wilson lower bound"},
            "splits": {name: len(subset) for name, subset in splits.items()},
            "thresholds": model.thresholds, "calibration": calibration, "test": audit,
            "ready": not reasons, "reasons": reasons,
            "train_seconds": round(time.perf_counter() - started, 3),
            "artifact_bytes": sum(p.stat().st_size for p in artifact.iterdir()),
            "limitations": ["Precision intervals assume representative independent examples.",
                            "Threshold selection uses calibration only. Reusing test data across experiments is adaptive evaluation.",
                            "Byte familiarity is a heuristic, not a guarantee against distribution shift."]}
        (artifact / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    def _artifact(self, model_id):
        if not isinstance(model_id, str) or not re.fullmatch(r"[0-9a-f]{32}", model_id):
            raise ValueError("invalid model ID")
        return self.path / "models" / model_id

    def promote(self, model_id):
        artifact = self._artifact(model_id)
        report = json.loads((artifact / "report.json").read_text())
        if not report["ready"]:
            raise ValueError("candidate failed its evidence gate: " + " ".join(report["reasons"]))
        if report["dataset_revision"] != self._revision(self._examples()):
            raise ValueError("examples changed since evaluation; fit and review a new candidate")
        Predictor.load(artifact)  # validate before replacing a working champion
        temp = self.path / f".champion-{uuid.uuid4().hex}"
        temp.write_text(model_id + "\n")
        os.replace(temp, self.path / "champion")
        self._cached = None

    def _champion(self):
        pointer = self.path / "champion"
        if not pointer.exists():
            return None, None
        model_id = pointer.read_text().strip()
        if self._cached is None or self._cached[0] != model_id:
            self._cached = (model_id, Predictor.load(self._artifact(model_id)))
        return self._cached

    def decide(self, text, *, fallback=None):
        """Fallback is your callable, invoked once only when the local model abstains.

        Passing it authorizes execution in the host application. Itchy itself has no
        provider integrations. Exceptions propagate; they never become training data.
        """
        normalize(text)
        model_id, model = self._champion()
        prediction = model.predict(text) if model else None
        label = prediction.label if prediction else None
        reason = prediction.reason if prediction else "no_champion"
        route = "local" if label is not None else "abstain"
        if label is None and fallback is not None:
            label = fallback(text)
            if label not in self.labels:
                raise ValueError("fallback returned a label outside the task taxonomy")
            route = "fallback"
        decision = Decision(uuid.uuid4().hex, label, route,
                            prediction.proposed_label if prediction else None,
                            prediction.score if prediction else 0.0, reason, model_id)
        with self._db() as db:
            db.execute("INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                       (decision.id, text, label, route, decision.proposed_label,
                        decision.score, reason, model_id, time.time()))
        return decision

    def review_queue(self, *, limit=20, route=None):
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be 1..1000")
        if route not in (None, "local", "fallback", "abstain"):
            raise ValueError("route must be local, fallback, or abstain")
        with self._db() as db:
            return [dict(row) for row in db.execute('''
                SELECT * FROM decisions WHERE reviewed=0 AND (? IS NULL OR route=?)
                ORDER BY CASE route WHEN 'abstain' THEN 0 WHEN 'fallback' THEN 1 ELSE 2 END,
                         score ASC, created ASC LIMIT ?''', (route, route, limit))]

    def correct(self, decision_id, label, *, group=None):
        with self._db() as db:
            row = db.execute("SELECT * FROM decisions WHERE id=?", (decision_id,)).fetchone()
            if row is None:
                raise ValueError("unknown decision ID")
            split = self._teach(db, row["text"], label, group, source="human_review")
            db.execute("UPDATE decisions SET reviewed=1 WHERE id=?", (decision_id,))
            return split

    def status(self):
        model_id, _ = self._champion()
        with self._db() as db:
            examples = {r[0]: r[1] for r in db.execute("SELECT split, COUNT(*) FROM examples GROUP BY split")}
            routes = {r[0]: r[1] for r in db.execute("SELECT route, COUNT(*) FROM decisions GROUP BY route")}
            pending = db.execute("SELECT COUNT(*) FROM decisions WHERE reviewed=0").fetchone()[0]
        return {"labels": self.labels, "champion": model_id, "examples": examples,
                "decisions": routes, "pending_review": pending}

    def export(self, destination):
        model_id, model = self._champion()
        if model is None:
            raise ValueError("promote a qualified model before exporting")
        destination = Path(destination)
        model.save(destination)
        shutil.copyfile(self._artifact(model_id) / "report.json", destination / "report.json")
        return str(destination)

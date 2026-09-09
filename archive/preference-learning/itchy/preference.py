"""Experimental contextual choice ranker for Whetstone. Advisory output only."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import zlib

import numpy as np

from .model import features, normalize, wilson_lower
from .whetstone import LEARNABLE_KINDS, consultation, reviewed_choices, revision, split_choices


def encode(question, option, kind, dimensions=16384):
    option = normalize(option)
    question = normalize(question)
    if len(question.encode()) > 8192 or len(option.encode()) > 4096:
        raise ValueError("question or option exceeds preference model input limit")
    ids, values = features(option, dimensions, 4096)
    vector = np.zeros(dimensions, dtype=np.float32)
    vector[ids] = values
    # Explicit interactions: the same option can be preferred in one context and
    # rejected in another. A bag of option text alone cannot learn that distinction.
    qwords = sorted(set(re.findall(r"\w+", question)))[:64]
    owords = sorted(set(re.findall(r"\w+", option)))[:64]
    interaction = np.zeros(dimensions, dtype=np.float32)
    for qword in ["kind=" + kind] + qwords:
        for oword in owords:
            key = ("context\0" + qword + "\0" + oword).encode()
            interaction[zlib.crc32(key) % dimensions] += 1
    norm = np.linalg.norm(interaction)
    if norm:
        vector += interaction / norm
    return vector / np.linalg.norm(vector)


class PreferenceModel:
    def __init__(self, weights, *, scope, dataset_revision, threshold=None, qualified=False):
        self.weights = weights
        self.scope = scope
        self.dataset_revision = dataset_revision
        self.threshold = threshold
        self.qualified = qualified

    @classmethod
    def train(cls, rows, *, scope, dataset_revision, dimensions=16384, epochs=40):
        if not rows:
            raise ValueError("no reviewed training choices")
        weights = np.zeros(dimensions, dtype=np.float32)
        encoded = [(np.stack([encode(r["question"], o, r["kind"], dimensions) for o in r["options"]]),
                    r["options"].index(r["chosen"])) for r in rows]
        rng = np.random.default_rng(0)
        for epoch in range(epochs):
            rate = 0.5 / math.sqrt(1 + epoch / 5)
            for index in rng.permutation(len(encoded)):
                matrix, chosen = encoded[index]
                logits = matrix @ weights
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                probabilities[chosen] -= 1
                weights -= rate * (probabilities @ matrix + 0.0001 * weights)
        return cls(weights, scope=scope, dataset_revision=dataset_revision)

    def rank(self, question, options, kind):
        if (kind not in LEARNABLE_KINDS or not isinstance(options, list) or not 2 <= len(options) <= 8
                or len({normalize(o) for o in options}) != len(options)):
            raise ValueError("expected 2–8 distinct options and a learnable decision kind")
        matrix = np.stack([encode(question, o, kind, len(self.weights)) for o in options])
        logits = matrix @ self.weights
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        # Tie ordering is deterministic but ties cannot pass the positive margin gate.
        ranked = sorted(zip(options, map(float, probs)), key=lambda x: (-x[1], x[0]))
        return ranked, ranked[0][1] - ranked[1][1]

    def save(self, destination, report):
        path = Path(destination).expanduser()
        path.mkdir(parents=True, exist_ok=False, mode=0o700)
        np.savez_compressed(path / "preference.npz", weights=self.weights)
        meta = {"format": 1, "scope": self.scope, "dataset_revision": self.dataset_revision,
                "threshold": self.threshold, "qualified": self.qualified,
                "weights_sha256": hashlib.sha256((path / "preference.npz").read_bytes()).hexdigest()}
        (path / "preference.json").write_text(json.dumps(meta, indent=2) + "\n")
        (path / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        for file in path.iterdir():
            os.chmod(file, 0o600)

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser()
        meta = json.loads((path / "preference.json").read_text())
        if meta["format"] != 1:
            raise ValueError("unsupported preference artifact")
        if hashlib.sha256((path / "preference.npz").read_bytes()).hexdigest() != meta["weights_sha256"]:
            raise ValueError("preference artifact checksum mismatch")
        with np.load(path / "preference.npz", allow_pickle=False) as data:
            weights = data["weights"]
        if weights.ndim != 1 or not 256 <= len(weights) <= 262144 or not np.isfinite(weights).all():
            raise ValueError("invalid preference weights")
        return cls(weights, scope=meta["scope"], dataset_revision=meta["dataset_revision"],
                   threshold=meta["threshold"], qualified=meta["qualified"])


def evaluate(model, rows, threshold):
    accepted = correct = all_correct = 0
    for row in rows:
        ranking, margin = model.rank(row["question"], row["options"], row["kind"])
        matches = ranking[0][0] == row["chosen"]
        all_correct += matches
        if threshold is not None and margin >= threshold:
            accepted += 1
            correct += matches
    return {"total": len(rows), "accepted": accepted, "correct": correct,
            "coverage": accepted / len(rows) if rows else 0,
            "accepted_accuracy": correct / accepted if accepted else None,
            "all_choice_accuracy": all_correct / len(rows) if rows else None,
            "wilson_lower_95": wilson_lower(correct, accepted)}


def fit_whetstone(db_path, *, scope, destination, target=0.9, min_support=30):
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("an explicit project scope is required")
    if not 0.5 < target < 1 or not isinstance(min_support, int) or min_support < 1:
        raise ValueError("invalid preference evaluation policy")
    rows, inspection = reviewed_choices(db_path, scope=scope)
    splits = split_choices(rows)
    sizes = {k: len(v) for k, v in splits.items()}
    if sizes["train"] < 20 or sizes["calibration"] < min_support or sizes["test"] < min_support:
        return {"ready": False, "artifact_written": False, "inspection": inspection, "splits": sizes,
                "reason": "insufficient_reviewed_choices", "scope": scope}
    model = PreferenceModel.train(splits["train"], scope=scope, dataset_revision=revision(rows))
    for threshold in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.9):
        calibration = evaluate(model, splits["calibration"], threshold)
        if calibration["accepted"] >= min_support and calibration["wilson_lower_95"] >= target:
            model.threshold = threshold
            break
    audit = evaluate(model, splits["test"], model.threshold)
    model.qualified = (model.threshold is not None and audit["accepted"] >= min_support
                       and audit["wilson_lower_95"] >= target and audit["coverage"] >= 0.1)
    report = {"ready": model.qualified, "artifact_written": True, "scope": scope,
              "dataset_revision": model.dataset_revision, "splits": sizes,
              "policy": {"target_lower_bound": target, "min_support": min_support, "min_coverage": 0.1},
              "learner": "byte features + hashed question-option interactions, linear choice ranking",
              "calibration": evaluate(model, splits["calibration"], model.threshold), "test": audit,
              "threshold": model.threshold, "mode": "shadow_only",
              "limitations": "Lexical prototype; no validated personal-judgment accuracy. Intervals assume independent representative choices; within-run correlation can violate that assumption."}
    model.save(destination, report)
    return report


def suggest(db_path, *, artifact, consultation_id):
    query = consultation(db_path, consultation_id)
    # Host gates remain authoritative, even if a candidate model is highly confident.
    result = {"mode": "shadow_only", "choice": None, "consultation_id": consultation_id,
              "executes_actions": False, "writes_whetstone": False}
    if query["gate"] != "advisory" or query["kind"] not in LEARNABLE_KINDS:
        return {**result, "reason": "host_gate"}
    model = PreferenceModel.load(artifact)
    if model.scope != query["scope"]:
        return {**result, "reason": "scope_mismatch"}
    rows, _ = reviewed_choices(db_path, scope=query["scope"])
    if revision(rows) != model.dataset_revision:
        return {**result, "reason": "review_data_changed"}
    ranking, margin = model.rank(query["question"], query["options"], query["kind"])
    accepted = model.qualified and model.threshold is not None and margin >= model.threshold
    return {**result, "choice": ranking[0][0] if accepted else None,
            "reason": "shadow_suggestion" if accepted else "insufficient_evidence",
            "ranking": [{"option": option, "score": score} for option, score in ranking],
            "margin": margin, "score_is_probability_of_correctness": False}

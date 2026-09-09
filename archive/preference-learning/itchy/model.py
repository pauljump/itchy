"""A small byte n-gram linear classifier. No pretrained weights or tokenizer.

The softmax score is a ranking signal, NOT a probability of being correct.
Calibration and an independent audit determine which labels may be served.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import unicodedata
import zlib

import numpy as np


def normalize(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a nonempty string")
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def features(text: str, dimensions: int, max_bytes: int) -> tuple[np.ndarray, np.ndarray]:
    data = normalize(text).encode("utf-8")
    if len(data) > max_bytes:
        raise ValueError(f"text exceeds {max_bytes} UTF-8 bytes; split it explicitly")
    data = b"\x00" + data + b"\x00"
    hashes = [zlib.crc32(data[i:i + n]) % dimensions
              for n in (2, 3, 4) for i in range(len(data) - n + 1)]
    ids, counts = np.unique(hashes, return_counts=True)
    values = np.log1p(counts).astype(np.float32)
    values /= np.linalg.norm(values)
    return ids, values


def wilson_lower(correct: int, total: int) -> float:
    """Lower end of a two-sided 95% Wilson interval, descriptive on IID samples."""
    if total == 0:
        return 0.0
    z = 1.959963984540054
    p = correct / total
    return (p + z*z/(2*total) - z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1+z*z/total)


@dataclass(frozen=True)
class Prediction:
    label: str | None
    proposed_label: str | None
    score: float
    familiarity: float
    reason: str


class Predictor:
    def __init__(self, labels, weights, bias, seen, *, max_bytes=8192,
                 min_familiarity=0.8, thresholds=None):
        self.labels = tuple(labels)
        self.weights = weights
        self.bias = bias
        self.seen = seen
        self.max_bytes = max_bytes
        self.min_familiarity = min_familiarity
        self.thresholds = thresholds or {}

    @classmethod
    def train(cls, examples, labels, *, dimensions=8192, epochs=40, seed=0):
        if not 256 <= dimensions <= 262144 or not 1 <= epochs <= 1000:
            raise ValueError("dimensions must be 256..262144 and epochs 1..1000")
        weights = np.zeros((len(labels), dimensions), dtype=np.float32)
        bias = np.zeros(len(labels), dtype=np.float32)
        seen = np.zeros(dimensions, dtype=bool)
        encoded = []
        for row in examples:
            ids, values = features(row["text"], dimensions, 8192)
            seen[ids] = True
            encoded.append((ids, values, labels.index(row["label"])))
        rng = np.random.default_rng(seed)
        for epoch in range(epochs):
            rate = 0.5 / math.sqrt(1 + epoch / 5)
            for index in rng.permutation(len(encoded)):
                ids, values, target = encoded[index]
                logits = weights[:, ids] @ values + bias
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                probs[target] -= 1
                weights[:, ids] -= rate * (probs[:, None] * values + 0.0001 * weights[:, ids])
                bias -= rate * probs
        return cls(labels, weights, bias, seen)

    def raw(self, text):
        ids, values = features(text, self.weights.shape[1], self.max_bytes)
        logits = self.weights[:, ids] @ values + self.bias
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        best = int(probs.argmax())
        familiarity = float(np.sum(values[self.seen[ids]] ** 2))
        return self.labels[best], float(probs[best]), min(1.0, familiarity)

    def predict(self, text: str) -> Prediction:
        # Invalid input must not be mistaken for a plausible prediction.
        normalize(text)
        if len(normalize(text).encode("utf-8")) > self.max_bytes:
            return Prediction(None, None, 0.0, 0.0, "input_too_long")
        label, score, familiarity = self.raw(text)
        threshold = self.thresholds.get(label)
        if familiarity < self.min_familiarity:
            reason = "unfamiliar_bytes"
        elif threshold is None:
            reason = "label_not_qualified"
        elif score < threshold:
            reason = "below_threshold"
        else:
            return Prediction(label, label, score, familiarity, "qualified")
        return Prediction(None, label, score, familiarity, reason)

    def save(self, path):
        """Write into a NEW directory. NPZ uses numeric arrays; never pickle."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(path / "weights.npz", weights=self.weights, bias=self.bias, seen=self.seen)
        metadata = {"format": 1, "labels": self.labels, "max_bytes": self.max_bytes,
                    "min_familiarity": self.min_familiarity, "thresholds": self.thresholds,
                    "weights_sha256": hashlib.sha256((path / "weights.npz").read_bytes()).hexdigest()}
        (path / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads((path / "model.json").read_text())
        if meta["format"] != 1:
            raise ValueError("unsupported artifact format")
        if hashlib.sha256((path / "weights.npz").read_bytes()).hexdigest() != meta["weights_sha256"]:
            raise ValueError("artifact checksum mismatch")
        with np.load(path / "weights.npz", allow_pickle=False) as arrays:
            weights, bias, seen = arrays["weights"], arrays["bias"], arrays["seen"]
        labels = meta["labels"]
        if (weights.ndim != 2 or weights.shape[0] != len(labels) or len(labels) < 2
                or weights.shape[1] > 262144 or bias.shape != (len(labels),)
                or seen.shape != (weights.shape[1],) or seen.dtype != bool
                or not np.isfinite(weights).all() or not np.isfinite(bias).all()):
            raise ValueError("invalid model arrays")
        return cls(labels, weights, bias, seen, max_bytes=meta["max_bytes"],
                   min_familiarity=meta["min_familiarity"], thresholds=meta["thresholds"])


def measure(model, rows):
    per_label = {label: {"accepted": 0, "correct": 0} for label in model.labels}
    confusion = {label: {other: 0 for other in model.labels} for label in model.labels}
    full_correct = 0
    for row in rows:
        prediction = model.predict(row["text"])
        if prediction.proposed_label is not None:
            confusion[row["label"]][prediction.proposed_label] += 1
        full_correct += prediction.proposed_label == row["label"]
        if prediction.label is not None:
            counts = per_label[prediction.label]
            counts["accepted"] += 1
            counts["correct"] += prediction.label == row["label"]
    accepted = sum(c["accepted"] for c in per_label.values())
    correct = sum(c["correct"] for c in per_label.values())
    for counts in per_label.values():
        n = counts["accepted"]
        counts["precision"] = counts["correct"] / n if n else None
        counts["wilson_lower_95"] = wilson_lower(counts["correct"], n)
    return {"total": len(rows), "accepted": accepted, "correct": correct,
            "coverage": accepted / len(rows) if rows else 0.0,
            "precision": correct / accepted if accepted else None,
            "wilson_lower_95": wilson_lower(correct, accepted),
            "ungated_accuracy": full_correct / len(rows) if rows else None,
            "per_label": per_label, "confusion": confusion}


def calibrate(model, rows, target_precision, min_support):
    """Select per-label thresholds ONLY on calibration rows, never test rows."""
    scored = [(model.raw(row["text"]), row["label"]) for row in rows]
    for label in model.labels:
        matching = [(score, truth == label) for (pred, score, familiarity), truth in scored
                    if pred == label and familiarity >= model.min_familiarity]
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]
        for threshold in thresholds:
            selected = [correct for score, correct in matching if score >= threshold]
            if len(selected) >= min_support and wilson_lower(sum(selected), len(selected)) >= target_precision:
                model.thresholds[label] = threshold
                break

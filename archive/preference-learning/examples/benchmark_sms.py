"""Reproducible CPU check on a local copy of UCI's SMS Spam Collection.

No automatic downloads. No task records or raw SMS texts are written into the repo.
This retrospective dataset is not a production/LLM comparison or a modern spam test.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import re
import tempfile
import time

import numpy as np

from itchy import Task
from itchy.model import fingerprint, normalize


def prepare(path):
    unique, conflicts = {}, set()
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line in lines:
        label, text = line.split("\t", 1)
        if label not in ("ham", "spam"):
            raise ValueError("expected UCI labels ham/spam")
        key = fingerprint(text)
        if key in unique and unique[key]["label"] != label:
            conflicts.add(key)
        # Related digit variants stay together; this is NOT full near-deduplication.
        group = re.sub(r"\d+", "#", normalize(text))
        unique[key] = {"text": text, "label": label,
                       "group": hashlib.sha256(group.encode()).hexdigest(), "source": "uci_sms"}
    return [row for key, row in sorted(unique.items()) if key not in conflicts], {
        "source_rows": len(lines), "normalized_unique": len(unique),
        "conflicting_texts_excluded": len(conflicts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="path to the extracted SMSSpamCollection file")
    parser.add_argument("--output", help="optional aggregate JSON report; contains no SMS text")
    args = parser.parse_args()
    rows, counts = prepare(args.data)
    with tempfile.TemporaryDirectory(prefix="itchy-sms-") as directory:
        task = Task(Path(directory) / "sms", labels=["ham", "spam"])
        seed = Path(directory) / "seed.jsonl"
        seed.write_text("".join(json.dumps(row) + "\n" for row in rows))
        task.import_jsonl(seed)
        # Fixed in advance. Do not tune after looking at this test split.
        report = task.fit(target_precision=0.95, min_support=50)
        all_rows = task._examples()
        train = [r for r in all_rows if r["split"] == "train"]
        test = [r for r in all_rows if r["split"] == "test"]
        majority = Counter(r["label"] for r in train).most_common(1)[0][0]
        from itchy import Predictor
        predictor = Predictor.load(task._artifact(report["model_id"]))
        elapsed = []
        for row in test:
            start = time.perf_counter()
            predictor.predict(row["text"])
            elapsed.append((time.perf_counter() - start) * 1000)
        report.update({
            "benchmark": "UCI SMS Spam Collection — retrospective, one fixed split",
            "source": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
            "citation": "Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection. DOI: 10.24432/C5CC84",
            "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            "preparation": counts,
            "grouping": "exact normalized deduplication; digit-normalized variants share a group; other near-duplicates may remain",
            "split_labels": {split: dict(Counter(r["label"] for r in all_rows if r["split"] == split))
                             for split in ("train", "calibration", "test")},
            "majority_baseline": {"label": majority, "coverage": 1.0,
                                  "accuracy": sum(r["label"] == majority for r in test) / len(test)},
            "inference_ms": {"median": float(np.median(elapsed)), "p95": float(np.percentile(elapsed, 95)),
                             "calls": len(elapsed), "includes_sqlite_logging": False},
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "numpy": np.__version__},
        })
        # Machine-specific UUID is not useful in the portable benchmark report.
        report.pop("model_id")
        output = json.dumps(report, indent=2) + "\n"
        if args.output:
            Path(args.output).write_text(output)
        print(output)


if __name__ == "__main__":
    main()

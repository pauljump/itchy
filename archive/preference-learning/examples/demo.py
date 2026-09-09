"""Offline, synthetic plumbing demonstration — not a generalization benchmark.

Run from the repository: python -m examples.demo
The vocabulary is deliberately easy. Templates are shared across splits; do not
mistake this fixture for independent real-world customer conversations.
"""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time

from itchy import Predictor, Task


def demo_rows():
    phrases = {
        "billing": ["charged twice for my subscription", "refund the duplicate payment",
                    "invoice has the wrong amount", "billing receipt shows an extra charge"],
        "access": ["password reset link is expired", "cannot sign in to my account",
                   "locked out after login attempt", "two factor authentication code is missing"],
        "bug": ["application crashes when uploading", "screen freezes after saving a file",
                "export button returns an error", "dashboard is blank after the update"],
    }
    for split, prefix in [("train", "Hello"), ("calibration", "Please help"), ("test", "Can you check")]:
        for label, variants in phrases.items():
            for i in range(100):
                yield {"text": f"{prefix}, {variants[i % len(variants)]}. Reference {split}-{i:03d}.",
                       "label": label, "split": split, "group": f"{split}-{label}-{i}"}


def main():
    with tempfile.TemporaryDirectory(prefix="itchy-demo-") as directory:
        task = Task(Path(directory) / "support", labels=["billing", "access", "bug"])
        source = Path(directory) / "examples.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in demo_rows()))
        task.import_jsonl(source)
        report = task.fit(target_precision=0.9, min_support=30)
        print("SYNTHETIC DEMO — shared templates, not a production benchmark")
        print(json.dumps({k: report[k] for k in ("ready", "splits", "artifact_bytes", "train_seconds")}, indent=2))
        print(json.dumps(report["test"], indent=2))
        if not report["ready"]:
            raise RuntimeError(report["reasons"])
        task.promote(report["model_id"])
        for text in ["Hello, charged twice for my subscription.",
                     "Please help, password reset link is expired.",
                     "Dashboard is blank after the update.",
                     "紫の宇宙船 🪐 мир"]:
            print(json.dumps(asdict(task.decide(text)), ensure_ascii=False))
        # Caller-owned fallback is a deterministic fixture, never an API call.
        fallback = task.decide("Where do I send my tax paperwork?", fallback=lambda _: "billing")
        print("Fallback:", json.dumps(asdict(fallback)))
        task.correct(fallback.id, "billing")
        print("Review recorded:", json.dumps(task.status()))
        exported = task.export(Path(directory) / "portable")
        standalone = Predictor.load(exported)
        elapsed = []
        for _ in range(100):
            start = time.perf_counter()
            standalone.predict("I was charged twice for my subscription.")
            elapsed.append((time.perf_counter() - start) * 1000)
        print(f"Standalone median inference: {sorted(elapsed)[50]:.3f} ms (100 local runs)")
        print("Portable artifact loads without the examples database. Temporary demo data is removed on exit.")


if __name__ == "__main__":
    main()

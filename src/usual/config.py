"""config.py — load usual.config.json and apply defaults per spec §2.7."""
import json
import os
from typing import Any


DEFAULTS: dict[str, Any] = {
    "user_name": "the user",
    "transcripts_glob": "~/.claude/projects/**/*.jsonl",
    "mining_dir": "mining",
    "corpus_path": "study-corpus.jsonl",
    "exam_dir": "exam",
    "domains": ["product", "design", "people", "money", "factory", "voice", "other"],
    "min_confidence": 0.5,
    "holdout_size": 20,
    "holdout_strategy": "random-seeded",
    "seed": 42,
    "redaction": {
        "regex": True,
        "llm_scan": True,
        "redact_emails": True,
        "redact_long_numbers": True,
    },
    "manifest_overrides": {
        "drops": [],
        "scrubs": [],
        "holdout": [],
        "keeps": [],
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str | None = None) -> dict[str, Any]:
    """
    Load usual.config.json from *path* (file path or directory containing it).
    Falls back to DEFAULTS for any missing key.  Returns a fully-merged config dict.
    """
    cfg: dict[str, Any] = {}

    if path is not None:
        # Accept a directory containing usual.config.json or a direct file path.
        if os.path.isdir(path):
            candidate = os.path.join(path, "usual.config.json")
        else:
            candidate = path

        if os.path.exists(candidate):
            with open(candidate) as fh:
                cfg = json.load(fh)
        else:
            raise FileNotFoundError(f"usual config not found: {candidate}")

    merged = _deep_merge(DEFAULTS, cfg)
    return merged

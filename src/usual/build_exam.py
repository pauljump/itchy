"""build_exam.py — config-driven exam builder (generalized from brain/usual/build-exam.py).

Pipeline:
  1. Load mining JSONL from config.mining_dir, assigning ids w<wave>b<batch>:<lineno>.
  2. Redact (regex via redact.redact_entries, then redact.assert_clean — ABORT on residual).
  3. Curate (curate.curate — drops/scrubs with reasons).
  4. Split holdout per config (holdout_strategy: random-seeded or manual id list).
  5. Write:
       study-corpus.jsonl          — kept study entries (full)
       exam/dayN-items.json        — holdout items: id/situation/domain/date ONLY
       exam/dayN-answers.json      — holdout entries (full)
       curation-manifest.json      — drops/scrubs/holdout/counts + reasons
  6. Assert items file contains NO `call` and NO `reasons` keys.
  7. Print: mined=… dropped=… scrubbed=… holdout=… study=…

Regression proof: feeding the reference user's wave-1 mining dir with manifest_overrides carrying
the exact DROPS/SCRUB/HOLDOUT from the original script reproduces:
  mined=167 dropped=11 scrubbed=1 holdout=20 study=135
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from typing import Any

from .config import load_config
from .redact import redact_entries, assert_clean
from .curate import curate


# ---------------------------------------------------------------------------
# ID assignment
# ---------------------------------------------------------------------------

def _wave_batch_from_filename(filename: str) -> tuple[int, int] | None:
    """Extract (wave, batch) from a mining filename like wave1-batch-2.jsonl."""
    m = re.search(r"wave(\d+)-batch-(\d+)", filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def load_mining_entries(mining_dir: str) -> dict[str, dict[str, Any]]:
    """
    Load all wave*/batch-* JSONL files from *mining_dir*.
    Assigns id `w<wave>b<batch>:<lineno>` to every entry (1-indexed line numbers).
    Returns an ordered dict {id: entry}.
    """
    entries: dict[str, dict[str, Any]] = {}

    if not os.path.isdir(mining_dir):
        raise FileNotFoundError(f"mining_dir not found: {mining_dir}")

    # Sort files so the order is deterministic: wave1-batch-1 < wave1-batch-2 < …
    filenames = sorted(
        f for f in os.listdir(mining_dir) if f.endswith(".jsonl")
    )

    for filename in filenames:
        wb = _wave_batch_from_filename(filename)
        if wb is None:
            continue  # skip non-wave files
        wave, batch = wb
        filepath = os.path.join(mining_dir, filename)
        with open(filepath) as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                entry = json.loads(raw)
                eid = f"w{wave}b{batch}:{lineno}"
                entry["id"] = eid
                entries[eid] = entry

    return entries


# ---------------------------------------------------------------------------
# Holdout splitting
# ---------------------------------------------------------------------------

def split_holdout(
    candidate_ids: list[str],
    config: dict[str, Any],
) -> list[str]:
    """
    Select holdout ids from *candidate_ids* (entries not dropped/scrubbed).

    Strategies:
      - "manual": use config.manifest_overrides.holdout list directly.
      - "random-seeded": random sample of config.holdout_size with config.seed.

    Validates that the manual/forced holdout ids are in candidate_ids.
    """
    strategy: str = config.get("holdout_strategy", "random-seeded")
    holdout_size: int = int(config.get("holdout_size", 20))
    overrides = config.get("manifest_overrides", {})
    forced_holdout: list[str] = overrides.get("holdout", [])

    if strategy == "manual" or forced_holdout:
        # Validate forced ids exist in candidates.
        missing = [i for i in forced_holdout if i not in set(candidate_ids)]
        assert not missing, (
            f"manifest_overrides.holdout contains ids not in curated candidates: {missing}"
        )
        assert len(set(forced_holdout)) == len(forced_holdout), (
            "manifest_overrides.holdout has duplicate ids"
        )
        if strategy == "manual":
            # Use the forced list exactly.
            return list(forced_holdout)
        # If strategy is random-seeded but a forced holdout is provided,
        # treat forced list as the holdout (override wins).
        return list(forced_holdout)

    # random-seeded
    assert holdout_size <= len(candidate_ids), (
        f"holdout_size={holdout_size} exceeds candidate pool size={len(candidate_ids)}"
    )
    rng = random.Random(config.get("seed", 42))
    pool = list(candidate_ids)
    rng.shuffle(pool)
    return pool[:holdout_size]


# ---------------------------------------------------------------------------
# Items-file anti-leak assert
# ---------------------------------------------------------------------------

def assert_items_answer_free(items: list[dict[str, Any]]) -> None:
    """ABORT if any item in the items file contains 'call' or 'reasons'."""
    violations = []
    for item in items:
        if "call" in item:
            violations.append(f"  item {item.get('id')!r} contains 'call' field")
        if "reasons" in item:
            violations.append(f"  item {item.get('id')!r} contains 'reasons' field")
    if violations:
        raise AssertionError(
            "ABORT: items file would leak answers:\n" + "\n".join(violations)
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_exam(
    config: dict[str, Any],
    base_dir: str,
    *,
    day: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    """
    Run the full exam-build pipeline.

    Args:
        config:   loaded usual config dict.
        base_dir: root directory for output files (corpus_path, exam_dir,
                  curation-manifest.json are resolved relative to this).
        day:      exam day number (for file naming: dayN-items.json etc.).
        force:    if True, skip the assert_clean abort and just warn (NOT recommended).

    Returns a summary dict with counts.
    """
    mining_dir = os.path.join(base_dir, config.get("mining_dir", "mining"))
    corpus_path = os.path.join(base_dir, config.get("corpus_path", "study-corpus.jsonl"))
    exam_dir = os.path.join(base_dir, config.get("exam_dir", "exam"))
    os.makedirs(exam_dir, exist_ok=True)

    # --- Step 1: Load ---
    entries_map = load_mining_entries(mining_dir)
    all_entries = list(entries_map.values())
    total_mined = len(all_entries)

    # Validate that all manifest override ids exist.
    overrides = config.get("manifest_overrides", {})
    all_override_ids = (
        list(overrides.get("drops", []))
        + list(overrides.get("scrubs", []))
        + list(overrides.get("holdout", []))
        + list(overrides.get("keeps", []))
    )
    missing = [i for i in all_override_ids if i not in entries_map]
    assert not missing, f"manifest_overrides ids not found in mining data: {missing}"

    # Holdout list must have exactly holdout_size unique entries (when manual).
    forced_holdout = overrides.get("holdout", [])
    if forced_holdout:
        assert len(set(forced_holdout)) == len(forced_holdout), (
            "manifest_overrides.holdout has duplicate ids"
        )
        expected_holdout_size = config.get("holdout_size", 20)
        assert len(forced_holdout) == expected_holdout_size, (
            f"manifest_overrides.holdout has {len(forced_holdout)} entries, "
            f"expected holdout_size={expected_holdout_size}"
        )

    # --- Step 2: Redact (MANDATORY, ABORT on residual) ---
    redact_cfg = config.get("redaction", {})
    redact_entries(all_entries, {"redaction": redact_cfg})

    try:
        assert_clean(all_entries)
    except RuntimeError as exc:
        if force:
            print(
                f"\n*** WARNING (--force): {exc}\n"
                "Continuing despite residual secrets. This is NOT recommended.\n",
                file=sys.stderr,
            )
        else:
            raise

    # --- Step 3: Curate (drops/scrubs) ---
    # NOTE: curate() operates on ALL entries (including future holdout candidates).
    # Holdout entries must not be dropped/scrubbed — the manifest_overrides.holdout
    # is protected by Step 1 validation (ids must exist) and the holdout ids are
    # NOT in force_drops/force_scrubs (caller's responsibility to not overlap).
    curation = curate(all_entries, config)

    dropped_count = len(curation.dropped)
    scrubbed_count = len(curation.scrubbed)
    kept_ids = {e["id"] for e in curation.kept}

    # --- Step 4: Split holdout ---
    holdout_ids_list = split_holdout(list(kept_ids), config)
    holdout_ids_set = set(holdout_ids_list)

    study_entries = [e for e in curation.kept if e["id"] not in holdout_ids_set]
    holdout_entries = [entries_map[eid] for eid in holdout_ids_list]

    # --- Step 5: Build items (answer-free) ---
    items = [
        {
            "id": e["id"],
            "situation": e["situation"],
            "domain": e["domain"],
            "date": e["date"],
        }
        for e in holdout_entries
    ]

    # --- Assert items file is answer-free (ABORT if violated) ---
    assert_items_answer_free(items)

    # --- Step 6: Write files ---
    with open(corpus_path, "w") as fh:
        for e in study_entries:
            fh.write(json.dumps(e) + "\n")

    items_path = os.path.join(exam_dir, f"day{day}-items.json")
    with open(items_path, "w") as fh:
        json.dump(items, fh, indent=1)

    answers_path = os.path.join(exam_dir, f"day{day}-answers.json")
    with open(answers_path, "w") as fh:
        json.dump(holdout_entries, fh, indent=1)

    # Curation manifest.
    manifest = {
        "wave": 1,  # TODO: derive from config when multi-wave support lands
        "date": _today(),
        "total_mined": total_mined,
        "drops": sorted(d["id"] for d in curation.dropped),
        "scrubs": sorted(s["id"] for s in curation.scrubbed),
        "holdout": holdout_ids_list,
        "study_count": len(study_entries),
        "drop_reasons": {d["id"]: d["reason"] for d in curation.dropped},
        "scrub_reasons": {s["id"]: s["reason"] for s in curation.scrubbed},
    }
    manifest_path = os.path.join(base_dir, "curation-manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=1)

    summary = {
        "mined": total_mined,
        "dropped": dropped_count,
        "scrubbed": scrubbed_count,
        "holdout": len(holdout_ids_list),
        "study": len(study_entries),
    }
    return summary


def _today() -> str:
    import datetime
    return datetime.date.today().isoformat()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Usual exam builder — generalized, config-driven."
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=".",
        help="Base directory (contains mining/, usual.config.json, etc.).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to usual.config.json (default: <base_dir>/usual.config.json).",
    )
    parser.add_argument(
        "--day",
        type=int,
        default=0,
        help="Exam day number for output file naming (default: 0).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Continue even if residual secrets are detected (NOT recommended).",
    )
    args = parser.parse_args()

    config_path = args.config or os.path.join(args.base_dir, "usual.config.json")
    if os.path.exists(config_path):
        cfg = load_config(config_path)
    else:
        # No config file — use pure defaults.
        from .config import DEFAULTS
        cfg = dict(DEFAULTS)

    summary = build_exam(cfg, args.base_dir, day=args.day, force=args.force)
    print(
        f"mined={summary['mined']} "
        f"dropped={summary['dropped']} "
        f"scrubbed={summary['scrubbed']} "
        f"holdout={summary['holdout']} "
        f"study={summary['study']}"
    )


if __name__ == "__main__":
    _main()

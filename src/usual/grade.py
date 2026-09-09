"""grade.py — compute call/reason percentages from judge per-item scores per arm.

Spec §2.5:
  call_pct   = mean(call_match) / 2 * 100      (over all items)
  reason_pct = mean(reason_match) / 2 * 100    (over NON-NULL reason_match only)

TRAP: reason_match can be null when ground-truth reasons is empty. Those items
are EXCLUDED from the reason denominator. See w1b2:7 and w1b2:12 in day0-grades.json.

Regression test: feeding the K and M columns from brain/usual/exam/day0-grades.json
must reproduce K=75.0/66.67 and M=62.5/52.78 exactly.

Output shape (same as proven day0-grades.json):
{
  "<arm_name>": [ {id, call_match, reason_match, note}, ... ],
  ...
  "totals": {
    "<arm_name>": { "call_pct": float, "reason_pct": float },
    ...
  }
}

CLI usage:
  python3 -m usual.grade pred-arm-A.json pred-arm-B.json --names K M
  or feed a day0-grades.json with arms already named (for re-grade / regression):
  python3 -m usual.grade day0-grades.json
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_arm_scores(items: list[dict[str, Any]]) -> dict[str, float]:
    """
    Given a list of per-item judge scores for one arm, return:
      {"call_pct": float, "reason_pct": float}

    call_match is always an integer 0-2.
    reason_match is an integer 0-2 OR null/None (excluded from denominator).
    """
    if not items:
        return {"call_pct": 0.0, "reason_pct": 0.0}

    call_sum = sum(item["call_match"] for item in items)
    call_count = len(items)
    call_pct = round(call_sum / (call_count * 2) * 100, 2)

    reason_vals = [
        item["reason_match"]
        for item in items
        if item.get("reason_match") is not None
    ]
    if reason_vals:
        reason_sum = sum(reason_vals)
        reason_pct = round(reason_sum / (len(reason_vals) * 2) * 100, 2)
    else:
        reason_pct = 0.0

    return {"call_pct": call_pct, "reason_pct": reason_pct}


def grade_arms(
    arms: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """
    Given a dict {arm_name: [per-item scores]}, compute totals per arm.
    Returns a dict with arm names as keys + a "totals" key.
    """
    result: dict[str, Any] = {}
    totals_map: dict[str, dict[str, float]] = {}

    for arm_name, items in arms.items():
        result[arm_name] = items
        totals_map[arm_name] = compute_arm_scores(items)

    result["totals"] = totals_map
    return result


def totals(
    arms: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    """
    Convenience wrapper around grade_arms — returns only the totals sub-dict.
    Accepts the same {arm_name: [per-item scores]} shape as grade_arms.

    Usage (regression proof):
        g = json.load(open("day0-grades.json"))
        t = grade.totals({"K": g["K"], "M": g["M"]})
        assert t["K"]["call_pct"] == 75.0
    """
    return grade_arms(arms)["totals"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_arm_file(path: str) -> list[dict[str, Any]]:
    """Load a per-arm prediction/score JSON file (list of per-item dicts)."""
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected a JSON array in {path}, got {type(data)}")


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Grade usual exam arms. "
            "Pass one or more per-arm score files, or a combined grades JSON."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "One or more per-arm score JSON files (lists of {id, call_match, reason_match, note}), "
            "OR a single combined grades JSON with named arm keys."
        ),
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Names for each arm file (positional, must match --inputs count).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for grades JSON (default: print to stdout).",
    )
    parser.add_argument(
        "--exam-dir",
        default=None,
        help=(
            "If set, write output to <exam-dir>/day<N>-grades.json "
            "(requires --day to set N)."
        ),
    )
    parser.add_argument(
        "--day",
        type=int,
        default=0,
        help="Exam day number for output filename (used with --exam-dir).",
    )
    args = parser.parse_args()

    # Detect if a single combined-grades file was passed.
    if len(args.inputs) == 1:
        with open(args.inputs[0]) as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "totals" not in data:
            # Single arm file.
            name = args.names[0] if args.names else "arm"
            arms = {name: data}
        elif isinstance(data, dict):
            # Could be a combined grades file (has named arm keys).
            # Re-grade it.
            arm_names = [k for k in data if k != "totals"]
            arms = {k: data[k] for k in arm_names}
        else:
            arms = {"arm": data}
    else:
        names = args.names or [f"arm-{i}" for i in range(len(args.inputs))]
        assert len(names) == len(args.inputs), (
            f"--names count ({len(names)}) must match inputs count ({len(args.inputs)})"
        )
        arms = {name: _load_arm_file(path) for name, path in zip(names, args.inputs)}

    grades = grade_arms(arms)

    # Determine output path.
    out_path = args.out
    if out_path is None and args.exam_dir:
        os.makedirs(args.exam_dir, exist_ok=True)
        out_path = os.path.join(args.exam_dir, f"day{args.day}-grades.json")

    output = json.dumps(grades, indent=2)

    if out_path:
        with open(out_path, "w") as fh:
            fh.write(output + "\n")
        # Print summary to stdout.
        for arm_name, scores in grades["totals"].items():
            print(
                f"{arm_name}: call={scores['call_pct']}%  reason={scores['reason_pct']}%"
            )
    else:
        print(output)


if __name__ == "__main__":
    _main()

"""The offline CLI. All machine-readable output is JSON."""
import argparse
from dataclasses import asdict
import json
import sqlite3
import sys

from .task import Task


def main(argv=None):
    parser = argparse.ArgumentParser(description="Itchy — teach tiny local models your repeated decisions")
    parser.add_argument("--task", default=".itchy", help="local task directory (default: .itchy)")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="create a finite-label task")
    init.add_argument("--labels", nargs="+", required=True)
    imp = sub.add_parser("import", help="import reviewed JSONL examples transactionally")
    imp.add_argument("file")
    teach = sub.add_parser("teach", help="add or correct trusted supervision")
    teach.add_argument("text")
    teach.add_argument("label")
    teach.add_argument("--group")
    teach.add_argument("--split", choices=["train", "calibration", "test"])
    fit = sub.add_parser("fit", help="train a candidate and audit it; does not promote")
    fit.add_argument("--precision", type=float, default=0.95)
    fit.add_argument("--min-support", type=int, default=50)
    fit.add_argument("--min-coverage", type=float, default=0.1)
    fit.add_argument("--dimensions", type=int, default=8192)
    fit.add_argument("--epochs", type=int, default=40)
    promote = sub.add_parser("promote", help="activate a candidate that passed its gate")
    promote.add_argument("model_id")
    predict = sub.add_parser("predict", help="decide locally or abstain; never calls a provider")
    predict.add_argument("text")
    review = sub.add_parser("review", help="show pending decisions, including local answers")
    review.add_argument("--limit", type=int, default=20)
    review.add_argument("--route", choices=["local", "fallback", "abstain"])
    correct = sub.add_parser("correct", help="review a logged decision and supply its true label")
    correct.add_argument("decision_id")
    correct.add_argument("label")
    correct.add_argument("--group")
    sub.add_parser("status", help="show example and routing counts")
    export = sub.add_parser("export", help="export champion weights and report without raw examples")
    export.add_argument("destination")
    ws = sub.add_parser("whetstone", help="learn reviewed Whetstone choices; read-only, shadow mode")
    ws.add_argument("--db", default="~/.whetstone/judgment.sqlite3")
    wsub = ws.add_subparsers(dest="ws_command", required=True)
    inspect = wsub.add_parser("inspect")
    inspect.add_argument("--scope")
    fit_ws = wsub.add_parser("fit")
    fit_ws.add_argument("--scope", required=True)
    fit_ws.add_argument("--output", required=True)
    fit_ws.add_argument("--target", type=float, default=0.9)
    fit_ws.add_argument("--min-support", type=int, default=30)
    suggest_ws = wsub.add_parser("suggest")
    suggest_ws.add_argument("--artifact", required=True)
    suggest_ws.add_argument("--consultation", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "whetstone":
            from .whetstone import inspect_store
            from .preference import fit_whetstone, suggest
            if args.ws_command == "inspect":
                result = inspect_store(args.db, scope=args.scope)
            elif args.ws_command == "fit":
                result = fit_whetstone(args.db, scope=args.scope, destination=args.output,
                                      target=args.target, min_support=args.min_support)
            else:
                result = suggest(args.db, artifact=args.artifact, consultation_id=args.consultation)
            print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
            return 2 if args.ws_command == "fit" and not result["ready"] else 0
        task = Task(args.task, labels=args.labels if args.command == "init" else None)
        command = args.command
        if command in ("init", "status"):
            result = task.status()
        elif command == "import":
            result = {"imported_rows": task.import_jsonl(args.file)}
        elif command == "teach":
            result = {"split": task.teach(args.text, args.label, group=args.group, split=args.split)}
        elif command == "fit":
            result = task.fit(target_precision=args.precision, min_support=args.min_support,
                              min_coverage=args.min_coverage, dimensions=args.dimensions, epochs=args.epochs)
        elif command == "promote":
            task.promote(args.model_id)
            result = task.status()
        elif command == "predict":
            result = asdict(task.decide(args.text))
        elif command == "review":
            result = task.review_queue(limit=args.limit, route=args.route)
        elif command == "correct":
            result = {"split": task.correct(args.decision_id, args.label, group=args.group)}
        elif command == "export":
            result = {"exported": task.export(args.destination)}
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 2 if command == "fit" and not result["ready"] else 0
    except (ValueError, TypeError, OSError, sqlite3.Error) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

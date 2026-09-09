"""Usual autopilot CLI. Stdlib only; no provider calls or background collection."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sqlite3
import sys

from .autopilot import Store, KINDS, MODES
from .transcripts import discover
from .migration import default_database


def markdown_report(report):
    def safe(value):
        return str(value).replace("<", "&lt;").replace(">", "&gt;")
    run, summary = report["run"], report["summary"]
    lines = [f"# Usual run: {safe(run['task'])}", "", f"Run: `{run['id']}` · {run['status']} · {run.get('mode', 'autopilot')}",
             f"{summary['predictions']} evidence-backed predictions · {summary['defaults']} agent defaults · {summary['escalations']} escalations", "",
             "Confidence labels are the agent's assessment, not measured accuracy. Pending choices are not training evidence.", ""]
    for i, c in enumerate(report["consultations"], 1):
        d = c["decision"]
        lines += [f"## {i}. {safe(c['question'])}", ""]
        if not d:
            lines += ["Not decided yet.", ""]
            continue
        lines += [f"**{safe(d['choice'])}**", "", safe(d["rationale"]), "",
                  f"Basis: {d['basis']} · Confidence: {d['confidence']} · Review: {d['review']}", ""]
        if d["correction"]:
            lines += [f"Your correction: {safe(d['correction'])}", ""]
        for e in c["evidence"]:
            if e["id"] in d["evidence_ids"]:
                quote = safe(e["quote"]).replace("\n", "\n> ")
                lines += [f"> {quote}", "", f"Source: {safe(e['source'])}:{e['line']} · {e['origin']} · `{e['id']}`", ""]
        lines += [f"Decision: `{d['id']}`", ""]
    return "\n".join(lines)


def report_html(report):
    content = html.escape(markdown_report(report))
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Usual decision report</title><style>body{max-width:900px;margin:48px auto;padding:0 24px;background:#f7f6f2;color:#23302a;font:17px/1.6 system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}</style><body><pre>' + content + '</pre></body></html>'


def parser():
    p = argparse.ArgumentParser(description="Usual — autopilot for your coding judgment.")
    p.add_argument("--db", default=str(default_database()), help="Private local database (never inside a public project)")
    commands = p.add_subparsers(dest="command")
    commands.add_parser("onboard", help="Start the guided setup; inventory history without reading its contents")
    mode = commands.add_parser("mode", help="View or change the mode for new runs, or one active run")
    mode.add_argument("--set", choices=sorted(MODES), dest="selected")
    mode.add_argument("--run")
    init = commands.add_parser("init", help="Import explicit human choices from recent local transcripts")
    init.add_argument("--provider", choices=["codex", "claude", "both"], default="both")
    init.add_argument("--limit", type=int, default=20)
    init.add_argument("--scope", default="global")
    init.add_argument("--dry-run", action="store_true")
    imp = commands.add_parser("import", help="Import selected transcript files; safe to repeat")
    imp.add_argument("paths", nargs="+")
    imp.add_argument("--scope", default="global")
    mine = commands.add_parser("mine", help="Mine linked decisions across native Claude/Codex history")
    mine.add_argument("--provider", choices=["codex", "claude", "both"], default="both")
    mine.add_argument("--all", action="store_true", help="Include all discoverable main-session transcripts, including Codex archives")
    mine.add_argument("--limit", type=int, default=20)
    mine.add_argument("--scope", default="global")
    mine.add_argument("--dry-run", action="store_true")
    mine.add_argument("--force", action="store_true", help="Reparse unchanged files (e.g. after a parser upgrade)")
    eps = commands.add_parser("episodes", help="Inspect private historical question-answer episodes")
    eps.add_argument("--search", default="")
    eps.add_argument("--limit", type=int, default=20)
    commands.add_parser("status")
    commands.add_parser("doctor")
    evidence = commands.add_parser("evidence")
    evidence.add_argument("--scope")
    evidence.add_argument("--limit", type=int, default=30)
    retire = commands.add_parser("retire", help="Exclude a source from future retrieval, preserving the audit trail")
    retire.add_argument("--evidence", required=True)
    start = commands.add_parser("start")
    start.add_argument("--task", required=True)
    start.add_argument("--scope", default=str(Path.cwd().resolve()))
    start.add_argument("--mode", choices=sorted(MODES))
    consult = commands.add_parser("consult")
    consult.add_argument("--run", required=True)
    consult.add_argument("--question", required=True)
    consult.add_argument("--option", action="append", required=True)
    consult.add_argument("--kind", choices=sorted(KINDS), default="implementation")
    consult.add_argument("--uncertain", action="store_true", help="Flag weak or conflicting evidence; escalation mode requires a check-in")
    record = commands.add_parser("record")
    record.add_argument("--consultation", required=True)
    record.add_argument("--choice", required=True)
    record.add_argument("--rationale", required=True)
    record.add_argument("--evidence", action="append", default=[])
    record.add_argument("--confidence", choices=["low", "medium", "high"], default="medium")
    record.add_argument("--basis", choices=["prediction", "agent_default", "escalated"], default="prediction")
    for name in ("finish", "report"):
        command = commands.add_parser(name)
        command.add_argument("--run", required=True)
        command.add_argument("--format", choices=["json", "markdown", "html"], default="json")
    review = commands.add_parser("review", help="Apply an explicit user review; agents must not self-endorse")
    review.add_argument("--decision", required=True)
    review.add_argument("--verdict", choices=["accepted", "corrected", "rejected"], required=True)
    review.add_argument("--statement", default="")
    review.add_argument("--confirm-user-review", action="store_true", help="Confirm the current user explicitly gave this verdict")
    serve = commands.add_parser("review-ui", help="Open a private, loopback-only decision review")
    serve.add_argument("--run", required=True)
    serve.add_argument("--port", type=int, default=0)
    backup = commands.add_parser("backup")
    backup.add_argument("destination")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "init":
            if not 1 <= args.limit <= 100:
                raise ValueError("Choose a limit between 1 and 100 transcripts.")
            paths = discover(args.provider, args.limit)
            if args.dry_run:
                print(json.dumps({"files": [str(p) for p in paths], "scope": args.scope, "network_calls": 0}, indent=2))
                return 0
        if args.command == "mine":
            from .episodes import history_files
            paths, exclusions = history_files(args.provider)
            if not args.all:
                if not 1 <= args.limit <= 100:
                    raise ValueError("Use limit 1–100, or --all for the complete discovered inventory.")
                paths = sorted(paths,key=lambda item:item[1].stat().st_mtime,reverse=True)[:args.limit]
            if args.dry_run:
                from collections import Counter
                print(json.dumps({"files":len(paths), "providers":dict(Counter(p for p,_ in paths)),
                                  "bytes":sum(f.stat().st_size for _,f in paths), "excluded":exclusions,
                                  "sample_sources":[str(f) for _,f in paths[:10]], "network_calls":0},indent=2))
                return 0
        store = Store(args.db)
        if args.command in (None, "onboard"):
            result = store.onboard()
        elif args.command == "mode":
            result = store.mode(args.selected, args.run)
        elif args.command == "mine":
            result = store.mine_history(paths,args.scope,args.force)
            result["excluded_files"] = exclusions
        elif args.command == "episodes":
            result = store.episodes(args.search,args.limit)
        elif args.command == "init":
            result = store.import_files(paths, args.scope)
        elif args.command == "import":
            result = store.import_files(args.paths, args.scope)
        elif args.command == "status":
            result = store.status()
        elif args.command == "doctor":
            with store.db() as db:
                integrity = db.execute("PRAGMA quick_check").fetchone()[0]
            result = {"ok": integrity == "ok", "sqlite": sqlite3.sqlite_version, "integrity": integrity,
                      "python": sys.version.split()[0], "database": str(store.path), "network_calls": 0,
                      "codex_skill": (Path.home()/".agents/skills/usual/SKILL.md").is_file(),
                      "claude_skill": (Path.home()/".claude/skills/usual/SKILL.md").is_file()}
        elif args.command == "evidence":
            if not 1 <= args.limit <= 20000:
                raise ValueError("Choose an evidence limit between 1 and 20000.")
            result = store.evidence(args.scope, args.limit)
        elif args.command == "retire":
            result = store.retire_evidence(args.evidence)
        elif args.command == "start":
            result = store.start(args.task, args.scope, args.mode)
        elif args.command == "consult":
            result = store.consult(args.run, args.question, args.option, args.kind, args.uncertain)
        elif args.command == "record":
            result = store.record(args.consultation, args.choice, args.rationale, args.evidence, args.confidence, args.basis)
        elif args.command in ("finish", "report"):
            result = store.finish(args.run) if args.command == "finish" else store.report(args.run)
            if args.format != "json":
                print(markdown_report(result) if args.format == "markdown" else report_html(result))
                return 0
        elif args.command == "review":
            if not args.confirm_user_review:
                raise ValueError("Human review is required. Use the private review UI, or pass --confirm-user-review only for a verdict the user explicitly gave.")
            result = store.review(args.decision, args.verdict, args.statement)
        elif args.command == "review-ui":
            from .review_server import serve
            serve(store, args.run, args.port)
            return 0
        elif args.command == "backup":
            result = store.backup(args.destination)
        else:
            raise ValueError("Unknown command")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if isinstance(result, dict) and (result.get("errors") or result.get("ok") is False):
            return 1
        return 0
    except (ValueError, OSError, sqlite3.Error) as error:
        # SQLite/OS diagnostics can contain paths or values: keep operational errors bounded.
        message = str(error) if isinstance(error, ValueError) else "Local storage operation failed. Check file permissions, free space, and database availability."
        print(json.dumps({"error": message}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

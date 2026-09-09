#!/usr/bin/env python3
"""Reproducible fixture replay through the real CLI/storage engine, not a live model run."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from usual.autopilot import Store
from usual.cli import report_html


def run_demo(database):
    store = Store(database)
    paths = [ROOT / 'demo/fixtures/sample-codex.jsonl', ROOT / 'demo/fixtures/sample-claude.jsonl']
    imported = store.import_files(paths, 'global')
    assert imported['added'] == 4 and not imported['errors']
    assert store.import_files(paths, 'global')['added'] == 0
    run = store.start('Build a local reading list with durable storage and clear controls.', 'sample-reading-list')
    specs = [
        ('Which storage fits this single-user local reading list?', ['Use SQLite', 'Use hosted Postgres'], 'implementation',
         'Use SQLite', 'The earlier user explicitly chose SQLite for single-user local storage to avoid another service.', 'SQLite'),
        ('Should the local Python web server use a framework dependency?', ['Use the Python standard library', 'Add a web framework'], 'dependency',
         'Use the Python standard library', 'The earlier user chose the standard library for a small local server and explained the dependency tradeoff.', 'standard library'),
        ('How should the reading list controls be labeled?', ['Use visible text labels', 'Use icons alone'], 'design',
         'Use visible text labels', 'The user previously chose visible text labels for this exact kind of control.', 'text labels'),
    ]
    for question, options, kind, choice, rationale, needle in specs:
        consultation = store.consult(run['id'], question, options, kind)
        source = next(e for e in consultation['evidence'] if needle in e['quote'])
        store.record(consultation['id'], choice, rationale, [source['id']], 'medium', 'prediction')
    permission = store.consult(run['id'], 'Should I publish the finished reading list now?', ['Publish now', 'Wait for the user'], 'permission')
    store.record(permission['id'], 'Wait for the user', 'Preferences are not permission to publish. Keep the built app local.', [], 'low', 'escalated')
    before = store.finish(run['id'])
    assert store.status()['evidence'] == 4
    decisions = [c['decision'] for c in before['consultations']]
    # These are explicit synthetic fixture reviews, never reviews on a real user's behalf.
    store.review(decisions[0]['id'], 'accepted')
    store.review(decisions[1]['id'], 'corrected', 'Keep the Python standard library for this small local server, but use an existing web framework when the project already has one.')
    after = store.report(run['id'])
    next_run = store.start('Add a reading list endpoint to a project with an existing web framework.', 'sample-reading-list')
    next_consultation = store.consult(next_run['id'], 'The project already has a web framework. Should this endpoint use that framework or the Python standard library?', ['Reuse the existing web framework', 'Write a separate standard-library server'], 'dependency')
    correction = next(e for e in next_consultation['evidence'] if e['origin'] == 'endorsed' and 'existing web framework' in e['call'])
    store.record(next_consultation['id'], 'Reuse the existing web framework', 'The explicitly corrected preference covers projects with an existing framework; it is more specific than the original small-server choice.', [correction['id']], 'medium', 'prediction')
    next_report = store.finish(next_run['id'])
    return {'mode': 'fixture_replay', 'disclosure': 'Synthetic coding transcripts and prewritten agent choices run through the real Usual engine. This is not a live Codex or Claude inference run.',
            'import': imported, 'evidence': store.evidence(), 'before_review': before, 'after_review': after,
            'next_run': next_report, 'checks': ['Native Codex and Claude human messages imported', 'System and tool messages excluded',
                'Repeated import deduplicated', 'Every predicted choice cites a stored source', 'Publication escalated',
                'Pending predictions never learned', 'Human correction changes the evidence for the next run', 'Original prediction preserved']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, help='Write a shareable synthetic run JSON here')
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix='usual-demo-') as directory:
        result = run_demo(Path(directory) / 'demo.sqlite3')
    output = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + '\n')
        args.out.with_suffix('.html').write_text(report_html(result['after_review']))
        print(json.dumps({'demo': str(args.out), 'checks': result['checks']}))
    else:
        print(output)


if __name__ == '__main__':
    main()

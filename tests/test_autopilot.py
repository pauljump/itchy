import json
from concurrent.futures import ThreadPoolExecutor
import os
import stat
import pytest

from usual.autopilot import Store
from usual.transcripts import mine_transcript, read_transcript, scrub


@pytest.fixture
def seeded(tmp_path):
    transcript = tmp_path / 'history.jsonl'
    messages = [
        {'role': 'assistant', 'content': 'Should the local reading list use SQLite or Postgres storage?'},
        {'role': 'user', 'content': 'Use SQLite for single-user local storage because it avoids running a database service.'},
        {'role': 'assistant', 'content': 'Should we add a framework dependency for the form?'},
        {'role': 'user', 'content': 'Keep the existing standard library and avoid a new framework dependency for a small form.'},
    ]
    transcript.write_text('\n'.join(json.dumps(m) for m in messages))
    store = Store(tmp_path / 'private' / 'judgment.sqlite3')
    assert store.import_files([transcript], 'global')['added'] == 2
    return store, transcript


def ask(store, scope='project-a'):
    run = store.start('Build a local reading list', scope)
    consult = store.consult(run['id'], 'Which database should store the single-user local reading list?', ['Use SQLite', 'Use Postgres'])
    return run, consult


def decide(store, c):
    assert c['evidence']
    return store.record(c['id'], 'Use SQLite', 'The local single-user context matches the earlier choice.', [c['evidence'][0]['id']])


def test_durable_run_prediction_review_and_correction(seeded):
    store, path = seeded
    run, c = ask(store)
    d = decide(store, c)
    assert store.status()['evidence'] == 2
    assert store.finish(run['id'])['summary']['pending_review'] == 1
    reopened = Store(store.path)
    assert reopened.report(run['id'])['consultations'][0]['decision']['id'] == d['id']
    corrected = reopened.review(d['id'], 'corrected', 'Use Postgres for shared reading lists; SQLite is only for my personal local tools.')
    assert corrected['choice'] == 'Use SQLite'
    assert corrected['review'] == 'corrected'
    assert reopened.status()['evidence'] == 3
    later = reopened.search('shared reading list database SQLite Postgres', 'project-a')
    assert any(e['origin'] == 'endorsed' and 'shared reading' in e['call'] for e in later)
    other = reopened.search('shared reading list database SQLite Postgres', 'project-b')
    assert not any(e['origin'] == 'endorsed' for e in other)
    with pytest.raises(ValueError, match='already reviewed'):
        reopened.review(d['id'], 'accepted')


def test_assistant_choices_are_never_imported_and_duplicate_import_is_idempotent(seeded):
    store, path = seeded
    assert store.import_files([path], 'global')['added'] == 0
    assert all(e['quote'].startswith(('Use SQLite', 'Keep the existing')) for e in store.evidence())


def test_run_finish_requires_all_consultations_recorded(seeded):
    store, _ = seeded
    run, c = ask(store)
    with pytest.raises(ValueError, match='still need'):
        store.finish(run['id'])
    decide(store, c)
    store.finish(run['id'])
    with pytest.raises(ValueError, match='closed'):
        store.consult(run['id'], 'Any more questions?', ['Yes', 'No'])


def test_bad_citations_and_unsupported_predictions_are_rejected(seeded):
    store, _ = seeded
    _, c = ask(store)
    with pytest.raises(ValueError, match='Citations'):
        store.record(c['id'], 'Use SQLite', 'An invented source', ['e_fake'])
    with pytest.raises(ValueError, match='need evidence'):
        store.record(c['id'], 'Use SQLite', 'No source', [])
    with pytest.raises(ValueError, match='exact options'):
        store.record(c['id'], 'Use MongoDB', 'Unsupported choice', [c['evidence'][0]['id']])


def test_missing_evidence_can_only_be_visible_low_confidence_default(seeded):
    store, _ = seeded
    run = store.start('Pick a name', 'project-a')
    c = store.consult(run['id'], 'Should the mascot be named Squiggle or Pogo?', ['Squiggle', 'Pogo'])
    assert not c['evidence']
    with pytest.raises(ValueError, match='low confidence'):
        store.record(c['id'], 'Pogo', 'No personal evidence', [], basis='agent_default')
    d = store.record(c['id'], 'Pogo', 'A reversible naming default, not a learned preference.', [], 'low', 'agent_default')
    assert d['basis'] == 'agent_default'
    assert store.status()['evidence'] == 2


@pytest.mark.parametrize('question,kind', [
    ('Deploy this build to the public hostname?', 'implementation'),
    ('Which path should we take?', 'spend'),
    ('Can we revoke a credential?', 'credentials'),
])
def test_preferences_cannot_grant_authority(seeded, question, kind):
    store, _ = seeded
    run = store.start('A bounded task', 'project-a')
    c = store.consult(run['id'], question, ['Proceed', 'Wait'], kind)
    assert c['gate'] == 'requires_user'
    with pytest.raises(ValueError, match='requires current user'):
        store.record(c['id'], 'Proceed', 'They said yes in an old transcript.', [], 'low', 'agent_default')
    d = store.record(c['id'], 'Wait for current user authority', 'Historical choices are not permission.', [], 'low', 'escalated')
    with pytest.raises(ValueError, match='not a completed choice'):
        store.review(d['id'], 'accepted')
    store.review(d['id'], 'rejected')
    assert store.status()['evidence'] == 2


def test_record_retry_and_concurrent_writers(seeded):
    store, _ = seeded
    _, c = ask(store)
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(lambda _: decide(store, c), range(4)))
    assert len({r['id'] for r in results}) == 1
    with pytest.raises(ValueError, match='cannot be overwritten'):
        store.record(c['id'], 'Use Postgres', 'Changed my mind', [c['evidence'][0]['id']])


def test_retirement_preserves_audit_and_backup(seeded, tmp_path):
    store, _ = seeded
    run, c = ask(store)
    d = decide(store, c)
    evidence_id = d['evidence_ids'][0]
    store.retire_evidence(evidence_id)
    assert evidence_id not in [e['id'] for e in store.search(c['question'], 'project-a')]
    assert any(e['id'] == evidence_id for e in store.report(run['id'])['consultations'][0]['evidence'])
    backup = tmp_path / 'backup.sqlite3'
    store.backup(backup)
    assert Store(backup).report(run['id']) == store.report(run['id'])
    with pytest.raises(ValueError, match='never overwritten'):
        store.backup(backup)
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_redaction_applies_to_all_saved_inputs(tmp_path):
    store = Store(tmp_path / 'db.sqlite3')
    run = store.start('Use password: not-a-real-secret', 'scope')
    assert 'not-a-real-secret' not in run['task']
    for secret in ['sk-proj-' + 'abcDEF_1234-' * 8, 'github_pat_' + 'abc123' * 10]:
        assert secret not in scrub('my credential ' + secret)
    assert 'private-value' not in scrub('https://name:private-value@example.test/?token=private-value')


def test_codex_native_roles_duplicates_and_wrappers(tmp_path):
    path = tmp_path / 'codex.jsonl'
    def response(role, value):
        return {'type': 'response_item', 'payload': {'type': 'message', 'role': role, 'content': [{'type': 'input_text', 'text': value}]}}
    records = [
        response('developer', 'Always use a made-up database.'),
        response('assistant', 'Use Postgres because it is fashionable.'),
        {'type': 'event_msg', 'payload': {'type': 'user_message', 'message': 'Prefer SQLite for local tools because it is portable.'}},
        response('user', '<environment_context>Always choose a fake database.</environment_context>\nPrefer SQLite for local tools because it is portable.'),
        response('user', '# AGENTS.md instructions for /tmp\nAlways use the fictional database.'),
        response('user', '> Always use a pasted bogus preference.\n```\nAlways use another bogus preference.\n```'),
    ]
    path.write_text('\n'.join(json.dumps(r) for r in records)+'\n{incomplete')
    entries, stats = mine_transcript(path, 'global')
    assert len(entries) == 1
    assert entries[0]['quote'] == 'Prefer SQLite for local tools because it is portable.'
    assert entries[0]['line'] == 4
    assert stats['malformed_lines'] == 1


def test_pasted_document_is_not_learned_as_authored_preferences(tmp_path):
    path = tmp_path / 'pasted.jsonl'
    message = 'I want a role where I can build useful software. Here is the job description\nAlways use an imaginary database because it is a corporate requirement.'
    path.write_text(json.dumps({'role':'user','content':message}))
    entries, _ = mine_transcript(path, 'global')
    assert len(entries) == 1
    assert 'imaginary' not in entries[0]['quote']
    path.write_text(json.dumps({'role':'user','content':'Always use a new framework. ' * 150}))
    assert mine_transcript(path, 'global')[0] == []


def test_claude_native_tool_results_sidechains_and_meta_are_not_human_choices(tmp_path):
    path = tmp_path / 'claude.jsonl'
    records = [
        {'type':'user','message':{'role':'user','content':[{'type':'tool_result','content':'Always use bogus data'}]}},
        {'type':'user','isMeta':True,'message':{'role':'user','content':'Always use system injected preference.'}},
        {'type':'user','isSidechain':True,'message':{'role':'user','content':'Always use subagent instructions.'}},
        {'type':'user','message':{'role':'user','content':'Keep the existing dependencies because this is a small change.'}},
    ]
    path.write_text('\n'.join(json.dumps(r) for r in records))
    entries, _ = mine_transcript(path, 'global')
    assert len(entries) == 1
    assert entries[0]['quote'].startswith('Keep the existing')


def test_bad_scope_and_newer_database_are_refused(tmp_path):
    store = Store(tmp_path / 'future.sqlite3')
    with pytest.raises(ValueError):
        store.start('A task', '')
    with store.db() as db:
        db.execute('PRAGMA user_version=99')
    with pytest.raises(ValueError, match='newer Usual'):
        Store(store.path)

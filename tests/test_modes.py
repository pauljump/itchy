import json

import pytest

from usual.autopilot import Store
from usual.cli import main


def consultation(store, run, **kwargs):
    return store.consult(run['id'], 'Which storage fits this local reading list?',
                         ['SQLite local storage', 'Postgres server storage'], **kwargs)


def seed(store, tmp_path):
    path = tmp_path / 'choices.jsonl'
    path.write_text('\n'.join(json.dumps(m) for m in [
        {'role': 'assistant', 'content': 'Which storage fits this local reading list?'},
        {'role': 'user', 'content': 'Use SQLite local storage for a personal reading list.'},
    ]))
    store.import_files([path], 'global')


def test_default_persists_and_active_runs_keep_their_mode(tmp_path):
    store = Store(tmp_path / 'private.sqlite3')
    original = store.start('First task', 'project')
    store.mode('check-in')
    reopened = Store(store.path)
    assert reopened.start('Next task', 'project')['mode'] == 'check-in'
    assert reopened.report(original['id'])['run']['mode'] == 'autopilot'
    assert reopened.start('Override', 'project', 'escalation')['mode'] == 'escalation'
    reopened.mode('check-in', original['id'])
    assert reopened.report(original['id'])['run']['mode'] == 'check-in'
    reopened.finish(original['id'])
    with pytest.raises(ValueError, match='closed'):
        reopened.mode('autopilot', original['id'])


def test_check_in_gate_cannot_be_bypassed_by_mode_change(tmp_path):
    store = Store(tmp_path / 'private.sqlite3')
    run = store.start('Build', 'project', 'check-in')
    c = consultation(store, run)
    assert c['gate_reason'] == 'check-in'
    store.mode('autopilot', run['id'])
    with pytest.raises(ValueError, match='current user authority'):
        store.record(c['id'], c['options'][0], 'Routine choice', [], 'low', 'agent_default')
    decision = store.record(c['id'], c['options'][0], 'User chose SQLite in this session.', [], 'high', 'escalated')
    assert decision['basis'] == 'escalated'
    assert store.status()['evidence'] == 0
    with pytest.raises(ValueError, match='not a completed choice'):
        store.review(decision['id'], 'accepted')


def test_stricter_mode_applies_to_an_open_consultation(tmp_path):
    store = Store(tmp_path / 'private.sqlite3')
    run = store.start('Build', 'project')
    c = consultation(store, run)
    store.mode('check-in', run['id'])
    with pytest.raises(ValueError, match='run mode'):
        store.record(c['id'], c['options'][0], 'Routine choice', [], 'low', 'agent_default')


def test_escalation_handles_absent_weak_and_conflicting_evidence(tmp_path):
    store = Store(tmp_path / 'private.sqlite3')
    run = store.start('Build', 'project', 'escalation')
    assert consultation(store, run)['gate'] == 'requires_user'
    seed(store, tmp_path)
    c = consultation(store, run)
    ids = [c['evidence'][0]['id']]
    assert c['gate'] == 'advisory'
    for basis in ['agent_default', 'prediction']:
        with pytest.raises(ValueError, match='run mode'):
            store.record(c['id'], c['options'][0], 'Weak evidence', ids, 'low', basis)
    store.record(c['id'], c['options'][0], 'Same personal local context', ids, 'medium', 'prediction')
    uncertain = consultation(store, run, uncertain=True)
    assert uncertain['gate_reason'] == 'uncertain_evidence'
    with pytest.raises(ValueError, match='current user authority'):
        store.record(uncertain['id'], uncertain['options'][0], 'Conflicting history', ids, 'high', 'prediction')


@pytest.mark.parametrize('mode', ['autopilot', 'check-in', 'escalation'])
@pytest.mark.parametrize('action', ['Delete the old notes file', 'Remove the unused folder',
                                   'Run rm -rf cache', 'Drop table drafts', 'Erase archived records'])
def test_deletion_requires_authority_in_every_mode(tmp_path, mode, action):
    store = Store(tmp_path / 'private.sqlite3')
    run = store.start('Tidy project', 'project', mode)
    c = store.consult(run['id'], 'How should we tidy the project?', [action, 'Keep existing resources'])
    assert c['gate_reason'] == 'current_permission'
    with pytest.raises(ValueError, match='authority|run mode'):
        store.record(c['id'], action, 'A former user said yes', [], 'low', 'agent_default')


def test_onboarding_inventories_without_import_or_transcript_read(tmp_path, monkeypatch, capsys):
    from usual import episodes
    source = tmp_path / 'never-open.jsonl'
    source.write_text('not a transcript')
    monkeypatch.setattr(episodes, 'history_files', lambda provider: ([('codex', source)], {}))
    def forbidden(*args, **kwargs):
        raise AssertionError('Onboarding must not mine contents')
    monkeypatch.setattr(episodes, 'mine_episodes', forbidden)
    path = tmp_path / 'private.sqlite3'
    assert main(['--db', str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['stage'] == 'choose-history'
    assert result['history']['available_files'] == 1
    assert result['learned']['evidence'] == 0
    assert result['transcript_contents_read'] is False
    store = Store(path)
    seed(store, tmp_path)
    assert store.onboard()['stage'] == 'ready'


def test_upgrade_preserves_legacy_runs_and_evidence(tmp_path):
    store = Store(tmp_path / 'private.sqlite3')
    seed(store, tmp_path)
    with store.db() as db:
        # Reproduce a pre-modes database, including its positional runs schema.
        db.execute('DROP TABLE run_modes')
        db.execute('DROP TABLE settings')
        db.execute("INSERT INTO runs VALUES('legacy','Earlier task','project','complete','2026-01-01','2026-01-01')")
    upgraded = Store(store.path)
    assert upgraded.report('legacy')['run']['mode'] == 'autopilot'
    assert upgraded.status()['evidence'] == 1
    upgraded.mode('escalation')
    assert upgraded.report('legacy')['run']['mode'] == 'autopilot'


def test_cli_mode_roundtrip(tmp_path, capsys):
    prefix = ['--db', str(tmp_path / 'private.sqlite3')]
    assert main(prefix + ['mode', '--set', 'escalation']) == 0
    capsys.readouterr()
    assert main(prefix + ['start', '--task', 'Build a tool']) == 0
    run = json.loads(capsys.readouterr().out)
    assert run['mode'] == 'escalation'
    assert main(prefix + ['mode', '--run', run['id'], '--set', 'check-in']) == 0
    assert json.loads(capsys.readouterr().out)['mode'] == 'check-in'

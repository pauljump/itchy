import json
from pathlib import Path
import sqlite3
import subprocess
import sys

from usual.migration import default_database, migrate_home

ROOT = Path(__file__).resolve().parents[1]


def test_install_retains_history_and_retires_old_skill_discovery(tmp_path):
    home = tmp_path / 'home with spaces'
    old = home / '.whetstone'
    old.mkdir(parents=True)
    db = old / 'judgment.sqlite3'
    with sqlite3.connect(db) as connection:
        connection.execute('CREATE TABLE preserved (id TEXT, decision TEXT)')
        connection.execute('INSERT INTO preserved VALUES (?, ?)', ('run_original', 'Keep the human wording'))
    (old / 'report.md').write_text('Original report')
    for client in ('.agents', '.claude'):
        skill = home / client / 'skills/whetstone'
        skill.mkdir(parents=True)
        (skill / 'SKILL.md').write_text('Original skill')
    assert default_database(home) == db
    result = subprocess.run([sys.executable, str(ROOT/'install.py'), '--home', str(home)], check=True, capture_output=True, text=True)
    output = json.loads(result.stdout)
    assert output['data']['status'] == 'migrated'
    assert old.is_symlink()
    assert default_database(home) == home / '.usual/judgment.sqlite3'
    with sqlite3.connect(default_database(home)) as connection:
        assert connection.execute('SELECT * FROM preserved').fetchall() == [('run_original', 'Keep the human wording')]
    assert (old/'report.md').read_text() == 'Original report'
    for item in output['installed']:
        assert (Path(item['previous_name'])/'SKILL.md').read_text() == 'Original skill'
        assert '/skills/' not in item['previous_name']
        assert not Path(item['skill']).with_name('whetstone').exists()
        assert (Path(item['skill'])/'scripts/usual.py').is_file()
    assert migrate_home(home)['status'] == 'already_migrated'


def test_existing_databases_are_never_merged_or_overwritten(tmp_path):
    for name, content in (('.whetstone', b'old data'), ('.usual', b'new data')):
        (tmp_path/name).mkdir()
        (tmp_path/name/'judgment.sqlite3').write_bytes(content)
    assert migrate_home(tmp_path)['status'] == 'kept_both'
    assert (tmp_path/'.whetstone/judgment.sqlite3').read_bytes() == b'old data'
    assert default_database(tmp_path).read_bytes() == b'new data'


def test_custom_legacy_symlink_is_retained(tmp_path):
    elsewhere = tmp_path/'external data'
    elsewhere.mkdir()
    (elsewhere/'judgment.sqlite3').write_bytes(b'private')
    (tmp_path/'.whetstone').symlink_to(elsewhere, target_is_directory=True)
    assert migrate_home(tmp_path)['status'] == 'kept_legacy'
    assert default_database(tmp_path).read_bytes() == b'private'
    assert not (tmp_path/'.usual').exists()

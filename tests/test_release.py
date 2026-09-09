import hashlib
import importlib.util
import json
import http.client
from pathlib import Path
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import zipfile

import pytest

from usual.server import PublicAutopilotHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]


def test_download_clean_install_runs_without_checkout_or_pip(tmp_path):
    public = tmp_path/'release-output'
    subprocess.run([sys.executable, str(ROOT/'scripts/build_release.py'), '--out', str(public)], check=True, capture_output=True)
    bundle = public/'usual.zip'
    assert (public/'learn.md').read_bytes() == (ROOT/'LEARN.md').read_bytes()
    manifest = json.loads((public/'release.json').read_text())
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == manifest['sha256']
    app_bundle = public/'reading-list.zip'
    assert hashlib.sha256(app_bundle.read_bytes()).hexdigest() == manifest['recorded_app']['sha256']
    recorded = json.loads((public/'build.json').read_text())
    assert recorded['mode'] == 'recorded_agent_build'
    assert recorded['report']['summary']['pending_review'] == 3
    with zipfile.ZipFile(app_bundle) as archive:
        for name, digest in recorded['artifact_sha256'].items():
            assert hashlib.sha256(archive.read('reading-list/' + name)).hexdigest() == digest
    extracted = tmp_path/'source with spaces'
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        assert all(not any(part in name for part in ['.sqlite', '.secrets', '.env', '.git/']) for name in names)
        assert all('sample-' in name for name in names if name.endswith('.jsonl'))
        archive.extractall(extracted)
    source = extracted/'usual'
    home = tmp_path/'home with spaces'
    result = subprocess.run([sys.executable,str(source/'install.py'),'--home',str(home)],check=True,capture_output=True,text=True)
    assert len(json.loads(result.stdout)['installed']) == 2
    # Move the original source; installed runtimes must remain self-contained.
    source.rename(extracted/'moved away')
    for prefix in ['.agents','.claude']:
        script=home/prefix/'skills/usual/scripts/usual.py'
        result=subprocess.run([sys.executable,str(script),'--db',str(tmp_path/(prefix+'.sqlite3')),'doctor'],check=True,capture_output=True,text=True,cwd=tmp_path)
        assert json.loads(result.stdout)['ok']
        shared = [sys.executable, str(script), '--db', str(tmp_path/'shared-private.sqlite3')]
        if prefix == '.agents':
            subprocess.run(shared + ['mode', '--set', 'check-in'], check=True, capture_output=True)
        result = subprocess.run(shared + ['start', '--task', 'Build a local tool'], check=True, capture_output=True, text=True)
        assert json.loads(result.stdout)['mode'] == 'check-in'
        assert (script.parent.parent/'references/onboarding.md').is_file()
    # Reinstall is recoverable and backups aren't competing discovered skills.
    result=subprocess.run([sys.executable,str(extracted/'moved away/install.py'),'--home',str(home)],check=True,capture_output=True,text=True)
    for item in json.loads(result.stdout)['installed']:
        assert Path(item['previous_version']).exists()
        assert '/skills/' not in item['previous_version']


def test_public_site_never_accepts_or_exposes_private_data():
    server=ThreadingHTTPServer(('127.0.0.1',0),PublicAutopilotHandler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(base) as response:
            page=response.read().decode()
            assert 'Less asking.' in page
            assert 'https://github.com/pauljump/itchy' in page
        with urllib.request.urlopen(base+'/demo.json') as response:
            demo=json.load(response)
            assert demo['mode']=='fixture_replay'
            assert demo['before_review']['summary']['predictions']==3
        for path in ['/api/corpus','/api/report','/studio','/sse','/../SKILL.md']:
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(base+path)
            assert error.value.code==404
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(base+'/api/onboarding/prepare',data=b'{}'))
        assert error.value.code==405
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port)
        connection.request('GET', '/whetstone.zip', headers={'Host': 'whetstone.polyfeeds.dev'})
        response = connection.getresponse()
        assert response.status == 308
        assert response.getheader('Location') == 'https://usual.polyfeeds.dev/usual.zip'
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()

import hashlib
import importlib.util
import json
import http.client
from html.parser import HTMLParser
import struct
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
        assert (script.parent.parent/'src/usual/public/og.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
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
            assert 'https://github.com/pauljump/usual' in page
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
        assert response.getheader('Location') == 'https://tryusual.com/usual.zip'
        response.read()
        connection.close()
        for hostname in ['usual.polyfeeds.dev', 'whetstone.polyfeeds.dev', 'WWW.TRYUSUAL.COM:443']:
            for method in ['GET', 'HEAD']:
                for path, target in [('/learn.md?from=share', '/learn.md?from=share'),
                                     ('/whetstone.zip?download=1', '/usual.zip?download=1')]:
                    connection = http.client.HTTPConnection('127.0.0.1', server.server_port)
                    connection.request(method, path, headers={'Host': hostname, 'X-Forwarded-Host': 'untrusted.example'})
                    response = connection.getresponse()
                    assert response.status == 308
                    assert response.getheader('Location') == 'https://tryusual.com' + target
                    assert response.read() == b''
                    connection.close()
    finally:
        server.shutdown()
        server.server_close()


def test_share_crawlers_can_fetch_the_declared_image():
    class Metadata(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tags = {}

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'meta':
                self.tags[attrs.get('property') or attrs.get('name')] = attrs.get('content')

    server = ThreadingHTTPServer(('127.0.0.1', 0), PublicAutopilotHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(base) as response:
            metadata = Metadata()
            metadata.feed(response.read().decode())
        tags = metadata.tags
        assert tags['twitter:card'] == 'summary_large_image'
        assert tags['og:image'] == tags['twitter:image']
        assert tags['og:image:alt'] and tags['twitter:image:alt']
        image_url = urllib.parse.urlparse(tags['og:image'])
        assert image_url.scheme == 'https' and image_url.netloc == 'tryusual.com'
        with urllib.request.urlopen(base + image_url.path) as response:
            data = response.read()
            assert response.headers.get_content_type() == 'image/png'
            assert data[:8] == b'\x89PNG\r\n\x1a\n'
            assert struct.unpack('>II', data[16:24]) == (int(tags['og:image:width']), int(tags['og:image:height']))
        with urllib.request.urlopen(urllib.request.Request(base + image_url.path, method='HEAD')) as response:
            assert int(response.headers['Content-Length']) == len(data)
            assert response.read() == b''
    finally:
        server.shutdown()
        server.server_close()

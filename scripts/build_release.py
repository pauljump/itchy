#!/usr/bin/env python3
"""Build a deterministic code-only download and a fixture-only public replay."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from run_demo import run_demo


def build(output=None):
    public = Path(output) if output else ROOT / 'src/usual/public'
    public.mkdir(parents=True, exist_ok=True)
    (public / 'learn.md').write_bytes((ROOT / 'LEARN.md').read_bytes())
    with tempfile.TemporaryDirectory(prefix='usual-public-demo-') as directory:
        demo = run_demo(Path(directory) / 'demo.sqlite3')
    (public / 'demo.json').write_text(json.dumps(demo, indent=2) + '\n')
    recorded = ROOT / 'demo/reading-list'
    (public / 'build.json').write_bytes((recorded / 'build.json').read_bytes())
    app_buffer = io.BytesIO()
    with zipfile.ZipFile(app_buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ['app.py', 'test_app.py', 'README.md', 'build.json']:
            info = zipfile.ZipInfo('reading-list/' + name, date_time=(2026, 9, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (recorded / name).read_bytes())
    app_bundle = app_buffer.getvalue()
    (public / 'reading-list.zip').write_bytes(app_bundle)
    # Explicit allowlist: never package private data, .git, credentials, or arbitrary output.
    source_files = [ROOT / name for name in ['SKILL.md', 'README.md', 'LEARN.md', 'LICENSE', 'install.py']]
    source_files += sorted((ROOT / 'scripts').glob('*.py'))
    source_files += sorted((ROOT / 'references').glob('*.md'))
    source_files += sorted((ROOT / 'demo/fixtures').glob('sample-*.jsonl'))
    source_files += [recorded / name for name in ['app.py', 'test_app.py', 'README.md', 'build.json']]
    source_files += sorted((ROOT / 'src/usual').glob('*.py'))
    source_files += sorted((ROOT / 'src/usual').glob('*.html'))
    contents = {}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in source_files:
            if source.is_symlink():
                raise ValueError('Release sources cannot be symlinks.')
            relative = str(source.relative_to(ROOT))
            data = source.read_bytes()
            info = zipfile.ZipInfo('usual/' + relative, date_time=(2026, 9, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
            contents[relative] = hashlib.sha256(data).hexdigest()
    bundle = buffer.getvalue()
    (public / 'usual.zip').write_bytes(bundle)
    manifest = {'version': '2.0.0b2', 'stage': 'local beta',
        'sha256': hashlib.sha256(bundle).hexdigest(), 'bytes': len(bundle), 'files': contents,
        'runtime': 'Python 3.11+, standard library only',
        'recorded_app': {'path': '/reading-list.zip', 'sha256': hashlib.sha256(app_bundle).hexdigest(), 'receipts': '/build.json'},
        'demo': 'Synthetic fixture replay through the real engine. No live model calls.',
        'model_execution': 'The invoking Codex or Claude Code session supplies reasoning and uses its normal quota.'}
    (public / 'release.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'bundle': str(public / 'usual.zip'), 'sha256': manifest['sha256'], 'files': len(contents)}))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', help='Build into a separate local directory without changing served downloads')
    build(parser.parse_args().out)

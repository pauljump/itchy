#!/usr/bin/env python3
"""Install the same self-contained skill into Codex and/or Claude Code. No network."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from datetime import datetime, timezone


def archive_skill(target):
    archive = target.parent.parent / 'usual-install-backups'
    archive.mkdir(parents=True, exist_ok=True, mode=0o700)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    backup = archive / (target.name + '-' + stamp)
    target.rename(backup)
    return backup


def install(root, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.usual-install-', dir=target.parent))
    backup = None
    try:
        shutil.copy2(root / 'SKILL.md', stage / 'SKILL.md')
        shutil.copytree(root / 'scripts', stage / 'scripts', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copytree(root / 'references', stage / 'references')
        shutil.copytree(root / 'demo/fixtures', stage / 'demo/fixtures')
        shutil.copytree(root / 'src/usual', stage / 'src/usual', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        if target.exists() or target.is_symlink():
            backup = archive_skill(target)
        stage.rename(target)
    except Exception:
        if backup and not target.exists():
            backup.rename(target)
        shutil.rmtree(stage, ignore_errors=True)
        raise
    legacy = target.with_name('whetstone')
    legacy_backup = archive_skill(legacy) if legacy.exists() or legacy.is_symlink() else None
    return {'skill': str(target), 'previous_version': str(backup) if backup else None,
            'previous_name': str(legacy_backup) if legacy_backup else None}


def main():
    parser = argparse.ArgumentParser(description='Install Usual for local Codex and Claude Code. No API key or pip install.')
    parser.add_argument('--client', choices=['codex', 'claude', 'both'], default='both')
    parser.add_argument('--home', type=Path, default=Path.home(), help='Install under another home directory (for isolated validation)')
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error('Python 3.11 or newer is required.')
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / 'src'))
    from usual.migration import migrate_home
    targets = []
    if args.client in ('codex', 'both'):
        targets.append(args.home / '.agents/skills/usual')
    if args.client in ('claude', 'both'):
        targets.append(args.home / '.claude/skills/usual')
    installed = [install(root, target) for target in targets]
    print(json.dumps({'installed': installed, 'data': migrate_home(args.home),
        'next': 'Say "use Usual" for guided setup. In Codex: $usual. In Claude Code: /usual. Then: "Build me something with Usual."'}, indent=2))


if __name__ == '__main__':
    main()

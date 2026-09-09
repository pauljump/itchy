"""Preserve local history when upgrading from Whetstone to Usual. No network."""
from pathlib import Path


def default_database(home=None):
    home = Path(home) if home is not None else Path.home()
    current = home / '.usual' / 'judgment.sqlite3'
    legacy = home / '.whetstone' / 'judgment.sqlite3'
    # A checkout can run before the installer; don't silently abandon its history.
    return legacy if not current.exists() and legacy.exists() else current


def migrate_home(home):
    home = Path(home)
    legacy, current = home / '.whetstone', home / '.usual'
    if not legacy.exists() and not legacy.is_symlink():
        return {'status': 'not_needed', 'home': str(current)}
    if current.exists() or current.is_symlink():
        if legacy.resolve() == current.resolve():
            return {'status': 'already_migrated', 'home': str(current)}
        return {'status': 'kept_both', 'home': str(current), 'legacy': str(legacy),
                'note': 'Both data folders exist. Nothing was merged or overwritten. Use --db to select the intended database.'}
    if legacy.is_symlink() or not legacy.is_dir():
        return {'status': 'kept_legacy', 'legacy': str(legacy),
                'note': 'Legacy data is not a regular directory. The CLI retains its existing database path.'}
    legacy.rename(current)
    try:
        # Existing tools and saved paths keep using the same database, not a stale copy.
        legacy.symlink_to(current.name, target_is_directory=True)
    except OSError:
        current.rename(legacy)
        raise
    return {'status': 'migrated', 'home': str(current), 'compatibility_link': str(legacy)}

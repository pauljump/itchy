# Upgrading from Whetstone

Usual is the current name of the same decision-history product. Its canonical source is
[pauljump/usual](https://github.com/pauljump/usual). The original Itchy language-model
research lives in `archive/itchy`; it is not required to install or use Usual.

Run `python3 install.py --client both` (or choose one client). The installer:

1. Installs the self-contained Usual skill in `~/.agents/skills/usual` and/or `~/.claude/skills/usual`.
2. Moves the corresponding old Whetstone skill outside discovery into the client's `usual-install-backups` directory. Existing Usual versions are backed up there too.
3. Renames `~/.whetstone` to `~/.usual` if only the former exists. A compatibility symlink keeps saved paths and existing tools pointed at the same files. There is no database copy or schema rewrite.

Reopen the coding client if necessary. Use `/usual` in Claude Code, `$usual` or the
skill picker in Codex, or say “use Usual.” Original conversations, run IDs, human
reviews, and evidence are retained. Historical records keep their original wording.

If both data folders already exist, the installer keeps both and reports that fact.
It never merges or overwrites databases. Use `--db /absolute/path/to/judgment.sqlite3`
before the subcommand to select one. Before installation, the CLI falls back to an
existing legacy database rather than starting over without your history.

Uninstalling Usual is separate from deleting history. An explicit full removal should
account for the Usual skill, its data directory, install backups, any compatibility
symlink, and custom database/export paths. Original transcripts are never removed.

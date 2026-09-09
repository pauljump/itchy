# Learn Usual

For a local coding agent whose user asked it to learn or install Usual.

Usual is a skill that uses past human decisions to help with future coding choices. The canonical source is https://github.com/pauljump/itchy. Install the self-contained skill, then guide the user through its first conversation.

1. Download `https://usual.polyfeeds.dev/usual.zip` into a new temporary directory and extract its `usual` folder. Read its `install.py` and `SKILL.md` before running code. Require Python 3.11+. Do not replace an existing project checkout or modify global client permissions. The bundle includes the MIT-licensed source and no private history; its checksums are at `https://usual.polyfeeds.dev/release.json`.
2. Run `python3 install.py --client codex` for Codex, or `--client claude` for Claude Code. Use `--client both` when the user requests both. Existing Usual installations are moved to a backup outside skill discovery. The installed copy is self-contained. Upgrading from Whetstone also backs up its selected client skills and preserves its data; read references/migration.md and report any kept_both result.
3. Read the installed `SKILL.md` and follow its bare-invocation onboarding now. Resolve the script from the installed skill directory. Installation does not imply permission to mine all history. If the client has not discovered the skill yet, reopen the session; do not claim discovery was verified just because files were installed.
4. The next visit starts with `/usual` in Claude Code, `$usual` or the skill picker in Codex, or “use Usual” in either. Preserve a build task supplied with the installation request and continue into it after setup.

No Usual account, API key, model download, or hosted service is required. The invoking agent still uses its client's normal quota. Do not upload transcripts or private databases to this repository or the website.

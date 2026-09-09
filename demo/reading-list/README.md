# An actual build using Usual

This reading-list app was implemented by Codex during development under the Whetstone name (now Usual) on September 7, 2026. Before writing the app, the agent imported the supplied synthetic history, retrieved evidence for three implementation choices, and logged its own judgments. The accompanying `build.json` contains the actual run receipts. These are sample preferences, not a real person's profile. The choices remain unreviewed.

## Run it

Unzip this folder, open a terminal here, and run:

```bash
python3 app.py
```

Open the private localhost link printed in the terminal. Add a title and URL, mark it read or unread, and filter the list. Close the server and start it again: your list survives. Python 3.11 or newer is the only requirement.

Your data lives at `~/.usual/demo-reading-list.sqlite3`. Use `--db /path/to/list.sqlite3` for a different file. There are no model calls, remote storage, tracking, or deployment steps in this app.

Run the backend checks with `python3 -m unittest -v`. They cover add/update/persistence, private API access, invalid links, and missing records. Browser interactions were not automatically tested: no browser was connected to this development session.

## What this demonstrates

- The current agent made actual choices, persisted them **before** implementation, and produced a working artifact.
- SQLite, the Python standard library, and visible control labels each correspond to a cited source in the receipts.
- This is one recorded Codex session using the skill's CLI workflow. It does not prove autonomous skill discovery or repeated-run reliability in both external clients.
- The separate site replay illustrates a synthetic human correction. No human review is fabricated for this recorded build.

To run your own loop, install Usual, initialize it from your history, and ask your coding agent to build something with Usual. Review the resulting choices using the private review UI.

# Demo: from transcripts to a reviewed coding run

The public walkthrough uses **synthetic transcripts and prewritten example choices**, passed through the real importer, SQLite store, retrieval, decision log, and review code. It is a reproducible fixture replay, not evidence that either client autonomously built an app. It demonstrates data flow and the correction loop without uploading personal transcripts or consuming model quota.

From a source checkout:

```bash
python3 scripts/run_demo.py --out /tmp/usual-demo.json
```

The replay imports native Codex and Claude fixtures, delegates storage/dependency/control-label choices, escalates publication, closes the run, accepts one synthetic choice, corrects another, and verifies that the correction is retrieved and cited in the next run. It produces JSON and a readable HTML report. Fixture review actions must never be applied to real-user decisions.

## Live agent demo

`demo/reading-list/` also contains an actual working app produced in the Codex development session. Its `build.json` records three model judgments made and logged before implementation, using only the synthetic history. The app has three backend tests, including persistence across restart. All three judgments remain pending human review. This demonstrates a real artifact from the current agent's CLI workflow; separate external-client skill discovery and browser interactions remain unverified. The public site serves this app and its receipts separately from the synthetic correction replay.

Install with `python3 install.py --client both`. In a disposable project, ask Codex with `$usual` or Claude Code with `/usual`:

> Build a small local reading list where I can add a title and URL, mark an item read, and filter the list. Use Usual to handle implementation choices. Keep it local, run the relevant checks, and show me the decision report when you're done.

Use the default private corpus only with the user's authorization to use their history. For a shareable run, import only the supplied synthetic fixtures into an isolated database, tell the agent the database path, and keep the run scope consistent. The agent should produce actual working code, cite real evidence IDs in its decisions, test the code, finish the run, and give the user the private review command.

The observable proof is the generated application **and** its decision receipts. A transcript replay alone is not live-agent validation. Record the exact tested client/version and test scope before claiming compatibility was exercised end to end. Running external model clients may consume subscription quota or incur API charges; obey the current user's spending constraints.

After the user corrects a specific decision in the private review UI, give the agent a second task where that correction matters. Check the new consultation's evidence and recorded choice. Do not self-endorse the first run to make the demo look successful.

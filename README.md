# Usual

**Your AI should know how you work.**

[Website](https://usual.polyfeeds.dev) · [Learn Usual](LEARN.md) · [Source](https://github.com/pauljump/usual)

Usual analyzes your local Claude and Codex conversations for the moments when an agent
asked you to choose. It preserves the question, offered alternatives, your actual answer,
and the surrounding context. The next time a similar decision arises, your agent can
consult that history, make a reasoned call, and show you what it decided afterward.

Tell your coding AI:

> Learn Usual: https://usual.polyfeeds.dev/learn.md

It installs the skill and walks you through learning from your past decisions. Then:

> Build me a reading-list app with Usual.

Usual helps your agent make the routine choices and gives you a decision review afterward.

Your old answers already contain useful evidence. You do not have to label every past
conversation again. Your review of **new** predictions closes the loop and makes future
evidence better. Usual retains context and exceptions rather than turning every
“Yes” into an unconditional rule.

MIT licensed. Mining, storage, retrieval, and review run locally without a Usual
account or hosted inference service. **Your current coding model supplies the reasoning.**
Usual makes no model API calls, trains no weights, and does not claim that local
storage makes your coding model's inference offline.

## Install and use

Requires Python 3.11+ and a local Codex or Claude Code installation. The runtime uses only the Python standard library. No extra model account, pip dependencies, or Usual API key.

The [learn guide](LEARN.md) gives your agent the installation steps. Once installed,
start with `/usual` in Claude Code, `$usual` or the skill picker in Codex,
or simply “use Usual.” The first conversation offers recent history, all local
history, or starting without history; shows supported patterns with examples; then
helps you start a build. You can skip mining and learn through future reviewed decisions.

| Mode | How your agent works |
| --- | --- |
| **Autopilot** (default) | Makes routine reversible choices; labels assumptions; gives you a review afterward. |
| **Check-in** | Asks before each material judgment call; carries out choices you've already agreed. |
| **Escalation** | Uses applicable history; asks when evidence is missing, weak, or conflicting. |

Say “use check-in mode for this build” or “make escalation my default.” Usual
stores the mode locally. Changing a default affects new builds; an active run keeps
its own mode unless you change it explicitly.

**Every mode asks before deleting existing files, data, or resources unless you've
already given permission for that deletion.** Spending, publication, sharing, and
credential changes also require current authority. Past approvals are never permission
for a new action. These are instructions and ledger gates for the calling agent;
Usual does not intercept shell commands or replace client permission controls.

Client invocation follows the official [Codex skills documentation](https://developers.openai.com/codex/skills)
and [Claude Code skills documentation](https://code.claude.com/docs/en/skills).

<details>
<summary>Manual installation and history commands</summary>

Obtain this repository, inspect the installer, and run:

```bash
python3 install.py --client both
```

Use `--client codex` or `--client claude` for one client. The installer copies a
self-contained skill into `~/.agents/skills/usual` and/or
`~/.claude/skills/usual`. Existing versions are backed up outside skill discovery.
Moving the source folder afterward will not break the skill. No global client settings
or permissions are changed. Reopen the client if the skill does not appear.

```bash
python3 scripts/usual.py onboard
python3 scripts/usual.py mine --provider both --all --dry-run
python3 scripts/usual.py mine --provider both --all
python3 scripts/usual.py mode --set autopilot
```

The agent uses its installed script's absolute path. `--all` includes Codex archives
and main-session Claude transcripts; omit it for a recent sample. Unchanged files are
skipped on later runs. Use `--force` after a miner upgrade.

</details>

Usual links native question controls to the actual human replies, including Claude's
`AskUserQuestion` and Codex's synchronous/asynchronous question tools. It also retains
adjacent assistant-question/user-reply pairs as candidates for interpretation. Short
answers such as “yes” retain the question that gives them meaning.

Each private episode includes source project, date, file and line references, options,
answer, and the next assistant statement when available. That follow-up is observed
context, not proof an action succeeded. Exact matching can identify a selected option;
freeform answers stay in your words. The agent interprets reasons and exceptions.

```bash
python3 scripts/usual.py episodes --search "storage" --limit 10
```

Mining is explicit and repeatable, with per-file checkpoints and reversible retirement
of candidate links invalidated by source reprocessing. It never modifies original
transcripts. Reports expose exclusions, malformed/oversized records, cancelled questions,
and import errors. Deterministic secret redaction runs before storage; it does not remove
all potentially private information. See [runtime semantics](references/runtime.md).

## From Itchy to Usual

Itchy started with a small question: how useful could a model be if it focused on one
narrow job? Building with coding agents made that job concrete. The agent kept asking
for decisions its user had already made in earlier conversations.

Usual carries those decisions forward. It finds the original question and human answer,
keeps the context, and gives your current agent evidence for the next call. Your
corrections improve what it can draw on next time.

The name changed; the thread stayed the same: make a focused tool useful through
feedback. Usual uses your existing coding model rather than the original Itchy weights.
The [original research and its results correction](archive/itchy/README.md) are preserved
in the archive. [Read the evolution](references/evolution.md).

## Already using Whetstone?

Run the new installer. It installs `/usual`, backs up the old skill, and preserves your
local history under `~/.usual`. Existing database paths keep working through a
compatibility link. If both old and new data directories exist, neither is overwritten.
See [migration details](references/migration.md).

## What actually runs

The **current coding model** supplies judgment. Usual is its local evidence and audit tool, not a second model or a set of fine-tuned weights. Using the skill still consumes the coding client's normal model usage. The runtime itself makes no inference calls.

Each task gets a durable run ID. The skill consults material implementation questions and records one of:

- **Prediction:** a proposed user-like choice with citations to retrieved evidence.
- **Agent default:** a visible low-confidence assumption for a reversible, in-scope choice when evidence is insufficient.
- **Escalation:** something needing the current user's input or authority.

Evidence retrieval is lexical and confidence is the agent's qualitative assessment. Neither is an accuracy percentage. Quotes can be context-dependent or contradictory; the skill must interpret them and follow current instructions. Usual does not execute actions or override the client's permission system. It cannot convert an old “yes” into permission to publish, spend, delete, or share now.

## Review and learn

At the end, the agent returns its report and a `review-ui --run RUN_ID` command. The private local review page shows the choice, rationale, source quotes, and review status. Choose **That's my call**, **I'd choose differently**, or **Don't learn this**.

Historical human replies enter as observed decision episodes. For new agent predictions, only explicit acceptance/correction adds project-scoped endorsed evidence. Unreviewed and rejected predictions are never fed back as human choices. The original prediction survives correction. Retired evidence is excluded from future retrieval while old receipts remain intact. The local confirmation mechanism is a contract with the calling skill, not separate human identity verification against another process on the same computer.

## Data and recovery

The default store is `~/.usual/judgment.sqlite3`, outside project repositories. SQLite uses private file permissions, transactions, WAL, foreign keys, and a busy timeout. Native transcript files are never modified. Common secrets are redacted before persistence; this is not complete PII detection. Evidence supplied to Codex or Claude is processed by that provider under its normal data handling.

The private review server binds to 127.0.0.1 and requires a fresh in-memory capability for data access. It checks Host/Origin and does not allow cross-origin data access. Never tunnel it. The public website serves only code downloads and synthetic demo data, not a private corpus API.

Use `backup /new/private/file.sqlite3` for a consistent snapshot; existing backups are not overwritten. Use `status` and `report --run RUN_ID` to resume interrupted work. See [runtime semantics](references/runtime.md) for import limits, scopes, retirement, and review behavior.

The public hosting layer has existing Pulse and Cloudflare visit/performance analytics. The CLI and private review UI contain no analytics and do not upload transcript data to the website.

## Demo and validation status

This is a **local beta**. The website source bundle includes guided setup, the decision-history miner, and autonomy modes. The local runtime and clean installation have been tested; live model compliance and prediction accuracy across users are not established by these checks. The public walkthrough is a labelled fixture replay: synthetic native transcripts and prewritten example choices pass through the real importer, retrieval, ledger, and review code. It does not claim a live model generated the example choices.

```bash
python3 scripts/run_demo.py --out /tmp/usual-demo.json
```

It verifies native imports, duplicate handling, source citations, permission escalation, isolation of unreviewed predictions, and retrieval of a correction in the next run. JSON and HTML receipts are generated. See [the end-to-end demo protocol](references/demo.md) for live-client testing and the distinction between a replay and an actual generated application.

Validation covers durable runs, source attribution, idempotent and concurrent recording, run closure, explicit human review, project isolation, backups, redaction, review authentication, native transcript role filtering, and clean installation. Live client/version results must be recorded separately; a test suite alone does not prove unattended production readiness or decision accuracy across users.

## Development and release

```bash
PYTHONPATH=src python3 -m pytest -q
```

From the repository root:

```bash
python3 scripts/usual.py --help
python3 scripts/build_release.py
PYTHONPATH=src python3 -m usual.server --autopilot-public --port 8794
```

The release builder uses an explicit code-only allowlist and emits a SHA-256 manifest. It never packages private SQLite files, local reports, credentials, or real transcripts. Public deployment uses the `usual.polyfeeds.dev` Cloudflare Tunnel route, process `usual-web`, port 8230, through the control-plane fleet registry/vault runner. See [deploy/web.json](deploy/web.json).

The earlier consumer onboarding and experimental studio remain local legacy interfaces. The old blinded-decision benchmark is historical, single-person research; it is not the production autopilot's accuracy score. Its original implementation remains in the repository's exam/grade modules and Git history.

Skill format references: [Codex](https://learn.chatgpt.com/docs/build-skills), [Claude Code](https://code.claude.com/docs/en/skills).

MIT licensed. See [LICENSE](LICENSE).

## Recorded build

The public site also offers a runnable reading-list app at `/reading-list.zip` and its actual pre-implementation decision receipts at `/build.json`. Codex built it in this development session using synthetic transcript evidence. Three backend tests pass; all three predictions remain unreviewed. See [the recorded build](demo/reading-list/README.md) for reproduction and precise validation limits.

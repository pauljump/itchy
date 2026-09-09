# Runtime and data contract

Usual v2 is a local, single-user tool for Codex and Claude Code. Python 3.11+ and SQLite are the only runtime dependencies. The installed skill is copied into each client's skill directory, with its runtime alongside it.

`python3 /path/to/skill/scripts/usual.py --help` lists commands. Each command writes one JSON result; errors go to stderr and return nonzero. `report` and `finish` also support `--format markdown|html`. Redirect private reports outside public repositories. `doctor` checks the SQLite file and skill discovery paths. `status` lists recent runs for resume.

## Decision history

`mine --provider both --all --dry-run` inventories main-session Codex JSONL (including
archives) and Claude project transcripts. Omit `--all` for the newest bounded selection.
`mine` streams files with a 16 MiB per-line bound rather than dropping whole large files.
Malformed/oversized records, tool cancellations, subagents, and unknown answer links are
counted. Unchanged files are skipped; `--force` reparses after a parser update. Imports
are transactional per file, and original transcripts are never modified.

Native Codex synchronous/asynchronous question responses and Claude AskUserQuestion
results are linked by their actual call IDs. Plain assistant questions followed by user
replies are retained as `adjacent_reply` candidates. Short answers retain their question.
Internal analysis, tool output, injected instruction wrappers and subagent instructions
are not treated as authored answers. Event-only Codex user records currently contribute
coverage counts but are not linked without native response-item messages.

Each redacted episode retains question/options, answer, selected options only when the
answer matches exactly, source project/date/lines, and the next assistant statement when
available. That statement is context, not a verified execution outcome. The evidence store
uses `origin: episode`; it is observed history, not an inferred universal preference or
an endorsed future-agent prediction. Public output never includes the private corpus.

Historical answers need no second review to inform the current agent's reasoning.
Predictions made now still require explicit review before becoming endorsed evidence.
`episodes --search TEXT --limit 10` inspects the private records. Retirement of their
associated evidence excludes them from retrieval and inspection without destroying receipts.
Default global retrieval retains original project context; do not assume a project-specific
choice is a universal rule. There is no hidden inference call, automatic fine-tuning, or
prediction-accuracy percentage supplied by the miner.

## Evidence

Supported imports: Codex rollout JSONL (`response_item` and fallback `event_msg`), Claude Code session JSONL (`user`/`assistant`), ChatGPT and Claude web JSON exports, and generic role/content JSONL. It excludes tool results, sidechains, system/developer messages, known provider instruction wrappers, code fences, block quotes, and short bare approvals. Format drift and quoted instructions can still affect classification: observed quotes are evidence to interpret, not verified general principles. The importer reports partial failures, malformed records, and duplicates. Files are bounded at 32 MiB; each import is bounded at 100 files through `init`.

Import defaults to `global` evidence scope. Give selected files `import --scope /project/root ...` to keep them within one project. Runs query their own scope and global evidence. Source filenames, line numbers, dates, exact redacted human quotes, and preceding assistant context are retained. Raw transcript files are never modified. Imported statements are heuristic candidates and may be irrelevant; there is no claim that the entire history is understood.

Retrieval uses weighted lexical overlap, at least two matching terms, and a small boost for explicitly endorsed choices. The current coding model reasons over the retrieved evidence. It must resolve negation, exceptions, and conflicts; Usual's retrieval code does not certify semantic support. User instructions and repo constraints remain current authority.

## Decisions and review

Every consultation snapshots its sources. Every choice references exactly one consultation. Retrying the same `record` is idempotent; changing an existing decision is refused. Runs cannot finish with unresolved consultations, and completed runs cannot acquire new decisions. Human reviews append an event; corrections preserve the original prediction. Accepted/corrected choices add evidence in that run's project scope. Predictions, defaults, and escalations remain separate. Confidence labels are agent assessments, not calibrated measurements.

Permission gates use explicit kinds plus conservative word matching. They can over-trigger or miss a phrasing; they are not a security classifier. The agent's own permission controls remain authoritative. Usual never executes an action from a choice. The CLI's user-review confirmation flag is an explicit contract with the calling agent, not proof of a separate human identity. A process with access to the same OS account can modify local files; this is not a multi-tenant authorization boundary.

## Privacy and recovery

SQLite runs with foreign keys, WAL, transactions, a busy timeout, and private file permissions. Use a private directory, not a synced/public repo. Common secrets are scrubbed before persistence; redaction isn't a complete PII detector. There is no provider call in the runtime, but the agent sees retrieved evidence and its provider processes it. No background collector, telemetry, or sync is included in the CLI or private review UI.

`backup` uses SQLite's consistent backup API and refuses overwrite. Stop work and restore a backup to a new private path if needed, then pass that path through `--db`. `retire` is exclusion from future retrieval, not erasure from old audit snapshots. Permanent erasure requires removing the private database and any backups/reports with the user's explicit intent.

The review server binds only to 127.0.0.1, verifies Host/Origin, and requires a fresh in-memory bearer capability for reads and writes. The link carries the capability in its fragment; it is removed from browser history after load. Do not publish or tunnel that server. Its capability is not persisted. A browser refresh requires reopening the original complete link.


## Onboarding and modes

`onboard` (also the no-argument CLI entry) returns available native-history counts,
existing evidence counts, the saved mode, and a suggested next step. It inventories
filenames without reading transcript contents or importing them. It can initialize
an empty private database. The conversational skill handles history selection,
source-grounded analysis, and the transition to an actual build.

`mode --set autopilot|check-in|escalation` persists the default for new runs.
`start --mode MODE` overrides it for one run; `mode --set MODE --run RUN_ID`
changes an active run. Closed runs cannot be changed. Existing databases are extended
with additive settings/run-mode tables; older runs retain autopilot semantics.
Reports include the run mode, and mode changes appear in the event ledger.

Check-in gates every material consultation. Escalation gates a consultation with no
retrieved evidence or with `--uncertain`; it also refuses agent-default and low-confidence
prediction records. The agent must still examine applicability and contradictions:
lexical retrieval cannot establish either. Autopilot permits low-confidence reversible
defaults. Current user input/permission is recorded as `escalated`, with a rationale
that distinguishes an actual answer from an unresolved question. It is not a prediction
and cannot be endorsed as learned permission in the review UI.

All modes retain permission gates. The text guard conservatively catches common deletion
and destructive-command wording even under an implementation kind; it can overmatch
(e.g. discussing a delete button) and is not a semantic classifier or an execution
sandbox. The skill must recognize actual destructive actions even if the text guard
misses them. A `requires_user` consultation never becomes delegated by changing mode;
recording an escalation does not prove or grant user authority.

# Whetstone — session handoff

Updated September 8, 2026.

## Product direction

Whetstone is an **open-source, local autopilot for vibe coding**, aimed at Codex and Claude Code users. It imports explicit choices from local chat transcripts, retrieves relevant evidence during a coding task, and records the current agent's judgments before implementation. Users review those judgments afterward; explicit corrections inform future consultations.

The lead positioning is **“Open source. Runs locally.”** Keep MIT licensing, inspectable source, local decision storage, and no Whetstone account prominent. Be precise: Whetstone's import, retrieval, storage, and review run locally; the invoking coding model processes retrieved evidence under its provider's normal policies. This is not offline model inference.

The site was redesigned around warm paper, dark typography, restrained blue, simple rules, and a shorter explanation. Preserve the directness: no fake terminal, repeated slogan sections, or decorative card grids. The walkthrough and three-step installation are the main product explanation.

## Shipped and installed

- Canonical source: this repository (`whetstone-release`), remote `pauljump/whetstone`.
- Public site: https://whetstone.polyfeeds.dev, replacing the previous consumer onboarding page.
- Deployment: existing Cloudflare Tunnel to port 8230, PM2 process `whetstone-web`, public-only `--autopilot-public` mode. [deploy/web.json](deploy/web.json) records the command and ownership.
- Current release: **2.0.0b1, local beta**. Do not present it as production validated.
- Self-contained skills installed at `~/.agents/skills/whetstone` and `~/.claude/skills/whetstone`. Installer backs up prior installations outside skill discovery.
- Public routes serve the landing page, health, release manifest, source ZIP, synthetic replay, and recorded sample build. Private corpus, ingestion, model execution, and review APIs are not exposed publicly.

The earlier consumer onboarding and experimental studio implementations are preserved as legacy local interfaces. The public process does not serve them.

## Evidence and review semantics

Native Codex and Claude transcript import is bounded, deduplicated, role-aware, and redacts common secrets. Extraction and lexical retrieval are heuristic; source quotes are evidence, not instructions or verified universal preferences. The current coding model makes the prediction; Whetstone does not invoke a second model or train weights.

SQLite stores durable runs, consultation snapshots, decisions, append-only reviews, and events. Predictions require citations from their consultation. Unsupported reversible assumptions are explicitly labelled low-confidence agent defaults. Permission gates are recorded as escalations and do not expand the host's authority.

Pending and rejected decisions never become new preference evidence. Only explicit human acceptance or correction does. Do not self-endorse predictions. The localhost review UI uses a temporary capability with Host/Origin checks; the CLI review confirmation is a calling-agent contract, not separate human identity verification.

See [the skill](SKILL.md) and [runtime semantics](references/runtime.md) for the operational workflow.

## Demo and validation

There are two distinct public demonstrations:

1. `/demo.json`: synthetic transcripts and prewritten choices passed through the real engine, including a synthetic correction and its retrieval in a subsequent run. This is a fixture replay.
2. `/reading-list.zip` and `/build.json`: a working local reading-list app built by the current Codex development session. Three judgments were recorded before implementation using synthetic history. All remain pending human review. This exercised the CLI workflow, not independent client skill discovery.

Validation completed in this session:

- Main suite: **129 passed, 5 skipped**.
- Recorded app: **3 backend tests passed**, covering persistence across restart, private API access, and input validation.
- Clean source-download installation for both clients, including paths with spaces, moving the original checkout, and recoverable reinstallation.
- Durable/concurrent logging, source attribution, project isolation, correction retrieval, retirement, backup, redaction, and private-review access controls.
- Skill format validation, JavaScript syntax, navigation anchors, and Git whitespace checks.
- Public health and route isolation, demo downloads, current source bundle contents, MIT license, and published SHA-256 hashes.
- The release tests were rerun after the visual and open-source/local positioning edits: **2 passed**.

Remaining before claiming production readiness:

- Independent Codex and Claude Code runs that discover and follow the installed skill, build an artifact, and return the decision report. The proposed test is one run per client capped at five minutes, using existing logins without API-key fallback. Approval to consume quota was requested but **not received**; neither external client run occurred.
- Browser visual and interaction QA on desktop and mobile. CUA reported no connected browsers, so this was not performed.
- Broader real-user evidence and prediction-quality evaluation. Test success does not establish judgment accuracy or reliable unattended operation.

## Private session data

Real user history and reports remain outside the repository, in `~/.whetstone`. The bounded import processed 19 files; one oversized transcript was skipped and nine malformed lines were reported. A long pasted-document candidate exposed a retrieval weakness; extraction and ranking were tightened and that candidate was retired while preserving its audit trail.

The private packaging run is `run_b41546e99cd84757a316`, now complete, with one prediction and one diagnostic escalation. Neither has been endorsed. Its report is `~/.whetstone/reports/whetstone-packaging.html`. Do not package this report, the private database, or real transcripts in public releases.

## Resuming and releasing

Use the commands in [README.md](README.md) for development, installation, and release generation. `scripts/build_release.py` builds the source ZIP and sample-only public assets from an explicit allowlist. Rebuild after changing bundled source; check the manifest against the public download. The public server reads the assets from the canonical checkout.

The source ZIP is the current beta source distribution. GitHub was inspected during the session and still showed earlier product code; no push was requested or performed. Closing-session commits are local. Deployment records in the control-plane and kit repositories contain other projects' unrelated edits; commit only Whetstone's portions.

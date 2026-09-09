# The first conversation

The user should leave setup knowing what Usual learned, when it will ask, and how to start a build. The agent runs the commands; the user should not need to copy terminal commands.

## Arrive

Run the installed script's `onboard` command. It inventories filenames and counts stored evidence without reading transcript contents or importing anything. Say briefly:

> Usual helps me make routine build decisions using choices you've already made. I'll show you what I learned, then use it in your next build.

If history is already imported, use it. Do not make a returning user repeat setup. If the user has already selected a history scope in this conversation, honor it. Otherwise offer one short choice: learn from the most recent 20 local conversations, all local conversations, or start without history. Name only the clients found. Explain that the local store stays on this machine and selected excerpts are read by the current coding agent under that client's normal usage and data handling.

No native history found? Say so and continue without history, or use a specific supported local export the user supplies. Never claim to have read cloud ChatGPT/Claude chats that are not present locally. Never turn choosing “start without history” into another setup questionnaire.

## Learn

For recent history, run `mine --provider both --limit 20 --dry-run`, then the same command without `--dry-run`. For all history, use `--all` in place of `--limit 20`. Follow SKILL.md's mining procedure. Show a concise result count and any meaningful parsing or coverage limitation.

Inspect `episodes --limit 12`, then search for specific topics and contrary examples. If only standalone quotes exist, inspect `evidence`. Present up to three supported patterns in this form:

> When [specific context], you chose [choice]. I'd use that as a starting point for [similar future decision]. [Source/date]

Keep these as contextual hypotheses. Include an exception when one exists. Do not fill three slots with weak guesses, describe extracted counts as learned rules, or ask the user to relabel their old answers. Offer correction without making a response a gate to the requested build. If no defensible pattern exists, explain that future choices will be explicit defaults until evidence develops.

## Build

Default to **autopilot**, unless a saved mode or the current request says otherwise. Explain it once:

> I'll make routine reversible choices and give you a decision review afterward. I'll ask before deleting anything or taking actions that need your permission. Say “check-in mode” to approve each material choice, or “escalation mode” to have me ask whenever your history doesn't give a clear answer.

Respect the full mode and permission rules in SKILL.md. A selected default is saved with `mode --set MODE`. The user can simply say:

> Build me a reading-list app with Usual.

Claude Code also accepts `/usual build me a reading-list app`; Codex CLI/IDE use `$usual build me a reading-list app`. Use the current client's actual invocation syntax, not a fabricated common slash command. If a build task is already supplied, begin it immediately. Otherwise end with one short invitation to describe what to build. Do not run `start` until there is a real task.

On later builds, skip this walkthrough, announce the saved mode, and follow the coding loop. Refresh history only when requested; there is no background watcher.

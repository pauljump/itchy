> **Archived intermediate experiment.** The active product is [Usual](../../README.md). The commands and integration names below are historical; this experiment is not part of the Usual installation.

# Itchy

**A tiny model that learns your judgment.**

Your coding agent keeps reaching decisions you have made before: how much complexity
is worth adding, which tradeoff you would choose, what “finished” means for this project.
Itchy's purpose is to learn from those choices so your agent gets better at anticipating
your answer.

[Whetstone](https://github.com/pauljump/whetstone) provides the feedback loop: the question,
the alternatives, the agent's prediction, and your explicit review. **Itchy is the
experimental learner for that loop.**

```text
Agent reaches a decision
        ↓
Whetstone records the question and alternatives
        ↓
Itchy ranks the alternatives using reviewed choices from this project
        ↓
Current agent considers the suggestion and records its actual decision
        ↓
You accept, correct, or reject it in Whetstone
        ↓
Explicit reviewed choices become the next training dataset
```

The goal is a small personal judgment model you own and can bring to different agents.
The first implementation runs locally, with NumPy and no pretrained model downloads.
**It is an experimental contextual ranker, not a demonstrated substitute for human judgment
or a trained language model of its owner.**

## What works today

- Read a Whetstone autopilot database **without writing to it**.
- Count usable reviewed choices separately from observed transcript quotes and guesses.
- Learn to rank **new option text in context**, rather than assign a fixed category label.
- Keep whole runs and repeated questions together across train/calibration/test splits.
- Evaluate agreement with the actual reviewed choice, then abstain below a measured margin.
- Produce a private, portable model and **shadow suggestions** for real Whetstone consultations.
- Respect Whetstone's gates and project scope; reject a model when its reviewed source data changes.

Shadow mode means Itchy makes a suggestion alongside the current workflow. It does not
record the agent's choice, accept a review, change a skill, execute actions, or grant
permission. Whetstone's current agent continues supplying reasoning and checking current
instructions. The integration is an explicit CLI call; it is not automatically installed
into Whetstone's coding loop.

## Start with your real evidence

```bash
git clone https://github.com/pauljump/itchy.git
cd itchy
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .

# Defaults to ~/.whetstone/judgment.sqlite3; prints aggregate counts, not quotes.
itchy whetstone inspect

# Keep the model outside source control; use the scope Whetstone recorded.
itchy whetstone fit \
  --scope /absolute/project/root \
  --output ~/.itchy/project-judgment-v1
```

If you have only imported transcripts and unreviewed predictions, the answer is
`insufficient_reviewed_choices`. Itchy does not manufacture preference labels from
those quotes. Review actual decisions through Whetstone's existing review workflow,
then fit again. A failed or insufficient fit exits 2; it never activates itself.

The model needs at least 20 training choices, enough calibration/test observations to
meet the minimum support, and enough correct accepted predictions to meet the requested
Wilson lower bound. The default target is 0.90 with support floor 30; 30/30 correct is
still insufficient for that lower bound. Entire runs stay together, so one large run
cannot stand in for several independent held-out runs.

```bash
# Explicit opt-in shadow query against an existing consultation.
itchy whetstone suggest \
  --artifact ~/.itchy/project-judgment-v1 \
  --consultation ask_ID_FROM_WHETSTONE
```

For a non-default database, put `--db /private/judgment.sqlite3` immediately after
`whetstone`. All commands emit JSON. `python -m itchy` also works.

## The choice is contextual

“Local SQLite” can be the right choice for a private prototype and the wrong choice for
a collaborative service. The ranker sees the question, decision kind, and each option's
text. Its first learner combines byte n-gram features with hashed question–option word
interactions and learns a score over each available option.

Option index is not a feature. Reordering the same alternatives preserves their scores.
The model does not generate rationales or explanations; the current agent must still
interpret the situation. Lexical overlap is a limited representation of judgment. A
passing fixture test establishes mechanics, not semantic understanding or personal accuracy.

## What counts as your choice

An example must have an explicit accepted/corrected review, matching active endorsed
evidence, and an eligible advisory decision kind. Pending, rejected, retired, escalated,
permission, spending, credential, destructive, and scope decisions are excluded.

For a correction, the user's statement must identify one offered alternative unambiguously
by normalized exact match. “It depends” is useful evidence for Whetstone, but not a winner
label for this learner. Itchy reports it as `needs_choice_mapping` and skips it. No model
silently interprets freeform corrections as a confirmed choice.

Models are project-scoped. When the source reviews change or are retired, an existing
artifact refuses suggestions until retrained and re-evaluated. This alpha does not pool
people's data or silently generalize one project's preferences across all projects.

## What has been verified

Tests cover reviewed-only ingestion, read-only database access, freeform correction
handling, retirement, scope isolation, host gates, option-order invariance, contextual
preference learning, and portable model loading. The live Whetstone schema was inspected
and the bridge was exercised against an actual local store. That store had **no eligible
reviewed choices**, so no personal model was trained and no personal prediction-accuracy
number is claimed.

The earlier generic classifier remains as a [supporting experiment](docs/classifier.md),
with a [reproducible public SMS benchmark](docs/benchmark.md). **SMS accuracy says nothing
about predicting a person's judgment.** It is not the evidence for this new direction.

See [the direction and next proof](docs/direction.md) and
[the preference evaluation contract](docs/whetstone.md).

## Why Itchy

Itchy began with a 16MB byte-level language model and the idea “build it the right size.”
The question now is: **how small can a model be when its job is learning one person's
repeated choices?** Whetstone supplies the missing training-and-correction loop.

The original language-model BPB scores were invalidated by future-byte leakage. Historical
scripts and notebooks remain available with a [clear correction](docs/research-correction.md).
This preference learner does not use those weights or those scores.

## Development

```bash
python -m pip install -e . pytest
python -m pytest -q
```

Requires Python 3.10+ for Itchy; Whetstone itself requires Python 3.11+.
`pyproject.toml` defines the new package. The old root `requirements.txt` belongs to the
historical LM experiments. The package is installed from source; no PyPI release is
announced. The package metadata name is `itchy-local`, and command/import are `itchy`.

Private models contain learned information and project scope metadata. Keep them and
Whetstone's database outside public repositories. Itchy makes no network or provider
calls. MIT licensed.

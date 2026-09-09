# Direction: a small personal judgment model, learned through Whetstone

Updated 2026-09-08 after the owner selected Whetstone as the intended application.
Status: local experimental implementation; public publication pending.

## The product

Whetstone captures and reviews judgment. Itchy learns from those reviews. Each person
can own a small model that becomes better at anticipating the choices they make while
working with an agent. That is the daily-use hypothesis: fewer repeated preference
questions, fewer unwanted implementation choices, and a visible correction loop.

The coding agent remains the general problem solver. Itchy answers a narrower question:
**given these alternatives and this context, which one is this person likely to prefer?**

The original Itchy thesis—give a narrow job an appropriately small model—fits this
application. The repeated work is contextual preference prediction. Whetstone already
has the question, options, proposed answer, source context, and eventual human verdict.

## Separate responsibilities

- Whetstone: evidence, consultation, decision receipt, explicit review, source retirement.
- Itchy: dataset eligibility, context-conditioned option ranking, held-out evaluation,
  private model artifact, and an abstaining shadow suggestion.
- Current agent: reasoning, current user instructions, project constraints, actual action.

These are separate public projects with a read-only interface. No Whetstone source,
installed skill, live service, or permission setting was changed for this prototype.

## Why the first generic classifier is not the product

Ticket categories have a fixed label set. Agent tradeoffs present new alternatives on
every question. A classifier trained on labels such as local/hosted would miss both
context and the open-ended nature of those alternatives. The new learner scores option
text conditioned on the question and kind; its outputs are invariant to option order.
The generic classifier code stays available as a baseline experiment, not the main pitch.

## Current implementation and its limits

The prototype uses byte n-gram option features and hashed word-pair interactions between
question and option, with a linear listwise ranking objective. It is inexpensive to train
locally and can represent opposing preferences for the same alternatives in different
lexical contexts. It is not a semantic foundation model or demonstrated personal judgment.

Only explicitly reviewed, active, unambiguous advisory choices become labels. Observed
quotes are available to Whetstone's reasoning but are not automatically a training target.
A freeform correction that does not identify an offered alternative remains unmapped.

The inspected live store had no eligible reviewed choices. That is an actual cold start,
not an excuse to train on the agent's guesses. The bridge reports missing data and writes
no model. We have verified the mechanics with fixtures and the read path against the live
schema; we have not established personal accuracy, avoided interruptions, or daily adoption.

## Next proof that matters

1. Collect explicit reviews through normal Whetstone work, including corrections and
   context-specific exceptions. Do not create fake dilemmas to inflate the count.
2. Hold out whole later runs. Group repeated questions/options across runs so the same
   question cannot appear as both training and novel evaluation.
3. Compare Itchy against a simple preference-frequency baseline and Whetstone's existing
   retrieval-assisted agent predictions, using the actual reviewed winner as ground truth.
   No new paid teacher calls are required to evaluate already-recorded predictions.
4. Measure agreement at each coverage level, error by kind, option-order stability,
   review effort, and disagreements where the general agent was right but Itchy was wrong.
5. Keep it advisory until prospective evidence shows it helps. A larger local encoder or
   fine-tuned small LM is a future learner candidate only if the collected failures justify it.

A synthetic preference test or the old SMS benchmark cannot settle this question. The
enduring asset is a person's reviewed, contextual decisions and a portable learned model
built from them. There is no claim that this alpha has already achieved that end state.

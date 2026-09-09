# Whetstone bridge and preference evaluation

The adapter targets Whetstone autopilot database schema 1, verified against the local
v2 code. It opens SQLite with `mode=ro` and `query_only`, reading a consistent transaction.
It never writes reviews, imports transcripts, starts a run, or executes a decision.

## Eligible labels

Join the decision, consultation, run, explicit review event, and active endorsed evidence.
Require accepted/corrected status to agree with the review event. An accepted example uses
the original choice. A corrected example uses the review statement, only if it maps to
one offered option by Unicode-normalized, case-folded exact match. Reject ambiguous joins,
retired evidence, pending/rejected predictions, freeform unmapped corrections, escalations,
host gates, and kinds outside implementation/design/dependency/testing.

`inspect` prints aggregate counts. Observed quote counts are store-wide and explicitly
not training labels. `fit` requires one explicit project scope and never pools across scopes.

## Split and train

Link each example's run ID to its normalized question + sorted options + kind. Connected
components stay together, transitively; two runs sharing a question cannot leak across
splits through another question. A stable hash assigns components approximately 60/20/20.
Repeated identical question/answer examples count once. Conflicting answers stay together
and remain visible to the learner; they are evidence of context loss or changed preferences.

Train the listwise choice ranker on training examples only. The model sees question, kind,
and option text—not decision IDs, human-review text, original agent rationale, correctness,
or original agent confidence. Calibration selects a margin from a fixed grid. Test applies
the frozen margin and cannot change it. Eligibility requires minimum accepted support,
Wilson lower bound, and at least 10% coverage on the held-out choices.

These Wilson intervals assume independent, representative choices. Grouped splitting avoids
cross-split leakage but does not eliminate correlation between multiple choices inside one
run. The current interval is descriptive, not a formal guarantee across correlated choices.
Fresh chronological evaluation is needed before a production claim. Repeatedly tuning to the
same test set invalidates its status as an independent test.

## Serving contract

A shadow suggestion reads an actual consultation and first honors its existing host gate.
The artifact must match the consultation's scope and current eligible-review fingerprint.
New or retired reviews invalidate the artifact. The ranker's margin is a ranking signal,
not calibrated probability of correctness. A failed candidate or low margin returns no
choice; a passing one still returns `mode: shadow_only`, never an authorized action.

Current instructions and the agent's permission system always take precedence. A host gate
is not a claim that every consequential action has been detected. The ranker is not a safety
classifier and cannot convert a learned preference into permission.

Artifact directories are created with private permissions; numeric arrays are loaded without
pickle and verified by checksum. Artifacts include scope metadata and learned information.
They are private user data, even though raw transcripts are not copied into them.

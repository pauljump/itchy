# Evaluation contract

Itchy answers a finite-label classification problem, with abstention. Its first learner
is not a generative LM and no BPB score is relevant to it.

## Three separate jobs

- **Train:** fit the byte-feature linear classifier. Only these rows determine weights,
  bias, and the feature-presence mask used by the familiarity heuristic.
- **Calibration:** choose the lowest passing threshold from the fixed grid
  0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99 for each predicted label. A qualifying
  subset meets both the minimum support and the requested Wilson lower bound.
- **Test:** apply those frozen thresholds once. Never pick another threshold or remove
  a failing label based on test performance. A failure blocks the candidate.

A label that never qualifies in calibration always abstains. A label that qualifies
there but fails the independent test blocks promotion for the entire candidate.
`min_support` counts accepted predictions of a label, not total rows with that truth label.
The default 50 is only a floor: at a target lower bound of 0.95, 50/50 correct does
not pass. Approximately 73 error-free accepted examples are needed in each evaluated
split for that label, and errors increase the requirement.

## What to inspect

- **Coverage:** accepted / all test cases. Zero answers is not a useful deployment.
- **Accepted precision:** correct / accepted, globally and separately per predicted label.
- **Wilson lower bound:** the lower endpoint of a two-sided 95% Wilson score interval.
- **Ungated accuracy and confusion matrix:** model choices before threshold/familiarity
  rejection, so failures are not hidden by abstention.
- **Fallback quality:** not measured here. Local accepted precision is not combined-system
  accuracy, and fallback count is not automatically money saved.

Wilson intervals assume representative, independent observations. They are descriptive
on these data: they are not a calibrated per-input probability, a simultaneous guarantee
across labels, or a guarantee about an adversarial or shifted future distribution.
Calibration involves selection; the separate test is the check against that selection.

## Avoid getting a great score for the wrong reason

Use `group` to keep conversations, customers, documents, or generated templates together.
Automatic splitting by text alone cannot detect paraphrases. Exact normalized duplicates
are deduplicated across the full dataset and cannot be reassigned between splits.
Prefer explicit chronological partitions for changing production traffic. Verify every
label appears in each split; fitting rejects missing classes rather than hiding them.

Repeatedly changing a model after reading the same test results turns that test into
another development set. This alpha stores the exact dataset revision for each run,
but does not enforce a one-shot evaluation protocol. Obtain fresh reviewed test data
before claiming a production improvement. The synthetic demo shares templates on
purpose to exercise the plumbing and is not evidence of generalization.

Review accepted local decisions, not only uncertain cases. Track class coverage and
precision over fresh traffic, preserve fallback capacity, and retrain deliberately.
There is no automatic drift detector or automatic production rollback in this alpha.

## Privacy and persistence

`Task` saves raw text in a local SQLite database. It is a trusted, single-user local tool;
it has no authentication layer, retention daemon, or encrypted storage implementation.
Use filesystem access controls and an appropriate data-retention policy in a deployment.
The artifact export omits raw examples. Learned weights and labels still disclose
information about the task and must be reviewed before sharing.

Model writes go to a new UUID directory; promotion atomically replaces a small pointer.
Failed fits/promotions do not modify the champion. SQLite supervision imports and
corrections are transactional. Serialize training/promotion in the host application;
this alpha does not provide a distributed training coordinator.

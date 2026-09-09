# One fixed public-data check

Run on 2026-09-08. This is a retrospective smoke benchmark, not a new state of the art,
a prospective deployment trial, or a comparison against an LLM.

**Source:** Almeida, T. & Hidalgo, J. (2011),
[SMS Spam Collection, UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection),
[DOI: 10.24432/C5CC84](https://doi.org/10.24432/C5CC84). UCI identifies the dataset license
as CC BY 4.0. Its messages predate contemporary spam and combine several source corpora.

## Fixed procedure

Read 5,574 source rows, deduplicate normalized exact texts to 5,159, and exclude any
conflicting-label duplicate texts (none in this file). Group digit-normalized variants
before applying Itchy's stable 60/20/20 group hash. Result: 3,117 train, 1,057 calibration,
985 test. Other near-duplicates may remain; original conversation/sender IDs are unavailable.
The grouping reduces one leakage route but does not establish complete independence.

Use the defaults without changing them after seeing the test: 8,192 feature dimensions,
40 epochs, seed 0; target lower bound 0.95, minimum support 50, minimum coverage 0.10.
Calibration selected thresholds of 0.5 for both labels. The baseline always predicts
the majority label learned from the training split; it is a simple sanity baseline,
not a competitive tuned text-classification system.

## Result

| Measurement | Result |
|---|---:|
| Test decisions answered | 985 / 985 (100%) |
| Test decisions correct | 966 / 985 (98.07%) |
| Majority-label baseline accuracy | 87.72% |
| Predicted ham correct | 863 / 881 |
| Predicted spam correct | 103 / 104 |
| Ham precision Wilson lower bound | 96.79% |
| Spam precision Wilson lower bound | 94.75% |
| Promotion gate | **Failed**: spam lower bound < 95% |
| Training + evaluation + model write | 1.612 seconds |
| Compressed weights + metadata | 62,201 bytes |
| Standalone median / p95 inference | 0.032 / 0.069 milliseconds |

Eighteen spam messages were classified as ham, and one ham message as spam. Precision
is not recall: this would miss 18/121 spam messages. Inference timing covers one pass over
the 985 test texts with a loaded model, excludes SQLite logging and model loading, and
is machine-specific. Artifact size excludes NumPy/Python, runtime RAM, and the report.

The model is small and useful enough to evaluate further, but **this candidate is not
eligible for promotion under the policy used in the run**. We did not lower the target,
retune on test mistakes, or remove the failing label after reading the test result.
More independent evidence or a newly evaluated candidate is needed to qualify it.

[Complete JSON report](benchmark-sms-2026-09-08.json), including class counts, confusion
matrices, environment, file checksum, dataset revision, and all policy settings.

## Reproduce

Download and extract `SMSSpamCollection` using the download link on the
[UCI dataset page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
The benchmark script itself makes no network calls:

```bash
python -m examples.benchmark_sms /path/to/SMSSpamCollection --output /tmp/itchy-sms-report.json
```

The source file checksum for this run is
`7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d`.
Raw data, the SQLite task, and the trained model remain outside the repository and are
removed from the script's temporary directory after the aggregate report is produced.
Only this report and the reproduction code are part of the release.

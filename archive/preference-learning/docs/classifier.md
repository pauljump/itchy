> Supporting classifier experiment. The project's primary direction is now [personal judgment with Whetstone](../README.md).

# Itchy

**Teach a tiny local model the decisions you keep asking an LLM to make.**

A support queue. A stream of documents. An agent choosing from a fixed set of intents.
The wording changes; the decision repeats. Itchy learns that decision from reviewed
examples and gives you a small model you can run on a CPU.

```text
Reviewed examples → tiny model → calibration → independent test → explicit promotion
                        ↑                                              ↓
                    corrections ← review queue ← local answer or fallback
```

**Working alpha:** Python SDK + CLI, local training, per-label acceptance thresholds,
SQLite feedback history, and portable model artifacts. NumPy is the only runtime
dependency. No model downloads, provider accounts, GPU, or network calls.

## Try it in a minute

```bash
git clone https://github.com/pauljump/itchy.git
cd itchy
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m examples.demo
```

The offline demo trains three support-routing labels, checks a separate test split,
promotes the candidate, answers familiar requests locally, abstains on unfamiliar input,
records a reviewed fallback, and loads an exported artifact without its training database.
Its data is **synthetic with shared templates**: it demonstrates the software, not
real-world accuracy. The demo prints the measurements on your machine.

Installation is from this repository; `itchy-local` is the package metadata name, **not
an announced PyPI release**. The command and Python import are both `itchy`.

## A real-data check, including the failure

On one fixed, deduplicated split of the public UCI SMS Spam Collection, the default
learner got **966 of 985 test messages right (98.1%)**, answering all 985. It trained
in **1.61 seconds** and produced **62.2 KB of weights and metadata** on an Apple Silicon
Mac. The majority-label baseline got 87.7% right. This is a retrospective dataset check,
not an LLM comparison or a modern production spam benchmark.

**Itchy still blocked promotion.** The spam class got 103/104 accepted predictions right,
but its 95% Wilson lower bound was 94.75%, below the requested 95%. We retained that
failure rather than adjusting the policy after seeing the test. See the
[reproduction, full report, and limitations](benchmark.md).

## Give one recurring decision its own model

```python
from itchy import Task

route = Task(".itchy/support", labels=["billing", "access", "bug"])

# Trusted examples: human labels or independently verified outputs.
route.teach("Charged twice for my subscription", "billing", group="ticket-102")
route.teach("My password reset link expired", "access", group="ticket-103")
route.teach("Export crashes after the latest update", "bug", group="ticket-104")

# No champion yet: this abstains and enters the review queue.
decision = route.decide("I cannot log in")
route.correct(decision.id, "access")
```

Three examples do not make a deployable model. Collect representative examples of each
label across training, calibration, and test splits, then:

```python
route.import_jsonl("reviewed.jsonl")
report = route.fit(target_precision=0.95, min_support=50)
print(report["test"], report["reasons"])

if report["ready"]:
    route.promote(report["model_id"])  # explicit activation after reviewing the report

# Your callable runs once if Itchy abstains. Itchy has no provider integrations.
def existing_router(text):
    return "access"  # replace this demonstration stub with your existing implementation

decision = route.decide("My password reset link expired", fallback=existing_router)
print(decision.label, decision.route)  # route is local, fallback, or abstain
```

Fallback answers are logged, **never automatically treated as truth**. A correction
adds trusted supervision for the next training run. Training creates a candidate;
it does not mutate the serving model.

## The daily loop

```bash
itchy --task .itchy/support init --labels billing access bug
itchy --task .itchy/support import reviewed.jsonl
itchy --task .itchy/support fit --precision 0.95 --min-support 50
itchy --task .itchy/support promote MODEL_ID_FROM_REPORT
itchy --task .itchy/support predict "I was charged twice"
itchy --task .itchy/support review
itchy --task .itchy/support review --route local
itchy --task .itchy/support correct DECISION_ID billing
itchy --task .itchy/support status
itchy --task .itchy/support export ./support-model
```

`fit` exits **2** when a completed evaluation fails its gate, **1** on invalid input,
and **0** when the candidate is eligible. Commands emit JSON. `python -m itchy` works
without installing the console entry point.

The review queue includes **local answers as well as fallbacks**. Review both: a stream
of fallback corrections alone cannot reveal confidently wrong local predictions.
Training and promotion are explicit; a running task reloads its champion when the
pointer changes.

## Bring your data

One JSON object per line:

```json
{"text":"The reset email never arrived", "label":"access", "group":"ticket-882"}
{"text":"Following up on that same login issue", "label":"access", "group":"ticket-882"}
```

A stable hash assigns groups to train/calibration/test in approximate 60/20/20 proportions.
Set `"split":"train"`, `"calibration"`, or `"test"` explicitly for chronological or
predefined splits. Provide a shared `group` for messages from the same conversation,
customer, document, or generated template. Related examples must stay together.

Normalized exact duplicates cannot cross splits, and corrections retain their original
assignment. **Near-duplicate discovery is not automatic.** Imports are transactional:
one malformed row rolls the batch back. `teach` and `import` are trusted-supervision
interfaces; review teacher outputs before using them.

## What “ready” means

1. Weights learn from **training** examples only.
2. **Calibration** examples select a score threshold separately for each predicted label.
3. The frozen thresholds face **test** examples. Each enabled label must meet the
   requested precision lower bound and sample count; total accepted coverage must also
   pass. Any enabled label failing the audit blocks the whole candidate.
4. Promotion refuses a failed candidate or a dataset changed since evaluation.

The report includes coverage, accepted precision, per-label counts, a confusion matrix,
95% Wilson lower bounds, artifact size, elapsed training time, and a dataset fingerprint.
Softmax scores are ranking signals, **not probabilities of correctness**. These are
empirical evidence gates, not guarantees about future traffic. A small sample can
legitimately qualify zero labels. See [evaluation details](evaluation.md).

## A small model you own

The first learner is **byte n-gram features + a linear softmax classifier**, trained
from scratch on your examples. This carries forward Itchy's byte-level, right-sized
thesis without needing a pretrained language model for a finite-label decision.
There is no learned tokenizer or vocabulary to download.

The compressed artifact contains numeric weights, feature-presence bits, label metadata,
and an evaluation report. Export does not copy raw texts or the decisions database.
Weights are learned from private examples, so export is not a privacy guarantee.

```python
from itchy import Predictor

model = Predictor.load("./support-model")
prediction = model.predict("Please send me my billing receipt")
if prediction.label is None:
    print("Needs another decision path:", prediction.reason)
```

Inputs are Unicode-normalized and case-folded, with an 8,192 UTF-8-byte limit. Oversized
prediction inputs abstain; training rejects them. A feature-familiarity heuristic
rejects some unfamiliar inputs; it cannot detect every distribution shift.

## Where it fits

Good starting tasks have a stable label set and repeat often: support routing, content
categories, inbox sorting, or narrow intent recognition. Start with one task and compare
against your current rules or classifier using fresh reviewed examples.

This version does **not** generate prose, extract arbitrary JSON, reason over documents,
or fine-tune a transformer. Nuance, changing taxonomies, and unfamiliar domains may need
a stronger learner or the original fallback. There is no validated production dataset
or claim of beating an LLM in this release.

[DSPy](https://github.com/stanfordnlp/dspy) optimizes language-model programs;
[RouteLLM](https://github.com/lm-sys/RouteLLM) chooses between language models;
[Adaptive Classifier](https://github.com/codelion/adaptive-classifier) supports adaptive
text classification. Itchy's specific focus is an inspectable local loop that learns a
finite-label decision, measures which part it can handle, and produces a portable artifact.
See the [design rationale](direction.md).

## Where it came from

Itchy began as a 16MB byte-level language-model experiment for Parameter Golf, with the
idea **“build it the right size.”** Its original headline scores were invalid: grouping
inputs into byte patches while shifting targets by one byte exposed future targets to
the model. The 12-byte path can see 11 bytes ahead.

The [original write-up](../ITCHY_README.md), notebooks, model variants, and training scripts
remain available as historical research. Their BPB claims are not valid causal
next-byte benchmarks. The new classifier does not use those models or metrics.
Read [the correction](research-correction.md) and [the existing lookahead investigation](../NOLOOKAHEAD.md).

## Development

```bash
python -m pip install -e . pytest
python -m pytest
python -m examples.demo
```

The old `requirements.txt` belongs to the historical LM experiments. Install from
`pyproject.toml` for the new package. Runtime task directories contain raw input text;
keep them outside version control. Itchy sends no telemetry.

MIT. Contributions with **one repeated decision, a shareable dataset, and an honest
before/after evaluation** are especially useful.

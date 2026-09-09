# Itchy

## Current product

**Status:** active local experiment, 2026-09-08. Primary direction: **a tiny personal
judgment model learned through Whetstone**. Public repo: https://github.com/pauljump/itchy.
Nothing from this evolution has been published yet.

Paul rejected The Board as the first use case and selected Whetstone. The product is
contextual preference prediction: given a question and new alternatives, anticipate the
user's choice from explicit reviewed decisions. Whetstone retains evidence, consultation,
review and permission responsibilities; Itchy supplies a small experimental learner.
See [the direction](docs/direction.md) and [the integration contract](docs/whetstone.md).

## Ownership

Canonical working code: `/Users/mini-home/projects/_factory/itchy`, owned by the `_factory`
Git root (`pauljump/monorepo`). Public `pauljump/itchy` is the release target. Temporary
release checkouts are non-canonical. No nested Git root was introduced. Whetstone's
canonical source is `/Users/mini-home/projects/whetstone-release`; it was inspected but
not edited, installed, or deployed by this work.

## Current architecture

- `itchy/whetstone.py`: read-only adapter to Whetstone schema 1. Joins actual review events
  and active endorsed evidence; skips predictions, observed quotes, freeform unmapped
  corrections and authority decisions. Groups entire runs plus repeated questions across
  train/calibration/test splits. Scope required for fitting.
- `itchy/preference.py`: byte n-gram option features plus hashed question–option word
  interactions; local listwise linear ranking; held-out margin calibration/test;
  private numeric model artifacts; scope, host gate, and current-review revision checks.
- `itchy/cli.py`: `whetstone inspect`, `whetstone fit`, `whetstone suggest`. Suggestions are
  shadow-only and do not write decisions/reviews or execute anything.
- `itchy/model.py` and `itchy/task.py`: retained supporting generic classifier and feedback
  experiment. Its SMS benchmark is not evidence of personal-judgment accuracy.

NumPy is the only runtime dependency. No provider calls, pretrained downloads, transformer
fine-tuning, Whetstone source changes, or automatic agent-skill integration.

## Commands

```bash
python3 -m pip install -e .
python3 -m itchy whetstone inspect
python3 -m itchy whetstone fit --scope /absolute/project/root --output ~/.itchy/private-model
python3 -m itchy whetstone suggest --artifact ~/.itchy/private-model --consultation ask_ID
python3 -m pytest -q
```

Package metadata is in `pyproject.toml`. Root `requirements.txt`, model variants, notebooks,
and training scripts belong to historical language-model research.

## Evidence and limits

The actual local Whetstone store had **zero eligible reviewed choices**. The bridge
successfully inspected it and a fit refuses to invent a model from observed quotes or
pending agent predictions. Fixture tests check contextual choice learning, option-order
invariance, source isolation, read-only access, retirement, gates, scope, and held-out
label corruption. This is not a validated personal model, an avoided-interruption result,
or a production accuracy claim.

The original patch=12 language model leaks up to 11 future bytes on the reproduced
lookahead probe. Historical BPB claims are invalidated in current docs. Original research
and the pre-existing local lookahead investigation are preserved.

## Next proof

Review actual Whetstone choices during normal work. Hold out whole later runs. Compare
Itchy against a simple preference baseline and already-recorded Whetstone agent predictions
using explicit human verdicts. Measure agreement at coverage, context exceptions, review
effort, and cases where Itchy disagrees with a correct agent. Stay advisory until the
prospective evidence supports useful delegation. Never learn permission from preference.

# From Itchy to Usual

Itchy began as a byte-level language-model experiment for Parameter Golf. The idea was
to build something small around a specific job, then improve it through feedback.

The original notebooks, training scripts, and competition records are preserved in
[the Itchy archive](../archive/itchy/README.md). The patched-model BPB scores were later
invalidated by future-byte leakage; [the correction](../archive/itchy/CORRECTION.md)
explains why. Those scores are not evidence for Usual, and Usual does not use the weights.

## The task became personal

Working with coding agents exposed a recurring problem. An agent reaches a tradeoff
and asks its user to choose. The technical options can all be reasonable. The choice
still depends on the person: what complexity is worth adding, what “finished” means,
and which compromises make sense in this project.

After many conversations, those answers already exist. So do the rejected suggestions,
exceptions, and corrections. The useful asset is the record of how the person decided.

An intermediate experiment explored a small learner trained on reviewed choices. It
remains in [the archive](../archive/preference-learning/README.md), with its limitations.
There was no validated personal model to announce. The evidence-and-review loop was
the useful product on its own.

## A record of decisions, carried forward

That loop first took shape under the working name **Whetstone**. It became **Usual**:
a local skill that helps an existing coding agent learn how its user works.

A preference list loses context. “Yes” needs its question. “Use the second one” needs
the alternatives. “Usually, except for this project” needs the exception. An assistant's
explanation must remain distinct from the user's actual words.

Usual preserves these decision episodes. Your agent retrieves relevant examples, reasons
about the current task, and records its call. You can accept it, correct it, or exclude
it from learning. A past approval never grants new permission to delete, spend, or publish.

**Past choice → relevant context → new decision → human correction.**

Mining, storage, retrieval, and review are local. The coding model you already use does
the reasoning. There is no separate Usual model service, and counting mined decisions
is not a claim of predictive accuracy.

Usual is the active product in `pauljump/usual`. The repository keeps its history while
the front door, installation, and ongoing development serve the work you do every day.

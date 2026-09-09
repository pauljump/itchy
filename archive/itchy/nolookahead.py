"""The lookahead test — 40 lines that tell you whether your LM can see its own answer.

Copy this file, or just the function, into your project. No install, no dependencies
beyond numpy, works with torch / mlx / jax / anything: you hand it a scoring closure.

WHY: in next-token prediction, the loss at output slot i may depend on inputs x[0..i]
and nothing else. If it also depends on x[j] for some j > i, the model is being scored
on a target it was allowed to read. Every metric downstream of that is fiction.

This is easy to introduce whenever the input is regrouped before scoring — byte/patch
models, multi-token prediction heads, blockwise or diffusion-style decoders, anything
where one hidden state emits several output slots. It is nearly impossible to notice by
staring at a loss curve, because a leaking model trains beautifully.

Runs on UNTRAINED weights in seconds. Run it before you train, not after you publish.
"""
from __future__ import annotations

import numpy as np


def lookahead_report(score, x, y, vocab_size, probes=12, tol=1e-4, seed=0):
    """Find output slots whose loss depends on future input.

    score(x, y) -> per-slot loss, shape (B, S), as numpy or anything with __array__.
                   Must be deterministic. Pass your *eval* scoring path, not a wrapper.
    x, y         -> one integer batch, shape (B, S), exactly as your eval loop builds it.

    Returns (max_lookahead, violations). max_lookahead is how many positions into the
    future the worst-affected slot can see; 0 means clean.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    base = np.asarray(score(x, y), dtype=np.float64)
    jitter = np.abs(np.asarray(score(x, y), dtype=np.float64) - base).max()
    if jitter > tol:
        raise RuntimeError(
            f"score() is not deterministic (rescoring the same batch moved the loss by "
            f"{jitter:.2e} > tol={tol:.0e}). Disable dropout/sampling and retry."
        )

    seq = x.shape[1]
    violations: list[tuple[int, int, float]] = []
    for j in sorted(rng.choice(np.arange(1, seq), size=min(probes, seq - 1), replace=False)):
        bumped = x.copy()
        offset = rng.integers(1, vocab_size, size=x.shape[0])
        bumped[:, j] = (bumped[:, j] + offset) % vocab_size  # guaranteed different token
        delta = np.abs(np.asarray(score(bumped, y), dtype=np.float64) - base).mean(axis=0)
        for i in np.flatnonzero(delta[:j] > tol):  # slot i < j moved => it saw x[j]
            violations.append((int(i), int(j), float(delta[i])))

    max_lookahead = max((j - i for i, j, _ in violations), default=0)
    return max_lookahead, violations


def assert_no_lookahead(score, x, y, vocab_size, **kw):
    """Raise if any output slot can see future input. Drop this in your test suite."""
    max_lookahead, violations = lookahead_report(score, x, y, vocab_size, **kw)
    if violations:
        worst = max(violations, key=lambda v: v[1] - v[0])
        raise AssertionError(
            f"LOOKAHEAD LEAK: {len(violations)} slot/input pairs violate causality. "
            f"Worst: loss at slot {worst[0]} moved by {worst[2]:.3f} nats when input "
            f"{worst[1]} changed ({max_lookahead} positions of lookahead). "
            f"Roughly {max_lookahead}/{max_lookahead + 1} of your scored positions may be "
            f"copying rather than predicting; your reported metric is not next-token loss."
        )
    return True

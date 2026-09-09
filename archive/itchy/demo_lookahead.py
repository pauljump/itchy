"""Runs the lookahead test against this repo's own model. Takes ~10s, no training.

patch=12 is the configuration whose numbers are in the README. It fails.
patch=1 is the same code with no regrouping. It passes.
"""
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from model_itchy_final import ItchyFinal
from nolookahead import lookahead_report

BATCH, VOCAB = 8, 260


def make_score(patch):
    model = ItchyFinal(dim=384, num_layers=4, num_heads=6, num_kv_heads=3,
                       patch_size=patch, decode_head_dim=128)
    mx.eval(model.parameters())

    def score(x, y):
        logits = model(mx.array(np.asarray(x))).astype(mx.float32)
        ce = nn.losses.cross_entropy(logits.reshape(-1, VOCAB),
                                     mx.array(np.asarray(y)).reshape(-1), reduction="none")
        return np.asarray(ce.reshape(x.shape[0], x.shape[1]))

    return score


for patch in (12, 1):
    seq = 24 * patch
    # exactly how train_itchy_final.py:209 builds a batch: one-BYTE shift, then patch
    stream = np.random.default_rng(0).integers(0, 256, BATCH * seq + 1)
    x, y = stream[:-1].reshape(BATCH, seq), stream[1:].reshape(BATCH, seq)

    lookahead, violations = lookahead_report(make_score(patch), x, y, VOCAB, probes=12)
    slots = len({i for i, _, _ in violations})
    print(f"\npatch_size={patch:<3} seq={seq}")
    print(f"  lookahead      : {lookahead} positions into the future")
    print(f"  leaking slots  : {slots} distinct output slots read an input they shouldn't")
    if violations:
        i, j, d = max(violations, key=lambda v: v[2])
        print(f"  worst          : loss at slot {i} moved {d:.3f} nats when input {j} changed")
        print(f"  implication    : ~{lookahead}/{lookahead+1} of scored positions can copy")
    else:
        print("  clean — every output slot depends only on its own past")

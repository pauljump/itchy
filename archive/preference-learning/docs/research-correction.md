# Correction to the original Itchy results

Verified locally on 2026-09-08 using the pre-existing `demo_lookahead.py` and
`nolookahead.py` investigation. No training or external model calls were used.

The original README reported 0.2903 BPB at patch size 12 and 0.2329 BPB with
per-position MLP decoding. **These are not valid causal next-byte language-model
results.** They must not be compared to Parameter Golf leaderboard scores.

`train_itchy_final.py` builds its input with `local_chunk[:-1]` and its targets with
`local_chunk[1:]`, then the model groups inputs into multi-byte patches. A hidden
state that reads bytes 0..11 is asked to predict bytes 1..12. Eleven target bytes
are already in that state's input. Causal attention between patches does not fix
lookahead inside a patch.

Reproduction on untrained weights:

```text
patch_size=12, sequence=288: 11 positions of lookahead; 67 affected output slots
patch_size=1,  sequence=24:   0 positions of lookahead;  0 affected output slots
```

This reproduces the qualitative result of the existing local investigation. Random
model initialization can change the magnitude and individual affected slots. The
12-byte configuration fails the causality test; the one-byte control passes.

```bash
# Optional historical diagnostic; requires Apple Silicon and MLX, in addition to NumPy.
python demo_lookahead.py
```

The old patch-size and decoder ablations measured a contaminated objective. Do not
carry their rankings forward as evidence for a corrected causal model. The original
scripts, notebooks, and write-up remain in place for historical inspection; they are
not the new package's training path. The new classifier does not load these models.

See [the original local investigation](../NOLOOKAHEAD.md) for the diagnostic approach.

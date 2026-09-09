# The lookahead test

**40 lines that tell you whether your language model can see its own answer.**

No install. No dependencies beyond numpy. Works with torch, mlx, jax, anything.
Runs on **untrained weights in seconds** — so you run it before you spend a GPU-hour,
not after you publish.

```python
from nolookahead import assert_no_lookahead

assert_no_lookahead(score, x, y, vocab_size=256)
```

`score(x, y)` is your own eval scoring path, returning per-position loss of shape `(B, S)`.
That's the whole interface.

## Why

In next-token prediction, the loss at output slot `i` may depend on inputs `x[0..i]` and
nothing else. If it also depends on `x[j]` for some `j > i`, your model is being scored on
a target it was allowed to read. Everything downstream of that number is fiction.

This is trivial to introduce and nearly impossible to notice. It appears whenever the input
gets regrouped before scoring — byte/patch models, multi-token prediction heads, blockwise or
diffusion-style decoders, any design where one hidden state emits several output slots. A
leaking model trains beautifully. The loss curve looks great. The ablations look clean.

## It caught the model in this repo

This repository published a 16MB byte-level LM claiming **0.2903 bits-per-byte** — roughly
twice as good as frontier models, from 17.5M parameters. The test, on random weights:

```
patch_size=12  seq=288
  lookahead      : 11 positions into the future
  leaking slots  : 67 distinct output slots read an input they shouldn't
  worst          : loss at slot 257 moved 0.135 nats when input 262 changed
  implication    : ~11/12 of scored positions can copy

patch_size=1   seq=24
  lookahead      : 0 positions into the future
  clean — every output slot depends only on its own past
```

The cause: targets were built with a one-**byte** shift on the flat stream and *then* grouped
into 12-byte patches. Patch `p` ingests bytes `[12p, 12p+12)`, and 11 of the 12 decode heads
hanging off it are scored against bytes inside that same window. Only the last head predicts
anything. The headline metric was an average over 11 free positions and one real one.

Trained, at patch=4, per-position bits-per-byte came out:

| slot 0 | slot 1 | slot 2 | slot 3 *(the only real prediction)* |
|---|---|---|---|
| 0.809 | 0.302 | 1.384 | **4.301** |

Reported-style average: **1.699**. Honest number: **4.301**.

## The tell, if you don't want to run anything

Leaked metrics scale as `1/group_size`, because the group dilutes one real prediction across
`k` free ones. This repo's own patch-size ablation, multiplied back out:

| patch size | reported BPB | × patch size |
|---|---|---|
| 2 | 1.1512 | 2.30 |
| 3 | 0.7800 | 2.34 |
| 4 | 0.5944 | 2.38 |
| 8 | 0.3298 | 2.64 |

Nearly constant — and landing on the token-level baseline's 2.56. The celebrated finding that
"patch size is the dominant hyperparameter, 40× larger than every other trick combined" was
the dilution factor, measured to four decimal places.

## The other thing to check

If your tiny model beats the best known result by 2×, the prior that you have a bug is
overwhelming. 0.29 BPB should have stopped this on day one. Check plausibility before you
check anything else.

## Run the demo

```bash
python demo_lookahead.py   # ~10s, no training, no data
```

MIT. If it finds something in your repo, I'd like to hear about it.

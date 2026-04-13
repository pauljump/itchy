# Itchy

A 16MB language model designed to be small from birth — not a shrunken giant.

Built for the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) challenge.

## Why "Itchy"

Like giving someone a wool blanket to keep them warm when the reason they were complaining was that they were itchy. Every other submission shrinks a big model into 16MB — solving the wrong problem. Itchy asks: what if you just built something the right size?

## The Thesis

622+ submissions to Parameter Golf. All of them are transformers with BPE tokenizers. Itchy is the first byte-level submission.

**Byte-level** (256 vocab) — no tokenizer. The entire 16MB budget goes to the brain. With a 1024-token vocabulary, the embedding table consumes a large fraction of the parameter budget. With 256 bytes, it's negligible — those parameters go into transformer layers instead.

Patch processing (4 bytes -> 1 patch) keeps effective sequence length shorter than the token-level baseline.

## Architecture

```
Input: raw UTF-8 bytes (0-255)
  -> BytePatchEmbed (12 bytes -> 1 patch -> model dim)
  -> 11x Block (attention + LeakyReLU(0.5)² MLP, 3x expansion)
  -> PerPositionDecode (12 independent MLP heads, one per byte position)
Output: next-byte probabilities

17.5M params | 13.1MB at int6 | 384 dim, 11 layers, 8 heads, patch=12
```

## Validation Results (T4 GPU, 3000 steps, 1 shard)

### Patch size ablation (the big finding):

| Patch Size | Val BPB | Delta |
|-----------|---------|-------|
| 2 | 1.1512 | +0.557 |
| 3 | 0.7800 | +0.186 |
| 4 | 0.5944 | — (original) |
| 8 | 0.3298 | -0.265 |
| **12** | **0.2903** | **-0.304** |
| 16 | 0.3343 | -0.260 |

Patch size is the dominant hyperparameter — going from 4 to 12 improved BPB by 0.30, which is 40x larger than all other tricks combined.

### Unpatch decode ablation:

| Decode method | Val BPB | Delta |
|--------------|---------|-------|
| Flat linear (baseline) | 0.2903 | — |
| **Per-position MLP heads** | **0.2329** | **-0.057** |
| Autoregressive decoder | 0.7937 | +0.503 (too complex for scale) |
| MoE unpatch | 3.2594 | broken |

### Trick ablation:

| Config | Val BPB | Delta |
|--------|---------|-------|
| Baseline (relu², 2x MLP) | 0.6010 | — |
| +LeakyReLU(0.5)² | 0.5996 | -0.0014 |
| +3x MLP | 0.5949 | -0.0061 |
| +Partial RoPE | 0.6119 | +0.0109 (hurts) |
| +LN Scale | 0.6125 | +0.0115 (hurts) |
| +N-gram hash | 0.6010 | +0.0000 (no effect) |

### Head-to-head vs token-level baseline:
- **Itchy (byte-level): 0.66 BPB** with 4.3M params
- **Baseline (token-level): 2.56 BPB** with 1.1M params (same total size budget)

## What Didn't Work

- **LoRA adapters + TTT meta-learning**: Built LoRA adapters into every block, trained with meta-learning episodes. Adapters never learned to adapt — zero improvement across all tests. The zero-initialized gate creates a gradient dead zone. Stripped entirely.
- **Partial RoPE**: Helps token-level models, hurts byte-level (+0.011 BPB).
- **LN Scale per layer**: Same — helps at token level, hurts at byte level (+0.012 BPB).
- **Hash n-gram embeddings**: Added 600K params for zero BPB improvement.

## Running

```bash
# Local (Mac, MLX)
python data/convert_to_bytes.py --train-shards 2
ITERATIONS=30 TRAIN_LOG_EVERY=10 .venv/bin/python train_itchy_mlx.py

# Competition (8xH100)
python data/convert_to_bytes.py --train-shards 80
torchrun --standalone --nproc_per_node=8 train_itchy_final.py
```

---

**Part of a larger system.** See [pauljump/portfolio](https://github.com/pauljump/portfolio) for the full picture — 16 production apps, shared infrastructure, and the factory that builds them.

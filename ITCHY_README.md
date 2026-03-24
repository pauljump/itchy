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
  -> BytePatchEmbed (4 bytes -> 1 patch -> model dim)
  -> 10x Block (attention + LeakyReLU(0.5)² MLP, 3x expansion)
  -> ByteUnpatch (model dim -> per-byte logits)
Output: next-byte probabilities

19.5M params | 14.6MB at int6 | 448 dim, 10 layers, 8 heads
```

## Validation Results (T4 GPU, 3000 steps, 1 shard)

Ablation study — each trick tested individually:

| Config | Params | Val BPB | Delta |
|--------|--------|---------|-------|
| Baseline (relu², 2x MLP) | 4.3M | 0.6010 | — |
| +LeakyReLU(0.5)² | 4.3M | 0.5996 | -0.0014 |
| +3x MLP | 5.3M | 0.5949 | -0.0061 |
| +Partial RoPE 8/32 | 4.3M | 0.6119 | +0.0109 (hurts) |
| +LN Scale | 4.3M | 0.6125 | +0.0115 (hurts) |
| +N-gram hash | 4.9M | 0.6010 | +0.0000 (no effect) |
| **+Leaky +3x MLP** | **5.3M** | **0.5944** | **-0.0066** |

Head-to-head vs token-level baseline at matched compute:
- **Itchy (byte-level): 0.66 BPB** with 4.3M params
- **Baseline (token-level): 2.56 BPB** with 1.1M params (same total size budget)

The token-level model can only fit 1.1M params because its 1024-vocab embedding table eats most of the budget. Byte-level puts 4x more params into the actual transformer.

## What Didn't Work

- **LoRA adapters + TTT meta-learning**: Built LoRA adapters into every block, trained with meta-learning episodes. Adapters never learned to adapt — zero improvement across all tests. The zero-initialized gate creates a gradient dead zone. Stripped entirely.
- **Partial RoPE**: Helps token-level models, hurts byte-level (+0.011 BPB).
- **LN Scale per layer**: Same — helps at token level, hurts at byte level (+0.012 BPB).
- **Hash n-gram embeddings**: Added 600K params for zero BPB improvement.

## Status

- [x] Model architecture (byte-level, LeakyReLU², 3x MLP)
- [x] Ablation study on T4 (7 configs tested)
- [x] Head-to-head vs token-level baseline
- [x] MLX training loop (local Mac)
- [x] CUDA training script (8xH100, ready to run)
- [x] Byte data pipeline
- [ ] Full training runs on H100s (waiting for compute grant)
- [ ] 3-seed validation
- [ ] Submission PR

## Running

```bash
# Local (Mac, MLX)
python data/convert_to_bytes.py --train-shards 2
ITERATIONS=30 TRAIN_LOG_EVERY=10 .venv/bin/python train_itchy_mlx.py

# Competition (8xH100)
python data/convert_to_bytes.py --train-shards 80
torchrun --standalone --nproc_per_node=8 train_itchy_final.py
```

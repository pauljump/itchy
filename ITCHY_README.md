# Itchy

A 16MB language model designed to be small from birth — not a shrunken giant.

Built for the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) challenge.

## The Name

Named after "the itchy blanket." When something can't tell you it's uncomfortable, you sometimes force what works for you onto it — like wrapping a kid in a scratchy wool blanket because *you* know it's warm. Every other submission in this competition takes a big model's architecture and forces it into 16MB. Itchy asks: what if the model was never big?

## The Thesis

622+ submissions to Parameter Golf. Essentially all of them are transformers with BPE tokenizers, playing two games:

- **Game 1**: Better transformer tricks (crowded, incremental)
- **Game 2**: Test-Time Training bolted on after training (big unlock, but hacky)

**Itchy plays Game 3:** Build a model native to 16MB with adaptation in its DNA.

### Three ideas, none tried before at this scale:

1. **Byte-level** (256 vocab) — no tokenizer. The entire 16MB budget goes to the brain. Patch processing (4 bytes -> 1 patch) keeps sequences manageable.

2. **TTT-as-architecture** — LoRA adapters are built into every transformer block. During training, 20% of steps are meta-learning episodes where the model practices adapting to new text. At eval, per-document adaptation isn't a surprise — it's what the model was trained for.

3. **Classical stacking** — PPM compression layered on neural predictions. Zero extra parameters, free bits.

## Architecture

```
Input: raw UTF-8 bytes (0-255)
  -> BytePatchEmbed (4 bytes -> 1 patch -> model dim)
  -> 12x AdaptiveBlock (attention + MLP + LoRA adapter)
  -> ByteUnpatch (model dim -> per-byte logits)
Output: next-byte probabilities

17.3M params | 13.0MB at int6 | 3MB headroom for code
```

## Status

- [x] Model architecture (byte-level transformer + LoRA adapters)
- [x] MLX training loop (local Mac iteration)
- [x] Byte data pipeline (sp1024 -> raw bytes conversion)
- [x] TTT meta-learning training episodes
- [ ] CUDA training script (8xH100)
- [ ] Full training runs + 3-seed validation
- [ ] Submission PR

## Running Locally

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm

# Download data + convert to bytes
python data/cached_challenge_fineweb.py --train-shards 2
python data/convert_to_bytes.py --train-shards 2

# Train (abbreviated, Mac is slow)
ITERATIONS=30 TRAIN_LOG_EVERY=10 VAL_LOSS_EVERY=0 python train_itchy_mlx.py
```

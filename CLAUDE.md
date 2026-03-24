# Itchy — Claude Operating File

## What This Is

**Itchy** is our entry to OpenAI's Parameter Golf competition. Train the best language model that fits in 16MB, trains in 10 minutes on 8xH100s, scored by bits per byte (BPB) on FineWeb validation set. Competition ends April 30, 2026.

Named after "the itchy blanket" — the thesis that every other submission shrinks a big model and forces it to work at small scale (like forcing an itchy blanket on a kid who can't tell you it's uncomfortable). Itchy is designed to be small from birth.

## The Thesis (Game 3)

Everyone in the competition plays two games:
- **Game 1**: Better transformer architecture (crowded, incremental, 622+ submissions)
- **Game 2**: Test-Time Training bolted on after the fact (the big unlock, but hacky)

**We play Game 3:** Build a model that was never big. Byte-level, adaptation-native, designed for 16MB.

### The Autism Analogy (How Paul Thinks About This)
Paul's autistic son can memorize a flight number from a year ago but can't choose between pizza and burgers. Parents adapt — but sometimes the adaptation is wrong (forcing an itchy blanket for warmth). Every other submission is the parent forcing big-model assumptions on a tiny model. Itchy is designed around how a small brain actually works, not how a big brain pretends to be small.

### Key Insights From Research
1. **#1 failure mode is the "generalist tax"** — one tiny model tries to handle all text types. TTT works because it specializes per-document. The gap is 0.56 BPB (1.12 → 0.56).
2. **Architecture space is barely explored** — 622 submissions, basically ALL transformers. Zero byte-level, zero MoE (tried and failed), zero RWKV.
3. **Two genuine gaps**: byte-level at tiny scale (never tried) and TTT-as-architecture (never tried below 3B params).
4. **The combination has zero precedent** — byte-level + adaptation-native + classical stacking at sub-10M params.

## Architecture

### Byte-Level Model
- **256 vocab** (raw bytes) instead of 1024 BPE tokens
- No tokenizer overhead — entire 16MB goes to the brain
- **Patch processing**: group 4 bytes → 1 patch to cut sequence length (net shorter than baseline)
- BPB calculation is trivial: `loss_nats / ln(2)` (no token-to-byte conversion)
- The repo already supports `byte260` data variant — pre-tokenized byte data available

### Adaptation-Native (TTT as Architecture)
- Built-in LoRA-style adapters in every transformer block (zero-initialized)
- Model trains WITH adapters present — TTT isn't bolted on, it's architectural
- 20% of training steps are meta-learning episodes (practice adapting)
- At eval: adapters specialize per-document chunk

### Classical Stacking
- PPM (Prediction by Partial Matching) layered on neural predictions
- Zero additional parameters, small but free BPB improvement

## Competition Rules
- **16,000,000 bytes** total artifact (model weights compressed + code). NOT 16 MiB.
- **10 min training** on 8xH100 SXM 80GB (wallclock, compile warmup is free)
- **10 min evaluation** (separate budget — TTT happens here)
- **Tokenizer NOT counted** in artifact size
- **External libraries NOT counted** (just imports)
- **Statistical significance**: 3 seeds, p < 0.01, improvement > 0.005 nats
- **TTT rule**: score tokens BEFORE training on them. `torch.inference_mode()` during scoring.
- **No paid prefix** — can't bake validation data into artifact

## Scoreboard Context (as of 2026-03-24)
- Baseline: 1.2244 BPB
- Best merged: 1.1194 BPB (LeakyReLU² + legal TTT)
- Best pending: 0.6430 BPB (aggressive LoRA TTT, 8 epochs per doc)
- Best pending (strict legal): ~0.56 BPB

## Project Setup

### Directory
`/Users/mini-home/Desktop/itchy/`

### Repo Structure
Forked from `openai/parameter-golf`. Key files:
- `train_gpt.py` — CUDA baseline training script (8xH100)
- `train_gpt_mlx.py` — MLX baseline for local Mac iteration
- `data/cached_challenge_fineweb.py` — downloads pre-tokenized shards from HuggingFace
- `data/datasets/fineweb10B_sp1024/` — downloaded training + validation data
- `data/tokenizers/` — SentencePiece BPE tokenizer
- `records/` — submission directories
- `docs/plans/2026-03-24-itty-parameter-golf.md` — full implementation plan

### Environment
- Python 3.12 (`/opt/homebrew/bin/python3.12`)
- venv at `.venv/` (created, deps installed)
- Installed: mlx, numpy, tqdm, huggingface-hub, datasets, tiktoken, sentencepiece
- MLX includes mlx.nn (no separate package needed)

### Data Downloaded
- 2 training shards: `fineweb_train_000000.bin`, `fineweb_train_000001.bin` (~191MB each)
- 1 validation shard: `fineweb_val_000000.bin` (~118MB)
- Tokenizer: `fineweb_1024_bpe.model` + `.vocab`

### What Hasn't Been Done Yet
- [ ] Baseline has NOT been run yet (venv activation issue in background shell — need to run with `.venv/bin/python`)
- [ ] No RunPod account or compute credits yet
- [ ] No Itchy model code written yet
- [ ] No byte-level data downloaded yet (use `--variant byte260`)

## How To Run Baseline Locally
```bash
cd /Users/mini-home/Desktop/itchy
ITERATIONS=30 TRAIN_LOG_EVERY=10 VAL_LOSS_EVERY=0 .venv/bin/python train_gpt_mlx.py
```
Expected: loss starts ~7.0, decreases toward ~5.0 in 30 steps. SLOW on Mac.

## Git Workflow
Commit to main. No branches.

## Key Research References
- TTT Layers paper (Sun et al., July 2024) — TTT as architecture, not bolt-on
- TTT-E2E (Sun et al., Dec 2025) — meta-learning for TTT, tested at 3B only
- Pico-MAML (Nov 2025) — meta-learning at 11M params, proved it works at our scale
- ByT5 (Google, 2021) — byte-level LM, smallest at 300M (never tried at 5M)
- cmix — SOTA text compression, mixes classical + neural (our stacking inspiration)
- Parameter Golf Issue #402 — TTT legality debate

## Kit Home Connection
The end goal: run this model on iPhone via CoreML for Kit Home.
- 16MB model = smaller than a photo, trivial for Neural Engine
- On-device TTT = model adapts to THIS family's documents on first upload
- "A brain that grows around your child" — privacy as architecture, not a promise
- Gemini fallback only for deep reasoning tasks

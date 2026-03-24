# Itty: Parameter Golf Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a competitive Parameter Golf submission using a novel byte-level, adaptation-native architecture — then run it on iPhone via CoreML for Kit Home.

**Architecture:** Three-phase approach: (1) get the proven baseline running and submit, (2) build our Game 3 architecture (byte-level + TTT-as-core), (3) stack classical compression on top. Each phase is independently submittable.

**Tech Stack:** Python, PyTorch, MLX (local iteration), CoreML (iPhone deployment), FlashAttention, CUDA

**Competition constraints:**
- 16,000,000 bytes total artifact (model + code)
- 10 minutes training on 8xH100 SXM 80GB
- 10 minutes evaluation (separate budget)
- Scored by bits per byte (BPB) on FineWeb validation set (50K docs)
- 3 seeds for statistical significance (p < 0.01, improvement > 0.005 nats)

---

## Phase 1: Foundation — Get Running + Apply Known Wins

**Why:** You can't innovate on something you can't run. Get the baseline working locally, understand it viscerally, apply proven improvements, and submit our first entry. This gets us on the leaderboard and gives us a benchmark for Game 3.

### Task 1.1: Fork + Project Setup

**Files:**
- Create: `README.md` (project readme)
- Create: `requirements.txt`
- Clone: full parameter-golf repo contents into itty

- [ ] **Step 1: Fork the Parameter Golf repo**

```bash
cd /Users/mini-home/Desktop/itty
gh repo fork openai/parameter-golf --clone=false
# Download the repo contents directly
git remote add upstream https://github.com/openai/parameter-golf.git
git fetch upstream
git checkout -b main upstream/main
```

- [ ] **Step 2: Create our project README**

```markdown
# Itty

A 16MB language model designed to be small from birth — not a shrunken giant.

Built for the OpenAI Parameter Golf challenge.
Designed to run on iPhone.

## Philosophy

Every other submission shrinks a big model. Itty was never big.
Byte-level. Adaptation-native. Classical compression stacked.
```

- [ ] **Step 3: Verify repo structure**

```bash
ls -la train_gpt.py train_gpt_mlx.py data/ records/
```
Expected: All baseline files present.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: fork parameter-golf, add project readme"
```

### Task 1.2: Get Baseline Running Locally (MLX)

**Files:**
- Read: `train_gpt_mlx.py`
- Read: `data/cached_challenge_fineweb.py`
- Read: `requirements.txt`

- [ ] **Step 1: Install dependencies**

```bash
cd /Users/mini-home/Desktop/itty
python3 -m venv .venv
source .venv/bin/activate
pip install mlx mlx-nn numpy tqdm huggingface-hub datasets tiktoken sentencepiece
```

- [ ] **Step 2: Download training data (one shard for local testing)**

```bash
python data/cached_challenge_fineweb.py
```

Note: This downloads pre-tokenized shards from HuggingFace. For local testing, we only need 1-2 shards. Full training data is ~8B tokens across ~80 shards.

- [ ] **Step 3: Run baseline training on Mac (abbreviated)**

```bash
python train_gpt_mlx.py
```

Expected: Training starts, prints loss every step. Will be SLOW on Mac (hours vs minutes on H100). Let it run 100 steps to verify it works, then Ctrl+C.

Watch for:
- Loss decreasing from ~7.0 toward ~5.0 in first 100 steps
- No OOM errors (MLX manages memory differently than CUDA)
- BPB calculation at end

- [ ] **Step 4: Document baseline local performance**

Record: steps completed, final loss, time per step, memory usage. Save to `docs/baseline-local.md`.

- [ ] **Step 5: Commit**

```bash
git add docs/baseline-local.md
git commit -m "docs: baseline local performance on Mac"
```

### Task 1.3: Apply for RunPod Compute Credits

- [ ] **Step 1: Apply for OpenAI compute credits**

Go to the Parameter Golf README, find the RunPod credits application form. Fill out with:
- Project name: Itty
- Approach: "Byte-level adaptation-native architecture — designing a model native to 16MB rather than shrinking a large one. Novel combination of byte-level processing + TTT-as-core-architecture, both untested at this scale."
- Expected usage: ~50 hours H100 time for experimentation

- [ ] **Step 2: Set up RunPod account**

Create RunPod account, configure SSH access, test connecting to their Parameter Golf template.

- [ ] **Step 3: Document access in project**

Save credentials setup instructions to `docs/runpod-setup.md` (no secrets in the file).

### Task 1.4: Run Official Baseline on H100s

**Files:**
- Read: `train_gpt.py`

- [ ] **Step 1: Launch RunPod pod with Parameter Golf template**

```bash
# SSH into RunPod pod
# Template pre-installs PyTorch, CUDA, FlashAttention
```

- [ ] **Step 2: Clone our repo on the pod**

```bash
git clone https://github.com/<username>/itty.git
cd itty
pip install -r requirements.txt
```

- [ ] **Step 3: Download full training data on pod**

```bash
python data/cached_challenge_fineweb.py
```

- [ ] **Step 4: Run baseline training (8xH100)**

```bash
torchrun --nproc_per_node=8 train_gpt.py
```

Expected: ~7000 steps in 10 minutes. Final val_loss ~3.28, val_bpb ~1.22.

- [ ] **Step 5: Record baseline H100 performance**

Save: steps, val_loss, val_bpb, training time, artifact size. Update `docs/baseline-h100.md`.

- [ ] **Step 6: Commit**

```bash
git add docs/baseline-h100.md
git commit -m "docs: baseline H100 performance benchmarks"
```

### Task 1.5: Apply Proven Improvements (Game 1 Speedrun)

**Files:**
- Create: `train_itty.py` (copy from baseline, apply improvements)

Apply the following proven techniques from top submissions (each worth -0.002 to -0.03 BPB):

- [ ] **Step 1: Create our training script from baseline**

```bash
cp train_gpt.py train_itty.py
```

- [ ] **Step 2: Implement sliding window evaluation**

Add sliding window function with stride=64, seq_len=2048. This alone is worth ~-0.03 BPB. Score only the last `stride` tokens per window (except first window which scores all). Each token gets maximum context.

- [ ] **Step 3: Apply architecture improvements**

In the GPT model class:
- Increase to 11 layers (from 9)
- 3x MLP expansion (from 2x)
- LeakyReLU(0.5) squared activation (from relu squared)
- Partial RoPE (16/64 dims)

- [ ] **Step 4: Apply quantization improvements**

- int6 per-row quantization (from int8)
- zstd compression (from zlib)
- GPTQ-lite clip search

- [ ] **Step 5: Apply training improvements**

- EMA with decay=0.997
- Weight decay 0.04 on Muon optimizer

- [ ] **Step 6: Run on H100s, compare to baseline**

```bash
torchrun --nproc_per_node=8 train_itty.py
```

Expected: BPB should drop from ~1.22 to ~1.13-1.15 range.

- [ ] **Step 7: Run 3 seeds for statistical significance**

```bash
for seed in 42 137 256; do
    torchrun --nproc_per_node=8 train_itty.py --seed=$seed
done
```

- [ ] **Step 8: Commit**

```bash
git add train_itty.py
git commit -m "feat: apply proven improvements — sliding window, deeper model, int6, EMA"
```

### Task 1.6: Implement Legal TTT (Game 2)

**Files:**
- Modify: `train_itty.py`

- [ ] **Step 1: Implement score-first TTT**

The legal TTT protocol:
1. Split validation into 32K-token non-overlapping chunks
2. For each chunk: SCORE under `torch.inference_mode()`, then TRAIN with SGD on already-scored tokens
3. Last chunk scored but never trained on

Key implementation detail: `torch.inference_mode()` context manager during scoring provides a hard guarantee that no weight updates happen while scoring.

TTT parameters: SGD with lr=0.002, momentum=0.9, 3 epochs per chunk, cosine LR decay within each chunk, grad clip 1.0.

- [ ] **Step 2: Test TTT locally (abbreviated)**

Run TTT on a small slice of validation data to verify the score-first protocol works.

- [ ] **Step 3: Run full TTT on H100s**

Expected: BPB should drop 0.03-0.10 below non-TTT score.

- [ ] **Step 4: Prepare first submission**

Create submission directory:
```
records/track_10min_16mb/2026-03-XX_itty_v1/
  README.md
  submission.json
  train_itty.py
  log_seed42.txt
  log_seed137.txt
  log_seed256.txt
```

- [ ] **Step 5: Submit PR to Parameter Golf**

```bash
gh pr create --repo openai/parameter-golf --title "itty v1: [BPB score]" --body "..."
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: legal score-first TTT + first submission"
```

---

## Phase 2: Game 3 — The Byte-Level Adaptive Architecture

**Why:** This is our novel contribution. A model designed from birth to be 16MB, working at byte level, with adaptation built into its architecture. No prior art exists for this combination at this scale.

### Task 2.1: Byte-Level Data Pipeline

**Files:**
- Create: `data/byte_stream.py`
- Create: `tests/test_byte_stream.py`

- [ ] **Step 1: Write failing test — byte stream reads raw text**

ByteStream should yield raw UTF-8 byte sequences (values 0-255), no tokenizer involved. Shape: (seq_len,), dtype: torch.long, max value 255.

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_byte_stream.py -v
```

- [ ] **Step 3: Implement ByteStream**

Load FineWeb data as raw bytes. 256 possible values (0-255). Simple sequential reader with shard support and wraparound. Supports get_batch(batch_size, device).

- [ ] **Step 4: Run test to verify it passes**

- [ ] **Step 5: Write byte-level BPB calculation**

For byte-level models, BPB is trivial: `bpb = cross_entropy_nats / ln(2)`. No token-to-byte conversion needed since each token IS a byte.

- [ ] **Step 6: Commit**

```bash
git add data/byte_stream.py tests/test_byte_stream.py
git commit -m "feat: byte-level data pipeline — no tokenizer"
```

### Task 2.2: Byte-Level Model Architecture

**Files:**
- Create: `model_itty.py`
- Create: `tests/test_model.py`

The core innovation: a model designed for bytes from the ground up.

**Key design decisions:**
- Vocab size: 256 (bytes) — embedding table is tiny (256 x dim)
- The saved embedding budget goes into MORE model capacity
- Patch-based processing: group bytes into patches (e.g., 4 bytes = 1 patch) to reduce sequence length
- Built-in adapter slots for TTT — zero-initialized, trained during adaptation

- [ ] **Step 1: Write failing test — model forward pass**

Test that Itty(dim=512, n_layers=12, n_heads=8, patch_size=4) takes input (1, 1024) of byte values and produces logits of shape (1, 1024, 256).

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement Itty model**

Components:
- **BytePatchEmbed**: Embed 256 byte values, group into patches (4 bytes per patch), project to model dim. This cuts effective sequence length by 4x.
- **AdaptiveBlock**: Transformer block with built-in LoRA-style adapter (zero-initialized). Adapter has down-projection (dim -> rank), up-projection (rank -> dim), and learnable gate scalar. The model trains WITH adapters present, so TTT is architectural.
- **ByteUnpatch**: Convert patch representations back to per-byte logits (project to patch_size * 256, reshape).

The adapter design is critical: gate starts at zero (adapters have no effect initially), but the model learns to USE the adapters during TTT-aware training.

- [ ] **Step 4: Size the model to fit 16MB**

Run parameter count experiments with different (dim, n_layers) configurations. At int6 quantization (0.75 bytes/param), 16MB fits ~21M params. At int8, ~16M params. Target ~14MB model leaving 2MB for code + overhead.

Configurations to try: (384, 12), (512, 8), (448, 10), (512, 10).

- [ ] **Step 5: Run test to verify forward pass**

- [ ] **Step 6: Commit**

```bash
git add model_itty.py tests/test_model.py
git commit -m "feat: Itty byte-level model with built-in adapters"
```

### Task 2.3: Byte-Level Training Loop

**Files:**
- Create: `train_itty_byte.py`
- Create: `train_itty_byte_mlx.py`

- [ ] **Step 1: Write byte-level training script**

Adapt the baseline training loop for byte-level:
- Use ByteStream instead of TokenStream
- Cross-entropy over 256 classes instead of 1024
- BPB = loss / ln(2) directly (no byte lookup tables)
- Same optimizer setup (Muon + Adam split)
- Same wallclock limit (600s)
- torch.compile the model

- [ ] **Step 2: Create MLX version for local iteration**

Mirror the CUDA training script using mlx.core / mlx.nn for local Mac testing.

- [ ] **Step 3: Verify training converges locally**

Run 200 steps, verify loss decreases steadily. Byte-level models start with ~ln(256) = 5.55 loss.

- [ ] **Step 4: Commit**

```bash
git add train_itty_byte.py train_itty_byte_mlx.py
git commit -m "feat: byte-level training loop with MLX local support"
```

### Task 2.4: Native TTT — Adaptation as Architecture

**Files:**
- Modify: `train_itty_byte.py`
- Modify: `model_itty.py`

**The key insight:** During training, randomly simulate TTT episodes. The model learns WITH adaptation happening, so TTT at test time isn't a surprise — it's what the model was born for.

- [ ] **Step 1: Implement TTT-aware training**

During training, periodically (20% of steps):
1. Take a batch of text
2. Split it into two halves: "context" and "target"
3. Freeze backbone, train adapters on "context" (inner loop: SGD, 3 steps)
4. Score "target" with adapted model (outer loss)
5. Backprop outer loss through backbone (meta-learning)
6. Reset adapters to zero for next step

This is meta-learning: optimize the backbone so that adapter adaptation on new text maximally improves predictions.

- [ ] **Step 2: Implement adapter utility methods**

Add to Itty class:
- `adapter_parameters()`: yield only adapter params
- `freeze_backbone()`: freeze everything except adapters
- `unfreeze_backbone()`: unfreeze all params
- `reset_adapters()`: zero out adapter weights and gates

- [ ] **Step 3: Implement training schedule**

Mix 80% regular next-byte prediction steps with 20% TTT meta-learning episodes. The ratio needs tuning on H100s.

- [ ] **Step 4: Test locally**

Run 100 steps with TTT training. Verify:
- Regular steps: loss decreases normally
- TTT steps: outer_loss (post-adaptation) is lower than pre-adaptation
- No NaN/inf gradients from the meta-learning loop

- [ ] **Step 5: Commit**

```bash
git add train_itty_byte.py model_itty.py
git commit -m "feat: TTT-aware meta-learning training — adaptation by design"
```

### Task 2.5: Evaluation with Native TTT

**Files:**
- Modify: `train_itty_byte.py`

- [ ] **Step 1: Implement byte-level TTT scoring**

Score-first TTT for byte-level model:
- Split validation bytes into chunks (8192 bytes each)
- For each chunk: SCORE under torch.inference_mode(), then ADAPT adapters only
- Cosine LR decay within adaptation, grad clip 1.0
- Last chunk scored but never adapted on

Because the model was TRAINED with TTT episodes, this adaptation should work better than bolt-on TTT.

- [ ] **Step 2: Compare native TTT vs bolt-on TTT**

Run both modes on the same model. Hypothesis: TTT-trained model adapts better at inference.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: native TTT scoring for byte-level model"
```

---

## Phase 3: Stack + Polish + Submit

### Task 3.1: Classical Compression Stacking

**Files:**
- Create: `classical_stack.py`

- [ ] **Step 1: Implement PPM predictor**

Prediction by Partial Matching — a classical text compression algorithm. Zero learnable parameters. Maintains byte n-gram counts and predicts based on longest matching context. Max order 5.

- [ ] **Step 2: Implement probability mixing**

Mix neural model probabilities with PPM probabilities: `mixed = alpha * neural + (1 - alpha) * ppm` where alpha ~0.95. The classical model catches simple repetitive patterns the neural model wastes capacity on.

- [ ] **Step 3: Integrate into scoring loop**

Run PPM alongside neural model during scoring. Measure BPB improvement. Expected: small but free (~0.001-0.005 BPB) with zero additional model parameters.

- [ ] **Step 4: Commit**

```bash
git add classical_stack.py
git commit -m "feat: classical PPM compression stacking"
```

### Task 3.2: Quantization + Artifact Packaging

**Files:**
- Create: `quantize.py`

- [ ] **Step 1: Implement int6 quantization for Itty**

Adapt baseline quantization: int6 per-row for attention/MLP weights, int8 for embeddings, GPTQ-lite clip search (5 percentiles, pick min MSE), zstd compression.

- [ ] **Step 2: Verify artifact fits in 16MB**

Total = compressed model bytes + code file bytes. Must be <= 16,000,000.

- [ ] **Step 3: Verify BPB after quantization**

Load quantized model, run scoring, measure degradation. Should be < 0.01 BPB degradation with int6 + GPTQ-lite.

- [ ] **Step 4: Commit**

```bash
git add quantize.py
git commit -m "feat: int6 quantization + artifact packaging"
```

### Task 3.3: Full Submission

- [ ] **Step 1: Run 3 seeds on 8xH100**

- [ ] **Step 2: Verify statistical significance**

All 3 runs within tight range, p < 0.01 that improvement > 0.005 nats over previous SOTA.

- [ ] **Step 3: Write submission README**

Explain our approach: byte-level (first in competition), adaptation-native (TTT trained in), classical stacking, designed for on-device deployment.

- [ ] **Step 4: Create submission.json**

```json
{
    "name": "itty_byte",
    "val_bpb": "<measured>",
    "bytes_total": "<measured>",
    "author": "pauljump",
    "github_id": "pauljump",
    "date": "2026-03-XX",
    "blurb": "Byte-level adaptive model native to 16MB. First byte-level submission."
}
```

- [ ] **Step 5: Submit PR**

```bash
gh pr create --repo openai/parameter-golf \
  --title "itty: byte-level adaptive model [BPB]" \
  --body "First byte-level submission. Designed from birth for 16MB, not shrunk."
```

---

## Phase 4: iPhone Deployment (Kit Home Integration)

### Task 4.1: CoreML Conversion

**Files:**
- Create: `export_coreml.py`

- [ ] **Step 1: Export Itty to CoreML**

Trace the base model (without TTT), convert via coremltools with FLOAT16 precision and ALL compute units (enables Neural Engine). Save as .mlpackage.

- [ ] **Step 2: Test CoreML model on Mac**

Load .mlpackage, run prediction, verify output matches PyTorch model within tolerance.

- [ ] **Step 3: Commit**

```bash
git add export_coreml.py
git commit -m "feat: CoreML export for on-device inference"
```

### Task 4.2: Kit Home Integration Design

- [ ] **Step 1: Write integration design doc**

Document how Itty integrates into Kit Home:
- On-device document classification (IEP, medical, insurance)
- On-device entity extraction (provider names, dates, medications)
- First-upload TTT adaptation: model specializes to this family's documents
- Gemini fallback only for deep reasoning tasks
- Privacy guarantee: model adapts on-device, data never leaves phone

- [ ] **Step 2: Commit**

```bash
git add docs/kit-home-integration.md
git commit -m "docs: Kit Home integration design"
```

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Byte-level 2x sequence length kills training speed | Can't train enough in 10 min | Patch processing reduces effective seq len by 4x. Net: 2x shorter than baseline. |
| Byte-level model scores worse than token-level | Not competitive | Phase 1 gives us a token-level fallback submission |
| Meta-learning (TTT training) is unstable | NaN gradients, no convergence | Start with simple adapter training, add meta-learning gradually |
| 16MB budget too tight for byte model + adapters | Artifact too large | Aggressive int5/int6 quantization, reduce adapter rank |
| RunPod credits denied or delayed | Can't iterate on H100s | Self-fund ~$50-100 of RunPod time for initial experiments |
| TTT rules tighten (Issue #402) | Our TTT approach invalidated | Our score-first protocol is the strictest legal version |
| Classical stacking adds negligible BPB | Wasted effort | PPM is simple to implement; if gains < 0.001, drop it |
| CoreML doesn't support all ops | Can't deploy to iPhone | Simplify architecture to CoreML-compatible ops only |

---

## Success Criteria

1. **Minimum viable**: Get on the leaderboard with any score (Phase 1)
2. **Competitive**: Beat the current merged SOTA (1.1194 BPB) with our byte-level model
3. **Novel**: First byte-level submission in the competition, recognized as a new approach
4. **Deployable**: Model runs on iPhone via CoreML with <100ms inference latency
5. **Win**: Beat ALL submissions including unmerged TTT entries (<0.56 BPB)

## Timeline

- **Day 1-2**: Phase 1 (foundation + proven wins + first submission)
- **Day 3-5**: Phase 2 (byte-level model + native TTT)
- **Day 6-7**: Phase 3 (stack + polish + submit Game 3)
- **Day 8+**: Iterate based on results, Phase 4 (iPhone)
- **April 30**: Competition deadline

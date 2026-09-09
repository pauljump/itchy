#!/usr/bin/env python3
"""
Test the core TTT thesis: does adapting on a document's first half
improve prediction on the second half?

1. Train a small model for N steps
2. Pick documents from validation data
3. For each doc: measure loss WITH vs WITHOUT adaptation
4. Report the gap
"""
import math
import time
import glob
import numpy as np
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from model_itchy import Itchy, COMPUTE_DTYPE, rms_norm

# Small model for fast local training
DIM = 192
NUM_LAYERS = 6
NUM_HEADS = 4
NUM_KV_HEADS = 2
MLP_MULT = 2
PATCH_SIZE = 4
ADAPTER_RANK = 16
SEQ_LEN = 512  # bytes per sequence
TRAIN_STEPS = 300
BATCH_TOKENS = 2048
LOG_EVERY = 50

# TTT test params
TTT_INNER_LR = 0.01
TTT_INNER_STEPS = 5
NUM_TEST_DOCS = 50


def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    return tokens.astype(np.int32, copy=False)


def get_batch(data: np.ndarray, pos: int, batch_tokens: int, seq_len: int):
    usable = (batch_tokens // seq_len) * seq_len
    chunk = data[pos : pos + usable + 1]
    if len(chunk) < usable + 1:
        chunk = data[:usable + 1]
    x = chunk[:-1].reshape(-1, seq_len)
    y = chunk[1:].reshape(-1, seq_len)
    return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


def main():
    print("=" * 60)
    print("TTT THESIS VALIDATION")
    print("=" * 60)

    # Load data
    train_path = sorted(glob.glob("data/datasets/fineweb10B_bytes/fineweb_train_*.bin"))[0]
    val_path = sorted(glob.glob("data/datasets/fineweb10B_bytes/fineweb_val_*.bin"))[0]
    train_data = load_data_shard(Path(train_path))
    val_data = load_data_shard(Path(val_path))
    print(f"Train: {len(train_data):,} bytes | Val: {len(val_data):,} bytes")

    # Build small model
    mx.random.seed(42)
    model = Itchy(
        dim=DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS, mlp_mult=MLP_MULT,
        patch_size=PATCH_SIZE, adapter_rank=ADAPTER_RANK,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
    )
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Model: {n_params:,} params ({n_params/1e6:.1f}M)")
    print(f"Config: dim={DIM} layers={NUM_LAYERS} heads={NUM_HEADS} rank={ADAPTER_RANK}")
    print()

    # ==========================================
    # PHASE 1: Train the model
    # ==========================================
    print("PHASE 1: Training...")
    loss_and_grad = nn.value_and_grad(model, lambda x, y: model.loss(x, y))

    # Simple Adam-ish SGD for speed
    lr = 0.001
    pos = 0
    t0 = time.perf_counter()

    for step in range(1, TRAIN_STEPS + 1):
        x, y = get_batch(train_data, pos, BATCH_TOKENS, SEQ_LEN)
        pos = (pos + BATCH_TOKENS) % (len(train_data) - BATCH_TOKENS - 1)

        loss, grads = loss_and_grad(x, y)
        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.parameters()))
        updated = {k: p - lr * flat_grads[k] for k, p in flat_params.items()}
        model.update(tree_unflatten(list(updated.items())))
        mx.synchronize()

        if step % LOG_EVERY == 0 or step == 1:
            l = float(loss.item())
            bpb = l / math.log(2.0)
            elapsed = time.perf_counter() - t0
            print(f"  step {step:>4}/{TRAIN_STEPS} | loss {l:.4f} | bpb {bpb:.4f} | {elapsed:.0f}s")

    elapsed = time.perf_counter() - t0
    print(f"Training done in {elapsed:.0f}s")
    print()

    # ==========================================
    # PHASE 2: Test TTT adaptation
    # ==========================================
    print("PHASE 2: Testing TTT thesis...")
    print(f"  {NUM_TEST_DOCS} documents | {TTT_INNER_STEPS} adaptation steps | lr={TTT_INNER_LR}")
    print()

    doc_len = SEQ_LEN * 2  # need enough for context + target
    half = SEQ_LEN

    losses_no_adapt = []
    losses_with_adapt = []

    for doc_idx in range(NUM_TEST_DOCS):
        # Pick a random document-sized chunk from validation
        start = doc_idx * doc_len
        if start + doc_len + 1 > len(val_data):
            break

        chunk = val_data[start : start + doc_len + 1]
        context_x = mx.array(chunk[:half].reshape(1, -1), dtype=mx.int32)
        context_y = mx.array(chunk[1:half + 1].reshape(1, -1), dtype=mx.int32)
        target_x = mx.array(chunk[half:half + half].reshape(1, -1), dtype=mx.int32)
        target_y = mx.array(chunk[half + 1:half + half + 1].reshape(1, -1), dtype=mx.int32)

        # Loss WITHOUT adaptation
        loss_no = model.loss(target_x, target_y)
        mx.synchronize()
        loss_no_val = float(loss_no.item())
        losses_no_adapt.append(loss_no_val)

        # Save adapter state
        adapter_state = model.get_adapter_state()

        # Adapt on context (inner loop)
        for _ in range(TTT_INNER_STEPS):
            adapt_loss, adapt_grads = nn.value_and_grad(
                model, lambda cx, cy: model.loss(cx, cy)
            )(context_x, context_y)

            flat_grads = dict(tree_flatten(adapt_grads))
            flat_params = dict(tree_flatten(model.parameters()))
            updated = {}
            for k, p in flat_params.items():
                if "adapter" in k and k in flat_grads:
                    updated[k] = p - TTT_INNER_LR * flat_grads[k]
                else:
                    updated[k] = p
            model.update(tree_unflatten(list(updated.items())))
            mx.synchronize()

        # Loss WITH adaptation
        loss_with = model.loss(target_x, target_y)
        mx.synchronize()
        loss_with_val = float(loss_with.item())
        losses_with_adapt.append(loss_with_val)

        # Restore adapter state
        model.set_adapter_state(adapter_state)

        if (doc_idx + 1) % 10 == 0:
            avg_no = np.mean(losses_no_adapt[-10:])
            avg_with = np.mean(losses_with_adapt[-10:])
            delta = avg_with - avg_no
            print(f"  docs {doc_idx - 8:>3}-{doc_idx + 1:>3} | "
                  f"no_adapt: {avg_no:.4f} | adapted: {avg_with:.4f} | "
                  f"delta: {delta:+.4f} ({'BETTER' if delta < 0 else 'worse'})")

    # Final summary
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    avg_no = np.mean(losses_no_adapt)
    avg_with = np.mean(losses_with_adapt)
    delta = avg_with - avg_no
    bpb_no = avg_no / math.log(2.0)
    bpb_with = avg_with / math.log(2.0)
    bpb_delta = bpb_with - bpb_no

    print(f"  Without adaptation: {avg_no:.4f} nats ({bpb_no:.4f} BPB)")
    print(f"  With adaptation:    {avg_with:.4f} nats ({bpb_with:.4f} BPB)")
    print(f"  Delta:              {delta:+.4f} nats ({bpb_delta:+.4f} BPB)")
    print()

    n_better = sum(1 for a, b in zip(losses_no_adapt, losses_with_adapt) if b < a)
    print(f"  Adaptation helped on {n_better}/{len(losses_no_adapt)} documents ({100*n_better/len(losses_no_adapt):.0f}%)")
    print()

    if delta < -0.01:
        print("  THESIS VALIDATED: TTT adaptation meaningfully improves prediction.")
    elif delta < 0:
        print("  THESIS WEAKLY SUPPORTED: Small improvement from adaptation.")
    else:
        print("  THESIS NOT SUPPORTED: Adaptation did not help.")
        print("  (This may improve with more training or TTT-aware meta-learning.)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
TTT thesis test v2: Three conditions:
A) No adaptation (baseline)
B) Adapt ALL parameters (like competition TTT)
C) Adapt only adapters with gate=1.0 (unblocked adapters)
"""
import math
import time
import glob
import numpy as np
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from model_itchy import Itchy

DIM = 192
NUM_LAYERS = 6
NUM_HEADS = 4
NUM_KV_HEADS = 2
MLP_MULT = 2
PATCH_SIZE = 4
ADAPTER_RANK = 16
SEQ_LEN = 512
TRAIN_STEPS = 500
BATCH_TOKENS = 2048
LOG_EVERY = 100

TTT_LR = 0.002
TTT_STEPS = 5
NUM_TEST_DOCS = 30


def load_data_shard(path):
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    return tokens.astype(np.int32, copy=False)


def get_batch(data, pos, batch_tokens, seq_len):
    usable = (batch_tokens // seq_len) * seq_len
    end = min(pos + usable + 1, len(data))
    chunk = data[pos:end]
    if len(chunk) < usable + 1:
        chunk = data[:usable + 1]
    x = chunk[:-1].reshape(-1, seq_len)
    y = chunk[1:].reshape(-1, seq_len)
    return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


def adapt_and_score(model, context_x, context_y, target_x, target_y,
                    lr, steps, adapter_only=False):
    """Adapt model on context, score on target, return target loss."""
    # Save full state
    saved_params = {k: mx.array(v) for k, v in dict(tree_flatten(model.parameters())).items()}

    for _ in range(steps):
        loss, grads = nn.value_and_grad(
            model, lambda cx, cy: model.loss(cx, cy)
        )(context_x, context_y)

        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.parameters()))
        updated = {}
        for k, p in flat_params.items():
            if adapter_only and "adapter" not in k:
                updated[k] = p
            elif k in flat_grads:
                updated[k] = p - lr * flat_grads[k]
            else:
                updated[k] = p
        model.update(tree_unflatten(list(updated.items())))
        mx.synchronize()

    # Score target
    target_loss = model.loss(target_x, target_y)
    mx.synchronize()
    result = float(target_loss.item())

    # Restore state
    model.update(tree_unflatten(list(saved_params.items())))

    return result


def main():
    print("=" * 70)
    print("TTT THESIS VALIDATION v2")
    print("=" * 70)

    train_data = load_data_shard(Path(sorted(glob.glob("data/datasets/fineweb10B_bytes/fineweb_train_*.bin"))[0]))
    val_data = load_data_shard(Path(sorted(glob.glob("data/datasets/fineweb10B_bytes/fineweb_val_*.bin"))[0]))
    print(f"Train: {len(train_data):,} | Val: {len(val_data):,}")

    mx.random.seed(42)
    model = Itchy(
        dim=DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS, mlp_mult=MLP_MULT,
        patch_size=PATCH_SIZE, adapter_rank=ADAPTER_RANK,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
    )
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Model: {n_params:,} params | dim={DIM} layers={NUM_LAYERS}")

    # ==========================================
    # PHASE 1: Train
    # ==========================================
    print(f"\nPHASE 1: Training for {TRAIN_STEPS} steps...")
    loss_and_grad = nn.value_and_grad(model, lambda x, y: model.loss(x, y))
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
            print(f"  step {step:>4} | loss {l:.4f} | bpb {l/math.log(2):.4f} | {time.perf_counter()-t0:.0f}s")

    print(f"Training done in {time.perf_counter()-t0:.0f}s\n")

    # ==========================================
    # PHASE 2: Test three conditions
    # ==========================================
    print("PHASE 2: Testing TTT...")
    print(f"  TTT params: lr={TTT_LR}, steps={TTT_STEPS}, docs={NUM_TEST_DOCS}")
    print()

    doc_len = SEQ_LEN * 2
    half = SEQ_LEN

    results_A = []  # no adaptation
    results_B = []  # adapt all params
    results_C = []  # adapt adapters only (gate=1)

    for doc_idx in range(NUM_TEST_DOCS):
        start = doc_idx * doc_len * 3  # spread docs apart
        if start + doc_len + 1 > len(val_data):
            break

        chunk = val_data[start : start + doc_len + 1]
        context_x = mx.array(chunk[:half].reshape(1, -1), dtype=mx.int32)
        context_y = mx.array(chunk[1:half + 1].reshape(1, -1), dtype=mx.int32)
        target_x = mx.array(chunk[half:half + half].reshape(1, -1), dtype=mx.int32)
        target_y = mx.array(chunk[half + 1:half + half + 1].reshape(1, -1), dtype=mx.int32)

        # A: No adaptation
        loss_a = model.loss(target_x, target_y)
        mx.synchronize()
        results_A.append(float(loss_a.item()))

        # B: Adapt ALL parameters
        loss_b = adapt_and_score(model, context_x, context_y, target_x, target_y,
                                 lr=TTT_LR, steps=TTT_STEPS, adapter_only=False)
        results_B.append(loss_b)

        # C: Adapt adapters only, but first set gates to 1.0
        for block in model.blocks:
            block.adapter.gate = mx.array(1.0)
        loss_c = adapt_and_score(model, context_x, context_y, target_x, target_y,
                                 lr=TTT_LR, steps=TTT_STEPS, adapter_only=True)
        # Reset gates to 0
        for block in model.blocks:
            block.adapter.gate = mx.array(0.0)
        results_C.append(loss_c)

        if (doc_idx + 1) % 10 == 0:
            a = np.mean(results_A[-10:])
            b = np.mean(results_B[-10:])
            c = np.mean(results_C[-10:])
            print(f"  docs {doc_idx-8:>2}-{doc_idx+1:>2} | "
                  f"none: {a:.4f} | all_params: {b:.4f} ({b-a:+.4f}) | "
                  f"adapters: {c:.4f} ({c-a:+.4f})")

    # Final
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    a, b, c = np.mean(results_A), np.mean(results_B), np.mean(results_C)
    print(f"  A) No adaptation:         {a:.4f} nats  ({a/math.log(2):.4f} BPB)")
    print(f"  B) Adapt ALL params:      {b:.4f} nats  ({b/math.log(2):.4f} BPB)  delta: {b-a:+.4f}")
    print(f"  C) Adapt adapters (gate=1): {c:.4f} nats  ({c/math.log(2):.4f} BPB)  delta: {c-a:+.4f}")
    print()

    n_b = sum(1 for x, y in zip(results_A, results_B) if y < x)
    n_c = sum(1 for x, y in zip(results_A, results_C) if y < x)
    n = len(results_A)
    print(f"  All-param TTT helped: {n_b}/{n} docs ({100*n_b/n:.0f}%)")
    print(f"  Adapter TTT helped:   {n_c}/{n} docs ({100*n_c/n:.0f}%)")
    print()

    if b - a < -0.01:
        print("  ALL-PARAM TTT WORKS. The model benefits from per-document adaptation.")
    if c - a < -0.01:
        print("  ADAPTER TTT WORKS. LoRA adapters can capture per-document patterns.")
    if b - a >= 0 and c - a >= 0:
        print("  NEITHER APPROACH HELPED. Model may need more training first.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Itchy byte-level training script (MLX, for local Mac iteration).

Key differences from baseline:
- Byte-level data (256 vocab, no tokenizer)
- Patch processing (4 bytes -> 1 patch)
- BPB = loss_nats / ln(2) directly (no token-to-byte conversion)
- Built-in LoRA adapters for future TTT
"""
from __future__ import annotations

import glob
import math
import os
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from model_itchy import Itchy, rms_norm, COMPUTE_DTYPE

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================

class Hyperparameters:
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_bytes")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    # Training loop
    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 131_072))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 131_072))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 4))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 1024))  # bytes, must be divisible by patch_size
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 4_096))
    mlx_eager_eval: bool = bool(int(os.environ.get("MLX_EAGER_EVAL", "1")))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 10))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 1200))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model
    model_dim: int = int(os.environ.get("MODEL_DIM", 384))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 12))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 3))
    patch_size: int = int(os.environ.get("PATCH_SIZE", 4))
    adapter_rank: int = int(os.environ.get("ADAPTER_RANK", 32))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Optimizer
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    embed_lr: float = float(os.environ.get("EMBED_LR", 0.05))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # TTT Meta-learning
    ttt_ratio: float = float(os.environ.get("TTT_RATIO", 0.2))  # fraction of steps that are TTT meta-learning
    ttt_inner_steps: int = int(os.environ.get("TTT_INNER_STEPS", 3))
    ttt_inner_lr: float = float(os.environ.get("TTT_INNER_LR", 0.01))
    ttt_seq_len: int = int(os.environ.get("TTT_SEQ_LEN", 0))  # 0 = use train_seq_len

    out_dir: str = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if warmdown_start <= step < self.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight", "gate",
)

# ==============================================================================
# DATA LOADING — BYTE LEVEL
# ==============================================================================

def load_data_shard(path: Path) -> np.ndarray:
    """Load a binary shard. byte260 shards use uint16 with values 0-259."""
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


class ByteStream:
    """Stream raw byte data from shards."""
    def __init__(self, pattern: str, log_fn: Callable[[str], None] | None = None):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn:
                self.log_fn(f"WARNING: starting epoch:{self.epoch}")
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


class ByteLoader:
    def __init__(self, pattern: str, log_fn: Callable[[str], None] | None = None):
        self.stream = ByteStream(pattern, log_fn=log_fn)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


# ==============================================================================
# MATH HELPERS
# ==============================================================================

def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


def token_chunks(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[int]:
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks: list[int] = []
    remaining = usable_total
    while remaining > 0:
        chunk = min(remaining, usable_chunk)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def accumulate_flat_grads(
    accum: dict[str, mx.array] | None,
    grads_tree: dict,
    scale: float,
) -> dict[str, mx.array]:
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum


# ==============================================================================
# OPTIMIZERS (MUON + ADAM SPLIT)
# ==============================================================================

class Muon:
    def __init__(self, keys: list[str], params: dict[str, mx.array], args: Hyperparameters):
        self.keys = keys
        self.args = args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params: dict[str, mx.array], grads: dict[str, mx.array], step: int, lr_mul: float) -> dict[str, mx.array]:
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        out: dict[str, mx.array] = {}
        for k in self.keys:
            p = params[k]
            g = grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            out[k] = p - lr * (g_ortho * scale).astype(p.dtype)
        return out


class SplitOptimizers:
    def __init__(self, model: Itchy, args: Hyperparameters):
        self.args = args
        params = dict(tree_flatten(model.parameters()))

        # Embedding keys: byte_embed + patch projections + unpatch
        self.embed_keys = [k for k in params if "embed" in k or "unpatch" in k]

        # Matrix keys: 2D params in blocks, excluding control tensors and adapters
        self.matrix_keys = [
            k for k, p in params.items()
            if k.startswith("blocks.") and p.ndim == 2
            and not any(pat in k for pat in CONTROL_TENSOR_NAME_PATTERNS)
            and "adapter" not in k
        ]

        # Scalar keys: everything else (control tensors, skip_weights, adapter params)
        used = set(self.embed_keys + self.matrix_keys)
        self.scalar_keys = [k for k in params if k not in used]

        self.muon = Muon(self.matrix_keys, params, args)
        self.adam_embed = optim.Adam(
            learning_rate=args.embed_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )
        self.adam_scalar = optim.Adam(
            learning_rate=args.scalar_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )

    def step(self, model: Itchy, grads_tree: dict, step: int, lr_mul: float) -> None:
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)

        # Muon for matrix params
        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))

        # Adam for embeddings
        self.adam_embed.learning_rate = self.args.embed_lr * lr_mul
        embed_grads = {k: grads[k] for k in self.embed_keys if k in grads}
        embed_params = {k: params[k] for k in self.embed_keys if k in grads}
        if embed_grads:
            updated.update(self.adam_embed.apply_gradients(embed_grads, embed_params))

        # Adam for scalars + adapters
        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys if k in grads}
        scalar_params = {k: params[k] for k in self.scalar_keys if k in grads}
        if scalar_grads:
            updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))

        model.update(tree_unflatten(list(updated.items())))


# ==============================================================================
# VALIDATION
# ==============================================================================

def load_validation_bytes(pattern: str, seq_len: int) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(f) for f in files], axis=0))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def eval_val_bytes(
    args: Hyperparameters,
    compiled_loss,
    val_bytes: np.ndarray,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[float, float]:
    """Returns (val_loss_nats, val_bpb). For byte-level: BPB = loss / ln(2)."""
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    seq_len = args.train_seq_len
    val_batch_seqs = val_batch_tokens // seq_len
    total_seqs = (val_bytes.size - 1) // seq_len
    total_batches = max((total_seqs + val_batch_seqs - 1) // val_batch_seqs, 1)
    total_loss_sum = 0.0
    total_count = 0.0

    for batch_idx, batch_seq_start in enumerate(range(0, total_seqs, val_batch_seqs), start=1):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * seq_len
        raw_end = batch_seq_end * seq_len + 1
        chunk = val_bytes[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, seq_len)
        y_np = chunk[1:].reshape(-1, seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        n = float(y.size)
        batch_loss = compiled_loss(x, y).astype(mx.float32)
        mx.synchronize()
        total_loss_sum += float(batch_loss.item()) * n
        total_count += n
        if log_fn and total_batches > 1 and (batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0):
            log_fn(f"val_progress:{batch_idx}/{total_batches}")

    val_loss = total_loss_sum / total_count
    val_bpb = val_loss / math.log(2.0)
    return val_loss, val_bpb


# ==============================================================================
# TTT META-LEARNING
# ==============================================================================

def ttt_adapter_sgd_step(model: Itchy, context_x: mx.array, context_y: mx.array, lr: float) -> mx.array:
    """One inner-loop SGD step: train ONLY adapter params on context, return context loss."""
    # Compute loss and gradients w.r.t. ALL model params
    loss, grads = nn.value_and_grad(model, lambda cx, cy: model.loss(cx, cy))(context_x, context_y)

    # Extract only adapter gradients, apply SGD update to adapter params only
    flat_grads = dict(tree_flatten(grads))
    flat_params = dict(tree_flatten(model.parameters()))
    updated = {}
    for k, p in flat_params.items():
        if "adapter" in k and k in flat_grads:
            updated[k] = p - lr * flat_grads[k]
        else:
            updated[k] = p
    model.update(tree_unflatten(list(updated.items())))
    return loss


def ttt_meta_step(
    args: Hyperparameters,
    model: Itchy,
    train_loader: ByteLoader,
) -> tuple[mx.array, dict]:
    """
    One TTT meta-learning episode:
    1. Get batch, split into context (first half) / target (second half)
    2. Save adapter state
    3. Inner loop: SGD on adapters using context
    4. Compute outer loss on target (with adapted model)
    5. Compute gradients of outer loss w.r.t. ALL params (meta-gradient)
    6. Restore adapter state
    7. Return (outer_loss, grads) for backbone update
    """
    seq_len = args.ttt_seq_len if args.ttt_seq_len > 0 else args.train_seq_len
    # Ensure seq_len is divisible by patch_size and even (for splitting)
    seq_len = (seq_len // (args.patch_size * 2)) * (args.patch_size * 2)
    half = seq_len // 2

    # Get a batch of data
    x_full, y_full = train_loader.next_batch(seq_len, seq_len)
    # x_full, y_full: (1, seq_len) — single sequence for meta-learning

    # Split into context and target halves
    context_x = x_full[:, :half]
    context_y = y_full[:, :half]
    target_x = x_full[:, half:]
    target_y = y_full[:, half:]

    # Save adapter state before inner loop
    adapter_snapshot = model.get_adapter_state()

    # Inner loop: train adapters on context via SGD
    for _inner in range(args.ttt_inner_steps):
        _ctx_loss = ttt_adapter_sgd_step(model, context_x, context_y, args.ttt_inner_lr)
        if args.mlx_eager_eval:
            mx.synchronize()

    # Outer loss: evaluate on target with adapted model
    # We need gradients w.r.t. ALL parameters (the backbone learns to be adaptable)
    outer_loss, outer_grads = nn.value_and_grad(model, lambda tx, ty: model.loss(tx, ty))(target_x, target_y)

    if args.mlx_eager_eval:
        mx.synchronize()

    # Restore adapter state (reset to pre-adaptation values)
    model.set_adapter_state(adapter_snapshot)

    return outer_loss, outer_grads


# ==============================================================================
# TRAINING
# ==============================================================================

def loss_and_grad_chunked(
    args: Hyperparameters,
    train_loader: ByteLoader,
    compiled_loss_and_grad,
) -> tuple[mx.array, dict]:
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    for chunk_tokens in chunk_sizes:
        x, y = train_loader.next_batch(chunk_tokens, args.train_seq_len)
        loss, grads = compiled_loss_and_grad(x, y)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
        if args.mlx_eager_eval:
            mx.synchronize()
    return loss_value, tree_unflatten(list(grad_accum.items()))


def main() -> None:
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"itchy_{args.run_id}.txt"
    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)

    # Load validation data
    val_bytes = load_validation_bytes(args.val_files, args.train_seq_len)
    log(f"val_bytes:{val_bytes.size - 1} unique_values:{len(np.unique(val_bytes))}")

    # Model
    mx.random.seed(args.seed)
    model = Itchy(
        dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        patch_size=args.patch_size,
        adapter_rank=args.adapter_rank,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    )

    n_params = sum(int(v.size) for _, v in tree_flatten(model.parameters()))
    log(f"model_params:{n_params} ({n_params/1e6:.1f}M)")
    log(f"dim:{args.model_dim} layers:{args.num_layers} heads:{args.num_heads} kv_heads:{args.num_kv_heads}")
    log(f"mlp_mult:{args.mlp_mult} patch_size:{args.patch_size} adapter_rank:{args.adapter_rank}")
    log(f"int8_size:{n_params/1e6:.1f}MB int6_size:{n_params*0.75/1e6:.1f}MB")
    log(f"seq_len:{args.train_seq_len} (effective patches: {args.train_seq_len // args.patch_size})")

    opt = SplitOptimizers(model, args)
    log(f"optimizer: muon_keys:{len(opt.matrix_keys)} embed_keys:{len(opt.embed_keys)} scalar_keys:{len(opt.scalar_keys)}")

    train_loader = ByteLoader(args.train_files, log_fn=log)

    # Compiled functions
    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state,
        outputs=model.state,
    )

    # Warmup
    if args.warmup_steps > 0:
        for warmup_step in range(args.warmup_steps):
            warmup_loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            mx.synchronize()
            if warmup_step + 1 == args.warmup_steps:
                log(f"warmup_done:{warmup_step + 1} steps")

        # Prime the standalone loss graph
        small_x = mx.array(val_bytes[:args.train_seq_len + 1][:-1].reshape(1, -1), dtype=mx.int32)
        small_y = mx.array(val_bytes[:args.train_seq_len + 1][1:].reshape(1, -1), dtype=mx.int32)
        prime_loss = compiled_loss(small_x, small_y)
        mx.synchronize()

        # Reset data loader
        train_loader = ByteLoader(args.train_files, log_fn=log)

    # Training loop
    ttt_enabled = args.ttt_ratio > 0.0
    ttt_step_interval = max(int(1.0 / args.ttt_ratio), 1) if ttt_enabled else 0
    log(f"starting training: {args.iterations} iterations, {args.max_wallclock_seconds}s wallclock")
    if ttt_enabled:
        log(f"ttt_meta_learning: ratio={args.ttt_ratio} inner_steps={args.ttt_inner_steps} inner_lr={args.ttt_inner_lr} every={ttt_step_interval}steps")
    t0 = time.perf_counter()
    train_loss_accum = 0.0
    train_loss_count = 0
    ttt_loss_accum = 0.0
    ttt_loss_count = 0

    for step in range(1, args.iterations + 1):
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if args.max_wallclock_seconds > 0 and elapsed_ms > args.max_wallclock_seconds * 1000.0:
            log(f"wallclock_limit_reached step:{step} elapsed:{elapsed_ms/1000:.1f}s")
            break

        lr_mul = args.lr_mul(step, elapsed_ms)

        # Decide: TTT meta-learning step or regular training step
        is_ttt_step = ttt_enabled and (step % ttt_step_interval == 0)

        if is_ttt_step:
            # TTT meta-learning episode
            ttt_loss, ttt_grads = ttt_meta_step(args, model, train_loader)
            mx.synchronize()

            opt.step(model, ttt_grads, step=step, lr_mul=lr_mul)

            # Reset adapters after meta-step (they should start fresh each time)
            model.reset_adapters()

            ttt_loss_accum += float(ttt_loss.item())
            ttt_loss_count += 1
            # Also count toward train loss for logging
            train_loss_accum += float(ttt_loss.item())
            train_loss_count += 1
        else:
            # Regular training step with gradient accumulation
            total_loss = mx.array(0.0, dtype=mx.float32)
            grad_accum: dict[str, mx.array] | None = None
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
                total_loss = total_loss + loss * grad_scale
                grad_accum = accumulate_flat_grads(grad_accum, grads, grad_scale)

            mx.synchronize()

            opt.step(model, tree_unflatten(list(grad_accum.items())), step=step, lr_mul=lr_mul)

            train_loss_accum += float(total_loss.item())
            train_loss_count += 1

        if step % args.train_log_every == 0 or step == 1:
            avg_loss = train_loss_accum / max(train_loss_count, 1)
            avg_bpb = avg_loss / math.log(2.0)
            elapsed_s = (time.perf_counter() - t0)
            ms_per_step = elapsed_ms / step
            ttt_info = ""
            if ttt_loss_count > 0:
                ttt_avg = ttt_loss_accum / ttt_loss_count
                ttt_info = f" ttt_loss:{ttt_avg:.4f}({ttt_loss_count}eps)"
            log(f"step:{step} loss:{avg_loss:.4f} bpb:{avg_bpb:.4f} lr_mul:{lr_mul:.4f} ms/step:{ms_per_step:.0f} elapsed:{elapsed_s:.1f}s{ttt_info}")
            train_loss_accum = 0.0
            train_loss_count = 0
            ttt_loss_accum = 0.0
            ttt_loss_count = 0

        if args.val_loss_every > 0 and step % args.val_loss_every == 0:
            val_loss, val_bpb = eval_val_bytes(args, compiled_loss, val_bytes, log_fn=log)
            log(f"val step:{step} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    # Final validation
    elapsed_s = time.perf_counter() - t0
    log(f"training_done elapsed:{elapsed_s:.1f}s")

    log("running final validation...")
    val_loss, val_bpb = eval_val_bytes(args, compiled_loss, val_bytes, log_fn=log)
    log(f"FINAL val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")
    log(f"expected_starting_loss: ln(260)={math.log(260):.4f} -> bpb={math.log(260)/math.log(2):.4f}")


if __name__ == "__main__":
    main()

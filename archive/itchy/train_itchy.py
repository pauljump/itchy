"""
Itchy: byte-level adaptive model — CUDA training script for 8xH100.

Usage:
    torchrun --standalone --nproc_per_node=8 train_itchy.py

Key differences from baseline train_gpt.py:
- Byte-level (256 vocab, no tokenizer, BPB = loss/ln2)
- Patch processing (4 bytes -> 1 patch)
- LoRA adapters in every block for TTT
- TTT meta-learning episodes during training (20% of steps)
"""
from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_bytes")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model
    model_dim = int(os.environ.get("MODEL_DIM", 384))
    num_layers = int(os.environ.get("NUM_LAYERS", 12))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    patch_size = int(os.environ.get("PATCH_SIZE", 4))
    adapter_rank = int(os.environ.get("ADAPTER_RANK", 32))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Optimizer
    embed_lr = float(os.environ.get("EMBED_LR", 0.05))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # TTT Meta-learning
    ttt_ratio = float(os.environ.get("TTT_RATIO", 0.2))
    ttt_inner_steps = int(os.environ.get("TTT_INNER_STEPS", 3))
    ttt_inner_lr = float(os.environ.get("TTT_INNER_LR", 0.01))


# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()


# -----------------------------
# DATA LOADING — BYTE LEVEL
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class ByteStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedByteLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = ByteStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def load_validation_bytes(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


# -----------------------------
# BYTE-LEVEL VALIDATION
# -----------------------------

def eval_val_bytes(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_bytes: Tensor,
) -> tuple[float, float]:
    """Returns (val_loss_nats, val_bpb). BPB = loss / ln(2) for byte-level."""
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_bytes.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_count = torch.zeros((), device=device, dtype=torch.float64)

    model_module = model.module if hasattr(model, "module") else model
    model_module.set_mode("eval")
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_bytes[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model_module.forward_loss(x, y).detach()
            n = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * n
            val_count += n

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_count, op=dist.ReduceOp.SUM)

    val_loss = float((val_loss_sum / val_count).item())
    val_bpb = val_loss / math.log(2.0)
    model_module.set_mode("train")
    return val_loss, val_bpb


# -----------------------------
# QUANTIZATION (INT8 + ZLIB)
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,mlp_scale,resid_mix,q_gain,skip_weight,gate",
    ).split(",") if p
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_CLIP_Q = 99.99984 / 100.0

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict] = {}
    stats = dict.fromkeys(("param_count", "int8_payload_bytes", "baseline_tensor_bytes"), 0)

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().float().contiguous()
        stats["param_count"] += t.numel()
        stats["baseline_tensor_bytes"] += t.numel() * t.element_size()

        if not t.is_floating_point() or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            stored = t.half().contiguous() if t.is_floating_point() else t
            passthrough[name] = stored
            if t.is_floating_point() and t.dtype != torch.float16:
                passthrough_orig_dtypes[name] = str(tensor.dtype).removeprefix("torch.")
            stats["int8_payload_bytes"] += stored.numel() * stored.element_size()
            continue

        if t.ndim == 2:
            clip_abs = torch.quantile(t.abs(), INT8_CLIP_Q, dim=1)
            clipped = t.clamp(-clip_abs[:, None], clip_abs[:, None])
            scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
            q = (clipped / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
            scales[name] = scale.half()
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        else:
            clip_abs = float(torch.quantile(t.abs().flatten(), INT8_CLIP_Q).item()) if t.numel() else 0.0
            scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0)
            q = t.clamp(-clip_abs, clip_abs).div(scale).round().clamp(-127, 127).to(torch.int8)
            scales[name] = scale

        quantized[name] = q.contiguous()
        dtypes[name] = str(tensor.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += q.numel() + scales[name].numel() * scales[name].element_size()

    obj = {"__quant_format__": "int8_clean_per_row_v1", "quantized": quantized,
           "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj: dict) -> dict[str, Tensor]:
    out = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name].float()
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype)
        else:
            out[name] = (q.float() * s.item()).to(dtype)
    for name, t in obj["passthrough"].items():
        orig = passthrough_orig_dtypes.get(name)
        out[name] = t.to(getattr(torch, orig)) if orig else t
    return out


# -----------------------------
# MODEL
# -----------------------------

class RMSNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class LoRAAdapter(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.down = nn.Parameter(torch.randn(rank, dim) / math.sqrt(dim))
        self.up = nn.Parameter(torch.zeros(dim, rank))
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: Tensor) -> Tensor:
        h = F.linear(x, self.down.to(x.dtype))
        h = F.linear(h, self.up.to(x.dtype))
        return self.gate.to(x.dtype) * h


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class AdaptiveBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, adapter_rank: int):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.adapter = LoRAAdapter(dim, adapter_rank)

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        attn_out = attn_out + self.adapter(attn_out)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class BytePatchEmbed(nn.Module):
    def __init__(self, dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.byte_embed = nn.Embedding(260, dim)
        self.patch_proj = CastedLinear(dim * patch_size, dim, bias=False)

    def forward(self, byte_ids: Tensor) -> Tensor:
        bsz, seq_len = byte_ids.shape
        n_patches = seq_len // self.patch_size
        x = self.byte_embed(byte_ids)
        x = x.reshape(bsz, n_patches, self.patch_size * x.size(-1))
        return self.patch_proj(x)


class ByteUnpatch(nn.Module):
    def __init__(self, dim: int, patch_size: int, vocab_size: int = 260):
        super().__init__()
        self.patch_size = patch_size
        self.vocab_size = vocab_size
        self.proj = CastedLinear(dim, patch_size * vocab_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        bsz = x.shape[0]
        logits = self.proj(x)
        return logits.reshape(bsz, -1, self.vocab_size)


class Itchy(nn.Module):
    def __init__(self, dim: int, num_layers: int, num_heads: int, num_kv_heads: int,
                 mlp_mult: int, patch_size: int, adapter_rank: int,
                 logit_softcap: float, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.logit_softcap = logit_softcap
        self._mode = "train"

        self.embed = BytePatchEmbed(dim, patch_size)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            AdaptiveBlock(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, adapter_rank)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.unpatch = ByteUnpatch(dim, patch_size, vocab_size=260)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def set_mode(self, mode: str):
        self._mode = mode

    def forward_features(self, byte_ids: Tensor) -> Tensor:
        x = F.rms_norm(self.embed(byte_ids), (self.dim,))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def forward_loss(self, byte_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.forward_features(byte_ids)
        logits = self.unpatch(x)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float().reshape(-1, 260), target_ids.reshape(-1), reduction="mean")

    def forward(self, byte_ids: Tensor, target_ids: Tensor) -> Tensor:
        return self.forward_loss(byte_ids, target_ids)

    def adapter_parameters(self) -> list[nn.Parameter]:
        return [p for n, p in self.named_parameters() if "adapter" in n]

    def non_adapter_parameters(self) -> list[nn.Parameter]:
        return [p for n, p in self.named_parameters() if "adapter" not in n]

    def get_adapter_state(self) -> list[tuple[Tensor, Tensor, Tensor]]:
        return [(b.adapter.down.data.clone(), b.adapter.up.data.clone(), b.adapter.gate.data.clone())
                for b in self.blocks]

    def set_adapter_state(self, state: list[tuple[Tensor, Tensor, Tensor]]):
        for block, (down, up, gate) in zip(self.blocks, state):
            block.adapter.down.data.copy_(down)
            block.adapter.up.data.copy_(up)
            block.adapter.gate.data.copy_(gate)

    def reset_adapters(self):
        for block in self.blocks:
            block.adapter.gate.data.zero_()
            block.adapter.up.data.zero_()


def restore_low_dim_params_to_fp32(module: nn.Module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


# -----------------------------
# TTT META-LEARNING
# -----------------------------

def ttt_meta_step(
    args: Hyperparameters,
    base_model: Itchy,
    train_loader: DistributedByteLoader,
    grad_accum_steps: int,
    device: torch.device,
) -> Tensor:
    """
    One TTT meta-learning episode:
    1. Get batch, split into context/target
    2. Save adapter state
    3. Inner loop: SGD on adapters using context
    4. Compute outer loss on target
    5. Backward through backbone (meta-gradient)
    6. Restore adapter state
    Returns outer_loss for logging.
    """
    seq_len = args.train_seq_len
    half = seq_len // 2

    x, y = train_loader.next_batch(args.train_batch_tokens, seq_len, grad_accum_steps)
    # Use first sequence in batch for meta-learning
    context_x = x[:1, :half]
    context_y = y[:1, :half]
    target_x = x[:1, half:]
    target_y = y[:1, half:]

    adapter_snapshot = base_model.get_adapter_state()

    # Inner loop: SGD on adapter params only
    adapter_params = base_model.adapter_parameters()
    for _inner in range(args.ttt_inner_steps):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ctx_loss = base_model.forward_loss(context_x, context_y)
        # Compute grads for adapter params only
        adapter_grads = torch.autograd.grad(ctx_loss, adapter_params, create_graph=False)
        with torch.no_grad():
            for p, g in zip(adapter_params, adapter_grads):
                p.data -= args.ttt_inner_lr * g

    # Outer loss on target with adapted model — gradients flow to ALL params
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outer_loss = base_model.forward_loss(target_x, target_y)

    outer_loss.backward()

    # Restore adapter state
    with torch.no_grad():
        base_model.set_adapter_state(adapter_snapshot)

    return outer_loss.detach()


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # Distributed + CUDA setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_flash_sdp, enable_cudnn_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/itchy_{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True):
        if not master_process:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Validation data
    val_bytes = load_validation_bytes(args.val_files, args.train_seq_len)
    log0(f"val_bytes:{val_bytes.numel() - 1}")

    # Model
    base_model = Itchy(
        dim=args.model_dim, num_layers=args.num_layers, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult, patch_size=args.patch_size,
        adapter_rank=args.adapter_rank, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_named_params
                     if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS) and "adapter" not in n]
    scalar_params = [p for n, p in block_named_params
                     if (p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)) and "adapter" not in n]
    adapter_params_list = [p for n, p in block_named_params if "adapter" in n]

    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    embed_params = list(base_model.embed.parameters()) + list(base_model.unpatch.parameters())

    optimizer_embed = torch.optim.Adam(
        [{"params": embed_params, "lr": args.embed_lr, "base_lr": args.embed_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                          backend_steps=args.muon_backend_steps)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params + adapter_params_list, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers = [optimizer_embed, optimizer_muon, optimizer_scalar]

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} ({n_params/1e6:.1f}M)")
    log0(f"dim:{args.model_dim} layers:{args.num_layers} heads:{args.num_heads} kv_heads:{args.num_kv_heads}")
    log0(f"mlp_mult:{args.mlp_mult} patch_size:{args.patch_size} adapter_rank:{args.adapter_rank}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"ttt: ratio={args.ttt_ratio} inner_steps={args.ttt_inner_steps} inner_lr={args.ttt_inner_lr}")

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Data loader
    train_loader = DistributedByteLoader(args.train_files, rank, world_size, device)

    # Warmup
    if args.warmup_steps > 0:
        initial_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                (loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_done:{warmup_step + 1}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedByteLoader(args.train_files, rank, world_size, device)

    # Training loop
    ttt_enabled = args.ttt_ratio > 0.0
    ttt_step_interval = max(int(1.0 / args.ttt_ratio), 1) if ttt_enabled else 0
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val_bytes(args, model, rank, world_size, device, grad_accum_steps, val_bytes)
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()

        is_ttt_step = ttt_enabled and step > 0 and (step % ttt_step_interval == 0)

        if is_ttt_step:
            # TTT meta-learning — uses base_model directly (not DDP wrapper)
            ttt_loss = ttt_meta_step(args, base_model, train_loader, grad_accum_steps, device)

            frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
            for group in optimizer_muon.param_groups:
                group["momentum"] = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = group["base_lr"] * scale
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            base_model.reset_adapters()
        else:
            # Regular training step
            train_loss = torch.zeros((), device=device)
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                train_loss += loss.detach()
                (loss * grad_scale).backward()
            train_loss /= grad_accum_steps

            frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
            for group in optimizer_muon.param_groups:
                group["momentum"] = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = group["base_lr"] * scale
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
            for opt in optimizers:
                opt.step()
            zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log = args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0)
        if should_log:
            loss_val = ttt_loss.item() if is_ttt_step else train_loss.item()
            ttt_tag = " [TTT]" if is_ttt_step else ""
            log0(f"step:{step}/{args.iterations} train_loss:{loss_val:.4f} "
                 f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms "
                 f"tok_s:{int(args.train_batch_tokens / (approx_training_time_ms / step / 1000))}{ttt_tag}")

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # Serialization
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
        quant_buf = io.BytesIO()
        torch.save(quant_obj, quant_buf)
        quant_blob = zlib.compress(quant_buf.getvalue(), level=9)
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_bytes = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"artifact: model={quant_bytes} code={code_bytes} total={quant_bytes + code_bytes} (limit=16000000)")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

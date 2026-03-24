"""
Itchy Final — CUDA training script for 8xH100.

Byte-level transformer with LeakyReLU(0.5)² and 3x MLP.
No adapters, no TTT meta-learning. Clean and simple.

Usage:
    torchrun --standalone --nproc_per_node=8 train_itchy_final.py

Validated via ablation on T4:
- LeakyReLU(0.5)²: -0.0014 BPB
- 3x MLP: -0.0061 BPB
- Combined: -0.0066 BPB (0.5944 vs 0.6010 baseline)
- Partial RoPE: HURTS (+0.011), excluded
- LN Scale: HURTS (+0.012), excluded
- N-gram hash: no effect, excluded
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

    # Model — validated config
    model_dim = int(os.environ.get("MODEL_DIM", 384))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    patch_size = int(os.environ.get("PATCH_SIZE", 12))
    decode_head_dim = int(os.environ.get("DECODE_HEAD_DIM", 128))
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

    # EMA
    ema_enabled = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))

    # TTT at eval (all-param, not adapter)
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))


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
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True):
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
            lr, momentum, backend_steps = group["lr"], group["momentum"], group["backend_steps"]
            total = sum(p.numel() for p in params)
            flat = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "buf" not in state:
                        state["buf"] = torch.zeros_like(g)
                    buf = state["buf"]
                    buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                p.add_(flat[curr:curr + p.numel()].view_as(p).to(p.dtype), alpha=-lr)
                curr += p.numel()


# -----------------------------
# DATA LOADING
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight")

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens = int(header[2])
    tokens = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens.astype(np.uint16, copy=False))


class ByteStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def take(self, n):
        chunks = []
        left = n
        while left > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self.idx = (self.idx + 1) % len(self.files)
                self.tokens = load_data_shard(self.files[self.idx])
                self.pos = 0
                continue
            k = min(left, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedByteLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = ByteStream(pattern)

    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local = global_tokens // (self.world_size * grad_accum_steps)
        span = local + 1
        chunk = self.stream.take(span * self.world_size)
        start = self.rank * span
        local_chunk = chunk[start:start + span].to(dtype=torch.int64)
        x = local_chunk[:-1].reshape(-1, seq_len)
        y = local_chunk[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def load_validation_bytes(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


# -----------------------------
# MODEL
# -----------------------------

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)), persistent=False)
        self._cache = None

    def forward(self, seq_len, device, dtype):
        if self._cache is None or self._cache[0] != seq_len or self._cache[1].device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cache = (seq_len, freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :])
        return self._cache[1].to(dtype), self._cache[2].to(dtype)


def apply_rotary_emb(x, cos, sin):
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain):
        super().__init__()
        self.nh, self.nkv = num_heads, num_kv_heads
        self.hd = dim // num_heads
        kv = num_kv_heads * self.hd
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv, bias=False)
        self.c_v = CastedLinear(dim, kv, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain))
        self.rotary = Rotary(self.hd, rope_base)

    def forward(self, x):
        B, S, D = x.shape
        q = self.c_q(x).reshape(B, S, self.nh, self.hd).transpose(1, 2)
        k = self.c_k(x).reshape(B, S, self.nkv, self.hd).transpose(1, 2)
        v = self.c_v(x).reshape(B, S, self.nkv, self.hd).transpose(1, 2)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(S, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.nkv != self.nh))
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, S, D))


class MLP(nn.Module):
    def __init__(self, dim, mult):
        super().__init__()
        self.fc = CastedLinear(dim, dim * mult, bias=False)
        self.proj = CastedLinear(dim * mult, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), 0.5).square())


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain):
        super().__init__()
        self.attn = Attention(dim, num_heads, num_kv_heads, rope_base, qk_gain)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))))

    def forward(self, x, x0):
        m = self.resid_mix.to(x.dtype)
        x = m[0][None, None, :] * x + m[1][None, None, :] * x0
        x = x + self.attn_scale.to(x.dtype)[None, None, :] * self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


class PerPositionDecode(nn.Module):
    """Per-position MLP heads: each byte position in a patch gets its own small MLP.
    Validated: -0.057 BPB over flat linear projection."""
    def __init__(self, dim, patch_size, head_dim=128, vocab_size=260):
        super().__init__()
        self.patch_size = patch_size
        self.vocab_size = vocab_size
        self.pos = nn.Parameter(torch.randn(patch_size, dim) * 0.02)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, head_dim, bias=False),
                nn.ReLU(),
                nn.Linear(head_dim, vocab_size, bias=False),
            )
            for _ in range(patch_size)
        ])

    def forward(self, x):
        # x: (B, n_patches, dim)
        logits = []
        for p in range(self.patch_size):
            xp = x + self.pos[p][None, None, :]
            logits.append(self.heads[p](xp))  # (B, n_patches, 260)
        return torch.stack(logits, dim=2)  # (B, n_patches, patch_size, 260)


class Itchy(nn.Module):
    def __init__(self, dim, num_layers, num_heads, num_kv_heads, mlp_mult,
                 patch_size, decode_head_dim, logit_softcap, rope_base, qk_gain):
        super().__init__()
        self.dim, self.patch_size = dim, patch_size
        self.logit_softcap = logit_softcap
        self.byte_embed = nn.Embedding(260, dim)
        self.patch_proj = CastedLinear(dim * patch_size, dim, bias=False)
        ne = num_layers // 2
        nd = num_layers - ne
        self.ne, self.nd = ne, nd
        self.skip_weights = nn.Parameter(torch.ones(min(ne, nd), dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain)
            for _ in range(num_layers)
        ])
        self.decode = PerPositionDecode(dim, patch_size, head_dim=decode_head_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def forward(self, byte_ids, targets):
        B, S = byte_ids.shape
        P = self.patch_size
        x = self.byte_embed(byte_ids).reshape(B, S // P, P * self.dim)
        x = F.rms_norm(self.patch_proj(x), (self.dim,))
        x0 = x
        skips = []
        for i in range(self.ne):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.nd):
            if skips:
                x = x + self.skip_weights[i].to(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.ne + i](x, x0)
        x = F.rms_norm(x, (x.size(-1),))
        logits = self.decode(x).reshape(-1, 260)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets.reshape(-1))


def restore_fp32(module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if (p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)) and p.dtype != torch.float32:
                p.data = p.data.float()


# -----------------------------
# QUANTIZATION
# -----------------------------

INT8_CLIP_Q = 99.99984 / 100.0

def quantize_state_dict_int6(state_dict):
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    stats = {"param_count": 0, "payload_bytes": 0}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().float().contiguous()
        stats["param_count"] += t.numel()
        if not t.is_floating_point() or t.numel() <= 65536:
            stored = t.half() if t.is_floating_point() else t
            passthrough[name] = stored
            stats["payload_bytes"] += stored.numel() * stored.element_size()
            continue
        clip_range = 31  # int6: -31 to +31
        if t.ndim == 2:
            best_q, best_s, best_err = None, None, float("inf")
            for pct in [0.999, 0.9995, 0.9999, 0.99999, 1.0]:
                row_clip = torch.quantile(t.abs(), pct, dim=1) if pct < 1.0 else t.abs().amax(dim=1)
                s = (row_clip / clip_range).clamp_min(1.0 / clip_range).half()
                q = (t / s.float()[:, None]).round().clamp(-clip_range, clip_range).to(torch.int8)
                err = (t - q.float() * s.float()[:, None]).pow(2).mean().item()
                if err < best_err:
                    best_q, best_s, best_err = q, s, err
            quantized[name] = best_q
            scales[name] = best_s
        else:
            amax = t.abs().max().item()
            s = torch.tensor(amax / clip_range if amax > 0 else 1.0).half()
            quantized[name] = (t / s.float()).round().clamp(-clip_range, clip_range).to(torch.int8)
            scales[name] = s
        dtypes[name] = str(tensor.dtype).removeprefix("torch.")
        stats["payload_bytes"] += quantized[name].numel() + scales[name].numel() * 2
    return {"quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough}, stats


# -----------------------------
# VALIDATION
# -----------------------------

def eval_val_bytes(args, model, rank, world_size, device, grad_accum_steps, val_bytes):
    local_batch = args.val_batch_size // (world_size * grad_accum_steps)
    seq_len = args.train_seq_len
    local_seqs = local_batch // seq_len
    total_seqs = (val_bytes.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    count = torch.zeros((), device=device, dtype=torch.float64)

    with torch.no_grad():
        for bs in range(seq_start, seq_end, local_seqs):
            be = min(bs + local_seqs, seq_end)
            rs, re = bs * seq_len, be * seq_len + 1
            local = val_bytes[rs:re].to(device=device, dtype=torch.int64)
            x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y).detach()
            n = float(y.numel())
            loss_sum += loss.to(torch.float64) * n
            count += n

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

    val_loss = float((loss_sum / count).item())
    return val_loss, val_loss / math.log(2.0)


# -----------------------------
# TRAINING
# -----------------------------

def main():
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # Distributed setup
    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_flash_sdp, enable_cudnn_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile = None
    if master:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/itchy_{args.run_id}.txt"
        print(logfile)

    def log0(msg, console=True):
        if not master: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a") as f: print(msg, file=f)

    log0(code, console=False)
    log0(f"Python {sys.version}", console=False)
    log0(f"PyTorch {torch.__version__}", console=False)

    # Seed
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # Data
    val_bytes = load_validation_bytes(args.val_files, args.train_seq_len)
    log0(f"val_bytes:{val_bytes.numel() - 1}")

    # Model
    base_model = Itchy(
        dim=args.model_dim, num_layers=args.num_layers, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult, patch_size=args.patch_size,
        decode_head_dim=args.decode_head_dim, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, qk_gain=args.qk_gain_init,
    ).to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m, CastedLinear): m.float()
    restore_fp32(base_model)

    compiled = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = DDP(compiled, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled

    # Optimizer
    block_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_params if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for n, p in block_params if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    decode_params = list(base_model.decode.parameters())
    embed_params = [base_model.byte_embed.weight, base_model.patch_proj.weight] + decode_params

    opt_embed = torch.optim.Adam([{"params": embed_params, "lr": args.embed_lr, "base_lr": args.embed_lr}],
                                  betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for g in opt_muon.param_groups: g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.Adam([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
                                   betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizers = [opt_embed, opt_muon, opt_scalar]

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} ({n_params/1e6:.1f}M)")
    log0(f"dim:{args.model_dim} layers:{args.num_layers} mlp:{args.mlp_mult}x patch:{args.patch_size} decode_head:{args.decode_head_dim}")
    log0(f"activation:LeakyReLU(0.5)^2 rope:full decode:per_position_mlp")
    log0(f"world_size:{world_size} grad_accum:{grad_accum_steps} seed:{args.seed}")

    def zero_all():
        for o in optimizers: o.zero_grad(set_to_none=True)

    max_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0 or max_ms is None: return 1.0
        step_ms = elapsed_ms / max(step, 1)
        wd_ms = args.warmdown_iters * step_ms
        rem_ms = max(max_ms - elapsed_ms, 0.0)
        return rem_ms / max(wd_ms, 1e-9) if rem_ms <= wd_ms else 1.0

    # Data loader
    train_loader = DistributedByteLoader(args.train_files, rank, world_size, device)

    # EMA state
    ema_state = None
    if args.ema_enabled:
        ema_state = {n: t.detach().float().clone() for n, t in base_model.state_dict().items()}

    # Warmup
    if args.warmup_steps > 0:
        init_model = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opts = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_all()
            for ms in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                (loss * grad_scale).backward()
            for o in optimizers: o.step()
            if ws + 1 == args.warmup_steps: log0(f"warmup_done:{ws+1}")
        base_model.load_state_dict(init_model, strict=True)
        for o, s in zip(optimizers, init_opts): o.load_state_dict(s)
        zero_all()
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistributedByteLoader(args.train_files, rank, world_size, device)
        if ema_state:
            ema_state = {n: t.detach().float().clone() for n, t in base_model.state_dict().items()}

    # Training loop
    training_time_ms = 0.0
    stop_after = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last = step == args.iterations or (stop_after is not None and step >= stop_after)

        if last or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            # Use EMA weights for validation if available
            if ema_state:
                orig = {n: t.clone() for n, t in base_model.state_dict().items()}
                base_model.load_state_dict({n: t.to(orig[n].dtype) for n, t in ema_state.items()}, strict=True)
            val_loss, val_bpb = eval_val_bytes(args, model, rank, world_size, device, grad_accum_steps, val_bytes)
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.1f}ms")
            if ema_state:
                base_model.load_state_dict(orig, strict=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last: break

        elapsed = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed)
        zero_all()

        train_loss = torch.zeros((), device=device)
        for ms in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Momentum warmup
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        for g in opt_muon.param_groups:
            g["momentum"] = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum

        for o in optimizers:
            for g in o.param_groups: g["lr"] = g["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for o in optimizers: o.step()
        zero_all()

        # EMA update
        if ema_state:
            with torch.no_grad():
                for n, t in base_model.state_dict().items():
                    ema_state[n].mul_(args.ema_decay).add_(t.detach().float(), alpha=1.0 - args.ema_decay)

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.1f}ms "
                 f"tok_s:{int(args.train_batch_tokens/(approx_ms/step/1000))}")

        reached = max_ms is not None and approx_ms >= max_ms
        if distributed and max_ms is not None:
            rt = torch.tensor(int(reached), device=device)
            dist.all_reduce(rt, op=dist.ReduceOp.MAX)
            reached = bool(rt.item())
        if stop_after is None and reached:
            stop_after = step

    log0(f"peak_memory:{torch.cuda.max_memory_allocated()//1024//1024}MiB")

    # Serialize
    if master:
        # Use EMA weights for final model
        final_sd = ema_state if ema_state else base_model.state_dict()
        final_sd = {n: t.to(base_model.state_dict()[n].dtype) for n, t in final_sd.items()}

        torch.save(final_sd, "final_model.pt")
        quant_obj, qstats = quantize_state_dict_int6(final_sd)
        buf = io.BytesIO()
        torch.save(quant_obj, buf)
        blob = zlib.compress(buf.getvalue(), level=9)
        with open("final_model.int6.ptz", "wb") as f:
            f.write(blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"artifact: model={len(blob)} code={code_bytes} total={len(blob)+code_bytes} (limit=16000000)")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

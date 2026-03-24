"""
Itchy Final: Byte-level transformer for 16MB Parameter Golf.

Architecture validated via ablation:
- Byte-level (256 vocab, no tokenizer)
- Patch processing (4 bytes -> 1 patch)
- LeakyReLU(0.5)² activation
- 3x MLP expansion
- Full RoPE (partial RoPE hurts at byte level)
- No LN scaling (hurts at byte level)
- No n-gram hashing (no benefit)
- Encoder-decoder skip architecture
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

COMPUTE_DTYPE = mx.bfloat16


def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    """LeakyReLU(0.5)² — validated -0.0014 BPB over relu²."""
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc(x)
        x = mx.where(x > 0, x, 0.5 * x)  # LeakyReLU(0.5)
        return self.proj(x * x)  # squared


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array([[1.0] * dim, [0.0] * dim])

    def __call__(self, x: mx.array, x0: mx.array) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(rms_norm(x))
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(rms_norm(x))
        return x


class BytePatchEmbed(nn.Module):
    def __init__(self, dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.byte_embed = nn.Embedding(260, dim)
        self.patch_proj = CastedLinear(dim * patch_size, dim)

    def __call__(self, byte_ids: mx.array) -> mx.array:
        bsz, seq_len = byte_ids.shape
        n_patches = seq_len // self.patch_size
        x = self.byte_embed(byte_ids)
        x = x.reshape(bsz, n_patches, self.patch_size * x.shape[-1])
        return self.patch_proj(x)


class ByteUnpatch(nn.Module):
    def __init__(self, dim: int, patch_size: int, vocab_size: int = 260):
        super().__init__()
        self.patch_size = patch_size
        self.vocab_size = vocab_size
        self.proj = CastedLinear(dim, patch_size * vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        bsz = x.shape[0]
        logits = self.proj(x)
        return logits.reshape(bsz, -1, self.vocab_size)


class ItchyFinal(nn.Module):
    """
    Final Itchy architecture. Simple and validated.
    Byte-level + patch processing + LeakyReLU(0.5)² + 3x MLP.
    """
    def __init__(
        self,
        dim: int = 384,
        num_layers: int = 11,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        mlp_mult: int = 3,
        patch_size: int = 4,
        logit_softcap: float = 30.0,
        rope_base: float = 10000.0,
        qk_gain_init: float = 1.5,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.logit_softcap = logit_softcap

        self.embed = BytePatchEmbed(dim, patch_size)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)

        self.blocks = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ]
        self.unpatch = ByteUnpatch(dim, patch_size, vocab_size=260)

        # Zero-init output projections
        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)

    def __call__(self, byte_ids: mx.array) -> mx.array:
        x = rms_norm(self.embed(byte_ids).astype(COMPUTE_DTYPE))
        x0 = x
        skips: list[mx.array] = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = rms_norm(x)
        logits = self.unpatch(x)
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def loss(self, byte_ids: mx.array, target_ids: mx.array) -> mx.array:
        logits = self(byte_ids).reshape(-1, 260)
        targets = target_ids.reshape(-1)
        return nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="mean")


def count_params(model: ItchyFinal) -> int:
    from mlx.utils import tree_flatten
    return sum(v.size for _, v in tree_flatten(model.parameters()))


if __name__ == "__main__":
    configs = [
        (384, 11), (384, 12), (448, 10), (448, 11),
    ]
    print(f"{'dim':>4} {'layers':>6} {'params':>12} {'int8_MB':>8} {'int6_MB':>8}")
    print("-" * 45)
    for dim, layers in configs:
        m = ItchyFinal(dim=dim, num_layers=layers)
        n = count_params(m)
        print(f"{dim:>4} {layers:>6} {n:>12,} {n/1e6:>7.1f}M {n*0.75/1e6:>7.1f}M")

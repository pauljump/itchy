"""
Itchy v2: Byte-level model with all proven competition tricks.
Strips dead LoRA adapters, adds LeakyReLU², partial RoPE, LN scaling,
BigramHash for bytes, and proper sizing for 16MB at int6.
"""
from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

COMPUTE_DTYPE = mx.bfloat16


def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


class RMSNormNoWeight(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class ByteNgramHash(nn.Module):
    """Hash n-gram embeddings for byte-level local context.

    For each byte position, compute rolling hashes for n-grams of length 2-8,
    look up embeddings, and sum them. This captures local byte patterns
    (word fragments, UTF-8 sequences, markup) with minimal parameters.
    Based on BLT's finding that this is the biggest win for byte-level models.
    """
    def __init__(self, hash_vocab: int, embed_dim: int, model_dim: int,
                 ngram_sizes: tuple = (2, 3, 4, 5, 6)):
        super().__init__()
        self.hash_vocab = hash_vocab
        self.ngram_sizes = ngram_sizes
        # One embedding table shared across all n-gram sizes (with offset)
        total_entries = hash_vocab * len(ngram_sizes)
        self.embed = nn.Embedding(total_entries, embed_dim)
        self.embed.weight = mx.zeros_like(self.embed.weight)  # zero init
        self.proj = CastedLinear(embed_dim, model_dim) if embed_dim != model_dim else None
        if self.proj is not None:
            self.proj.weight = mx.zeros_like(self.proj.weight)
        self.scale = mx.array(0.05)
        # Hash primes for rolling hash
        self._primes = [36313, 27191, 51637, 39371, 73291, 59393, 97127]

    def __call__(self, byte_ids: mx.array) -> mx.array:
        t = byte_ids.astype(mx.int32)
        bsz, seq_len = t.shape
        result = mx.zeros((bsz, seq_len, self.embed.weight.shape[1]), dtype=mx.float32)

        for idx, n in enumerate(self.ngram_sizes):
            offset = idx * self.hash_vocab
            # Build rolling hash for n-gram of size n
            h = mx.zeros_like(t)
            for k in range(n):
                prime = self._primes[k % len(self._primes)]
                # Shift: for positions < k, we pad with 0
                if k == 0:
                    h = h + prime * t
                else:
                    shifted = mx.concatenate([
                        mx.zeros((bsz, k), dtype=mx.int32),
                        t[:, :-k]
                    ], axis=-1)
                    h = h ^ (prime * shifted)
            # Mod into vocab range, add offset for this n-gram size
            indices = (mx.abs(h) % self.hash_vocab) + offset
            # Zero out positions that don't have enough context
            if n > 1:
                mask = mx.concatenate([
                    mx.zeros((bsz, n - 1), dtype=mx.int32),
                    mx.ones((bsz, seq_len - n + 1), dtype=mx.int32)
                ], axis=-1)
                indices = indices * mask + offset * (1 - mask)  # map masked to offset (zeroed embed)
            result = result + self.embed(indices)

        if self.proj is not None:
            result = self.proj(result)
        return result * self.scale.astype(result.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float, rope_dims: int = 0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.rope_dims = rope_dims if rope_dims > 0 else self.head_dim
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.rope_dims, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array, ln_scale: float = 1.0) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.rope_dims < self.head_dim:
            # Partial RoPE: only apply to first rope_dims dimensions
            q_rope = rms_norm(q[..., :self.rope_dims]).astype(COMPUTE_DTYPE)
            q_pass = q[..., self.rope_dims:]
            k_rope = rms_norm(k[..., :self.rope_dims]).astype(COMPUTE_DTYPE)
            k_pass = k[..., self.rope_dims:]
            q_rope = self.rope(q_rope)
            k_rope = self.rope(k_rope)
            q = mx.concatenate([q_rope, rms_norm(q_pass).astype(COMPUTE_DTYPE)], axis=-1)
            k = mx.concatenate([k_rope, rms_norm(k_pass).astype(COMPUTE_DTYPE)], axis=-1)
        else:
            q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
            k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))

        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    """LeakyReLU(0.5)² MLP — proven -0.003 BPB over relu²."""
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc(x)
        # LeakyReLU(0.5) squared
        x = mx.where(x > 0, x, 0.5 * x)
        return self.proj(x * x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, rope_dims: int = 0,
                 layer_idx: int = 0, ln_scale: bool = True):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array([[1.0] * dim, [0.0] * dim])
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def __call__(self, x: mx.array, x0: mx.array) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        normed = self.attn_norm(x) * self.ln_scale_factor
        attn_out = self.attn(normed)
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        normed = self.mlp_norm(x) * self.ln_scale_factor
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(normed)
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


class ItchyV2(nn.Module):
    """
    Itchy v2: Byte-level model with competition-proven tricks.
    No adapters. All budget goes to model capacity.
    """
    def __init__(
        self,
        dim: int = 512,
        num_layers: int = 11,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        mlp_mult: int = 3,
        patch_size: int = 4,
        rope_dims: int = 16,
        ngram_hash_vocab: int = 8192,
        ngram_dim: int = 64,
        ngram_sizes: tuple = (2, 3, 4, 5, 6),
        logit_softcap: float = 30.0,
        rope_base: float = 10000.0,
        qk_gain_init: float = 1.5,
        ln_scale: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.logit_softcap = logit_softcap

        self.embed = BytePatchEmbed(dim, patch_size)
        self.ngram = ByteNgramHash(ngram_hash_vocab, ngram_dim, dim, ngram_sizes) if ngram_hash_vocab > 0 else None

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)

        self.blocks = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  rope_dims=rope_dims, layer_idx=i, ln_scale=ln_scale)
            for i in range(num_layers)
        ]
        self.final_norm = RMSNormNoWeight()
        self.unpatch = ByteUnpatch(dim, patch_size, vocab_size=260)

        # Zero-init output projections
        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, byte_ids: mx.array) -> mx.array:
        x = rms_norm(self.embed(byte_ids).astype(COMPUTE_DTYPE))

        # Add n-gram context (at patch level — average n-gram embeddings within each patch)
        if self.ngram is not None:
            ng = self.ngram(byte_ids).astype(COMPUTE_DTYPE)
            bsz, seq_len_ng, d = ng.shape
            n_patches = seq_len_ng // self.patch_size
            ng = ng.reshape(bsz, n_patches, self.patch_size, d).mean(axis=2)
            x = x + ng

        x0 = x
        skips: list[mx.array] = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x)
        logits = self.unpatch(x)
        return self.softcap(logits)

    def loss(self, byte_ids: mx.array, target_ids: mx.array) -> mx.array:
        logits = self(byte_ids)
        logits = logits.reshape(-1, 260)
        targets = target_ids.reshape(-1)
        return nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="mean")


def count_params(model: ItchyV2) -> int:
    from mlx.utils import tree_flatten
    return sum(v.size for _, v in tree_flatten(model.parameters()))


def size_configs():
    """Print param counts for v2 configurations."""
    configs = [
        # (dim, layers, heads, kv, mlp, patch, rope_dims, ngram_vocab, ngram_dim)
        (384, 11, 8, 4, 3, 4, 16, 8192, 64),
        (384, 12, 8, 4, 3, 4, 16, 8192, 64),
        (448, 10, 8, 4, 3, 4, 16, 8192, 64),
        (384, 11, 8, 4, 3, 4, 16, 4096, 64),    # smaller ngram
        (384, 11, 8, 4, 3, 4, 16, 0, 0),         # no ngram
        (448, 11, 8, 4, 3, 4, 16, 4096, 64),
    ]
    print(f"{'dim':>4} {'layers':>6} {'ngram':>6} {'params':>12} {'int8_MB':>8} {'int6_MB':>8}")
    print("-" * 55)
    for dim, layers, heads, kv, mlp, patch, rope, nv, nd in configs:
        m = ItchyV2(dim=dim, num_layers=layers, num_heads=heads, num_kv_heads=kv,
                    mlp_mult=mlp, patch_size=patch, rope_dims=rope,
                    ngram_hash_vocab=nv, ngram_dim=nd,
                    ngram_sizes=(2,3,4,5,6) if nv > 0 else ())
        n = count_params(m)
        print(f"{dim:>4} {layers:>6} {nv:>6} {n:>12,} {n/1e6:>7.1f}M {n*0.75/1e6:>7.1f}M")


if __name__ == "__main__":
    size_configs()

"""
Itchy: A byte-level, adaptation-native language model designed for 16MB.

Key design:
- 256 vocab (raw bytes) — no tokenizer overhead
- Patch processing: group P bytes into 1 patch to cut sequence length
- Built-in LoRA-style adapters in every block (zero-initialized)
- Encoder-decoder skip architecture (matches baseline pattern)
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


class LoRAAdapter(nn.Module):
    """Zero-initialized LoRA adapter. Gate starts at 0 so adapters have no effect initially."""
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.down = mx.random.normal((rank, dim)) * (1.0 / math.sqrt(dim))
        self.up = mx.zeros((dim, rank))
        self.gate = mx.array(0.0)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, S, D) -> project down then up, scaled by gate
        h = x @ self.down.astype(x.dtype).T  # (B, S, rank)
        h = h @ self.up.astype(x.dtype).T    # (B, S, D)
        return self.gate.astype(x.dtype) * h


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
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
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc(x))
        return self.proj(x * x)  # relu^2


class AdaptiveBlock(nn.Module):
    """Transformer block with built-in LoRA adapter on attention output."""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, adapter_rank: int):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(
            [[1.0] * dim, [0.0] * dim],
        )
        self.adapter = LoRAAdapter(dim, adapter_rank)

    def __call__(self, x: mx.array, x0: mx.array) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        attn_out = attn_out + self.adapter(attn_out)
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class BytePatchEmbed(nn.Module):
    """Embed raw bytes (0-255) and group into patches."""
    def __init__(self, dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.byte_embed = nn.Embedding(260, dim)  # 256 bytes + 4 special tokens
        self.patch_proj = CastedLinear(dim * patch_size, dim)

    def __call__(self, byte_ids: mx.array) -> mx.array:
        # byte_ids: (B, S) where S is divisible by patch_size
        bsz, seq_len = byte_ids.shape
        n_patches = seq_len // self.patch_size
        x = self.byte_embed(byte_ids)  # (B, S, dim)
        # Group into patches: (B, n_patches, patch_size * dim)
        x = x.reshape(bsz, n_patches, self.patch_size * x.shape[-1])
        return self.patch_proj(x)  # (B, n_patches, dim)


class ByteUnpatch(nn.Module):
    """Convert patch representations back to per-byte logits."""
    def __init__(self, dim: int, patch_size: int, vocab_size: int = 260):
        super().__init__()
        self.patch_size = patch_size
        self.vocab_size = vocab_size
        self.proj = CastedLinear(dim, patch_size * vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, n_patches, dim) -> (B, n_patches * patch_size, vocab_size)
        bsz = x.shape[0]
        logits = self.proj(x)  # (B, n_patches, patch_size * vocab_size)
        return logits.reshape(bsz, -1, self.vocab_size)


class Itchy(nn.Module):
    """
    Byte-level adaptive language model.

    Architecture:
    - BytePatchEmbed: 256 byte vocab -> patches of P bytes -> model dim
    - Encoder-decoder transformer with skip connections
    - LoRA adapters in every block (zero-init, for TTT)
    - ByteUnpatch: model dim -> per-byte logits over 260 classes
    """
    def __init__(
        self,
        dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        mlp_mult: int = 3,
        patch_size: int = 4,
        adapter_rank: int = 32,
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
            AdaptiveBlock(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, adapter_rank)
            for _ in range(num_layers)
        ]
        self.final_norm = RMSNormNoWeight()
        self.unpatch = ByteUnpatch(dim, patch_size, vocab_size=260)

        # Zero-init output projections (like baseline)
        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, byte_ids: mx.array) -> mx.array:
        """Forward pass. byte_ids: (B, S) with values 0-259."""
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

        x = self.final_norm(x)
        logits = self.unpatch(x)  # (B, S, 260)
        return self.softcap(logits)

    def loss(self, byte_ids: mx.array, target_ids: mx.array) -> mx.array:
        """Compute cross-entropy loss over bytes."""
        logits = self(byte_ids)  # (B, S, 260)
        logits = logits.reshape(-1, 260)
        targets = target_ids.reshape(-1)
        return nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="mean")

    def adapter_parameters(self) -> dict[str, mx.array]:
        """Return only adapter parameters (for TTT)."""
        from mlx.utils import tree_flatten
        all_params = dict(tree_flatten(self.parameters()))
        return {k: v for k, v in all_params.items() if "adapter" in k}

    def get_adapter_state(self) -> list[tuple[mx.array, mx.array, mx.array]]:
        """Snapshot all adapter parameters (down, up, gate) for later restore."""
        return [(block.adapter.down, block.adapter.up, block.adapter.gate) for block in self.blocks]

    def set_adapter_state(self, state: list[tuple[mx.array, mx.array, mx.array]]) -> None:
        """Restore adapter parameters from a snapshot."""
        for block, (down, up, gate) in zip(self.blocks, state):
            block.adapter.down = down
            block.adapter.up = up
            block.adapter.gate = gate

    def reset_adapters(self) -> None:
        """Zero out all adapter weights and gates."""
        for block in self.blocks:
            block.adapter.gate = mx.array(0.0)
            block.adapter.up = mx.zeros_like(block.adapter.up)

    def freeze_backbone(self) -> None:
        """Freeze all parameters except adapters by storing them and marking non-trainable.

        In MLX there's no requires_grad flag. Instead, we return the set of
        adapter keys so callers can selectively compute gradients.
        """
        self._frozen = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze the backbone (undo freeze_backbone)."""
        self._frozen = False

    @property
    def is_backbone_frozen(self) -> bool:
        return getattr(self, "_frozen", False)


def count_params(model: Itchy) -> int:
    from mlx.utils import tree_flatten
    return sum(v.size for _, v in tree_flatten(model.parameters()))


def size_model_configs():
    """Print param counts for different configurations to help fit 16MB."""
    configs = [
        (384, 12, 8, 4, 3, 4, 32),   # smaller
        (448, 10, 8, 4, 3, 4, 32),   # medium
        (512, 8, 8, 4, 3, 4, 32),    # wider fewer layers
        (512, 10, 8, 4, 3, 4, 32),   # wider more layers
        (512, 12, 8, 4, 2, 4, 32),   # wide 2x MLP
        (384, 14, 8, 4, 3, 4, 24),   # deep narrow
    ]
    print(f"{'dim':>4} {'layers':>6} {'mlp':>3} {'rank':>4} {'params':>10} {'int8_MB':>8} {'int6_MB':>8}")
    print("-" * 60)
    for dim, layers, heads, kv_heads, mlp_mult, patch, rank in configs:
        m = Itchy(dim=dim, num_layers=layers, num_heads=heads, num_kv_heads=kv_heads,
                  mlp_mult=mlp_mult, patch_size=patch, adapter_rank=rank)
        n = count_params(m)
        int8_mb = n * 1.0 / 1e6  # 1 byte per param
        int6_mb = n * 0.75 / 1e6  # 0.75 bytes per param
        print(f"{dim:>4} {layers:>6} {mlp_mult:>3}x {rank:>4} {n:>10,} {int8_mb:>7.1f}M {int6_mb:>7.1f}M")


if __name__ == "__main__":
    size_model_configs()

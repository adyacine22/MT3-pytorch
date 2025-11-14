"""Low-level layers shared by the PyTorch MT3 transformer."""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "linear":
        return lambda x: x
    if hasattr(F, name):
        return getattr(F, name)
    raise ValueError(f"Unsupported activation {name}")


class T5LayerNorm(nn.Module):
    """RMS-based layer norm without bias, mirroring T5."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return normed * self.weight


class SinusoidalPositions(nn.Module):
    """Fixed positional encodings reused by encoder and decoder."""

    def __init__(self, max_length: int, dim: int):
        super().__init__()
        min_scale, max_scale = 1.0, 10000.0
        half = dim // 2
        position = torch.arange(max_length).unsqueeze(1)
        scale = -math.log(max_scale / min_scale) / max(1, half - 1)
        div_term = min_scale * torch.exp(torch.arange(half) * scale)
        pe = torch.zeros(max_length, dim)
        pe[:, :half] = torch.sin(position * div_term)
        pe[:, half : 2 * half] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        pe_tensor = cast(torch.Tensor, self.pe)
        flat = positions.reshape(-1)
        gathered = pe_tensor.index_select(0, flat)
        return gathered.view(*positions.shape, pe_tensor.size(-1))


class MultiHeadAttention(nn.Module):
    """Multi-head dot-product attention with optional caching."""

    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, dropout: float):
        super().__init__()
        inner_dim = num_heads * head_dim
        if inner_dim != embed_dim:
            raise ValueError("embed_dim must equal num_heads * head_dim for MT3.")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        if key_value_states is None:
            key_value_states = hidden_states

        query = self._shape(self.q_proj(hidden_states))
        key = self._shape(self.k_proj(key_value_states))
        value = self._shape(self.v_proj(key_value_states))

        if past_key_value is not None:
            pk, pv = past_key_value
            key = torch.cat([pk, key], dim=2)
            value = torch.cat([pv, value], dim=2)

        mask = None
        if attention_mask is not None:
            mask = attention_mask
            if mask.size(1) == 1 and self.num_heads > 1:
                mask = mask.expand(mask.size(0), self.num_heads, mask.size(-2), mask.size(-1))
        if hasattr(F, "scaled_dot_product_attention"):
            dropout_p = self.dropout.p if self.training else 0.0
            context = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=dropout_p,
            )
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            context = torch.matmul(attn, value)
        context = context.transpose(1, 2).contiguous()
        context = context.view(hidden_states.shape[0], -1, self.num_heads * self.head_dim)
        output = self.out_proj(context)
        present = (key, value) if use_cache else None
        return output, present

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        return x.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)


class FeedForward(nn.Module):
    """T5-style MLP with optional gated activations."""

    def __init__(
        self,
        embed_dim: int,
        intermediate_dim: int,
        activations: Sequence[str],
        dropout: float,
    ):
        super().__init__()
        self.intermediate = nn.ModuleList(
            nn.Linear(embed_dim, intermediate_dim, bias=False)
            for _ in activations
        )
        self.activations = [_get_activation(a) for a in activations]
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(intermediate_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [act(proj(x)) for proj, act in zip(self.intermediate, self.activations)]
        hidden = parts[0]
        for part in parts[1:]:
            hidden = hidden * part
        hidden = self.dropout(hidden)
        return self.output(hidden)


def make_attention_mask(
    query_mask: torch.Tensor,
    key_mask: torch.Tensor,
) -> torch.Tensor:
    """Broadcasted attention mask for [batch, heads, q, k]."""

    mask = query_mask.unsqueeze(-1) & key_mask.unsqueeze(-2)
    return mask.unsqueeze(1)


def make_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
    return causal.unsqueeze(0).unsqueeze(0)


def combine_masks(*masks: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    valid_masks: list[torch.Tensor] = [m for m in masks if m is not None]
    if not valid_masks:
        return None
    out = valid_masks[0]
    for mask in valid_masks[1:]:
        out = out & mask
    return out

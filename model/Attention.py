"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/layers.py#L489 to pytorch
"""

import torch
import torch.nn as nn

from model.Layers import *
from model.Mask import *

device = "cuda" if torch.cuda.is_available() else "cpu"

from typing import Callable, Dict, Optional, Tuple


class Multi_Head_Attention(nn.Module):
    """Multi-head dot-product attention."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        dropout_rate: float = 0.0,
        kernel_init: Optional[Callable] = None,
        float32_logits: bool = False,
        use_flash_attention: bool = True,
        separate_qkv_proj: bool = True,  # Use separate Q/K/V projections (modern, more flexible)
        use_rope: bool = False,  # Use Rotary Position Embeddings
        rope_base: float = 10000.0,  # RoPE base frequency
        max_seq_len: int = 2048,  # Maximum sequence length for RoPE
    ):
        """
        Initializes the Multi-head Attention layer.

        Args:
            num_heads: Number of attention heads.
            head_dim: The dimension of each attention head.
            dtype: The dtype of the computation.
            dropout_rate: The dropout rate.
            kernel_init: The kernel initializer.
            float32_logits: Whether to use float32 for logits.
            use_flash_attention: Whether to use Flash Attention (PyTorch 2.0+).
            separate_qkv_proj: Whether to use separate Q/K/V projections (default: True).
                - True: Separate Linear layers for Q, K, V (modern, more parameters, more flexible)
                - False: Shared projection for Q, K, V (legacy MT3 style, fewer parameters)
            use_rope: Whether to use Rotary Position Embeddings (default: False).
            rope_base: Base frequency for RoPE (default: 10000.0).
            max_seq_len: Maximum sequence length for RoPE precomputation (default: 2048).
        """
        super(Multi_Head_Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.dropout_rate = dropout_rate
        self.separate_qkv_proj = separate_qkv_proj
        self.use_rope = use_rope
        
        # Separate or shared Q/K/V projections
        if separate_qkv_proj:
            # Modern approach: separate projections for Q, K, V
            # Allows independent learning of query, key, value representations
            self.query_projection = nn.Linear(
                self.num_heads * self.head_dim, self.num_heads * self.head_dim, bias=False
            )
            self.key_projection = nn.Linear(
                self.num_heads * self.head_dim, self.num_heads * self.head_dim, bias=False
            )
            self.value_projection = nn.Linear(
                self.num_heads * self.head_dim, self.num_heads * self.head_dim, bias=False
            )
        else:
            # Legacy approach: single shared projection
            # Saves parameters but less flexible
            self.projection = nn.Linear(
                self.num_heads * self.head_dim, self.num_heads * self.head_dim, bias=False
            )
        
        # Rotary Position Embeddings (optional)
        if use_rope:
            from model.RotaryEmbedding import RotaryEmbedding
            self.rope = RotaryEmbedding(
                dim=head_dim,
                max_seq_len=max_seq_len,
                base=rope_base,
            )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.kernel_init = kernel_init if kernel_init is not None else nn.init.xavier_uniform_
        self.float32_logits = float32_logits
        self.use_flash_attention = use_flash_attention and hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.output = nn.Linear(self.num_heads * self.head_dim, self.num_heads * self.head_dim)

    def dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Computes dot-product attention given query, key, and value.

        Args:
            query: A [batch, q_length, num_heads, qk_depth] tensor.
            key: A [batch, kv_length, num_heads, qk_depth] tensor.
            value: A [batch, kv_length, num_heads, v_depth] tensor.
            bias: A bias tensor.
            deterministic: Whether to use dropout.

        Returns:
            The attention output.
        """
        assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
        assert (
            query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
        ), "q, k, v batch dims must match."
        assert query.shape[-2] == key.shape[-2] == value.shape[-2], "q, k, v num_heads must match."
        assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
        assert query.shape[-1] == key.shape[-1], "q, k depths must match."

        # Use Flash Attention if available (PyTorch 2.0+)
        if self.use_flash_attention:
            # Reshape: [batch, length, num_heads, head_dim] -> [batch, num_heads, length, head_dim]
            q = query.transpose(1, 2)  # [batch, num_heads, q_length, head_dim]
            k = key.transpose(1, 2)    # [batch, num_heads, kv_length, head_dim]
            v = value.transpose(1, 2)  # [batch, num_heads, kv_length, head_dim]
            
            # Prepare attention mask from bias
            attn_mask = None
            if bias is not None:
                # bias is [batch, num_heads, q_length, kv_length]
                attn_mask = bias.to(q.dtype).to(q.device)
            
            # Flash Attention: automatically uses memory-efficient implementation
            dropout_p = 0.0 if deterministic else self.dropout_rate
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
            
            # Reshape back: [batch, num_heads, q_length, head_dim] -> [batch, q_length, num_heads, head_dim]
            return attn_output.transpose(1, 2)
        
        # Fallback: Original implementation
        # Casting logits and softmax computation for float32 for model stability.
        if self.float32_logits:
            query = query.float()
            key = key.float()

        # `attn_weights`: [batch, num_heads, q_length, kv_length]
        attn_weights = torch.einsum("bqhd,bkhd->bhqk", query, key)

        # Apply attention bias: masking, dropout, proximity bias, etc.
        if bias is not None:
            attn_weights = attn_weights + bias.to(attn_weights.dtype).to(attn_weights.device)

        # Normalize the attention weights across `kv_length` dimension.
        attn_weights = F.softmax(attn_weights, dim=-1).to(self.dtype)

        if not deterministic:
            attn_weights = self.dropout(attn_weights)

        # Take the linear combination of `value`.
        return torch.einsum("bhqk,bkhd->bqhd", attn_weights, value)

    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        decode: bool = False,
        cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Computes multi-head dot-product attention.

        Args:
            inputs_q: A [batch, q_length, features] tensor.
            inputs_kv: A [batch, kv_length, features] tensor.
            mask: An attention mask.
            bias: A bias tensor.
            decode: Whether to use autoregressive caching.
            cache: The autoregressive cache.

        Returns:
            A tuple of the attention output and the updated cache.
        """
        # In Original MT3, they initialize the parameter with query_init, using customized Dense Layer
        # Now supports separate Q/K/V projections for modern training
        if self.separate_qkv_proj:
            # Separate projections: more parameters, more flexibility
            query = self.query_projection(inputs_q).view(
                inputs_q.size(0), inputs_q.size(1), self.num_heads, self.head_dim
            )
            key = self.key_projection(inputs_kv).view(
                inputs_kv.size(0), inputs_kv.size(1), self.num_heads, self.head_dim
            )
            value = self.value_projection(inputs_kv).view(
                inputs_kv.size(0), inputs_kv.size(1), self.num_heads, self.head_dim
            )
        else:
            # Shared projection: legacy MT3 style
            query = self.projection(inputs_q).view(
                inputs_q.size(0), inputs_q.size(1), self.num_heads, self.head_dim
            )
            key = self.projection(inputs_kv).view(
                inputs_kv.size(0), inputs_kv.size(1), self.num_heads, self.head_dim
            )
            value = self.projection(inputs_kv).view(
                inputs_kv.size(0), inputs_kv.size(1), self.num_heads, self.head_dim
            )
        
        # Apply RoPE if enabled
        if self.use_rope:
            # RoPE is applied only to Q and K, not V
            # Shape: [batch, seq_len, num_heads, head_dim]
            query = self.rope(query, start_idx=0)
            key = self.rope(key, start_idx=0)

        if decode:
            if cache is None:
                cache = {
                    "cached_key": torch.zeros_like(key),
                    "cached_value": torch.zeros_like(value),
                }

            key = torch.cat([cache["cached_key"], key], dim=1)
            value = torch.cat([cache["cached_value"], value], dim=1)

            cache["cached_key"] = key
            cache["cached_value"] = value

        if mask is not None:
            attention_bias = torch.where(
                mask > 0,
                torch.zeros_like(mask).to(self.dtype),
                -1e10 * torch.ones_like(mask).to(self.dtype),
            )
        else:
            attention_bias = None

        if bias is not None:
            attention_bias = combine_biases(attention_bias, bias)

        x = self.dot_product_attention(
            query, key, value, bias=attention_bias, deterministic=not self.training
        )

        out = self.output(x.reshape(x.size(0), x.size(1), x.size(2) * x.size(3)))

        if decode:
            return out, cache
        return out, None

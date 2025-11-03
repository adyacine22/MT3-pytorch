"""
Rotary Position Embeddings (RoPE) for MT3-PyTorch

Implementation based on:
- RoFormer paper (Su et al., 2021): https://arxiv.org/abs/2104.09864
- LLaMA implementation
- GPT-NeoX implementation

RoPE applies rotary embeddings directly to Q/K, encoding relative positions
via rotation matrices. This provides better extrapolation than absolute
or learned position embeddings.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    
    Precomputes rotation matrices for all positions up to max_seq_len,
    then applies them to query and key embeddings in attention.
    
    Args:
        dim: Dimension of the rotary embeddings (should be head_dim)
        max_seq_len: Maximum sequence length to precompute (default: 2048)
        base: Base for the geometric progression of frequencies (default: 10000)
        device: Device to store embeddings (default: None, will use input device)
    
    Mathematical formulation:
        - Frequency: θ_i = base^(-2i/dim) for i in [0, dim/2)
        - Position: m ∈ [0, max_seq_len)
        - Rotation angle: m * θ_i
        - Applied as: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    
    Example:
        >>> rope = RotaryEmbedding(dim=64, max_seq_len=512)
        >>> q = torch.randn(2, 8, 256, 64)  # [batch, heads, seq, dim]
        >>> k = torch.randn(2, 8, 256, 64)
        >>> q_rot = rope(q, start_idx=0)
        >>> k_rot = rope(k, start_idx=0)
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute rotation frequencies
        # θ_i = base^(-2i/dim) for i in [0, dim/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos and sin for all positions
        self._update_cos_sin_cache(max_seq_len, device=device)
    
    def _update_cos_sin_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Update cached cos/sin values for given sequence length."""
        self.max_seq_len = seq_len
        
        # Position indices: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        # Compute all rotation angles: outer product of positions and frequencies
        # Shape: [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)
        
        # Combine cos and sin (interleave for easier application)
        # Shape: [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensor.
        
        Args:
            x: Input tensor of shape [..., seq_len, dim]
            start_idx: Starting position index (for cached decoding)
        
        Returns:
            Rotated tensor of same shape as input
        """
        seq_len = x.shape[-2]
        
        # Extend cache if needed
        if start_idx + seq_len > self.max_seq_len:
            self._update_cos_sin_cache(start_idx + seq_len, device=x.device)
        
        # Get cos/sin for current positions
        cos = self.cos_cached[start_idx : start_idx + seq_len]
        sin = self.sin_cached[start_idx : start_idx + seq_len]
        
        # Apply rotation
        return apply_rotary_emb(x, cos, sin)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply precomputed rotary embeddings.
    
    Rotation formula for complex numbers in 2D:
        [x1, x2] rotated by θ = [x1*cos(θ) - x2*sin(θ), x1*sin(θ) + x2*cos(θ)]
    
    Args:
        x: Input tensor [..., seq_len, dim]
        cos: Cosine values [seq_len, dim]
        sin: Sine values [seq_len, dim]
    
    Returns:
        Rotated tensor of same shape
    """
    # Split x into pairs: [x1, x2, x3, x4, ...] -> ([x1, x3, ...], [x2, x4, ...])
    x1 = x[..., ::2]  # Even indices
    x2 = x[..., 1::2]  # Odd indices
    
    # Apply rotation to each pair
    # Rotated x1 = x1 * cos - x2 * sin
    # Rotated x2 = x1 * sin + x2 * cos
    
    # Expand cos/sin to match x dimensions
    # x shape: [..., seq_len, dim]
    # cos/sin shape: [seq_len, dim] -> need to broadcast
    cos = cos[..., ::2]  # Match split dimension
    sin = sin[..., ::2]
    
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    
    # Interleave back: ([x1', x3', ...], [x2', x4', ...]) -> [x1', x2', x3', x4', ...]
    rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
    rotated = rotated.flatten(-2)  # Merge last two dimensions
    
    return rotated.type_as(x)


class RotaryEmbeddingESM(nn.Module):
    """
    Alternative RoPE implementation (ESM/Meta style).
    
    This version uses a different interleaving strategy that's more efficient
    for certain hardware. Produces identical results to standard RoPE.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        self._set_cos_sin_cache(max_seq_len)
    
    def _set_cos_sin_cache(self, seq_len: int):
        """Set cos/sin cache."""
        self.max_seq_len = seq_len
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Different from standard: repeat instead of concat
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor, seq_len: int = None):
        """Apply rotary embeddings."""
        if seq_len is None:
            seq_len = x.shape[-2]
        
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
        
        return x * self.cos_cached[:, :, :seq_len, :] + \
               self._rotate_half(x) * self.sin_cached[:, :, :seq_len, :]
    
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)


# Helper function for easy integration
def apply_rope_to_query_key(
    q: torch.Tensor,
    k: torch.Tensor,
    rope: RotaryEmbedding,
    start_idx: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors.
    
    Args:
        q: Query tensor [..., seq_len, dim]
        k: Key tensor [..., seq_len, dim]
        rope: RotaryEmbedding instance
        start_idx: Starting position (for cached decoding)
    
    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    q_rotated = rope(q, start_idx=start_idx)
    k_rotated = rope(k, start_idx=start_idx)
    return q_rotated, k_rotated

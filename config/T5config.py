"""
Referenced original MT3 github code,
https://github.com/magenta/mt3/blob/main/mt3/network.py
"""

import torch
from typing import Any, Sequence
from data.constants import VOCAB_SIZE, DEFAULT_NUM_MEL_BINS


class Magenta_T5Config:
    """MT3 T5 Model Configuration.

    Token Types:
    1) Instrument(128 values)
    2) Note(128 values)
    3) On/Off(2 values)
    4) Time(205 values)
    5) Drum(128 values)
    6) End Tie Section(1 value)
    7) EOS(1 value)
    """

    def __init__(self):
        # Vocabulary configuration
        self.vocab_size: int = VOCAB_SIZE

        # Data types
        self.dtype: Any = torch.float32

        # Input/Output dimensions
        self.input_depth: int = DEFAULT_NUM_MEL_BINS  # Number of mel bins for encoder input
        self.emb_dim: int = 512  # Embedding dimension
        self.d_model: int = 512  # Model dimension (same as emb_dim)

        # Attention configuration
        self.num_heads: int = 8
        self.head_dim: int = 64
        
        # Modern attention features
        self.separate_qkv_proj: bool = True  # Use separate Q/K/V projections (modern, more flexible)
        self.use_rope: bool = False  # Use Rotary Position Embeddings (experimental, modern)
        self.rope_base: float = 10000.0  # RoPE base frequency
        self.max_seq_len: int = 2048  # Maximum sequence length for RoPE

        # Layer configuration
        self.num_encoder_layers: int = 6
        self.num_decoder_layers: int = 6

        # Feed-forward configuration
        self.mlp_dim: int = 2048  # Changed from 1024 to match legacy MT3
        self.d_ff: int = 2048  # Feed-forward dimension (same as mlp_dim)
        self.mlp_activations: Sequence[str] = ("gelu",)  # Changed from 'relu' to match legacy MT3

        # Regularization
        self.dropout_rate: float = 0.1

        # Model behavior
        self.logits_via_embeddings: bool = False

        # Sequence length (for compatibility)
        self.max_length: int = 1024

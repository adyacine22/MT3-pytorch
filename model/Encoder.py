"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/network.py#L158 to pytorch
"""

import torch
import torch.nn as nn
from model.Layers import *
from model.Attention import Multi_Head_Attention

from typing import Any, Optional

class EncoderLayer(nn.Module):
    """A single layer of the encoder."""
    def __init__(self, config: Any, use_flash_attention: bool = True):
        """
        Initializes the EncoderLayer.

        Args:
            config: The configuration object.
            use_flash_attention: Whether to use Flash Attention.
        """
        super(EncoderLayer, self).__init__()
        self.config = config

        self.pre_attention_layer_norm = LayerNorm(config.emb_dim)
        self.attention = Multi_Head_Attention(
            num_heads=config.num_heads, 
            head_dim=config.head_dim, 
            dropout_rate=config.dropout_rate,
            use_flash_attention=use_flash_attention,
            separate_qkv_proj=getattr(config, 'separate_qkv_proj', True),
            use_rope=getattr(config, 'use_rope', False),
            rope_base=getattr(config, 'rope_base', 10000.0),
            max_seq_len=getattr(config, 'max_seq_len', 2048),
        )
        self.dropout1 = nn.Dropout(config.dropout_rate)

        self.pre_mlp_layer_norm = LayerNorm(config.emb_dim)
        self.mlp = MlpBlock(emb_dim=config.emb_dim, intermediate_dim=config.mlp_dim, activations=config.mlp_activations, intermediate_dropout_rate=config.dropout_rate)
        self.dropout2 = nn.Dropout(config.dropout_rate)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs a forward pass through the encoder layer.

        Args:
            inputs: The encoder input.
            mask: The attention mask.

        Returns:
            The encoder layer output.
        """
        x = self.pre_attention_layer_norm(inputs)
        x, _ = self.attention(x, x, mask=mask)
        x = self.dropout1(x)
        x = x + inputs

        y = self.pre_mlp_layer_norm(x)
        y = self.mlp(y)
        y = self.dropout2(y)
        y = y + x

        return y

class Encoder(nn.Module):
    """The encoder of the T5 model."""
    def __init__(self, config: Any, use_flash_attention: bool = True):
        """
        Initializes the Encoder.

        Args:
            config: The configuration object.
            use_flash_attention: Whether to use Flash Attention.
        """
        super(Encoder, self).__init__()
        self.config = config

        self.token_embed = nn.Linear(in_features=config.input_depth, out_features=config.emb_dim)
        self.fixed_embed = FixedEmbed(config.emb_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config, use_flash_attention=use_flash_attention) 
            for _ in range(config.num_encoder_layers)
        ])
        self.layer_norm = LayerNorm(config.emb_dim)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None, deterministic: bool = False) -> torch.Tensor:
        """
        Performs a forward pass through the encoder.

        Args:
            inputs: The encoder input.
            mask: The attention mask.
            deterministic: Whether to use dropout.

        Returns:
            The encoder output.
        """
        cfg = self.config
        assert inputs.ndim == 3  # [batch, length, depth]

        seq_length = inputs.shape[1]
        positions = torch.arange(seq_length)[None, :].to(inputs.device)

        # [batch, length, depth] -> [batch, length, emb_dim]
        y = self.token_embed(inputs)
        y = y + self.fixed_embed(positions)
        if not deterministic:
            y = self.dropout(y)
        y = y.type(cfg.dtype)

        for layer in self.encoder_layers:
            # [batch, length, emb_dim] -> [batch, length, emb_dim]
            y = layer(y, mask=mask)

        y = self.layer_norm(y)
        if not deterministic:
            y = self.dropout(y)

        return y

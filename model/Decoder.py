"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/network.py#L158 to pytorch
"""

import torch
import torch.nn as nn
from model.Layers import *
from model.Attention import Multi_Head_Attention

from typing import Any, Dict, List, Optional, Tuple

class DecoderLayer(nn.Module):
    """A single layer of the decoder."""
    def __init__(self, config: Any, use_flash_attention: bool = True):
        """
        Initializes the DecoderLayer.

        Args:
            config: The configuration object.
            use_flash_attention: Whether to use Flash Attention.
        """
        super(DecoderLayer, self).__init__()
        self.config = config

        self.pre_self_attention_layer_norm = LayerNorm(config.emb_dim)
        self.self_attention = Multi_Head_Attention(
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

        self.pre_cross_attention_layer_norm = LayerNorm(config.emb_dim)
        self.encoder_decoder_attention = Multi_Head_Attention(
            num_heads=config.num_heads, 
            head_dim=config.head_dim, 
            dropout_rate=config.dropout_rate,
            use_flash_attention=use_flash_attention,
            separate_qkv_proj=getattr(config, 'separate_qkv_proj', True),
            use_rope=getattr(config, 'use_rope', False),
            rope_base=getattr(config, 'rope_base', 10000.0),
            max_seq_len=getattr(config, 'max_seq_len', 2048),
        )
        self.dropout2 = nn.Dropout(config.dropout_rate)

        self.pre_mlp_layer_norm = LayerNorm(config.emb_dim)
        self.mlp = MlpBlock(emb_dim=config.emb_dim, intermediate_dim=config.mlp_dim, activations=config.mlp_activations, intermediate_dropout_rate=config.dropout_rate)
        self.dropout3 = nn.Dropout(config.dropout_rate)

    def forward(self, inputs: torch.Tensor, encoded: torch.Tensor, decoder_mask: Optional[torch.Tensor] = None, 
                encoder_decoder_mask: Optional[torch.Tensor] = None, decode: bool = False, 
                cache: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Performs a forward pass through the decoder layer.

        Args:
            inputs: The decoder input.
            encoded: The encoder output.
            decoder_mask: The decoder self-attention mask.
            encoder_decoder_mask: The encoder-decoder cross-attention mask.
            decode: Whether to use autoregressive caching.
            cache: The autoregressive cache.

        Returns:
            A tuple of the decoder layer output and the updated cache.
        """
        x = self.pre_self_attention_layer_norm(inputs)
        x, cache = self.self_attention(x, x, mask=decoder_mask, decode=decode, cache=cache)
        x = self.dropout1(x)
        x = x + inputs

        y = self.pre_cross_attention_layer_norm(x)
        y, _ = self.encoder_decoder_attention(y, encoded, mask=encoder_decoder_mask)
        y = self.dropout2(y)
        y = y + x

        z = self.pre_mlp_layer_norm(y)
        z = self.mlp(z)
        z = self.dropout3(z)
        z = z + y

        if decode:
            return z, cache
        return z, None

class Decoder(nn.Module):
    """The decoder of the T5 model."""
    def __init__(self, config: Any, use_flash_attention: bool = True):
        """
        Initializes the Decoder.

        Args:
            config: The configuration object.
            use_flash_attention: Whether to use Flash Attention.
        """
        super(Decoder, self).__init__()
        self.config = config
        
        self.token_embed = Embed(config.vocab_size, config.emb_dim, one_hot=True)
        self.fixed_embed = FixedEmbed(config.emb_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config, use_flash_attention=use_flash_attention) 
            for _ in range(config.num_decoder_layers)
        ])
        self.layer_norm = LayerNorm(config.emb_dim)
        self.dense = nn.Linear(in_features=config.emb_dim, out_features=config.vocab_size)
        
        # Weight tying: Share embedding weights with output layer (legacy MT3 behavior)
        # This reduces parameters and improves generalization
        self.dense.weight = self.token_embed.embedding
        
    def forward(self, encoded: torch.Tensor, decoder_input_tokens: torch.Tensor, 
                decoder_positions: Optional[torch.Tensor] = None, decoder_mask: Optional[torch.Tensor] = None, 
                encoder_decoder_mask: Optional[torch.Tensor] = None, deterministic: bool = False, 
                decode: bool = False, max_decode_length: Optional[int] = None, 
                cache: Optional[List[Dict[str, torch.Tensor]]] = None) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
        """
        Performs a forward pass through the decoder.

        Args:
            encoded: The encoder output.
            decoder_input_tokens: The decoder input tokens.
            decoder_positions: The decoder positions.
            decoder_mask: The decoder self-attention mask.
            encoder_decoder_mask: The encoder-decoder cross-attention mask.
            deterministic: Whether to use dropout.
            decode: Whether to use autoregressive caching.
            max_decode_length: The maximum decoding length.
            cache: The autoregressive cache.

        Returns:
            A tuple of the decoder output logits and the updated cache.
        """
        cfg = self.config
        assert decoder_input_tokens.ndim == 2  # [batch, len]

        seq_length = decoder_input_tokens.shape[-1]
        if decoder_positions is None:
            decoder_positions = torch.arange(seq_length)[None, :].to(decoder_input_tokens.device)

        # [batch, length] -> [batch, length, emb_dim]
        y = self.token_embed(decoder_input_tokens)
        y = y + self.fixed_embed(decoder_positions)
        y = self.dropout(y)
        y = y.type(cfg.dtype)

        if decode and cache is None:
            cache = [None] * len(self.decoder_layers)

        new_cache = []
        for i, layer in enumerate(self.decoder_layers):
            # [batch, length, emb_dim] -> [batch, length, emb_dim]
            if decode:
                y, layer_cache = layer(y, encoded, decoder_mask=decoder_mask, encoder_decoder_mask=encoder_decoder_mask, decode=decode, cache=cache[i])
                new_cache.append(layer_cache)
            else:
                y, _ = layer(y, encoded, decoder_mask=decoder_mask, encoder_decoder_mask=encoder_decoder_mask, decode=decode)

        y = self.layer_norm(y)
        if not deterministic:
            y = self.dropout(y)

        # [batch, length, emb_dim] -> [batch, length, vocab_size]
        logits = self.dense(y)

        if decode:
            return logits, new_cache
        return logits, None

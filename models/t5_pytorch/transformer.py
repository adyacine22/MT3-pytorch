"""PyTorch implementation of the MT3 encoder-decoder transformer."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .config import T5Config
from .layers import (
    FeedForward,
    MultiHeadAttention,
    SinusoidalPositions,
    T5LayerNorm,
    combine_masks,
    make_attention_mask,
    make_causal_mask,
)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: T5Config):
        super().__init__()
        self.self_norm = T5LayerNorm(cfg.emb_dim)
        self.self_attn = MultiHeadAttention(
            cfg.emb_dim, cfg.num_heads, cfg.head_dim, cfg.dropout_rate
        )
        self.dropout = nn.Dropout(cfg.dropout_rate)
        self.ff_norm = T5LayerNorm(cfg.emb_dim)
        self.ff = FeedForward(
            cfg.emb_dim, cfg.mlp_dim, cfg.mlp_activations, cfg.dropout_rate
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        y, _ = self.self_attn(self.self_norm(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + self.dropout(y)
        y = self.ff(self.ff_norm(hidden_states))
        hidden_states = hidden_states + self.dropout(y)
        return hidden_states


class DecoderLayer(nn.Module):
    def __init__(self, cfg: T5Config):
        super().__init__()
        self.self_norm = T5LayerNorm(cfg.emb_dim)
        self.self_attn = MultiHeadAttention(
            cfg.emb_dim, cfg.num_heads, cfg.head_dim, cfg.dropout_rate
        )
        self.cross_norm = T5LayerNorm(cfg.emb_dim)
        self.cross_attn = MultiHeadAttention(
            cfg.emb_dim, cfg.num_heads, cfg.head_dim, cfg.dropout_rate
        )
        self.dropout = nn.Dropout(cfg.dropout_rate)
        self.ff_norm = T5LayerNorm(cfg.emb_dim)
        self.ff = FeedForward(
            cfg.emb_dim, cfg.mlp_dim, cfg.mlp_activations, cfg.dropout_rate
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoded: torch.Tensor,
        self_mask: Optional[torch.Tensor],
        cross_mask: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        normed = self.self_norm(hidden_states)
        y, present = self.self_attn(
            normed,
            attention_mask=self_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + self.dropout(y)

        y, _ = self.cross_attn(
            self.cross_norm(hidden_states),
            key_value_states=encoded,
            attention_mask=cross_mask,
        )
        hidden_states = hidden_states + self.dropout(y)

        y = self.ff(self.ff_norm(hidden_states))
        hidden_states = hidden_states + self.dropout(y)
        return hidden_states, present


class Encoder(nn.Module):
    def __init__(self, cfg: T5Config):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_depth, cfg.emb_dim, bias=False)
        self.pos_emb = SinusoidalPositions(cfg.max_position_embeddings, cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.dropout_rate)
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.num_encoder_layers))
        self.final_norm = T5LayerNorm(cfg.emb_dim)

    def forward(
        self,
        encoder_input_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = self.input_proj(encoder_input_tokens)
        positions = torch.arange(
            encoder_input_tokens.size(1), device=encoder_input_tokens.device
        ).unsqueeze(0).expand(encoder_input_tokens.size(0), -1)
        hidden_states = hidden_states + self.pos_emb(positions)
        hidden_states = self.dropout(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        hidden_states = self.final_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class Decoder(nn.Module):
    def __init__(self, cfg: T5Config):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = SinusoidalPositions(cfg.max_position_embeddings, cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.dropout_rate)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_decoder_layers))
        self.final_norm = T5LayerNorm(cfg.emb_dim)

    def forward(
        self,
        decoder_input_tokens: torch.Tensor,
        encoded: torch.Tensor,
        self_mask: Optional[torch.Tensor],
        cross_mask: Optional[torch.Tensor],
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        positions = torch.arange(
            decoder_input_tokens.size(1), device=decoder_input_tokens.device
        ).unsqueeze(0).expand(decoder_input_tokens.size(0), -1)
        hidden_states = self.token_embed(decoder_input_tokens) + self.pos_emb(positions)
        hidden_states = self.dropout(hidden_states)

        next_past: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for idx, layer in enumerate(self.layers):
            past = past_key_values[idx] if past_key_values is not None else None
            hidden_states, present = layer(
                hidden_states,
                encoded,
                self_mask,
                cross_mask,
                past_key_value=past,
                use_cache=use_cache,
            )
            if use_cache:
                next_past.append(present)

        hidden_states = self.final_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, (next_past if use_cache else None)


class MT3Transformer(nn.Module):
    """End-to-end encoder-decoder network with continuous inputs."""

    def __init__(self, cfg: T5Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.output_layer: Optional[nn.Linear]
        if cfg.logits_via_embedding:
            self.output_layer = None
        else:
            self.output_layer = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(
        self,
        encoder_input_tokens: torch.Tensor,
        decoder_input_tokens: torch.Tensor,
        decoder_target_tokens: Optional[torch.Tensor] = None,
        *,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        decoder_padding_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        device = encoder_input_tokens.device
        batch_size, encoder_len = encoder_input_tokens.shape[:2]
        decoder_len = decoder_input_tokens.size(1)

        if encoder_padding_mask is None:
            encoder_padding_mask = torch.ones(
                batch_size, encoder_len, dtype=torch.bool, device=device
            )
        if decoder_padding_mask is None:
            if decoder_target_tokens is None:
                decoder_padding_mask = torch.ones(
                    batch_size, decoder_len, dtype=torch.bool, device=device
                )
            else:
                decoder_padding_mask = decoder_target_tokens.gt(0)

        encoder_mask = make_attention_mask(encoder_padding_mask, encoder_padding_mask)
        decoder_mask = combine_masks(
            make_attention_mask(decoder_padding_mask, decoder_padding_mask),
            make_causal_mask(decoder_len, device),
        )
        cross_mask = make_attention_mask(decoder_padding_mask, encoder_padding_mask)

        encoded = self.encoder(encoder_input_tokens, encoder_mask)
        decoded, cache = self.decoder(
            decoder_input_tokens,
            encoded,
            decoder_mask,
            cross_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if self.cfg.logits_via_embedding:
            logits = torch.matmul(
                decoded, self.decoder.token_embed.weight.t()
            ) / decoded.size(-1) ** 0.5
        else:
            output_layer = self.output_layer
            if output_layer is None:
                raise RuntimeError("Output projection missing despite logits_via_embedding=False")
            logits = output_layer(decoded)
        return logits, cache

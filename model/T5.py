"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/network.py#L158 to pytorch
"""

import torch
import torch.nn as nn
from model.Encoder import Encoder
from model.Decoder import Decoder
from model.Layers import *
from model.Mask import *
from data.constants import *

device = "cuda" if torch.cuda.is_available() else "cpu"

from typing import Any, Optional


class Transformer(nn.Module):
    """The T5 model."""

    def __init__(self, config: Any, use_flash_attention: bool = True):
        """
        Initializes the Transformer model.

        Args:
            config: The configuration object.
            use_flash_attention: Whether to use Flash Attention (PyTorch 2.0+).
        """
        super(Transformer, self).__init__()
        self.config = config
        self.encoder = Encoder(config=config, use_flash_attention=use_flash_attention)
        self.decoder = Decoder(config=config, use_flash_attention=use_flash_attention)

    def encode(
        self,
        encoder_input_tokens: torch.Tensor,
        encoder_segment_ids: Optional[torch.Tensor] = None,
        enable_dropout: bool = True,
    ) -> torch.Tensor:
        """
        Encodes the input tokens.

        Args:
            encoder_input_tokens: The encoder input tokens.
            encoder_segment_ids: The encoder segment IDs.
            enable_dropout: Whether to use dropout.

        Returns:
            The encoder output.
        """
        assert encoder_input_tokens.ndim == 3  # (batch, length, depth)

        encoder_mask = make_attention_mask(
            torch.ones(encoder_input_tokens.shape[:-1]),
            torch.ones(encoder_input_tokens.shape[:-1]),
            dtype=self.config.dtype,
        )

        if encoder_segment_ids is not None:
            encoder_mask = combine_masks(
                encoder_mask,
                make_attention_mask(
                    encoder_segment_ids, encoder_segment_ids, torch.equal, dtype=self.config.dtype
                ),
            )

        return self.encoder(encoder_input_tokens, encoder_mask, deterministic=not enable_dropout)

    def decode(
        self,
        encoded: torch.Tensor,
        encoder_input_tokens: torch.Tensor,
        decoder_input_tokens: torch.Tensor,
        decoder_target_tokens: torch.Tensor,
        encoder_segment_ids: Optional[torch.Tensor] = None,
        decoder_segment_ids: Optional[torch.Tensor] = None,
        decoder_positions: Optional[torch.Tensor] = None,
        enable_dropout: bool = True,
        decode: bool = False,  # decode: Whether to prepare and use an autoregressive cache
        max_decode_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Decodes the encoder output.

        Args:
            encoded: The encoder output.
            encoder_input_tokens: The encoder input tokens.
            decoder_input_tokens: The decoder input tokens.
            decoder_target_tokens: The decoder target tokens.
            encoder_segment_ids: The encoder segment IDs.
            decoder_segment_ids: The decoder segment IDs.
            decoder_positions: The decoder positions.
            enable_dropout: Whether to use dropout.
            decode: Whether to use autoregressive caching.
            max_decode_length: The maximum decoding length.

        Returns:
            The decoder output logits.
        """

        if decode:
            # For decoding, we need to create a causal mask that grows with the sequence.
            # However, the current implementation of make_decoder_mask expects the full target sequence.
            # We will create a simpler causal mask here.
            seq_len = decoder_input_tokens.shape[1]
            decoder_mask = torch.tril(
                torch.ones((seq_len, seq_len), dtype=torch.bool, device=decoder_input_tokens.device)
            )
            encoder_decoder_mask = make_attention_mask(
                torch.ones_like(decoder_input_tokens).to(device),
                torch.ones(encoder_input_tokens.shape[:-1]).to(device),
                dtype=self.config.dtype,
            )
        else:
            decoder_mask = make_decoder_mask(
                decoder_target_tokens,
                dtype=self.config.dtype,
                decoder_segment_ids=decoder_segment_ids,
            )
            encoder_decoder_mask = make_attention_mask(
                decoder_target_tokens > 0,
                torch.ones(encoder_input_tokens.shape[:-1], device=encoder_input_tokens.device),
                dtype=self.config.dtype,
            )

        if encoder_segment_ids is not None:
            if decode:
                raise ValueError(
                    "During decoding, packing should not be used but `encoder_segment_ids` was passed to `Transformer.decode`."
                )

            encoder_decoder_mask = combine_masks(
                encoder_decoder_mask,
                make_attention_mask(
                    decoder_segment_ids, encoder_segment_ids, torch.equal, dtype=self.config.dtype
                ),
            )

        logits, _ = self.decoder(
            encoded,
            decoder_input_tokens=decoder_input_tokens,
            decoder_positions=decoder_positions,
            decoder_mask=decoder_mask,
            encoder_decoder_mask=encoder_decoder_mask,
            deterministic=not enable_dropout,
            decode=decode,
            max_decode_length=max_decode_length,
        )
        return logits.type(self.config.dtype)

    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Shifts the input IDs to the right for decoder input."""
        decoder_start_token_id = TOKEN_START

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        return shifted_input_ids.to(device)

    def forward(
        self,
        encoder_input_tokens: torch.Tensor,
        decoder_target_tokens: torch.Tensor,
        decoder_input_tokens: Optional[torch.Tensor] = None,
        encoder_segment_ids: Optional[torch.Tensor] = None,
        decoder_segment_ids: Optional[torch.Tensor] = None,
        encoder_positions: Optional[torch.Tensor] = None,
        decoder_positions: Optional[torch.Tensor] = None,
        enable_dropout: bool = True,
        decode: bool = False,
    ) -> torch.Tensor:
        """
        Performs a forward pass through the transformer.

        Args:
            encoder_input_tokens: The encoder input tokens.
            decoder_target_tokens: The decoder target tokens.
            decoder_input_tokens: The decoder input tokens.
            encoder_segment_ids: The encoder segment IDs.
            decoder_segment_ids: The decoder segment IDs.
            encoder_positions: The encoder positions.
            decoder_positions: The decoder positions.
            enable_dropout: Whether to use dropout.
            decode: Whether to use autoregressive caching.

        Returns:
            The decoder output logits.
        """
        if decoder_input_tokens is None:
            decoder_input_tokens = self._shift_right(decoder_target_tokens)
        encoded = self.encode(
            encoder_input_tokens,
            encoder_segment_ids=encoder_segment_ids,
            enable_dropout=enable_dropout,
        )
        return self.decode(
            encoded,
            encoder_input_tokens,
            decoder_input_tokens,
            decoder_target_tokens,
            encoder_segment_ids=encoder_segment_ids,
            decoder_segment_ids=decoder_segment_ids,
            decoder_positions=decoder_positions,
            enable_dropout=enable_dropout,
            decode=decode,
        )

    def generate(
        self,
        encoder_input_tokens: torch.Tensor,
        max_length: int = 1024,
        start_token: int = TOKEN_START,
    ) -> torch.Tensor:
        """Generates a sequence of tokens given an encoder input."""
        self.eval()
        batch_size = encoder_input_tokens.shape[0]

        # Encode the input
        encoded = self.encode(encoder_input_tokens, enable_dropout=False)

        # Initialize the decoder input with the start token
        decoder_input_tokens = torch.full(
            (batch_size, 1), start_token, dtype=torch.long, device=encoder_input_tokens.device
        )

        # Autoregressive decoding loop
        for _ in range(max_length):
            # Decode the next token (decode=False to avoid KV-cache bugs)
            logits = self.decode(
                encoded,
                encoder_input_tokens,
                decoder_input_tokens,
                decoder_input_tokens,  # Not used in decode mode
                enable_dropout=False,
                decode=False,  # KV-caching disabled due to dimension mismatch bugs
            )

            # Get the last token logits and apply softmax
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)

            # Append the new token to the decoder input
            decoder_input_tokens = torch.cat(
                [decoder_input_tokens, next_token.unsqueeze(-1)], dim=-1
            )

            # Stop if all sequences have generated an EOS token
            if torch.all(next_token == TOKEN_END):
                break

        return decoder_input_tokens

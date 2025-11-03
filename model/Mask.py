"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/layers.py#L489 to pytorch
"""

import torch
from typing import Callable, Optional

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_attention_mask(
    query_input: torch.Tensor,
    key_input: torch.Tensor,
    pairwise_fn: Callable = torch.mul,
    extra_batch_dims: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Creates an attention mask.

    Args:
        query_input: The query input tensor.
        key_input: The key input tensor.
        pairwise_fn: The pairwise function to use.
        extra_batch_dims: The number of extra batch dimensions.
        dtype: The dtype of the mask.

    Returns:
        The attention mask.
    """
    mask = pairwise_fn(query_input.unsqueeze(-1), key_input.unsqueeze(-2))
    mask = mask.unsqueeze(-3)

    for i in range(extra_batch_dims):
        mask = mask.unsqueeze(i)

    return mask.type(dtype)


def make_causal_mask(
    x: torch.Tensor, extra_batch_dims: int = 0, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Creates a causal mask.

    Args:
        x: The input tensor.
        extra_batch_dims: The number of extra batch dimensions.
        dtype: The dtype of the mask.

    Returns:
        The causal mask.
    """
    idxs = torch.arange(x.shape[-1], dtype=torch.int32, device=x.device).expand(x.shape)
    return make_attention_mask(
        idxs, idxs, torch.greater_equal, extra_batch_dims=extra_batch_dims, dtype=dtype
    )


def combine_masks(
    *masks: Optional[torch.Tensor], dtype: torch.dtype = torch.float32
) -> Optional[torch.Tensor]:
    """
    Combines multiple masks.

    Args:
        masks: The masks to combine.
        dtype: The dtype of the combined mask.

    Returns:
        The combined mask.
    """
    # Filter out None masks (keep device-agnostic)
    filtered_masks = [m for m in masks if m is not None]

    if not filtered_masks:
        return None

    assert all(
        map(lambda x: x.ndim == filtered_masks[0].ndim, filtered_masks)
    ), f"masks must have same rank: {tuple(map(lambda x: x.ndim, filtered_masks))}"

    mask = filtered_masks[0]
    for other_mask in filtered_masks[1:]:
        mask = torch.logical_and(mask, other_mask)

    return mask.to(dtype)


def combine_biases(*masks: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Combines multiple biases.

    Args:
        masks: The biases to combine.

    Returns:
        The combined bias.
    """
    filtered_masks = [m for m in masks if m is not None]

    if not filtered_masks:
        return None

    assert all(
        map(lambda x: x.ndim == filtered_masks[0].ndim, filtered_masks)
    ), f"masks must have same rank: {tuple(map(lambda x: x.ndim, filtered_masks))}"

    mask = filtered_masks[0]
    for other_mask in filtered_masks[1:]:
        mask = mask + other_mask

    return mask


def make_decoder_mask(
    decoder_target_tokens: torch.Tensor,
    dtype: torch.dtype,
    decoder_causal_attention: Optional[torch.Tensor] = None,
    decoder_segment_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Creates a decoder mask.

    Args:
        decoder_target_tokens: The decoder target tokens.
        dtype: The dtype of the mask.
        decoder_causal_attention: The decoder causal attention tensor.
        decoder_segment_ids: The decoder segment IDs.

    Returns:
        The decoder mask.
    """
    masks = []

    causal_mask = make_causal_mask(decoder_target_tokens, dtype=dtype)

    if decoder_causal_attention is not None:
        inputs_mask = make_attention_mask(
            decoder_causal_attention, decoder_causal_attention, torch.logical_and, dtype=dtype
        )
        masks.append(torch.logical_or(causal_mask, inputs_mask).to(dtype))
    else:
        masks.append(causal_mask)

    masks.append(
        make_attention_mask(decoder_target_tokens > 0, decoder_target_tokens > 0, dtype=dtype)
    )

    # Packing mask
    if decoder_segment_ids is not None:
        masks.append(
            make_attention_mask(decoder_segment_ids, decoder_segment_ids, torch.equal, dtype=dtype)
        )

    return combine_masks(*masks, dtype=dtype)

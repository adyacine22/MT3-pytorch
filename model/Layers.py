"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/layers.py#L489 to pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

from typing import Callable, Optional, Tuple


def sinusoidal(
    shape: Tuple[int, int],
    min_scale: float = 1.0,
    max_scale: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sinusoidal init."""
    if dtype != torch.float32:
        raise ValueError("The sinusoidal initializer only supports float32.")
    if len(list(shape)) != 2:
        raise ValueError(f"Expected a 2D shape (max_len, features), but got {shape}.")
    max_len, features = shape
    pe = torch.zeros((max_len, features), dtype=dtype)
    position = torch.arange(0, max_len)[:, None]
    scale_factor = -np.log(max_scale / min_scale) / (features // 2 - 1)
    div_term = min_scale * torch.exp(torch.arange(0, features // 2) * torch.tensor(scale_factor))
    pe[:, : features // 2] = torch.sin(position * div_term)
    pe[:, features // 2 : 2 * (features // 2)] = torch.cos(position * div_term)

    return pe


class MlpBlock(nn.Module):
    """MLP block for the transformer."""

    def __init__(
        self,
        emb_dim: int = 512,
        intermediate_dim: int = 2048,
        activations: Tuple[str, ...] = ("relu",),
        kernel_init: Callable = nn.init.xavier_uniform_,
        intermediate_dropout_rate: float = 0.1,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initializes the MlpBlock.

        Args:
            emb_dim: The embedding dimension.
            intermediate_dim: The intermediate dimension.
            activations: The activation functions.
            kernel_init: The kernel initializer.
            intermediate_dropout_rate: The dropout rate for the intermediate layer.
            dtype: The dtype of the computation.
        """
        super(MlpBlock, self).__init__()
        self.dtype = dtype
        self.intermediate_layers = nn.ModuleList(
            [nn.Linear(in_features=emb_dim, out_features=intermediate_dim), nn.GELU()]  # Changed from ReLU to GELU (legacy MT3)
        )
        self.dropout = nn.Dropout(intermediate_dropout_rate)
        self.dense_layer = nn.Linear(in_features=intermediate_dim, out_features=emb_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the MLP block.

        Args:
            inputs: The input tensor.

        Returns:
            The output tensor.
        """
        x = inputs

        for layer in self.intermediate_layers:
            x = layer(x)

        x = self.dropout(x)
        output = self.dense_layer(x)
        return output


class Embed(nn.Module):
    """Embedding layer."""

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        dtype: torch.dtype = torch.float32,
        attend_dtype: Optional[torch.dtype] = None,
        embedding_init: Callable = nn.init.normal_,
        one_hot: bool = False,
    ):
        """
        Initializes the Embed layer.

        Args:
            num_embeddings: The number of embeddings.
            features: The embedding dimension.
            dtype: The dtype of the computation.
            attend_dtype: The dtype of the attention computation.
            embedding_init: The embedding initializer.
            one_hot: Whether to use one-hot encoding.
        """
        super(Embed, self).__init__()
        self.num_embeddings = num_embeddings
        self.dtype = dtype
        self.embedding_init = embedding_init
        self.one_hot = one_hot

        self.embedding = nn.Parameter(torch.Tensor(num_embeddings, features))
        self.reset_parameters()

    def reset_parameters(self):
        """Resets the embedding parameters."""
        self.embedding_init(self.embedding)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the embedding layer.

        Args:
            inputs: The input tensor.

        Returns:
            The output tensor.
        """
        if self.one_hot:
            iota = torch.arange(self.num_embeddings, dtype=torch.int32, device=inputs.device)
            one_hot = (inputs[..., None] == iota).to(dtype=self.dtype, device=self.embedding.device)
            output = torch.matmul(one_hot, self.embedding)
        else:
            output = self.embedding[inputs]

        return output


class FixedEmbed(nn.Module):
    """Fixed embedding layer for positional encoding."""

    def __init__(self, features: int, max_length: int = 2048, dtype: torch.dtype = torch.float32):
        """
        Initializes the FixedEmbed layer.

        Args:
            features: The embedding dimension.
            max_length: The maximum sequence length.
            dtype: The dtype of the computation.
        """
        super(FixedEmbed, self).__init__()
        self.features = features
        self.max_length = max_length
        self.dtype = dtype
        # Register as buffer so it moves with the model to correct device
        embedding = sinusoidal(shape=(self.max_length, self.features), dtype=self.dtype)
        self.register_buffer("embedding", embedding)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the fixed embedding layer.

        Args:
            inputs: The input tensor.

        Returns:
            The output tensor.
        """
        # The `decode` logic is removed as it's not idiomatic in PyTorch.
        # The decoder will handle passing the correct positions.
        return self.embedding[inputs, :]


# Referenced https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/t5/modeling_t5.py#L238
class LayerNorm(nn.Module):
    """Layer normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Initializes the LayerNorm layer.

        Args:
            hidden_size: The hidden size.
            eps: The epsilon value for numerical stability.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the layer normalization.

        Args:
            hidden_states: The input tensor.

        Returns:
            The output tensor.
        """
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

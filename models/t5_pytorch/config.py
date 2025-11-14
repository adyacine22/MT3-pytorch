"""Configuration helpers for the PyTorch MT3 model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from configs.project_config import PROJECT_CONFIG


@dataclass(frozen=True)
class T5Config:
    """PyTorch-friendly replica of the legacy T5Config."""

    vocab_size: int
    emb_dim: int
    num_heads: int
    num_encoder_layers: int
    num_decoder_layers: int
    head_dim: int
    mlp_dim: int
    mlp_activations: Sequence[str]
    dropout_rate: float
    logits_via_embedding: bool
    max_position_embeddings: int
    input_depth: int

    @property
    def hidden_size(self) -> int:
        return self.emb_dim

    @property
    def head_dim_total(self) -> int:
        return self.num_heads * self.head_dim


def load_t5_config(
    *,
    vocab_size: int | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> T5Config:
    """Load a T5Config using PROJECT_CONFIG defaults."""

    base = dict(PROJECT_CONFIG["model"]["t5_pytorch"])
    if overrides:
        base.update(overrides)
    if vocab_size is not None:
        base["vocab_size"] = vocab_size
    if base.get("vocab_size", 0) <= 0:
        raise ValueError("vocab_size must be provided via config or argument.")

    return T5Config(
        vocab_size=int(base["vocab_size"]),
        emb_dim=int(base["emb_dim"]),
        num_heads=int(base["num_heads"]),
        num_encoder_layers=int(base["num_encoder_layers"]),
        num_decoder_layers=int(base["num_decoder_layers"]),
        head_dim=int(base["head_dim"]),
        mlp_dim=int(base["mlp_dim"]),
        mlp_activations=tuple(base["mlp_activations"]),
        dropout_rate=float(base["dropout_rate"]),
        logits_via_embedding=bool(base["logits_via_embedding"]),
        max_position_embeddings=int(base["max_position_embeddings"]),
        input_depth=int(base["input_depth"]),
    )


def with_overrides(cfg: T5Config, **kw: Any) -> T5Config:
    """Return a copy of cfg with selected fields replaced."""

    return replace(cfg, **kw)

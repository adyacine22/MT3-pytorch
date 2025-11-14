"""Helper to build DataLoaders for chunk datasets."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Iterable

import torch
from torch.utils.data import DataLoader

from configs import load_project_config
from data.datasets.chunk_dataset import ChunkDataset, ChunkDatasetConfig
from data.datasets.collate import chunk_collate_fn
from data.datasets.samplers import TemperatureSampler
from data.datasets.tokenizer import OnTheFlyTokenizer
from data.audio.mixing import AudioMixer

TRAINING_CFG = load_project_config().get("training", {}).get("dataloader", {})


def build_chunk_dataloader(
    manifest_path: str | Path,
    *,
    batch_size: int | None = None,
    feature_type: str | None = None,
    load_tokens: bool | None = None,
    max_examples_per_mix: int | None = None,
    temperature: float | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    compute_log_mel_in_collate: bool | None = None,
    collate_device: str | torch.device | None = None,
    seed: int | None = None,
    shard_cache_size: int | None = None,
    prefetch_factor: int | None = None,
    split: str | Iterable[str] | None = None,
    max_midi_cache_size: int | None = None,
) -> DataLoader:
    """Create a DataLoader that yields ready-to-train MT3 batches."""
    batch_size = batch_size or int(TRAINING_CFG.get("batch_size", 32))
    feature_type = feature_type or TRAINING_CFG.get("feature_type", "waveform")
    load_tokens = load_tokens if load_tokens is not None else bool(TRAINING_CFG.get("load_tokens", True))
    max_examples_per_mix = max_examples_per_mix or int(TRAINING_CFG.get("max_examples_per_mix", 1))
    if temperature is None:
        temp_cfg = TRAINING_CFG.get("temperature")
        temperature = float(temp_cfg) if temp_cfg is not None else None
    num_workers = num_workers if num_workers is not None else int(TRAINING_CFG.get("num_workers", 0))
    pin_memory = pin_memory if pin_memory is not None else bool(TRAINING_CFG.get("pin_memory", True))
    compute_log_mel_in_collate = (
        compute_log_mel_in_collate
        if compute_log_mel_in_collate is not None
        else bool(TRAINING_CFG.get("compute_log_mel_in_collate", True))
    )
    shard_cache_size = shard_cache_size or int(TRAINING_CFG.get("shard_cache_size", 8))
    prefetch_factor = prefetch_factor or int(TRAINING_CFG.get("prefetch_factor", 2))

    midi_cache_size = (
        max_midi_cache_size
        if max_midi_cache_size is not None
        else TRAINING_CFG.get("max_midi_cache_size")
    )

    config = ChunkDatasetConfig(
        feature_type=feature_type,
        load_tokens=load_tokens,
        shard_cache_size=shard_cache_size,
        max_examples_per_mix=max_examples_per_mix,
    )
    dataset = ChunkDataset(
        manifest_path,
        config=config,
        tokenizer=OnTheFlyTokenizer(max_cache_size=midi_cache_size),
        audio_mixer=AudioMixer() if max_examples_per_mix > 1 else None,
        split_filter=split,
    )
    sampler = (
        TemperatureSampler(dataset, temperature=temperature, seed=seed)
        if temperature is not None
        else None
    )
    collate = partial(
        chunk_collate_fn,
        compute_log_mel=compute_log_mel_in_collate and feature_type == "waveform",
        log_mel_on_device=collate_device,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    return dataloader

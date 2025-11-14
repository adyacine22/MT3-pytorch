"""Manifest helpers and schema definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

MANIFEST_REQUIRED_COLUMNS = [
    "chunk_id",
    "track_id",
    "dataset",
    "split",
    "audio_path",
    "midi_path",
    "chunk_start_ms",
    "chunk_end_ms",
    "chunk_duration_ms",
    "chunk_frames",
    "chunk_samples",
    "sample_rate",
    "hop_length",
    "chunk_storage",
    "chunk_shard_path",
    "chunk_shard_index",
]


def build_manifest_row(
    chunk_meta: Dict[str, Any],
    chunk_path: Path,
    tokens: Sequence[int],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    chunk_path_str = chunk_path.resolve().as_posix()
    track_id = chunk_meta.get("track_id") or chunk_meta.get("track_slug")
    chunk_index = int(chunk_meta["chunk_index"])
    chunk_id = f"{track_id}-chunk{chunk_index:05d}" if track_id else f"chunk{chunk_index:05d}"
    chunk_start_ms = float(chunk_meta["chunk_start_s"]) * 1000.0
    chunk_end_ms = float(chunk_meta["chunk_end_s"]) * 1000.0
    chunk_duration_ms = max(chunk_end_ms - chunk_start_ms, 0.0)
    augment_profiles = chunk_meta.get("augment_profiles") or ["none"]
    primary_augment = augment_profiles[0] if augment_profiles else "none"
    rendered_stems = chunk_meta.get("rendered_stem_ids") or []
    stem_id = chunk_meta.get("stem_id")
    stem_ids = rendered_stems or ([stem_id] if stem_id else [])
    metadata_hash = f"{track_id}-{int(round(chunk_start_ms))}-{int(round(chunk_end_ms))}"
    shard_path_obj = chunk_meta.get("chunk_shard_path")
    shard_path = (
        shard_path_obj.resolve().as_posix()
        if isinstance(shard_path_obj, Path)
        else str(shard_path_obj) if shard_path_obj else chunk_path_str
    )
    shard_index = int(chunk_meta.get("chunk_shard_index", chunk_index))
    chunk_storage = chunk_meta.get("chunk_storage", "per_chunk")
    return {
        "chunk_id": chunk_id,
        "track_id": track_id,
        "dataset": chunk_meta["dataset"],
        "split": chunk_meta.get("split"),
        "audio_path": chunk_meta.get("audio_path"),
        "midi_path": chunk_meta.get("midi_path"),
        "chunk_index": chunk_index,
        "chunk_start_s": chunk_meta["chunk_start_s"],
        "chunk_end_s": chunk_meta["chunk_end_s"],
        "chunk_start_ms": chunk_start_ms,
        "chunk_end_ms": chunk_end_ms,
        "chunk_duration_ms": chunk_duration_ms,
        "chunk_frames": int(chunk_meta["frames"]),
        "chunk_samples": int(options["chunk_samples"]),
        "sample_rate": int(options["sample_rate"]),
        "hop_length": int(options["hop_length"]),
        "tokens": int(len(tokens)),
        "chunk_path": chunk_path_str,
        "precomputed_path": chunk_path_str,
        "chunk_storage": chunk_storage,
        "chunk_shard_path": shard_path,
        "chunk_shard_index": shard_index,
        "spectrogram_device": chunk_meta["spectrogram_device"],
        "augment_profiles": augment_profiles,
        "augmentation_profile": primary_augment,
        "batch_size": options["batch_size"],
        "tokenize_workers_used": chunk_meta["tokenize_workers"],
        "max_tokenize_workers": options["max_tokenize_workers"],
        "store_tokens": chunk_meta["tokens_included"],
        "instrument_programs": chunk_meta.get("instrument_programs", []),
        "instrument_names": chunk_meta.get("instrument_names", []),
        "instrument_classes": chunk_meta.get("instrument_classes", []),
        "stem_ids": stem_ids,
        "is_drum": bool(chunk_meta.get("is_drum", False)),
        "mix_group_id": chunk_meta.get("mix_group_id"),
        "metadata_hash": metadata_hash,
    }


__all__ = ["MANIFEST_REQUIRED_COLUMNS", "build_manifest_row"]

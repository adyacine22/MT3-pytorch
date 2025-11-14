"""Chunk planning and storage helpers used by preprocessing pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch


def plan_chunk_positions(
    total_samples: int, sample_rate: int, chunk_samples: int
) -> List[Dict[str, float]]:
    """Return a list of start positions ensuring we cover the full audio."""
    if total_samples <= 0 or chunk_samples <= 0:
        return [{"start_sample": 0, "start_s": 0.0}]
    positions: List[Dict[str, float]] = []
    start = 0
    while start <= total_samples - chunk_samples:
        positions.append({"start_sample": start, "start_s": start / sample_rate})
        start += chunk_samples
    if not positions or positions[-1]["start_sample"] < total_samples - chunk_samples:
        last_start = max(0, total_samples - chunk_samples)
        positions.append({"start_sample": last_start, "start_s": last_start / sample_rate})
    return positions


def batched(plans: List[Dict[str, float]], batch_size: int) -> Iterable[List[Dict[str, float]]]:
    if batch_size <= 1:
        for plan in plans:
            yield [plan]
        return
    for idx in range(0, len(plans), batch_size):
        yield plans[idx : idx + batch_size]


def slugify(text: str | None, fallback: str) -> str:
    if not text:
        text = fallback
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in text)
    return safe.strip("_") or fallback


def build_chunk_payload(
    log_mel: torch.Tensor | None,
    waveform: torch.Tensor | None,
    frame_times: Sequence[float],
    tokens: Sequence[int],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "frame_times": torch.tensor(frame_times, dtype=torch.float32),
        "chunk_start_s": float(meta["chunk_start_s"]),
        "chunk_end_s": float(meta["chunk_end_s"]),
        "tokens": torch.tensor(tokens, dtype=torch.int32),
        "spectrogram_device": meta["spectrogram_device"],
        "dataset": meta["dataset"],
        "track_id": meta["track_id"],
        "audio_path": meta["audio_path"],
        "midi_path": meta["midi_path"],
        "augment_profiles": meta["augment_profiles"],
    }
    if log_mel is not None:
        payload["log_mel"] = log_mel.cpu()
    if waveform is not None:
        payload["waveform"] = waveform.detach().cpu().to(dtype=torch.float16)
    return payload


def _write_chunk_file(
    payload: Dict[str, Any],
    chunk_path: Path,
    dry_run: bool,
    overwrite: bool,
) -> Path:
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    if chunk_path.exists() and not overwrite:
        return chunk_path
    if dry_run:
        return chunk_path
    torch.save(payload, chunk_path)
    return chunk_path


@dataclass
class ChunkShardWriter:
    """Handle chunk persistence, supporting per-chunk or per-track sharding."""

    output_root: Path
    dataset: str
    track_slug: str
    track_id: str
    chunk_storage: str
    dry_run: bool
    overwrite: bool
    dataset_dir: Path = field(init=False)
    _buffer: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _track_dir: Path | None = field(default=None, init=False)
    _shard_path: Path | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.strategy = self.chunk_storage if self.chunk_storage in {"per_chunk", "per_track"} else "per_chunk"
        self.dataset_dir = self.output_root / self.dataset
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        if self.strategy == "per_chunk":
            self._track_dir = self.dataset_dir / self.track_slug
            self._track_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._shard_path = self.dataset_dir / f"{self.track_slug}.pt"

    @property
    def shard_path(self) -> Path | None:
        return self._shard_path

    def should_skip_track(self, skip_existing: bool, write_manifest: bool) -> bool:
        if self.strategy != "per_track":
            return False
        if not skip_existing or write_manifest or self.overwrite:
            return False
        return self._shard_path is not None and self._shard_path.exists()

    def chunk_exists(self, chunk_slug: str) -> bool:
        path = self._chunk_path(chunk_slug)
        return path.exists()

    def write_chunk(self, chunk_slug: str, payload: Dict[str, Any]) -> Path:
        if self.strategy == "per_chunk":
            chunk_path = self._chunk_path(chunk_slug)
            return _write_chunk_file(payload, chunk_path, self.dry_run, self.overwrite)
        payload_with_slug = dict(payload)
        payload_with_slug["chunk_slug"] = chunk_slug
        self._buffer.append(payload_with_slug)
        if not self._shard_path:
            raise RuntimeError("Shard path missing for per-track storage.")
        return self._shard_path

    def finalize(self) -> None:
        if self.strategy != "per_track" or not self._buffer:
            self._buffer.clear()
            return
        if self.dry_run:
            self._buffer.clear()
            return
        if self._shard_path is None:
            raise RuntimeError("Shard path missing for per-track storage.")
        if self._shard_path.exists() and not self.overwrite:
            self._buffer.clear()
            return
        payload = {
            "track_id": self.track_id,
            "dataset": self.dataset,
            "chunks": self._buffer,
            "chunk_lookup": {entry["chunk_slug"]: idx for idx, entry in enumerate(self._buffer)},
        }
        torch.save(payload, self._shard_path)
        self._buffer.clear()

    def _chunk_path(self, chunk_slug: str) -> Path:
        if self.strategy == "per_track":
            if self._shard_path is None:
                raise RuntimeError("Shard path missing for per-track storage.")
            return self._shard_path
        if self._track_dir is None:
            raise RuntimeError("Track directory missing for per-chunk storage.")
        return self._track_dir / f"{chunk_slug}.pt"

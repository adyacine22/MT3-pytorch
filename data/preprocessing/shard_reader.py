"""Utilities for reading chunk shards emitted during preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import torch


class ChunkShard:
    """In-memory view over a per-track shard file."""

    def __init__(self, shard_path: Path | str, *, map_location: str | torch.device = "cpu") -> None:
        self.path = Path(shard_path)
        self.payload: Dict[str, Any] = torch.load(self.path, map_location=map_location)
        chunks = self.payload.get("chunks")
        if not isinstance(chunks, Sequence):
            raise ValueError(f"Shard file {self.path} missing 'chunks' list")
        self._chunks: Sequence[Dict[str, Any]] = chunks
        lookup: MutableMapping[str, int] = dict(self.payload.get("chunk_lookup") or {})
        if not lookup:
            lookup.update(
                {
                    entry.get("chunk_slug", f"chunk_{idx:05d}"): idx
                    for idx, entry in enumerate(self._chunks)
                }
            )
        self._lookup = lookup

    def __len__(self) -> int:
        return len(self._chunks)

    def chunk_slugs(self) -> Sequence[str]:
        return list(self._lookup.keys())

    def get_by_index(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(self._chunks):
            raise IndexError(f"Chunk index {index} out of range for shard {self.path}")
        return self._chunks[index]

    def get_by_slug(self, slug: str) -> Dict[str, Any]:
        if slug not in self._lookup:
            raise KeyError(f"Chunk slug '{slug}' missing from shard {self.path}")
        return self.get_by_index(self._lookup[slug])


def _infer_chunk_slug(manifest_row: Mapping[str, Any]) -> str:
    if "chunk_slug" in manifest_row:
        return str(manifest_row["chunk_slug"])
    chunk_index = int(manifest_row.get("chunk_index", 0))
    return f"chunk_{chunk_index:05d}"


def load_chunk_from_manifest_row(
    manifest_row: Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Load a chunk payload given a manifest row describing its storage."""
    storage = str(manifest_row.get("chunk_storage", "per_chunk")).lower()
    shard_path = Path(manifest_row["chunk_shard_path"])
    if storage == "per_track":
        slug = _infer_chunk_slug(manifest_row)
        shard = ChunkShard(shard_path, map_location=map_location)
        return shard.get_by_slug(slug)
    chunk_path = Path(manifest_row.get("chunk_path") or shard_path)
    return torch.load(chunk_path, map_location=map_location)


__all__ = ["ChunkShard", "load_chunk_from_manifest_row"]

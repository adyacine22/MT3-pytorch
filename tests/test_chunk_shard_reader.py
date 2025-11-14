from __future__ import annotations

from pathlib import Path

import torch

from data.preprocessing.shard_reader import ChunkShard, load_chunk_from_manifest_row


def _make_payload(slug: str) -> dict:
    return {
        "chunk_slug": slug,
        "tokens": torch.tensor([1, 2, 3]),
        "dataset": "testset",
    }


def test_chunk_shard_reader_builds_lookup(tmp_path: Path):
    shard_path = tmp_path / "test.pt"
    torch.save(
        {
            "track_id": "track",
            "dataset": "testset",
            "chunks": [_make_payload("chunk_00000")],
        },
        shard_path,
    )
    shard = ChunkShard(shard_path)
    assert len(shard) == 1
    payload = shard.get_by_slug("chunk_00000")
    assert torch.equal(payload["tokens"], torch.tensor([1, 2, 3]))


def test_load_chunk_from_manifest_row_handles_per_track(tmp_path: Path):
    shard_path = tmp_path / "track.pt"
    torch.save(
        {
            "track_id": "track",
            "dataset": "testset",
            "chunks": [_make_payload("chunk_00000")],
        },
        shard_path,
    )
    manifest_row = {
        "chunk_storage": "per_track",
        "chunk_shard_path": str(shard_path),
        "chunk_index": 0,
        "chunk_id": "track-chunk00000",
    }
    payload = load_chunk_from_manifest_row(manifest_row)
    assert payload["dataset"] == "testset"


def test_load_chunk_from_manifest_row_handles_per_chunk(tmp_path: Path):
    chunk_path = tmp_path / "chunk_00000.pt"
    torch.save({"tokens": [1, 2]}, chunk_path)
    manifest_row = {
        "chunk_storage": "per_chunk",
        "chunk_shard_path": str(chunk_path),
        "chunk_path": str(chunk_path),
        "chunk_index": 0,
    }
    payload = load_chunk_from_manifest_row(manifest_row)
    assert payload["tokens"] == [1, 2]

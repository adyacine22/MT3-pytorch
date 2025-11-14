from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import torch

from data.preprocessing.storage import (
    ChunkShardWriter,
    build_chunk_payload,
    plan_chunk_positions,
)


def _make_meta() -> Dict[str, Any]:
    return {
        "chunk_index": 0,
        "chunk_start_s": 0.0,
        "chunk_end_s": 2.0,
        "frames": 10,
        "spectrogram_device": "cpu",
        "dataset": "testset",
        "track_id": "track",
        "track_slug": "track",
        "split": "train",
        "audio_path": "audio.wav",
        "midi_path": "midi.mid",
        "augment_profiles": ["none"],
        "tokenize_workers": 0,
        "max_tokenize_workers": 0,
        "tokens_included": False,
        "chunk_storage": "per_track",
        "chunk_shard_index": 0,
    }


def test_plan_chunk_positions_covers_tail():
    plans = plan_chunk_positions(total_samples=1000, sample_rate=100, chunk_samples=256)
    assert plans[0]["start_sample"] == 0
    # ensure last chunk backfills remaining samples
    assert plans[-1]["start_sample"] == 1000 - 256


def test_build_chunk_payload_preserves_meta():
    meta = _make_meta()
    payload = build_chunk_payload(
        log_mel=None,
        waveform=torch.zeros(1, 256),
        frame_times=[0.0, 0.1],
        tokens=[1, 2, 3],
        meta=meta,
    )
    assert payload["tokens"].shape[0] == 3
    assert payload["dataset"] == "testset"
    assert "waveform" in payload


def test_chunk_shard_writer_per_track(tmp_path: Path):
    writer = ChunkShardWriter(
        output_root=tmp_path,
        dataset="testset",
        track_slug="track",
        track_id="track",
        chunk_storage="per_track",
        dry_run=False,
        overwrite=True,
    )
    meta = _make_meta()
    payload = build_chunk_payload(
        log_mel=None,
        waveform=torch.zeros(1, 10),
        frame_times=[0.0],
        tokens=[],
        meta=meta,
    )
    shard_path = writer.write_chunk("chunk_00000", payload)
    assert shard_path.name.endswith(".pt")
    writer.finalize()
    data = torch.load(shard_path)
    assert data["track_id"] == "track"
    assert data["chunks"][0]["chunk_slug"] == "chunk_00000"


def test_chunk_shard_writer_per_chunk(tmp_path: Path):
    writer = ChunkShardWriter(
        output_root=tmp_path,
        dataset="testset",
        track_slug="track",
        track_id="track",
        chunk_storage="per_chunk",
        dry_run=False,
        overwrite=True,
    )
    meta = _make_meta()
    payload = build_chunk_payload(
        log_mel=None,
        waveform=torch.zeros(1, 10),
        frame_times=[0.0],
        tokens=[],
        meta=meta,
    )
    chunk_path = writer.write_chunk("chunk_00000", payload)
    assert chunk_path.exists()
    writer.finalize()

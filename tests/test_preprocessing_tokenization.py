from __future__ import annotations

import note_seq  # type: ignore

from data.preprocessing.tokenization import (
    estimate_work_units,
    select_tokenize_workers,
    tokenize_chunk,
)


def _note_sequence(num_notes: int, duration: float) -> note_seq.NoteSequence:
    ns = note_seq.NoteSequence()
    for idx in range(num_notes):
        note = ns.notes.add()
        note.pitch = 60 + idx
        note.start_time = idx * (duration / max(1, num_notes))
        note.end_time = note.start_time + 0.1
    ns.total_time = duration
    return ns


def test_estimate_work_units_scales_with_density():
    dense = _note_sequence(100, 1.0)
    sparse = _note_sequence(10, 10.0)
    assert estimate_work_units(dense, chunk_count=5) > estimate_work_units(sparse, chunk_count=5)


def test_select_tokenize_workers_caps_at_max():
    ns = _note_sequence(500, 2.0)
    workers = select_tokenize_workers(ns, chunk_count=100, max_tokenize_workers=8)
    assert 1 <= workers <= 8


def test_tokenize_chunk_returns_tokens():
    ns = _note_sequence(4, 2.0)
    payload = {
        "chunk_index": 0,
        "chunk_start_s": 0.0,
        "chunk_end_s": 2.0,
        "frame_times": [0.0, 0.1, 0.2],
        "note_sequence_bytes": ns.SerializeToString(),
    }
    result = tokenize_chunk(payload)
    assert "tokens" in result
    assert isinstance(result["tokens"], (list, tuple))
    assert result["tokenize_time"] >= 0.0

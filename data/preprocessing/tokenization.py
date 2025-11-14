"""Tokenization workload helpers for preprocessing."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence, Tuple

import note_seq

from data.symbolic import tokenizer as tokenizer_module


TOKENIZE_WORKER_THRESHOLDS: List[Tuple[float, int]] = [
    (500.0, 1),
    (1500.0, 2),
    (2500.0, 4),
    (3500.0, 8),
    (5000.0, 16),
    (7000.0, 24),
    (9000.0, 32),
]


def estimate_work_units(note_sequence: note_seq.NoteSequence, chunk_count: int) -> float:
    duration = max(float(note_sequence.total_time), 1e-6)
    density = len(note_sequence.notes) / duration if duration > 0 else 0.0
    return density * max(1, chunk_count)


def select_tokenize_workers(
    note_sequence: note_seq.NoteSequence,
    chunk_count: int,
    max_tokenize_workers: int,
) -> int:
    if max_tokenize_workers <= 0:
        return 0
    work_units = estimate_work_units(note_sequence, chunk_count)
    for threshold, worker_target in TOKENIZE_WORKER_THRESHOLDS:
        if work_units < threshold:
            return min(worker_target, max_tokenize_workers)
    return min(TOKENIZE_WORKER_THRESHOLDS[-1][1], max_tokenize_workers)


def tokenize_chunk(payload: Dict[str, Any]) -> Dict[str, Any]:
    ns = note_seq.NoteSequence()
    ns.ParseFromString(payload["note_sequence_bytes"])
    start = time.perf_counter()
    output = tokenizer_module.tokenize_note_sequence(
        ns,
        chunk_start_s=payload["chunk_start_s"],
        chunk_end_s=payload["chunk_end_s"],
        frame_times=payload["frame_times"],
    )
    duration = time.perf_counter() - start
    return {
        "chunk_index": payload["chunk_index"],
        "tokens": output.tokens,
        "tokenize_time": duration,
    }


__all__ = [
    "estimate_work_units",
    "select_tokenize_workers",
    "tokenize_chunk",
    "TOKENIZE_WORKER_THRESHOLDS",
]

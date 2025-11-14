"""On-the-fly tokenization helpers used by ChunkDataset."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping

import note_seq
import torch

from data.symbolic import tokenizer as tokenizer_module


@dataclass
class TokenizerConfig:
    """Runtime tokenizer settings."""

    dtype: torch.dtype = torch.int32


def _combine_segments(segments: Iterable[note_seq.NoteSequence]) -> note_seq.NoteSequence:
    combined = note_seq.NoteSequence()
    for segment in segments:
        notes = combined.notes
        for note in segment.notes:
            new_note = notes.add()
            new_note.CopyFrom(note)
    return combined


def tokenize_note_sequence_segments(
    segments: Iterable[note_seq.NoteSequence],
    chunk_duration_s: float,
    frame_times: Iterable[float],
    *,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Tokenize one or more chunk-aligned segments into MT3 tokens."""
    combined = _combine_segments(segments)
    output = tokenizer_module.tokenize_note_sequence(
        combined,
        chunk_start_s=0.0,
        chunk_end_s=float(chunk_duration_s),
        frame_times=list(frame_times),
    )
    return torch.as_tensor(output.tokens, dtype=dtype)


class OnTheFlyTokenizer:
    """Cache-aware tokenization utility for datasets."""

    def __init__(
        self,
        config: TokenizerConfig | None = None,
        max_cache_size: int | None = None,
    ) -> None:
        self.config = config or TokenizerConfig()
        self.max_cache_size = max_cache_size
        self._ns_cache: MutableMapping[str, note_seq.NoteSequence] = (
            OrderedDict() if max_cache_size else {}
        )

    def tokens_for(
        self,
        midi_path: str | Path,
        chunk_start_s: float,
        chunk_end_s: float,
        frame_times: Iterable[float],
    ) -> torch.Tensor:
        segment = self.segment_sequence(
            midi_path=midi_path, chunk_start_s=chunk_start_s, chunk_end_s=chunk_end_s
        )
        duration = float(chunk_end_s) - float(chunk_start_s)
        return tokenize_note_sequence_segments(
            [segment],
            chunk_duration_s=duration,
            frame_times=frame_times,
            dtype=self.config.dtype,
        )

    def tokens_for_segments(
        self,
        segments: Iterable[note_seq.NoteSequence],
        chunk_duration_s: float,
        frame_times: Iterable[float],
    ) -> torch.Tensor:
        return tokenize_note_sequence_segments(
            segments,
            chunk_duration_s=chunk_duration_s,
            frame_times=frame_times,
            dtype=self.config.dtype,
        )

    def segment_sequence(
        self,
        midi_path: str | Path,
        chunk_start_s: float,
        chunk_end_s: float,
    ) -> note_seq.NoteSequence:
        """Return a NoteSequence containing only notes within the chunk window."""
        ns = self._load_note_sequence(str(midi_path))
        segment = note_seq.NoteSequence()
        window_start = float(chunk_start_s)
        window_end = float(chunk_end_s)
        for note in ns.notes:
            if note.end_time <= window_start or note.start_time >= window_end:
                continue
            new_note = segment.notes.add()
            new_note.pitch = note.pitch
            new_note.velocity = note.velocity
            new_note.program = note.program
            new_note.is_drum = note.is_drum
            new_note.start_time = max(note.start_time, window_start) - window_start
            new_note.end_time = min(note.end_time, window_end) - window_start
        segment.total_time = max((n.end_time for n in segment.notes), default=0.0)
        return segment

    def _load_note_sequence(self, midi_path: str) -> note_seq.NoteSequence:
        cache = self._ns_cache
        cached = cache.get(midi_path)
        if cached is not None:
            if isinstance(cache, OrderedDict):
                cache.move_to_end(midi_path)
            return cached
        ns = note_seq.midi_file_to_note_sequence(midi_path)
        if isinstance(cache, OrderedDict):
            cache[midi_path] = ns
            if self.max_cache_size and len(cache) > self.max_cache_size:
                cache.popitem(last=False)
        else:
            cache[midi_path] = ns
        return ns

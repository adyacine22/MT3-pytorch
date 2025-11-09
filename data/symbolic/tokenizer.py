"""Chunk-aware symbolic tokenizer."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Sequence, Tuple

import note_seq  # type: ignore
import numpy as np

from configs import load_project_config
from data.symbolic import vocabulary

CONFIG = load_project_config()
SYMBOLIC_CFG = CONFIG["symbolic"]
AUDIO_CFG = CONFIG["audio"]
FEATURE_CFG = AUDIO_CFG["features"]
IO_CFG = AUDIO_CFG["io"]
CODEC_CFG = SYMBOLIC_CFG["codec"]
TOKENIZER_CFG = SYMBOLIC_CFG.get("tokenizer", {})
STEPS_PER_SECOND = CODEC_CFG["steps_per_second"]
NUM_VELOCITY_BINS = max(1, CODEC_CFG["num_velocity_bins"])


@dataclass
class NoteEventData:
    """Lightweight representation of an onset/offset event."""

    pitch: int
    velocity: int | None = None
    program: int | None = None
    is_drum: bool = False


@dataclass
class NoteEncodingState:
    """Tracks active notes for tie handling."""

    active_pitches: Dict[Tuple[int, int], int] = field(default_factory=dict)


@dataclass
class TokenizerOutput:
    tokens: List[int]
    event_start_indices: List[int]
    event_end_indices: List[int]
    state_events: List[int]
    state_event_indices: List[int]

    def to_dict(self) -> Dict[str, List[int]]:
        return {
            "tokens": self.tokens,
            "event_start_indices": self.event_start_indices,
            "event_end_indices": self.event_end_indices,
            "state_events": self.state_events,
            "state_event_indices": self.state_event_indices,
        }


def tokenize_note_sequence(
    note_sequence: note_seq.NoteSequence,
    *,
    chunk_start_s: float,
    chunk_end_s: float,
    frame_times: Sequence[float] | None = None,
    num_frames: int | None = None,
    sample_rate: int | None = None,
    hop_length: int | None = None,
    include_ties: bool | None = None,
    onsets_only: bool | None = None,
) -> TokenizerOutput:
    """Tokenize a NoteSequence segment into MT3-style events.

    Args:
        note_sequence: Full track NoteSequence.
        chunk_start_s: Start time of chunk in seconds.
        chunk_end_s: End time of chunk in seconds.
        frame_times: Optional precomputed audio frame times (seconds).
        num_frames: Number of audio frames if ``frame_times`` not provided.
        sample_rate: Audio sample rate for computing frame times.
        hop_length: STFT hop length for computing frame times.
        include_ties: Override config for tie handling.
        onsets_only: Override config to emit only onset events.

    Returns:
        TokenizerOutput with tokens, alignment indices, and state events.
    """

    if chunk_end_s <= chunk_start_s:
        raise ValueError("chunk_end_s must be greater than chunk_start_s")

    include_ties_flag: bool = (
        TOKENIZER_CFG.get("include_ties", True) if include_ties is None else include_ties
    )
    onsets_only_flag: bool = (
        TOKENIZER_CFG.get("onsets_only", False) if onsets_only is None else onsets_only
    )

    sr = sample_rate or IO_CFG["sample_rate"]
    hop = hop_length or FEATURE_CFG["hop_length"]
    frame_times = list(frame_times or _default_frame_times(num_frames, hop, sr))
    if not frame_times:
        raise ValueError("frame_times cannot be empty")
    if any(frame_times[i] > frame_times[i + 1] for i in range(len(frame_times) - 1)):
        raise ValueError("frame_times must be non-decreasing")

    working_sequence = _prepare_sequence(note_sequence)
    state = _initial_state(
        working_sequence, chunk_start_s, include_ties=include_ties_flag
    )

    event_times, event_values = _collect_note_events(
        working_sequence,
        chunk_start_s=chunk_start_s,
        chunk_end_s=chunk_end_s,
        onsets_only=onsets_only_flag,
    )

    (
        tokens,
        event_start_indices,
        event_end_indices,
        state_events,
        state_event_indices,
    ) = _encode_and_index_events(
        state=state if include_ties_flag else None,
        event_times=event_times,
        event_values=event_values,
        frame_times=frame_times,
    )

    return TokenizerOutput(
        tokens=tokens,
        event_start_indices=event_start_indices,
        event_end_indices=event_end_indices,
        state_events=state_events,
        state_event_indices=state_event_indices,
    )


def _default_frame_times(
    num_frames: int | None,
    hop_length: int,
    sample_rate: int,
) -> List[float]:
    frames = num_frames or FEATURE_CFG["chunk_frames"]
    if frames <= 0:
        raise ValueError("num_frames must be positive")
    hop_seconds = hop_length / sample_rate
    return [i * hop_seconds for i in range(frames)]


def _prepare_sequence(ns: note_seq.NoteSequence) -> note_seq.NoteSequence:
    """Return a copy of the NoteSequence with optional processing."""
    clone = note_seq.NoteSequence()
    clone.CopyFrom(ns)
    if TOKENIZER_CFG.get("apply_sustain", True):
        clone = note_seq.apply_sustain_control_changes(clone)
    if TOKENIZER_CFG.get("trim_overlapping_notes", True):
        clone = _trim_overlapping_notes(clone)
    return clone


def _initial_state(
    ns: note_seq.NoteSequence,
    chunk_start_s: float,
    include_ties: bool,
) -> NoteEncodingState | None:
    if not include_ties:
        return None
    state = NoteEncodingState()
    for note in ns.notes:
        if note.is_drum or note.velocity == 0:
            continue
        if note.start_time < chunk_start_s and note.end_time > chunk_start_s:
            program = int(note.program) if note.program is not None else 0
            state.active_pitches[(note.pitch, program)] = _velocity_to_bin(note.velocity)
    return state


def _collect_note_events(
    ns: note_seq.NoteSequence,
    chunk_start_s: float,
    chunk_end_s: float,
    onsets_only: bool,
) -> Tuple[List[float], List[NoteEventData]]:
    events: List[Tuple[float, int, NoteEventData]] = []
    for note in ns.notes:
        if note.end_time <= chunk_start_s or note.start_time >= chunk_end_s:
            continue
        program = int(note.program) if note.program is not None else 0
        start_in_chunk = note.start_time >= chunk_start_s
        end_in_chunk = note.end_time <= chunk_end_s

        if start_in_chunk:
            rel_start = note.start_time - chunk_start_s
            events.append(
                (
                    rel_start,
                    1,
                    NoteEventData(
                        pitch=note.pitch,
                        velocity=None if onsets_only else note.velocity,
                        program=None if onsets_only else program,
                        is_drum=bool(note.is_drum),
                    ),
                )
            )

        if not onsets_only and end_in_chunk and not note.is_drum:
            rel_end = note.end_time - chunk_start_s
            events.append(
                (
                    rel_end,
                    0,
                    NoteEventData(
                        pitch=note.pitch,
                        velocity=0,
                        program=program,
                        is_drum=False,
                    ),
                )
            )

    if not events:
        return [], []

    # Sort by time, offsets before onsets, then program/pitch for determinism.
    events.sort(
        key=lambda item: (
            item[0],
            item[1],
            int(item[2].is_drum),
            -1 if item[2].program is None else item[2].program,
            item[2].pitch,
        )
    )
    times = [t for t, _, _ in events]
    values = [value for _, _, value in events]
    return times, values


def _encode_and_index_events(
    *,
    state: NoteEncodingState | None,
    event_times: Sequence[float],
    event_values: Sequence[NoteEventData],
    frame_times: Sequence[float],
) -> Tuple[List[int], List[int], List[int], List[int], List[int]]:
    indices = np.argsort(event_times, kind="stable")
    event_steps = [round(event_times[i] * STEPS_PER_SECOND) for i in indices]
    event_values = [event_values[i] for i in indices]

    events: List[int] = []
    state_events: List[int] = []
    event_start_indices: List[int] = []
    state_event_indices: List[int] = []

    cur_step = 0
    cur_event_idx = 0
    cur_state_event_idx = 0

    def fill_indices() -> None:
        nonlocal event_start_indices, state_event_indices
        while (
            len(event_start_indices) < len(frame_times)
            and frame_times[len(event_start_indices)] < cur_step / STEPS_PER_SECOND
        ):
            event_start_indices.append(cur_event_idx)
            state_event_indices.append(cur_state_event_idx)

    for event_step, value in zip(event_steps, event_values):
        while event_step > cur_step:
            events.append(vocabulary.shift_id(1))
            cur_step += 1
            fill_indices()
            cur_event_idx = len(events)
            cur_state_event_idx = len(state_events)

        if state is not None:
            tie_events = _state_to_tokens(state)
            if tie_events:
                state_events.extend(tie_events)
                cur_state_event_idx = len(state_events)

        encoded = _encode_note_event(state, value)
        events.extend(encoded)

    if frame_times:
        while cur_step / STEPS_PER_SECOND <= frame_times[-1]:
            events.append(vocabulary.shift_id(1))
            cur_step += 1
            fill_indices()
            cur_event_idx = len(events)

    # Ensure indices cover all frames.
    while len(event_start_indices) < len(frame_times):
        event_start_indices.append(cur_event_idx)
        state_event_indices.append(cur_state_event_idx)

    event_end_indices = event_start_indices[1:] + [len(events)]
    return events, event_start_indices, event_end_indices, state_events, state_event_indices


def _encode_note_event(
    state: NoteEncodingState | None,
    value: NoteEventData,
) -> List[int]:
    if value.velocity is None:
        return [vocabulary.pitch_id(value.pitch)]

    velocity_bin = _velocity_to_bin(value.velocity)
    if value.program is None:
        if state is not None:
            state.active_pitches[(value.pitch, 0)] = velocity_bin
        return [vocabulary.velocity_id(velocity_bin), vocabulary.pitch_id(value.pitch)]

    if value.is_drum:
        return [vocabulary.velocity_id(velocity_bin), vocabulary.drum_id(value.pitch)]

    program = int(value.program)
    if state is not None:
        state.active_pitches[(value.pitch, program)] = velocity_bin
    return [
        vocabulary.program_id(program),
        vocabulary.velocity_id(velocity_bin),
        vocabulary.pitch_id(value.pitch),
    ]


def _state_to_tokens(state: NoteEncodingState) -> List[int]:
    tokens: List[int] = []
    for pitch, program in sorted(state.active_pitches.keys(), key=lambda k: (k[1], k[0])):
        if state.active_pitches[(pitch, program)]:
            tokens.append(vocabulary.program_id(program))
            tokens.append(vocabulary.pitch_id(pitch))
    if not tokens:
        return []
    tokens.append(vocabulary.tie_id())
    return tokens


def _velocity_to_bin(velocity: int) -> int:
    if velocity <= 0:
        return 0
    bin_idx = int(
        math.ceil(NUM_VELOCITY_BINS * velocity / note_seq.MAX_MIDI_VELOCITY)
    )
    return min(bin_idx, NUM_VELOCITY_BINS)


def _trim_overlapping_notes(ns: note_seq.NoteSequence) -> note_seq.NoteSequence:
    trimmed = note_seq.NoteSequence()
    trimmed.CopyFrom(ns)
    channels = {
        (
            note.pitch,
            int(note.program) if note.program is not None else 0,
            bool(note.is_drum),
        )
        for note in trimmed.notes
    }
    for pitch, program, is_drum in channels:
        channel_notes = [
            note
            for note in trimmed.notes
            if note.pitch == pitch
            and (int(note.program) if note.program is not None else 0) == program
            and bool(note.is_drum) == is_drum
        ]
        sorted_notes = sorted(channel_notes, key=lambda note: note.start_time)
        for idx in range(1, len(sorted_notes)):
            if sorted_notes[idx - 1].end_time > sorted_notes[idx].start_time:
                sorted_notes[idx - 1].end_time = sorted_notes[idx].start_time
    valid_notes = [
        note for note in trimmed.notes if note.start_time < note.end_time
    ]
    del trimmed.notes[:]
    trimmed.notes.extend(valid_notes)
    trimmed.total_time = max((note.end_time for note in trimmed.notes), default=0.0)
    return trimmed


__all__ = [
    "TokenizerOutput",
    "tokenize_note_sequence",
]

"""Symbolic vocabulary and codec utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from configs import load_project_config

CONFIG = load_project_config()
CODEC_CFG = CONFIG["symbolic"]["codec"]
STEPS_PER_SECOND = CODEC_CFG["steps_per_second"]
MAX_SHIFT_MS = CODEC_CFG["max_shift_ms"]
NUM_VELOCITY_BINS = CODEC_CFG["num_velocity_bins"]
MIN_MIDI_PITCH = 0
MAX_MIDI_PITCH = 127
PITCH_RANGE = MAX_MIDI_PITCH - MIN_MIDI_PITCH + 1


PAD_ID = 0
EOS_ID = 1
UNK_ID = 2
SPECIAL_OFFSET = 3  # number of reserved tokens


@dataclass
class VocabularyRange:
    name: str
    min_id: int
    max_id: int


def build_event_ranges() -> List[VocabularyRange]:
    """Create the event ranges following legacy MT3 semantics."""
    ranges: List[VocabularyRange] = []

    def add_range(name: str, size: int) -> None:
        min_id = SPECIAL_OFFSET + sum(r.max_id - r.min_id + 1 for r in ranges)
        ranges.append(VocabularyRange(name, min_id, min_id + size - 1))

    # Shift events: 1 step per token up to max shift
    max_shift_steps = (STEPS_PER_SECOND * MAX_SHIFT_MS) // 1000
    add_range("shift", max_shift_steps)

    add_range("pitch", PITCH_RANGE)
    add_range("velocity", max(1, NUM_VELOCITY_BINS) + 1)
    add_range("tie", 1)
    add_range("program", 128)
    add_range("drum", PITCH_RANGE)
    return ranges


EVENT_RANGES = build_event_ranges()


def id_for_event(event_type: str, value: int) -> int:
    """Map event type/value to token id."""
    for r in EVENT_RANGES:
        if r.name == event_type:
            if value < 0 or r.min_id + value > r.max_id:
                raise ValueError(f"Value {value} out of range for {event_type}")
            return r.min_id + value
    raise ValueError(f"Unknown event type {event_type}")


def shift_id(steps: int) -> int:
    return id_for_event("shift", steps - 1)


def pitch_id(midi_pitch: int) -> int:
    return id_for_event("pitch", midi_pitch - MIN_MIDI_PITCH)


def velocity_id(velocity_bin: int) -> int:
    return id_for_event("velocity", velocity_bin)


def tie_id() -> int:
    return id_for_event("tie", 0)


def program_id(program: int) -> int:
    return id_for_event("program", program)


def drum_id(drum_pitch: int) -> int:
    return id_for_event("drum", drum_pitch - MIN_MIDI_PITCH)


def decode_event(token_id: int) -> Tuple[str, int]:
    """Reverse lookup of token id → (event_type, value)."""
    if token_id < SPECIAL_OFFSET:
        raise ValueError("Special tokens should be handled separately.")
    for r in EVENT_RANGES:
        if r.min_id <= token_id <= r.max_id:
            return r.name, token_id - r.min_id
    raise ValueError(f"Unknown token id {token_id}")


__all__ = [
    "PAD_ID",
    "EOS_ID",
    "UNK_ID",
    "STEPS_PER_SECOND",
    "MAX_SHIFT_MS",
    "NUM_VELOCITY_BINS",
    "MIN_MIDI_PITCH",
    "MAX_MIDI_PITCH",
    "EVENT_RANGES",
    "shift_id",
    "pitch_id",
    "velocity_id",
    "tie_id",
    "program_id",
    "drum_id",
    "decode_event",
]

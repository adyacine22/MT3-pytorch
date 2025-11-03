# Copyright 2024 The MT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model vocabulary."""

import dataclasses
import math
from typing import Callable, List, Optional, Sequence, Tuple, Mapping

import note_seq

# Default vocabulary configurations (matching legacy MT3)
DEFAULT_STEPS_PER_SECOND: int = 100
DEFAULT_MAX_SHIFT_SECONDS: int = 1  # Changed from 10 to match legacy MT3
DEFAULT_NUM_VELOCITY_BINS: int = 127  # Already correct (1-127, not 0-127)

# Special token IDs for decoding
DECODED_EOS_ID: int = -1
DECODED_INVALID_ID: int = -2


@dataclasses.dataclass
class EventRange:
    """Defines a range of values for a given event type."""

    type: str
    min_value: int
    max_value: int


@dataclasses.dataclass
class Event:
    """Represents a musical event with a type and a value."""

    type: str
    value: int


class Codec:
    """Encodes and decodes musical events."""

    def __init__(
        self,
        max_shift_steps: int,
        steps_per_second: float,
        event_ranges: List[EventRange],
    ):
        """
        Initializes the Codec.

        Args:
            max_shift_steps: Maximum number of shift steps to encode.
            steps_per_second: The duration of a single shift step (1 / steps_per_second).
            event_ranges: A list of EventRange objects for other event types.
        """
        self.steps_per_second: float = steps_per_second
        # Shift range: 0 to (max_shift_steps - 1)
        # e.g., max_shift_steps=100 → range 0-99 (100 values)
        self._shift_range: EventRange = EventRange(
            type="shift", min_value=0, max_value=max_shift_steps - 1
        )
        self._event_ranges: List[EventRange] = [self._shift_range] + event_ranges
        # Ensure all event types are unique
        assert len(self._event_ranges) == len(
            set([er.type for er in self._event_ranges])
        )

    @property
    def num_classes(self) -> int:
        """Returns the total number of event classes."""
        return sum(er.max_value - er.min_value + 1 for er in self._event_ranges)

    def is_shift_event_index(self, index: int) -> bool:
        """Checks if a given index corresponds to a shift event."""
        return self._shift_range.min_value <= index <= self._shift_range.max_value

    @property
    def max_shift_steps(self) -> int:
        """Returns the maximum number of shift steps."""
        return self._shift_range.max_value

    def encode_event(self, event: Event) -> int:
        """Converts an Event to its integer ID."""
        offset = 0
        for er in self._event_ranges:
            if event.type == er.type:
                if not er.min_value <= event.value <= er.max_value:
                    raise ValueError(
                        f"Event value {event.value} is outside the valid range "
                        f"[{er.min_value}, {er.max_value}] for type {event.type}"
                    )
                return offset + event.value - er.min_value
            offset += er.max_value - er.min_value + 1
        raise ValueError(f"Unknown event type: {event.type}")

    def event_type_range(self, event_type: str) -> Tuple[int, int]:
        """Returns the (min_id, max_id) for a given event type."""
        offset = 0
        for er in self._event_ranges:
            if event_type == er.type:
                return offset, offset + (er.max_value - er.min_value)
            offset += er.max_value - er.min_value + 1
        raise ValueError(f"Unknown event type: {event_type}")

    def decode_event_index(self, index: int) -> Event:
        """Converts an integer ID back to an Event."""
        offset = 0
        for er in self._event_ranges:
            if offset <= index <= offset + er.max_value - er.min_value:
                return Event(type=er.type, value=er.min_value + index - offset)
            offset += er.max_value - er.min_value + 1
        raise ValueError(f"Unknown event index: {index}")


@dataclasses.dataclass
class VocabularyConfig:
    """Configuration for the vocabulary."""

    steps_per_second: int = DEFAULT_STEPS_PER_SECOND
    max_shift_seconds: int = DEFAULT_MAX_SHIFT_SECONDS
    num_velocity_bins: int = DEFAULT_NUM_VELOCITY_BINS

    @property
    def abbrev_str(self) -> str:
        """Returns an abbreviated string representation of the config."""
        s = ""
        if self.steps_per_second != DEFAULT_STEPS_PER_SECOND:
            s += f"ss{self.steps_per_second}"
        if self.max_shift_seconds != DEFAULT_MAX_SHIFT_SECONDS:
            s += f"ms{self.max_shift_seconds}"
        if self.num_velocity_bins != DEFAULT_NUM_VELOCITY_BINS:
            s += f"vb{self.num_velocity_bins}"
        return s


def build_codec(vocab_config: VocabularyConfig) -> Codec:
    """
    Builds the event codec from a vocabulary config.
    
    Token order matches legacy MT3:
    - [0-99]:      shift events (0-990ms in 10ms steps) 
    - [100-226]:   velocity events (velocity 1-127, 127 bins)
    - [227-354]:   pitch events (MIDI pitch 0-127)
    - [355-482]:   drum events (drum pitch 0-127)
    - [483-610]:   program events (program 0-127)
    - [611]:       tie token
    Total: 612 tokens (matches legacy MT3)
    """
    # Legacy MT3 order: velocity → pitch → drum → program → tie
    # NOTE: velocity starts at 1, not 0 (velocity 0 is invalid)
    event_ranges = [
        EventRange("velocity", 1, vocab_config.num_velocity_bins),  # 1-127 (127 values)
        EventRange("pitch", note_seq.MIN_MIDI_PITCH, note_seq.MAX_MIDI_PITCH),  # 0-127
        EventRange("drum", note_seq.MIN_MIDI_PITCH, note_seq.MAX_MIDI_PITCH),  # 0-127
        EventRange("program", note_seq.MIN_MIDI_PROGRAM, note_seq.MAX_MIDI_PROGRAM),  # 0-127
        EventRange("tie", 0, 0),  # Single token
    ]
    return Codec(
        max_shift_steps=(
            vocab_config.steps_per_second * vocab_config.max_shift_seconds
        ),  # 100 * 1 = 100 steps (0-99)
        steps_per_second=vocab_config.steps_per_second,
        event_ranges=event_ranges,
    )


def num_velocity_bins_from_codec(codec: Codec) -> int:
    """Get number of velocity bins from event codec."""
    lo, hi = codec.event_type_range("velocity")
    return hi - lo


def velocity_to_bin(velocity: int, num_velocity_bins: int) -> int:
    """Converts a MIDI velocity to a velocity bin."""
    if velocity == 0:
        return 0
    return math.ceil(num_velocity_bins * velocity / note_seq.MAX_MIDI_VELOCITY)


def bin_to_velocity(velocity_bin: int, num_velocity_bins: int) -> int:
    """Converts a velocity bin back to a MIDI velocity."""
    if velocity_bin == 0:
        return 0
    return int(note_seq.MAX_MIDI_VELOCITY * velocity_bin / num_velocity_bins)


class GenericTokenVocabulary:
    """A vocabulary that maps tokens to integer IDs."""

    def __init__(self, regular_ids: int, extra_ids: int = 0):
        self._num_special_tokens: int = 3  # PAD, EOS, UNK
        self._num_regular_tokens: int = regular_ids
        self.extra_ids: int = extra_ids

    @property
    def eos_id(self) -> Optional[int]:
        return 1

    @property
    def unk_id(self) -> Optional[int]:
        return 2

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size, including special and extra tokens."""
        return self._num_special_tokens + self._num_regular_tokens + self.extra_ids

    def encode(self, token_ids: Sequence[int]) -> Sequence[int]:
        """Encodes a sequence of raw token IDs."""
        encoded = []
        for token_id in token_ids:
            if not 0 <= token_id < self._num_regular_tokens:
                raise ValueError(
                    f"Token ID {token_id} is out of range for regular tokens."
                )
            encoded.append(token_id + self._num_special_tokens)
        return encoded

    def decode(self, ids: Sequence[int]) -> Sequence[int]:
        """Decodes a sequence of vocabulary IDs."""
        decoded = []
        for i in ids:
            if i == self.eos_id:
                decoded.append(DECODED_EOS_ID)
                break  # Stop at EOS
            elif i < self._num_special_tokens or i >= self.vocab_size - self.extra_ids:
                decoded.append(DECODED_INVALID_ID)
            else:
                decoded.append(i - self._num_special_tokens)
        return decoded


@dataclasses.dataclass
class ProgramGranularity:
    """Defines how to handle program change events."""

    tokens_map_fn: Callable[[Sequence[int], Codec], Sequence[int]]
    program_map_fn: Callable[[int], int]


def drop_programs(tokens: Sequence[int], codec: Codec) -> Sequence[int]:
    """Removes program change events from a token sequence."""
    min_program_id, max_program_id = codec.event_type_range("program")
    return [t for t in tokens if not min_program_id <= t <= max_program_id]


def programs_to_midi_classes(tokens: Sequence[int], codec: Codec) -> Sequence[int]:
    """Maps program events to the first program in their MIDI class."""
    min_program_id, _ = codec.event_type_range("program")
    mapped_tokens = []
    for t in tokens:
        if min_program_id <= t <= min_program_id + 127:
            mapped_tokens.append(min_program_id + 8 * ((t - min_program_id) // 8))
        else:
            mapped_tokens.append(t)
    return mapped_tokens


PROGRAM_GRANULARITIES: Mapping[str, ProgramGranularity] = {
    "flat": ProgramGranularity(tokens_map_fn=drop_programs, program_map_fn=lambda p: 0),
    "midi_class": ProgramGranularity(
        tokens_map_fn=programs_to_midi_classes, program_map_fn=lambda p: 8 * (p // 8)
    ),
    "full": ProgramGranularity(
        tokens_map_fn=lambda tokens, codec: tokens, program_map_fn=lambda p: p
    ),
}

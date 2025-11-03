import torch
import numpy as np
from . import vocabularies
import note_seq
import dataclasses
from typing import Any, Dict, List, MutableMapping, Optional, Tuple


def shift_tokens(
    tokens: torch.Tensor, start_token: int = vocabularies.DECODED_EOS_ID
) -> torch.Tensor:
    """Shifts the tokens for autoregressive input."""
    shifted_tokens = torch.zeros_like(tokens)
    shifted_tokens[..., 1:] = tokens[..., :-1]
    shifted_tokens[..., 0] = start_token
    return shifted_tokens


class ContinuousInputsEncDecFeatureConverter:
    """Converts features for a continuous input encoder-decoder model."""

    def __init__(self, pack: bool = False):
        self.pack = pack

    def __call__(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Converts a dictionary of features.

        Args:
            features: A dictionary containing 'inputs' and 'targets'.

        Returns:
            A dictionary with the converted features.
        """
        decoder_input_tokens = shift_tokens(features["targets"])

        d = {
            "encoder_input_tokens": features["inputs"],
            "decoder_target_tokens": features["targets"],
            "decoder_input_tokens": decoder_input_tokens,
            "decoder_loss_weights": (features["targets"] != 0).to(torch.int32),
        }

        if self.pack:
            # Packing is not yet implemented
            pass

        return d


@dataclasses.dataclass
class TimedEvent:
    time: float
    type: str
    value: Any


def note_sequence_to_timed_events(ns: note_seq.NoteSequence) -> List[TimedEvent]:
    """Converts a NoteSequence to a list of timed events."""
    events: List[TimedEvent] = []
    for note in ns.notes:
        events.append(
            TimedEvent(
                note.start_time,
                "note_on",
                (note.pitch, note.velocity, note.program, note.is_drum),
            )
        )
        events.append(
            TimedEvent(
                note.end_time, "note_off", (note.pitch, note.program, note.is_drum)
            )
        )
    events.sort(key=lambda e: e.time)
    return events


def timed_events_to_tokens(
    events: List[TimedEvent], codec: vocabularies.Codec, frame_times: np.ndarray
) -> Tuple[List[int], List[int], List[int]]:
    """Converts a list of timed events to a sequence of tokens."""
    tokens: List[int] = []
    event_start_indices: List[int] = []
    current_time: float = 0.0
    token_idx: int = 0

    # Get num_velocity_bins from codec
    num_velocity_bins = vocabularies.num_velocity_bins_from_codec(codec)

    for event in events:
        # Add time shifts
        time_delta = event.time - current_time
        if time_delta > 0:
            shift_steps = int(round(time_delta * codec.steps_per_second))
            while shift_steps > 0:
                shift_amount = min(shift_steps, codec.max_shift_steps)
                tokens.append(
                    codec.encode_event(vocabularies.Event("shift", shift_amount))
                )
                shift_steps -= shift_amount

        # Add event
        if event.type == "note_on":
            pitch, velocity, program, is_drum = event.value
            if not is_drum:
                tokens.append(
                    codec.encode_event(vocabularies.Event("program", program))
                )
            velocity_bin = vocabularies.velocity_to_bin(velocity, num_velocity_bins)
            tokens.append(
                codec.encode_event(vocabularies.Event("velocity", velocity_bin))
            )
            if is_drum:
                tokens.append(codec.encode_event(vocabularies.Event("drum", pitch)))
            else:
                tokens.append(codec.encode_event(vocabularies.Event("pitch", pitch)))
        elif event.type == "note_off":
            pitch, program, is_drum = event.value
            if not is_drum:
                tokens.append(
                    codec.encode_event(vocabularies.Event("program", program))
                )
            # Note: Legacy MT3 uses velocity=1 for note_off events
            # Velocity must be in range [1, 127] (velocity 0 not allowed in new vocab)
            tokens.append(codec.encode_event(vocabularies.Event("velocity", 1)))
            if is_drum:
                tokens.append(codec.encode_event(vocabularies.Event("drum", pitch)))
            else:
                tokens.append(codec.encode_event(vocabularies.Event("pitch", pitch)))

        current_time = event.time

        while (
            len(event_start_indices) < len(frame_times)
            and frame_times[len(event_start_indices)] < current_time
        ):
            event_start_indices.append(token_idx)
        token_idx = len(tokens)

    event_end_indices = event_start_indices[1:] + [len(tokens)]

    return tokens, event_start_indices, event_end_indices


@dataclasses.dataclass
class NoteDecodingState:
    """Decoding state for note transcription."""

    current_time: float = 0.0
    current_velocity: int = 100
    current_program: int = 0
    active_pitches: MutableMapping[Tuple[int, int], Tuple[float, int]] = (
        dataclasses.field(default_factory=dict)
    )
    tied_pitches: MutableMapping[Tuple[int, int], bool] = dataclasses.field(
        default_factory=dict
    )
    is_tie_section: bool = False
    note_sequence: note_seq.NoteSequence = dataclasses.field(
        default_factory=lambda: note_seq.NoteSequence(ticks_per_quarter=220)
    )


def _add_note_to_sequence(
    ns: note_seq.NoteSequence,
    start_time: float,
    end_time: float,
    pitch: int,
    velocity: int,
    program: int,
    is_drum: bool,
):
    """Helper function to add a note to a NoteSequence."""
    end_time = max(end_time, start_time + 0.01)  # Ensure minimum duration
    ns.notes.add(
        start_time=start_time,
        end_time=end_time,
        pitch=pitch,
        velocity=velocity,
        program=program,
        is_drum=is_drum,
    )
    ns.total_time = max(ns.total_time, end_time)


def decode_note_event(
    state: NoteDecodingState,
    time: float,
    event: vocabularies.Event,
    codec: vocabularies.Codec,
    num_velocity_bins: int,
):
    """Processes a single note event and updates the decoding state."""
    if time < state.current_time:
        raise ValueError(
            f"Event time {time} is before current time {state.current_time}"
        )

    state.current_time = time

    if event.type == "pitch":
        pitch = event.value
        program = state.current_program
        if state.is_tie_section:
            if (pitch, program) not in state.active_pitches:
                raise ValueError(
                    f"Inactive pitch/program in tie section: {pitch}/{program}"
                )
            if (pitch, program) in state.tied_pitches:
                raise ValueError(f"Pitch/program is already tied: {pitch}/{program}")
            state.tied_pitches[(pitch, program)] = True
        elif state.current_velocity == 0:
            # Note off
            if (pitch, program) in state.active_pitches:
                start_time, velocity = state.active_pitches.pop((pitch, program))
                _add_note_to_sequence(
                    state.note_sequence,
                    start_time,
                    time,
                    pitch,
                    velocity,
                    program,
                    False,
                )
        else:
            # Note on
            if (pitch, program) in state.active_pitches:
                # End previous note
                start_time, velocity = state.active_pitches.pop((pitch, program))
                _add_note_to_sequence(
                    state.note_sequence,
                    start_time,
                    time,
                    pitch,
                    velocity,
                    program,
                    False,
                )
            state.active_pitches[(pitch, program)] = (time, state.current_velocity)

    elif event.type == "drum":
        if state.current_velocity == 0:
            raise ValueError("Velocity cannot be zero for a drum event")
        _add_note_to_sequence(
            state.note_sequence,
            time,
            time + 0.01,
            event.value,
            state.current_velocity,
            0,
            True,
        )

    elif event.type == "velocity":
        state.current_velocity = vocabularies.bin_to_velocity(
            event.value, num_velocity_bins
        )

    elif event.type == "program":
        state.current_program = event.value

    elif event.type == "tie":
        if not state.is_tie_section:
            raise ValueError("Tie event outside of a tie section")
        for pitch, program in list(state.active_pitches.keys()):
            if (pitch, program) not in state.tied_pitches:
                start_time, velocity = state.active_pitches.pop((pitch, program))
                _add_note_to_sequence(
                    state.note_sequence,
                    start_time,
                    time,
                    pitch,
                    velocity,
                    program,
                    False,
                )
        state.is_tie_section = False

    else:
        raise ValueError(f"Unknown event type: {event.type}")


def flush_note_decoding_state(state: NoteDecodingState) -> note_seq.NoteSequence:
    """Ends all active notes and returns the resulting NoteSequence."""
    for (pitch, program), (start_time, velocity) in state.active_pitches.items():
        _add_note_to_sequence(
            state.note_sequence,
            start_time,
            state.current_time,
            pitch,
            velocity,
            program,
            False,
        )
    return state.note_sequence


def tokens_to_note_sequence(
    tokens: List[int], codec: vocabularies.Codec
) -> note_seq.NoteSequence:
    """Converts a sequence of tokens to a NoteSequence."""
    state = NoteDecodingState()
    num_velocity_bins = vocabularies.num_velocity_bins_from_codec(codec)

    # This is a simplified implementation of the decoding process.
    # A proper implementation would need to handle segments and tie sections.
    state.is_tie_section = False  # Start not in a tie section

    for token in tokens:
        event = codec.decode_event_index(token)
        if event.type == "shift":
            state.current_time += event.value / codec.steps_per_second
        else:
            decode_note_event(
                state, state.current_time, event, codec, num_velocity_bins
            )

    return flush_note_decoding_state(state)


def merge_events(event_lists: List[List[TimedEvent]]) -> List[TimedEvent]:
    """Merges multiple event lists into a single sorted list."""
    merged: List[TimedEvent] = []
    for events in event_lists:
        merged.extend(events)
    merged.sort(key=lambda e: e.time)
    return merged

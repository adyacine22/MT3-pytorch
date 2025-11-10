"""Utilities to reconstruct NoteSequence objects from tokenizer outputs."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple, List

import note_seq  # type: ignore

from data.symbolic import vocabulary

STEPS_PER_SECOND = vocabulary.STEPS_PER_SECOND
MIN_MIDI_PITCH = vocabulary.MIN_MIDI_PITCH
SPECIAL_TOKENS = {vocabulary.PAD_ID, vocabulary.EOS_ID, vocabulary.UNK_ID}


def _steps_to_seconds(steps: int) -> float:
    return steps / STEPS_PER_SECOND


def _velocity_from_bin(bin_id: int) -> int:
    if bin_id <= 0:
        return 0
    max_velocity = note_seq.MAX_MIDI_VELOCITY
    bin_size = max_velocity / max(1, vocabulary.NUM_VELOCITY_BINS)
    approx = int(round(bin_id * bin_size))
    return max(1, min(max_velocity, approx))


def _seed_active_state(
    ns: note_seq.NoteSequence,
    active: Dict[Tuple[int, int, bool], note_seq.NoteSequence.Note],
    state_events: Sequence[int],
    limit: int,
    chunk_start_time: float,
) -> None:
    idx = 0
    program = 0
    while idx < limit:
        token = state_events[idx]
        if token == vocabulary.tie_id():
            break
        event_type, value = vocabulary.decode_event(token)
        if event_type == "program":
            program = value
        elif event_type == "pitch":
            pitch = value + MIN_MIDI_PITCH
            note = ns.notes.add()
            note.pitch = pitch
            note.velocity = 100
            note.start_time = chunk_start_time
            note.end_time = chunk_start_time
            note.program = program
            note.is_drum = False
            active[(pitch, program, False)] = note
        idx += 1


def tokens_to_note_sequence(
    tokens: Sequence[int],
    chunk_start_time: float = 0.0,
    state_events: Sequence[int] | None = None,
    state_event_indices: Sequence[int] | None = None,
    initial_active: Sequence[Tuple[int, int, bool, float]] | None = None,
) -> note_seq.NoteSequence:
    """Best-effort reconstruction of a NoteSequence from tokenizer tokens."""

    ns = note_seq.NoteSequence()
    cur_step = 0
    idx = 0
    active: Dict[Tuple[int, int, bool], note_seq.NoteSequence.Note] = {}

    if initial_active:
        for pitch, program, is_drum, start_time in initial_active:
            note = ns.notes.add()
            note.pitch = pitch
            note.velocity = 80
            note.start_time = start_time
            note.end_time = start_time
            note.program = program
            note.is_drum = is_drum
            active[(pitch, program, is_drum)] = note

    if state_events and not initial_active:
        initial_tokens = len(state_events)
        if state_event_indices:
            initial_tokens = min(initial_tokens, state_event_indices[0])
        _seed_active_state(ns, active, state_events, initial_tokens, chunk_start_time)

    def add_note(pitch: int, program: int, velocity_bin: int, is_drum: bool) -> None:
        key = (pitch, program, is_drum)
        if velocity_bin == 0:
            note = active.pop(key, None)
            if note is None:
                # Fall back to matching by pitch only to tolerate malformed sequences.
                for active_key in list(active.keys()):
                    if active_key[0] == pitch:
                        note = active.pop(active_key)
                        break
            if note is not None:
                note.end_time = chunk_start_time + _steps_to_seconds(cur_step)
        else:
            note = ns.notes.add()
            note.pitch = pitch
            note.velocity = _velocity_from_bin(velocity_bin)
            note.start_time = chunk_start_time + _steps_to_seconds(cur_step)
            note.program = program
            note.is_drum = is_drum
            if is_drum:
                # Percussive hits do not emit explicit note-offs; give them a minimal duration.
                note.end_time = note.start_time + _steps_to_seconds(1)
            else:
                note.end_time = note.start_time  # extended on note-off
                active[key] = note

    while idx < len(tokens):
        token = tokens[idx]
        if token in SPECIAL_TOKENS:
            idx += 1
            continue

        event_type, value = vocabulary.decode_event(token)
        if event_type == "shift":
            cur_step += value + 1
            idx += 1
            continue

        # Multi-token note event
        program = 0
        is_drum = False
        if event_type == "program":
            program = value
            idx += 1
            if idx >= len(tokens):
                break
            event_type, value = vocabulary.decode_event(tokens[idx])

        if event_type != "velocity":
        if event_type == "pitch":
            # onsets-only mode without velocities
            pitch = value + MIN_MIDI_PITCH
                velocity_bin = vocabulary.NUM_VELOCITY_BINS
                add_note(pitch, program, velocity_bin, is_drum=False)
                idx += 1
                continue
            # Unknown token sequence – skip
            idx += 1
            continue

        velocity_bin = value
        idx += 1
        if idx >= len(tokens):
            break
        event_type, value = vocabulary.decode_event(tokens[idx])
        if event_type == "pitch":
            pitch = value + MIN_MIDI_PITCH
            add_note(pitch, program, velocity_bin, is_drum=False)
        elif event_type == "drum":
            pitch = value + MIN_MIDI_PITCH
            add_note(pitch, program, velocity_bin, is_drum=True)
        else:
            # Malformed sequence, skip token
            idx += 1
            continue
        idx += 1

    # Close lingering notes at chunk end
    end_time = chunk_start_time + _steps_to_seconds(cur_step)
    for note in active.values():
        if note.end_time <= note.start_time:
            note.end_time = end_time

    ns.total_time = max((n.end_time for n in ns.notes), default=0.0)
    return ns


__all__ = ["tokens_to_note_sequence"]

from __future__ import annotations

import os
from pathlib import Path
import sys

import note_seq  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import load_project_config
from data.symbolic import tokenizer, vocabulary

CONFIG = load_project_config()
AUDIO_CFG = CONFIG["audio"]
FEATURE_CFG = AUDIO_CFG["features"]
IO_CFG = AUDIO_CFG["io"]

DEBUG_OUTPUT = os.environ.get("TOKENIZER_DEBUG", "").strip().lower() in {
    "1",
    "true",
    "yes",
}


def _debug_print(label: str, output: tokenizer.TokenizerOutput) -> None:
    if not DEBUG_OUTPUT:
        return
    print(f"--- {label} ---")
    print(f"tokens ({len(output.tokens)}): {output.tokens}")
    print("------------------")


def _build_sequence() -> note_seq.NoteSequence:
    ns = note_seq.NoteSequence()
    note = ns.notes.add()
    note.pitch = 60
    note.velocity = 90
    note.start_time = 0.0
    note.end_time = 1.0
    note.program = 0
    note.is_drum = False

    note2 = ns.notes.add()
    note2.pitch = 64
    note2.velocity = 80
    note2.start_time = 0.5
    note2.end_time = 1.5
    note2.program = 0
    note2.is_drum = False

    ns.total_time = 1.5
    return ns


def test_tokenize_basic_sequence():
    ns = _build_sequence()
    output = tokenizer.tokenize_note_sequence(
        ns,
        chunk_start_s=0.0,
        chunk_end_s=1.0,
        num_frames=FEATURE_CFG["chunk_frames"],
        sample_rate=IO_CFG["sample_rate"],
        hop_length=FEATURE_CFG["hop_length"],
    )
    _debug_print("basic_sequence", output)
    assert output.tokens, "Tokenizer should emit tokens for active notes"
    assert vocabulary.pitch_id(60) in output.tokens


def test_tokenize_emits_tie_tokens_for_continuations():
    ns = _build_sequence()
    output = tokenizer.tokenize_note_sequence(
        ns,
        chunk_start_s=0.75,
        chunk_end_s=1.5,
        num_frames=FEATURE_CFG["chunk_frames"],
        sample_rate=IO_CFG["sample_rate"],
        hop_length=FEATURE_CFG["hop_length"],
    )
    _debug_print("tie_state_sequence", output)
    tokens = output.tokens
    idx = 0
    tie_notes = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == vocabulary.tie_id():
            break
        event_type, _ = vocabulary.decode_event(token)
        assert event_type == "program"
        idx += 1
        velocity_type, _ = vocabulary.decode_event(tokens[idx])
        assert velocity_type == "velocity"
        idx += 1
        pitch_type, _ = vocabulary.decode_event(tokens[idx])
        assert pitch_type == "pitch"
        idx += 1
        tie_notes += 1
    assert tie_notes > 0
    assert idx < len(tokens) and tokens[idx] == vocabulary.tie_id()


def _run_as_script() -> None:
    global DEBUG_OUTPUT
    DEBUG_OUTPUT = True
    print("Running tokenizer debug harness...\n")
    test_tokenize_basic_sequence()
    test_tokenize_emits_tie_tokens_for_continuations()
    print("\nAll tokenizer checks passed.")


if __name__ == "__main__":
    _run_as_script()

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
    print(
        f"event_start_indices ({len(output.event_start_indices)}): "
        f"{output.event_start_indices}"
    )
    print(
        f"event_end_indices ({len(output.event_end_indices)}): "
        f"{output.event_end_indices}"
    )
    print(f"state_events ({len(output.state_events)}): {output.state_events}")
    print(
        f"state_event_indices ({len(output.state_event_indices)}): "
        f"{output.state_event_indices}"
    )
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
    assert len(output.event_start_indices) == FEATURE_CFG["chunk_frames"]
    assert output.event_end_indices[-1] == len(output.tokens)
    assert vocabulary.pitch_id(60) in output.tokens


def test_tokenize_emits_tie_state_for_continuations():
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
    assert vocabulary.tie_id() in output.state_events
    assert output.state_events.count(vocabulary.tie_id()) >= 1


def _run_as_script() -> None:
    global DEBUG_OUTPUT
    DEBUG_OUTPUT = True
    print("Running tokenizer debug harness...\n")
    test_tokenize_basic_sequence()
    test_tokenize_emits_tie_state_for_continuations()
    print("\nAll tokenizer checks passed.")


if __name__ == "__main__":
    _run_as_script()

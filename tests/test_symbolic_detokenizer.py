from __future__ import annotations

import json
from pathlib import Path
import sys

import note_seq  # type: ignore
import pytest
import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"pydub\.utils",
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import load_project_config
from data.symbolic import tokenizer, detokenizer, vocabulary

CONFIG = load_project_config()
DATASET_PATHS = CONFIG["paths"]["datasets"]
UNIFIED_INDEX_PATH = ROOT / DATASET_PATHS["unified_index"]
AUDIO_CFG = CONFIG["audio"]
FEATURE_CFG = AUDIO_CFG["features"]
IO_CFG = AUDIO_CFG["io"]


def _load_entries() -> list[dict]:
    if not UNIFIED_INDEX_PATH.exists():
        pytest.skip("unified_index.json missing – run data/create_unified_index.py")
    with UNIFIED_INDEX_PATH.open() as fp:
        payload = json.load(fp)
    return payload.get("entries", [])


def _pick_entry(entries: list[dict], dataset: str) -> dict:
    for entry in entries:
        if entry.get("dataset") == dataset:
            return entry
    pytest.skip(f"No entry found for dataset {dataset}")


def _chunk_iterator(total_time: float, chunk_duration: float) -> list[tuple[float, float]]:
    num_chunks = max(1, int(round(total_time / chunk_duration)))
    ranges = []
    for idx in range(num_chunks):
        start = idx * chunk_duration
        end = min(total_time, start + chunk_duration)
        if end <= start:
            end = start + 1e-6
        ranges.append((start, end))
    return ranges


def _trim_sequence(
    ns: note_seq.NoteSequence, start: float, end: float
) -> note_seq.NoteSequence:
    trimmed = note_seq.NoteSequence()
    for note in ns.notes:
        if note.end_time <= start or note.start_time >= end:
            continue
        if note.is_drum and note.start_time < start:
            # Drum hits cannot be recovered when their onset happens before the chunk boundary.
            continue
        new_note = trimmed.notes.add()
        new_note.pitch = note.pitch
        new_note.velocity = note.velocity
        new_note.start_time = max(note.start_time, start) - start
        new_note.end_time = min(note.end_time, end) - start
        new_note.program = note.program
        new_note.is_drum = note.is_drum
    trimmed.total_time = max((n.end_time for n in trimmed.notes), default=0.0)
    return trimmed


def _sequence_signature(
    sequence: note_seq.NoteSequence, window_start: float, window_end: float
) -> list[tuple[int, int, bool, int, int]]:
    entries: list[tuple[int, int, bool, int, int]] = []
    for note in sequence.notes:
        program = int(note.program) if note.program is not None else 0
        is_drum = bool(note.is_drum)
        rel_start = max(0.0, min(note.start_time, window_end) - window_start)
        rel_end = max(rel_start, min(note.end_time, window_end) - window_start)
        start_steps = round(rel_start * vocabulary.STEPS_PER_SECOND)
        end_steps = round(rel_end * vocabulary.STEPS_PER_SECOND)
        entries.append((note.pitch, program, is_drum, start_steps, end_steps))
    entries.sort()
    return entries


def _compare_sequences(
    reference: note_seq.NoteSequence,
    reconstructed: note_seq.NoteSequence,
    chunk_start: float,
    chunk_end: float,
) -> None:
    step_tolerance = 2
    ref = _sequence_signature(reference, 0.0, chunk_end - chunk_start)
    rec = _sequence_signature(reconstructed, chunk_start, chunk_end)
    assert len(ref) == len(rec)
    for (rpitch, rprog, ris_drum, r_start, r_end), (
        cpitch,
        cprog,
        cis_drum,
        c_start,
        c_end,
    ) in zip(ref, rec):
        assert rpitch == cpitch
        assert rprog == cprog
        assert ris_drum == cis_drum
        if abs(r_start - c_start) > step_tolerance:
            print(
                "[debug] start mismatch pitch="
                f"{rpitch} ref_steps={r_start} rec_steps={c_start}"
                f" chunk=[{chunk_start:.3f},{chunk_end:.3f}]"
            )
        assert abs(r_start - c_start) <= step_tolerance
        if ris_drum:
            continue
        if abs(r_end - c_end) > step_tolerance:
            print(
                "[debug] end mismatch pitch="
                f"{rpitch} ref_steps={r_end} rec_steps={c_end}"
                f" chunk=[{chunk_start:.3f},{chunk_end:.3f}]"
            )
        assert abs(r_end - c_end) <= step_tolerance


@pytest.mark.slow
def test_chunkwise_tokenize_detokenize_matches_original():
    entries = _load_entries()
    datasets = ["maestro", "slakh_full_mix", "slakh_stem"]
    chunk_duration = FEATURE_CFG["chunk_samples"] / IO_CFG["sample_rate"]

    for dataset in datasets:
        entry = _pick_entry(entries, dataset)
        midi_path = ROOT / entry["midi_path"]
        if not midi_path.exists():
            pytest.skip(f"MIDI missing: {midi_path}")
        print(f"[info] Dataset={dataset} MIDI={midi_path}")
        ns_raw = note_seq.midi_file_to_note_sequence(str(midi_path))
        ns_prepared = tokenizer._prepare_sequence(ns_raw)
        total_time = float(entry.get("duration_seconds", ns_prepared.total_time))

        checked = 0
        start_s = 0.0
        carry_state = []
        while start_s < total_time:
            end_s = min(total_time, start_s + chunk_duration)
            chunk_ns = _trim_sequence(ns_prepared, start_s, end_s)
            if chunk_ns.notes:
                print(f"  - Chunk starting at {start_s:.3f}s ({checked + 1}/10)")
                token_output = tokenizer.tokenize_note_sequence(
                    ns_raw,
                    chunk_start_s=start_s,
                    chunk_end_s=end_s,
                    num_frames=FEATURE_CFG["chunk_frames"],
                    sample_rate=IO_CFG["sample_rate"],
                    hop_length=FEATURE_CFG["hop_length"],
                )
                reconstructed = detokenizer.tokens_to_note_sequence(
                    token_output.tokens,
                    chunk_start_time=start_s,
                    initial_active=list(carry_state),
                )
                _compare_sequences(
                    chunk_ns,
                    reconstructed,
                    chunk_start=start_s,
                    chunk_end=end_s,
                )
                print(f"    ✓ Matching notes count={len(chunk_ns.notes)}")
                carry_state = [
                    (
                        note.pitch,
                        note.program,
                        bool(note.is_drum),
                        note.start_time,
                    )
                    for note in reconstructed.notes
                    if note.end_time >= end_s - 1e-3
                ]
                checked += 1
                if checked >= 10:
                    break
            start_s += chunk_duration
        assert checked > 0, f"No chunks with notes for dataset {dataset}"
        print(f"[summary] {dataset}: validated {checked} chunks")


def _run_as_script() -> None:
    test_chunkwise_tokenize_detokenize_matches_original()


if __name__ == "__main__":
    _run_as_script()

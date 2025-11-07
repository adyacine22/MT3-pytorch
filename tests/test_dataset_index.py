from __future__ import annotations

import json
from pathlib import Path

from configs import load_project_config
from configs.project_config import PROJECT_ROOT as CONFIG_PROJECT_ROOT


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_project_config()
DATASET_PATHS = CONFIG["paths"]["datasets"]
INDEX_PATH = (REPO_ROOT / DATASET_PATHS["unified_index"]).resolve()


def _load_entries():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Unified index missing at {INDEX_PATH}. Run data/create_unified_index.py first."
        )
    with INDEX_PATH.open() as fp:
        payload = json.load(fp)
    return payload.get("entries", [])


def test_unified_index_contains_expected_metadata():
    entries = _load_entries()

    assert entries, "Unified index should not be empty."

    maestro_entries = [e for e in entries if e["dataset"] == "maestro"]
    assert maestro_entries, "Should include MAESTRO examples."
    first_maestro = maestro_entries[0]
    assert first_maestro["instrument_programs"] == [0]
    assert first_maestro["instrument_names"] == ["Acoustic Grand Piano"]
    assert first_maestro["instrument_classes"] == ["Piano"]
    assert first_maestro["split"] in {"train", "validation", "test"}
    assert first_maestro["audio_path"].startswith("datasets/")
    assert first_maestro["midi_path"].startswith("datasets/")
    assert (REPO_ROOT / first_maestro["audio_path"]).exists()
    assert (REPO_ROOT / first_maestro["midi_path"]).exists()
    assert first_maestro["duration_seconds"] > 0
    assert first_maestro["audio_sample_rate"] > 0
    assert first_maestro["is_drum"] is False

    slakh_mix = next(e for e in entries if e["dataset"] == "slakh_full_mix")
    assert slakh_mix["dataset"] == "slakh_full_mix"
    assert slakh_mix["audio_path"].startswith("datasets/")
    assert slakh_mix["midi_path"].startswith("datasets/")
    assert (REPO_ROOT / slakh_mix["audio_path"]).exists()
    assert (REPO_ROOT / slakh_mix["midi_path"]).exists()
    assert len(slakh_mix["instrument_programs"]) == len(slakh_mix["instrument_names"]) == len(
        slakh_mix["instrument_classes"]
    )
    assert len(slakh_mix["instrument_programs"]) > 0
    assert slakh_mix["num_rendered_stems"] == len(slakh_mix["instrument_programs"])
    assert slakh_mix["audio_path"].endswith("mix.flac")
    assert slakh_mix["midi_path"].endswith("all_src.mid")
    assert slakh_mix["duration_seconds"] > 0

    slakh_stem = next(e for e in entries if e["dataset"] == "slakh_stem")
    assert slakh_stem["dataset"] == "slakh_stem"
    assert len(slakh_stem["instrument_programs"]) == 1
    assert len(slakh_stem["instrument_names"]) == 1
    assert len(slakh_stem["instrument_classes"]) == 1
    assert slakh_stem["audio_path"].startswith("datasets/")
    assert slakh_stem["midi_path"].startswith("datasets/")
    assert slakh_stem["duration_seconds"] > 0
    assert "stem_id" in slakh_stem
    assert "program_num" not in slakh_stem
    assert "/stems/" in slakh_stem["audio_path"]
    assert "/MIDI/" in slakh_stem["midi_path"]
    assert (REPO_ROOT / slakh_stem["audio_path"]).exists()
    assert (REPO_ROOT / slakh_stem["midi_path"]).exists()

    # Validate root field matches config path
    with INDEX_PATH.open() as fp:
        persisted = json.load(fp)
    assert persisted["root"] == str(CONFIG_PROJECT_ROOT.resolve())
    assert len(persisted["entries"]) == len(entries)

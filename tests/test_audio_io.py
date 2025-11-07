from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from data.audio.io import prepare_waveform
from configs import load_project_config


CONFIG = load_project_config()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATHS = CONFIG["paths"]["datasets"]
UNIFIED_INDEX_PATH = PROJECT_ROOT / DATASET_PATHS["unified_index"]
TEST_OUTPUT_DIR = PROJECT_ROOT / "test_files"


def _get_random_entry(dataset: str) -> dict:
    with UNIFIED_INDEX_PATH.open() as fp:
        entries = json.load(fp)["entries"]
    dataset_entries = [e for e in entries if e["dataset"] == dataset]
    assert dataset_entries, f"No entries found for dataset {dataset}"
    # deterministic pick for reproducibility (first entry)
    return dataset_entries[0]


def _write_wave(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)


def test_prepare_waveform_on_sample_entries(tmp_path):
    datasets_to_test = ["maestro", "slakh_full_mix", "slakh_stem"]
    output_root = PROJECT_ROOT / "test_files" / "test_audio_io"
    output_root.mkdir(exist_ok=True)
    for dataset in datasets_to_test:
        entry = _get_random_entry(dataset)
        audio_path = PROJECT_ROOT / entry["audio_path"]
        processed_audio, sr = prepare_waveform(str(audio_path))
        assert isinstance(processed_audio, np.ndarray)
        assert processed_audio.ndim == 1
        assert sr == CONFIG["audio"]["io"]["sample_rate"]

        # Save original and processed snippets for manual inspection
        dataset_dir = output_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        original_copy = dataset_dir / f"{Path(entry['audio_path']).stem}_original{audio_path.suffix}"
        shutil.copy(audio_path, original_copy)

        processed_copy = dataset_dir / f"{Path(entry['audio_path']).stem}_processed.wav"
        _write_wave(processed_copy, processed_audio, sr)
        print(f"Wrote original sample to {original_copy}")
        print(f"Wrote processed sample for {dataset} to {processed_copy}")

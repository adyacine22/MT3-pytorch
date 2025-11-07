from __future__ import annotations

import json
from pathlib import Path
import sys

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "soundfile is required for tests/test_audio_augment.py. "
        "Install it via `pip install soundfile` or run inside the mt3-pytorch environment."
    ) from exc

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import load_project_config
from data.audio.io import prepare_waveform, slice_chunk
from data.audio import augment


CONFIG = load_project_config()
DATASET_PATHS = CONFIG["paths"]["datasets"]
UNIFIED_INDEX_PATH = ROOT / DATASET_PATHS["unified_index"]
FEATURE_CFG = CONFIG["audio"]["features"]
PREPROCESS_CFG = CONFIG["audio"]["preprocess"]


def _first_entry(dataset: str) -> dict:
    with UNIFIED_INDEX_PATH.open() as fp:
        entries = json.load(fp)["entries"]
    for entry in entries:
        if entry["dataset"] == dataset:
            return entry
    raise ValueError(f"No entry found for dataset {dataset}")


def _save_audio(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.cpu().numpy(), sample_rate)


def test_apply_augmentation_profiles(tmp_path):
    entry = _first_entry("maestro")
    audio_path = ROOT / entry["audio_path"]
    waveform_np, sr = prepare_waveform(str(audio_path))
    chunk_samples = FEATURE_CFG["chunk_samples"]
    total_samples = waveform_np.shape[-1]
    start_sample = max(0, (total_samples // 2) - (chunk_samples // 2))
    chunk_np = slice_chunk(
        waveform_np,
        start_sample=start_sample,
        chunk_samples=chunk_samples,
        pad_value=PREPROCESS_CFG["pad_value"],
    )
    chunk = torch.from_numpy(chunk_np)

    output_root = ROOT / "test_files" / "test_audio_augment"
    output_root.mkdir(parents=True, exist_ok=True)

    profiles = ["rir", "noise", "eq", "clip", "gain"]
    for profile in profiles:
        augmented = augment.apply_augmentation(chunk, sr, profile=profile, seed=42)
        audio_out = output_root / f"{Path(entry['audio_path']).stem}_{profile}.wav"
        _save_audio(audio_out, augmented, sr)
        print(f"Saved {profile} augmentation to {audio_out}")

    # Combined augmentation sequence
    combined = chunk.clone()
    for profile in profiles:
        combined = augment.apply_augmentation(combined, sr, profile=profile, seed=42)
        combined_out = output_root / f"{Path(entry['audio_path']).stem}_combined.wav"
        _save_audio(combined_out, combined, sr)
        print(f"Saved combined augmentation to {combined_out}")


if __name__ == "__main__":
    import tempfile

    temp_dir = Path(tempfile.mkdtemp(prefix="test_audio_augment_"))
    test_apply_augmentation_profiles(temp_dir)

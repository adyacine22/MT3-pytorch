from __future__ import annotations

import json
from pathlib import Path
import soundfile as sf
import torch
import sys

import pytest

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import load_project_config
from configs.project_config import PROJECT_ROOT
from data.audio.io import prepare_waveform, slice_chunk
from data.audio.spectrogram import compute_features


CONFIG = load_project_config()
DATASET_PATHS = CONFIG["paths"]["datasets"]
UNIFIED_INDEX_PATH = PROJECT_ROOT / DATASET_PATHS["unified_index"]
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


def _save_spectrogram(path: Path, log_mel: torch.Tensor) -> None:
    if plt is None:
        pytest.skip("matplotlib required for spectrogram visualization")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(log_mel.cpu().numpy(), origin="lower", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel bins")
    ax.set_title(path.stem)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def test_waveform_to_logmel_artifacts(tmp_path):
    datasets = ["maestro", "slakh_full_mix", "slakh_stem"]
    output_root = PROJECT_ROOT / "test_files" / "test_audio_spectrogram"
    output_root.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        entry = _first_entry(dataset)
        audio_path = PROJECT_ROOT / entry["audio_path"]
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
        features = compute_features(chunk, sample_rate=sr)
        log_mel = features["log_mel"]
        assert log_mel.shape[0] == FEATURE_CFG["num_mels"]

        dataset_dir = output_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        audio_out = dataset_dir / f"{Path(entry['audio_path']).stem}_chunk.wav"
        image_out = dataset_dir / f"{Path(entry['audio_path']).stem}_spectrogram.png"

        _save_audio(audio_out, chunk, sr)
        _save_spectrogram(image_out, log_mel)

        print(f"Saved chunk for {dataset} to {audio_out}")
        print(f"Saved spectrogram for {dataset} to {image_out}")

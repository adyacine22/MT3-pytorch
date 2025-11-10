"""Centralised project configuration for MT3-PyTorch.

This module exposes configuration values that other parts of the codebase can
import directly. For now we only store dataset paths needed by the dataset
index creator and its associated tests, but future parameters should also live
here.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Paths relative to the project root; callers can resolve them as needed.
PROJECT_CONFIG = {
    "paths": {
        "datasets": {
            "maestro_root": "datasets/maestro-v3.0.0",
            "slakh_root": "datasets/slakh2100_flac_redux",
            "unified_index": "datasets/unified_index.json",
        },
        "cache": {
            "root": "cache",
            "chunk_manifest": "cache/chunk_manifest.parquet",
            "precomputed_chunks": "cache/precomputed_chunks",
        },
    },
    "audio": {
        "io": {
            "sample_rate": 16000,
            "convert_to_mono": True,
            "dtype": "float32",
        },
        "preprocess": {
            "dc_block": True,
            "high_pass_hz": 30.0,
            "high_pass_order": 2,
            "loudness_target_lufs": -23.0,
            "pre_emphasis": 0.97,
            "dither_amplitude": 1e-5,
            "pad_value": 0.0,
        },
        "features": {
            "hop_length": 128,
            "chunk_frames": 256,
            "chunk_samples": 32768,
            "num_mels": 512,
            "n_fft": 2048,
            "win_length": 2048,
            "window": "hann",
            "center": True,
            "mel_fmin": 20.0,
            "mel_fmax": 8000.0,
            "mel_power": 2.0,
            "log_epsilon": 1e-5,
        },
        "augment": {
            "noise_snr_db": [15.0, 30.0],
            "eq_gain_db": [-3.0, 3.0],
            "clip_dbfs": -3.0,
            "gain_jitter_db": [-3.0, 3.0],
            "rir_params": {
                "room_dim": [6.0, 5.0, 3.0],
                "absorption": 0.4,
                "max_order": 4,
                "source_height": 1.5,
                "mic_height": 1.5,
                "position_jitter": 0.3,
            },
        },
    },
    "symbolic": {
        "codec": {
            "steps_per_second": 125,
            "max_shift_ms": 2048,
            "num_velocity_bins": 1,
        },
        "tokenizer": {
            "include_ties": True,
            "onsets_only": False,
            "apply_sustain": True,
            "trim_overlapping_notes": True,
            "max_token_length": 1024,
        }
    },
    "compute": {
        "preprocessing": {
            "chunk_device": "cuda:0",
            "augment_enabled": False,
            "augment_profiles": ["noise"],
            "batch_size": 512,
            "max_tokenize_workers": 32,
            "datasets": ["maestro", "slakh_full_mix", "slakh_stem"],
            # "datasets": ["maestro"],
            "max_tracks_per_dataset": 5,
            "max_chunks_per_track": None,
            "dry_run": False,
            "overwrite": True,
            "store_spectrogram": True,
            "write_manifest": False,
            "skip_existing": True,
        }
    },
}

__all__ = ["PROJECT_CONFIG", "PROJECT_ROOT"]

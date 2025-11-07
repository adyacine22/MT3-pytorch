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
        },
    },
    "symbolic": {
        "codec": {
            "steps_per_second": 125,
            "max_shift_ms": 2048,
            "num_velocity_bins": 1,
        }
    },
}

__all__ = ["PROJECT_CONFIG", "PROJECT_ROOT"]

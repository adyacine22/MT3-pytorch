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
            "num_velocity_bins": 32,
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
            "default_profile": "light",
            "profiles": {
                "full": {
                    "chunk_device": "cuda:0",
                    "chunk_storage": "per_track",
                    "augment_enabled": True,
                    "augment_profiles": ["noise", "eq", "rir", "gain"],
                    "batch_size": 512,
                    "max_tokenize_workers": 32,
                    "store_tokens": True,
                    "track_workers": 1,
                    "datasets": ["maestro", "slakh_full_mix", "slakh_stem"],
                    "max_tracks_per_dataset": 5,
                    "max_chunks_per_track": None,
                    "dry_run": False,
                    "overwrite": True,
                    "store_spectrogram": True,
                    "write_manifest": False,
                    "skip_existing": True,
                },
                "light": {
                    "chunk_device": "auto",
                    "chunk_storage": "per_track",
                    "augment_enabled": False,
                    "augment_profiles": ["none"],
                    "batch_size": 512,
                    "store_tokens": False,
                    "track_workers": 32,
                    "datasets": ["maestro", "slakh_full_mix", "slakh_stem"],
                    "max_tracks_per_dataset": None,
                    "max_chunks_per_track": None,
                    "dry_run": False,
                    "overwrite": True,
                    "store_spectrogram": False,
                    "write_manifest": False,
                    "skip_existing": False,
                },
            },
        }
    },
    "training": {
        "dataloader": {
            "batch_size": 256,
            "num_workers": 32,
            "pin_memory": True,
            "temperature": 10/3,
            "feature_type": "waveform",
            "load_tokens": True,
            "max_examples_per_mix": 4,
            "shard_cache_size": 128,
            "compute_log_mel_in_collate": False,
            "prefetch_factor": 8,
            "max_midi_cache_size": 2048,
        },
        "optimization": {
            "optimizer": {
                "type": "adamw",
                "lr": 5e-4,
                "betas": [0.9, 0.999],
                "weight_decay": 0.01,
                "eps": 1e-8,
            },
            "gradient_clipping": 1.0,
        },
        "scheduler": {
            "type": "cosine",
            "warmup_steps": 4000,
            "min_lr_ratio": 0.1,
        },
        "loop": {
            "max_steps": 200000,
            "val_every_steps": 2000,
            "log_every_steps": 100,
            "val_max_batches": 32,
        },
        "precision": {
            "dtype": "bf16",
            "use_grad_scaler": False,
        },
        "augmentation": {
            "enabled": False,
            "profiles": ["none"],
            "probability": 1.0,
        },
        "runtime": {
            "device": "cuda",
            "seed": 42,
            "deterministic": False,
            "cudnn_benchmark": True,
            "compile": {
                "enabled": True,
                "mode": "default",
                "fullgraph": False,
            },
        },
        "checkpointing": {
            "save_dir": "checkpoints/mt3_step1",
            "save_every_steps": 5000,
            "keep_last": 3,
            "keep_best": 1,
            "metric": "val_loss",
            "mode": "min",
        },
    },
    "model": {
        "t5_pytorch": {
            "vocab_size": 0,  # override at runtime when tokenizer is known
            "emb_dim": 512,
            "num_heads": 8,
            "num_encoder_layers": 8,
            "num_decoder_layers": 8,
            "head_dim": 64,
            "mlp_dim": 1024,
            "mlp_activations": ["gelu", "linear"],
            "dropout_rate": 0.1,
            "logits_via_embedding": False,
            "max_position_embeddings": 2048,
            "input_depth": 512,
        }
    },
}

__all__ = ["PROJECT_CONFIG", "PROJECT_ROOT"]

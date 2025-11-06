"""
Training configuration for MT3-PyTorch.
Optimized for A100 80GB GPU with cached datasets.
"""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoint"
LOG_DIR = PROJECT_ROOT / "test_outputs"

# Create directories if they don't exist
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

# Cache directories (optimized layout by default)
MAESTRO_CACHE_TRAIN = PROJECT_ROOT / "cache" / "maestro_opt_fp16" / "train"
MAESTRO_CACHE_VAL = PROJECT_ROOT / "cache" / "maestro_opt_fp16" / "validation"
SLAKH_CACHE_TRAIN = PROJECT_ROOT / "cache" / "slakh_opt_fp16" / "train"
SLAKH_CACHE_VAL = PROJECT_ROOT / "cache" / "slakh_opt_fp16" / "validation"


# ==============================================================================
# A100 OPTIMIZED TRAINING CONFIGURATIONS
# ==============================================================================

# A100 Pretraining - Full (Production)
A100_PRETRAINING_FULL = {
    "phase": "pretrain",
    "config_name": "A100_pretraining_full",
    
    # Dataset configuration - use cached datasets
    "use_cached_data": False,
    "maestro_cache_dir": str(MAESTRO_CACHE_TRAIN),
    "maestro_val_cache_dir": str(MAESTRO_CACHE_VAL),
    "slakh_cache_dir": str(SLAKH_CACHE_TRAIN),
    "slakh_val_cache_dir": str(SLAKH_CACHE_VAL),
    "use_maestro": True,
    "use_slakh_stems": True,
    "use_slakh_mix": True,
    "use_slakh_mixed": True,  # Disable for speed
    "maestro_max_samples": None,  # Use all available
    "slakh_max_tracks": None,  # Use all available
    
    # Memory management for large datasets
    "cache_size": 400,  # LRU cache size - reduced to fit 200GB host RAM targets
    "preload_all": True,  # Keep LRU/on-demand for very large full pretraining
    # Validation-specific preload override (default: do NOT preload validation)
    "preload_all_val": False,
    
    # Audio mixing for multitrack learning
    "use_audio_mixing": True,
    "max_examples_per_mix": 8,
    "mix_probability": 1.0,
    
    # Temperature-based sampling (balanced)
    "temperatures": {
        "maestro": 3.0,
        "slakh_stems": 3.0,
        "slakh_mix": 3.0,
        "slakh_mixed": 3.0,
    },
    
    # Training hyperparameters (matching legacy MT3)
    "batch_size": 128,  # A100 can handle this
    "gradient_accumulation_steps": 8,  # Effective batch = 128 * 8 = 1024
    "max_steps": 500000,  # Full pretraining (LEGACY: 500k)
    "log_interval": 100,
    "val_interval": 5000,
    "save_interval": 5000,  # Match legacy checkpoint period
    
    # Optimizer (matching legacy MT3)
    "learning_rate": 0.001,  # LEGACY: 0.001 constant LR
    "weight_decay": 0.1,
    "gradient_clip_norm": 3.0,
    "warmup_steps": 1000,
    "scheduler_type": "constant",  # LEGACY: constant (no decay!)
    "optimizer_betas": (0.9, 0.98),
    "label_smoothing": 0.1,  # LEGACY: 0.1
    
        # A100 optimizations
    "use_mixed_precision": True,
    "use_fused_optimizer": True,
    "use_torch_compile": False,
    
    # Data loading (optimized for medium test dataset)
        # Data loading
    "num_workers": 24,  # Use more workers for machines with many CPU cores
    "pin_memory": True,
    "prefetch_factor": 4,
    "persistent_workers": True,
    
    # Validation
    
    # Validation
    "val_max_samples": None,
    
    # Checkpointing
    "checkpoint_dir": str(CHECKPOINT_DIR),
    "log_dir": str(LOG_DIR),
}

# A100 Pretraining - Test (Quick validation)
A100_PRETRAINING_TEST = {
    "phase": "pretrain",
    "config_name": "A100_pretraining_test",
    
    # Dataset configuration - small subset
    "use_cached_data": True,
    "maestro_cache_dir": str(MAESTRO_CACHE_TRAIN),
    "maestro_val_cache_dir": str(MAESTRO_CACHE_VAL),
    "slakh_cache_dir": str(SLAKH_CACHE_TRAIN),
    "slakh_val_cache_dir": str(SLAKH_CACHE_VAL),
    "use_maestro": True,
    "use_slakh_stems": True,
    "use_slakh_mix": True,
    "use_slakh_mixed": True,
    "maestro_max_samples": 100,  # Increased sample size for stronger cache impact tests
    "slakh_max_tracks": 100,  # Increased track count to exercise larger caches
    
    # Memory management (moderate dataset)
    "cache_size": 200,  # Used only when cached datasets are enabled
    "preload_all": True,
    # Validation-specific preload override (default: do NOT preload validation)
    "preload_all_val": True,
    
    # Audio mixing
    "use_audio_mixing": False,
    "max_examples_per_mix": 8,
    "mix_probability": 1.0,
    
    # Temperature-based sampling
    "temperatures": {
        "maestro": 3.0,
        "slakh_stems": 3.0,
        "slakh_mix": 3.0,
        "slakh_mixed": 3.0,
    },
    
    
    # Training hyperparameters (same as full but fewer steps)
    "batch_size": 256,
    "gradient_accumulation_steps": 4,  # Effective batch = 1024
    "max_steps": 10,  # Quick test - reduced from 500
    "log_interval": 5,  # Log more frequently for testing
    "val_interval": 20,
    "save_interval": 100,
    
    # Optimizer
    "learning_rate": 1e-4,
    "weight_decay": 0.1,
    "gradient_clip_norm": 3.0,
    "warmup_steps": 50,
    "scheduler_type": "constant",  # Match production
    "optimizer_betas": (0.9, 0.98),
    "label_smoothing": 0.1,
    
    # A100 optimizations
    "use_mixed_precision": True,
    "use_fused_optimizer": True,
    "use_torch_compile": True,
    
    # Data loading (reduced for small test dataset)
    "num_workers": 0,
    "pin_memory": True,
    "prefetch_factor": 4,  # Lower prefetch to reduce per-worker queue memory
    "persistent_workers": True,  # Keep workers alive to avoid reinitialization
    
    # Validation
    "val_max_samples": 20,
    
    # Checkpointing
    "checkpoint_dir": str(CHECKPOINT_DIR),
    "log_dir": str(LOG_DIR),
}

# A100 Finetuning - Full (Production)
A100_FINETUNING_FULL = {
    "phase": "finetune",
    "config_name": "A100_finetuning_full",
    
    # Dataset configuration
    "use_cached_data": True,
    "maestro_cache_dir": str(MAESTRO_CACHE_TRAIN),
    "maestro_val_cache_dir": str(MAESTRO_CACHE_VAL),
    "slakh_cache_dir": str(SLAKH_CACHE_TRAIN),
    "slakh_val_cache_dir": str(SLAKH_CACHE_VAL),
    "use_maestro": True,
    "use_slakh_stems": True,
    "use_slakh_mix": True,
    "use_slakh_mixed": True,
    "maestro_max_samples": None,
    "slakh_max_tracks": None,
    
    # NO audio mixing for finetuning
    "use_audio_mixing": False,
    "max_examples_per_mix": 1,
    "mix_probability": 0.0,
    
    # Temperature-based sampling
    "temperatures": {
        "maestro": 3.0,
        "slakh_stems": 3.0,
        "slakh_mix": 3.0,
    },
    
    # Training hyperparameters (smaller batch for finetuning)
    "batch_size": 128,
    "gradient_accumulation_steps": 2,  # Effective batch = 128 * 2 = 256
    "max_steps": 50000,  # Full finetuning
    "log_interval": 100,
    "val_interval": 2000,
    "save_interval": 5000,
    
    # Optimizer (lower LR for finetuning)
    "learning_rate": 5e-4,  # Lower for finetuning
    "weight_decay": 0.1,
    "gradient_clip_norm": 3.0,
    "warmup_steps": 500,
    "scheduler_type": "constant",  # Constant LR
    "optimizer_betas": (0.9, 0.98),
    "label_smoothing": 0.1,
    
    # A100 optimizations
    "use_mixed_precision": True,
    "use_fused_optimizer": True,
    "use_torch_compile": False,
    
    # Data loading
    "num_workers": 6,
    "pin_memory": True,
    "prefetch_factor": 2,
    "persistent_workers": True,
    
    # Validation
    "val_max_samples": 100,
    
    # Restore from pretrain checkpoint
    "restore_from": str(CHECKPOINT_DIR / "pretrain_final.pt"),
    
    # Checkpointing
    "checkpoint_dir": str(CHECKPOINT_DIR),
    "log_dir": str(LOG_DIR),
}

# A100 Finetuning - Test (Quick validation)
A100_FINETUNING_TEST = {
    "phase": "finetune",
    "config_name": "A100_finetuning_test",
    
    # Dataset configuration - small subset
    "use_cached_data": True,
    "maestro_cache_dir": str(MAESTRO_CACHE_TRAIN),
    "maestro_val_cache_dir": str(MAESTRO_CACHE_VAL),
    "slakh_cache_dir": str(SLAKH_CACHE_TRAIN),
    "slakh_val_cache_dir": str(SLAKH_CACHE_VAL),
    "use_maestro": True,
    "use_slakh_stems": True,
    "use_slakh_mix": True,
    "use_slakh_mixed": False,
    "maestro_max_samples": 50,
    "slakh_max_tracks": 20,
    
    # NO audio mixing
    "use_audio_mixing": False,
    "max_examples_per_mix": 1,
    "mix_probability": 0.0,
    
    # Temperature-based sampling
    "temperatures": {
        "maestro": 3.0,
        "slakh_stems": 3.0,
        "slakh_mix": 3.0,
    },
    
    # Training hyperparameters
    "batch_size": 128,
    "gradient_accumulation_steps": 2,  # Effective batch = 256
    "max_steps": 500,  # Quick test
    "log_interval": 10,
    "val_interval": 100,
    "save_interval": 200,
    
    # Optimizer
    "learning_rate": 5e-4,
    "weight_decay": 0.1,
    "gradient_clip_norm": 3.0,
    "warmup_steps": 50,
    "scheduler_type": "constant",
    "optimizer_betas": (0.9, 0.98),
    "label_smoothing": 0.1,
    
    # A100 optimizations
    "use_mixed_precision": True,
    "use_fused_optimizer": True,
    "use_torch_compile": False,
    
    # Data loading
    "num_workers": 6,
    "pin_memory": True,
    "prefetch_factor": 2,
    "persistent_workers": True,
    
    # Validation
    "val_max_samples": 20,
    
    # Optional: restore from pretrain if exists
    "restore_from": None,  # Set manually if needed
    
    # Checkpointing
    "checkpoint_dir": str(CHECKPOINT_DIR),
    "log_dir": str(LOG_DIR),
}


# ==============================================================================
# CONFIGURATION GETTER
# ==============================================================================

def get_config(config_name="A100_pretraining_test"):
    """
    Get training configuration by name.

    Args:
        config_name: Name of configuration
            - 'A100_pretraining_full': Full pretraining (165k steps, batch=1024)
            - 'A100_pretraining_test': Test pretraining (500 steps, batch=1024)
            - 'A100_finetuning_full': Full finetuning (50k steps, batch=256)
            - 'A100_finetuning_test': Test finetuning (500 steps, batch=256)

    Returns:
        dict: Training configuration
    """
    configs = {
        "A100_pretraining_full": A100_PRETRAINING_FULL,
        "A100_pretraining_test": A100_PRETRAINING_TEST,
        "A100_finetuning_full": A100_FINETUNING_FULL,
        "A100_finetuning_test": A100_FINETUNING_TEST,
    }

    if config_name not in configs:
        raise ValueError(
            f"Unknown config: {config_name}. "
            f"Available: {list(configs.keys())}"
        )

    return configs[config_name].copy()


def update_config(base_config, **kwargs):
    """
    Update configuration with custom parameters.

    Args:
        base_config: Base configuration dict
        **kwargs: Parameters to update

    Returns:
        dict: Updated configuration
    """
    config = base_config.copy()
    config.update(kwargs)
    return config

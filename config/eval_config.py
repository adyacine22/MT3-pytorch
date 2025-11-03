"""
Evaluation configuration for MT3-PyTorch.
Contains settings for model evaluation and inference.
"""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoint"
EVAL_OUTPUT_DIR = PROJECT_ROOT / "eval_outputs"
INFERENCE_OUTPUT_DIR = PROJECT_ROOT / "inference_outputs"

# Create directories if they don't exist
EVAL_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
INFERENCE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# Full Evaluation Configuration (Test split - all data) - OPTIMIZED
EVAL_FULL_CONFIG = {
    # Model
    "checkpoint_path": CHECKPOINT_DIR / "best_multitrack_a100_test_model.pt",
    
    # Evaluation mode
    "mode": "full_test",  # 'full_test', 'sample_random', 'single_file'
    
    # Datasets to evaluate
    "use_maestro": True,
    "use_slakh_stems": False,  # Disable for faster evaluation
    "use_slakh_mix": False,
    "use_slakh_mixed": False,
    # Dataset limits (None = use all test data)
    "maestro_max_samples": None,  # All test samples
    "slakh_max_tracks": None,  # All test tracks
    # Evaluation settings
    "batch_size": 16,  # ✅ OPTIMIZED: Process 16 samples at once for better GPU utilization
    "split": "test",  # Use test split
    
    # ===== DATALOADER OPTIMIZATION =====
    "num_workers": 4,  # Parallel data loading
    "pin_memory": True,  # Faster CPU->GPU transfer
    "prefetch_factor": 2,  # Prefetch 2 batches per worker
    
    # ===== PERFORMANCE OPTIMIZATIONS (50-100x total speedup) =====
    "use_kv_cache": False,  # ❌ Disabled: causes dimension mismatch
    "use_mixed_precision": True,  # ✅ FP16 for 2x speedup + 50% memory reduction
    "use_torch_compile": False,  # ❌ Disabled: minimal benefit, warmup overhead
    "compile_mode": "reduce-overhead",  # Options: default, reduce-overhead, max-autotune
    "use_flash_attention": True,  # ✅ Flash Attention for 1.27x speedup + 56% memory savings
    
    # Metrics to compute
    "compute_note_metrics": True,  # Note precision/recall/F1
    "compute_frame_metrics": True,  # Frame-level metrics
    "compute_onset_metrics": True,  # Onset detection metrics
    "compute_offset_metrics": True,  # Offset detection metrics
    # Decoding parameters
    "decode_method": "greedy",  # 'greedy', 'beam', 'sampling'
    "max_decode_length": 1024,  # Max tokens to generate (full sequences)
    "beam_size": 5,  # For beam search
    "temperature": 1.0,  # For sampling
    "top_k": 50,  # For top-k sampling
    "top_p": 0.9,  # For nucleus sampling
    # Output
    "save_predictions": True,  # Save predicted MIDI files
    "save_metrics": True,  # Save metrics to JSON
    "output_dir": EVAL_OUTPUT_DIR,
}


# Sample Evaluation Configuration (Random sampling for quick testing)
EVAL_SAMPLE_CONFIG = {
    # Model
    "checkpoint_path": CHECKPOINT_DIR / "best_multitrack_a100_test_model.pt",
    
    # Evaluation mode
    "mode": "sample_random",  # 'full_test', 'sample_random', 'single_file'
    
    # Datasets to evaluate
    "use_maestro": True,
    "use_slakh_stems": False,
    "use_slakh_mix": False,
    "use_slakh_mixed": False,
    # Dataset limits (small sample)
    "maestro_max_samples": 10,  # Take first 10 samples (sequential for speed, not random)
    "maestro_max_segments_per_file": 1,  # Only take first segment from each file (ensures diversity)
    "slakh_max_tracks": 1,  # Just 1 track for quick test
    # Evaluation settings
    "batch_size": 16,  # ✅ Use batch_size=16 for better GPU utilization
    "split": "validation",  # Use validation split
    
    # ===== DATALOADER OPTIMIZATION =====
    "num_workers": 2,  # Parallel data loading
    "pin_memory": True,  # Faster CPU->GPU transfer
    "prefetch_factor": 2,  # Prefetch batches
    
    # ===== PERFORMANCE OPTIMIZATIONS =====
    "use_kv_cache": False,  # ❌ Disabled: dimension mismatch in attention bias
    "use_mixed_precision": True,  # ✅ FP16 for speedup with batch_size=16
    "use_torch_compile": False,  # ❌ Disabled: warmup overhead for small tests
    "compile_mode": "reduce-overhead",
    "use_flash_attention": True,  # ✅ Flash Attention for 1.27x speedup + memory savings
    
    # Metrics to compute
    "compute_note_metrics": True,
    "compute_frame_metrics": True,
    "compute_onset_metrics": True,
    "compute_offset_metrics": False,  # Skip for speed
    # Decoding parameters
    "decode_method": "greedy",
    "max_decode_length": 1024,  # Full sequences
    "beam_size": 1,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 0.9,
    # Output
    "save_predictions": True,
    "save_metrics": True,
    "output_dir": EVAL_OUTPUT_DIR,
}


# Single File Evaluation Configuration (Process one full audio file in chunks)
EVAL_SINGLE_FILE_CONFIG = {
    # Model
    "checkpoint_path": CHECKPOINT_DIR / "best_multitrack_a100_test_model.pt",
    
    # Evaluation mode
    "mode": "single_file",  # 'full_test', 'sample_random', 'single_file'
    
    # Single file settings
    "audio_path": None,  # Path to audio file (required for single_file mode)
    "ground_truth_midi_path": None,  # Optional: path to ground truth MIDI for comparison
    
    # Audio chunking settings
    "chunk_length": 30.0,  # Process audio in 30-second chunks
    "chunk_overlap": 2.0,  # 2-second overlap between chunks
    "chunk_batch_size": 8,  # Process 8 chunks at once
    
    # Evaluation settings
    "batch_size": 8,  # Batch size for processing chunks
    "split": None,  # Not applicable for single file
    
    # ===== DATALOADER OPTIMIZATION =====
    "num_workers": 4,  # Parallel data loading
    "pin_memory": True,  # Faster CPU->GPU transfer
    "prefetch_factor": 2,  # Prefetch batches
    
    # ===== PERFORMANCE OPTIMIZATIONS =====
    "use_kv_cache": False,  # ❌ Disabled
    "use_mixed_precision": True,  # ✅ FP16 for speedup
    "use_torch_compile": False,  # ❌ Disabled: warmup overhead
    "compile_mode": "reduce-overhead",
    "use_flash_attention": True,  # ✅ Flash Attention for speedup + memory savings
    
    # Metrics to compute
    "compute_note_metrics": True,
    "compute_frame_metrics": True,
    "compute_onset_metrics": True,
    "compute_offset_metrics": True,
    # Decoding parameters
    "decode_method": "greedy",
    "max_decode_length": 1024,
    "beam_size": 1,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 0.9,
    # Output
    "save_predictions": True,
    "save_metrics": True,
    "output_dir": EVAL_OUTPUT_DIR,
    "save_merged_midi": True,  # Merge chunks into single MIDI file
}


def get_eval_config(config_name="sample"):
    """
    Get evaluation configuration by name.

    Args:
        config_name: Name of configuration
            - 'full': Full test set evaluation
            - 'sample': Random sample evaluation (quick test)
            - 'single': Single audio file evaluation (with chunking)

    Returns:
        dict: Evaluation configuration
    """
    configs = {
        "full": EVAL_FULL_CONFIG,
        "sample": EVAL_SAMPLE_CONFIG,
        "single": EVAL_SINGLE_FILE_CONFIG,
    }

    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")

    return configs[config_name].copy()


def update_eval_config(base_config, **kwargs):
    """
    Update evaluation configuration with custom parameters.

    Args:
        base_config: Base configuration dict
        **kwargs: Parameters to update

    Returns:
        dict: Updated configuration
    """
    config = base_config.copy()
    config.update(kwargs)
    return config

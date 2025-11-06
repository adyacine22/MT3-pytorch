"""
Cached SLAKH dataset loaders using preprocessed audio.
Eliminates disk I/O bottleneck by loading from pre-cached .pt files.

HYBRID APPROACH (matching legacy MT3 pipeline):
Returns audio FRAMES instead of mel-spectrograms.
Mel-spectrogram computation happens on GPU in training loop for maximum speed.

Memory Management Strategies:
1. Small datasets (< 100 files): Load all into memory
2. Large datasets (> 100 files): Use LRU cache with configurable size
"""

import os
import torch
import random
import psutil
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional
from functools import lru_cache
from data.lru_cache import LRUCache
import logging

logger = logging.getLogger(__name__)


def _load_cached_pt(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def _ensure_token_tensor(tokens) -> torch.Tensor:
    if isinstance(tokens, torch.Tensor):
        return tokens.to(dtype=torch.long, device="cpu").flatten()
    return torch.as_tensor(tokens, dtype=torch.long, device="cpu").flatten()


def _concat_token_list(token_seq) -> torch.Tensor:
    if not token_seq:
        return torch.empty(0, dtype=torch.long)
    tensors = []
    for token in token_seq:
        tensors.append(_ensure_token_tensor(token))
    if not tensors:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(tensors, dim=0)

from data.constants import (
    DEFAULT_NUM_MEL_BINS, DEFAULT_SAMPLE_RATE, FFT_SIZE,
    DEFAULT_HOP_WIDTH, MEL_FMIN, MEL_FMAX
)
from data.dataset_utils import FrameProcessingMixin
from data.spectrogram import MelSpectrogram

# Constants
TOKEN_PAD = 0
MEL_LENGTH = 256  # Number of frames per chunk
EVENT_LENGTH = 1024

# Memory management thresholds
PRELOAD_THRESHOLD = 100  # Files below this are fully loaded into memory
DEFAULT_CACHE_SIZE = 50  # LRU cache size for large datasets

# Dynamic memory management
def get_available_memory_gb():
    """Get available system RAM in GB."""
    try:
        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3)  # Convert to GB
    except:
        return 32.0  # Default assumption if psutil fails


def auto_determine_strategy(num_files, estimated_mb_per_file=20, preload_threshold=PRELOAD_THRESHOLD, num_workers=1):
    """
    Automatically determine loading strategy based on available RAM.
    
    Args:
        num_files: Number of files in dataset
        estimated_mb_per_file: Estimated memory per file in MB
        preload_threshold: Threshold for automatic preload
        num_workers: Number of DataLoader workers (each gets its own cache!)
        
    Returns:
        tuple: (should_preload, optimal_cache_size)
    """
    # Get available RAM
    available_gb = get_available_memory_gb()
    available_mb = available_gb * 1024
    
    # Estimate total memory needed for full preload
    estimated_total_mb = num_files * estimated_mb_per_file
    
    # Reserve 10% of RAM for other processes and training (more aggressive for high-RAM systems)
    # For systems with <64GB RAM, use 30% safety margin
    # For systems with >500GB RAM, use 10% safety margin (you have tons of headroom!)
    if available_gb > 500:
        usable_mb = available_mb * 0.90  # Use 90% on high-RAM systems
    elif available_gb > 128:
        usable_mb = available_mb * 0.80  # Use 80% on medium-RAM systems
    else:
        usable_mb = available_mb * 0.70  # Use 70% on low-RAM systems
    
    # CRITICAL: Account for multiple DataLoader workers
    # Each worker gets its own process with its own LRU cache!
    if num_workers > 1:
        usable_mb = usable_mb // num_workers  # Divide RAM across workers
        # Add extra safety margin for worker overhead
        usable_mb = int(usable_mb * 0.8)  # Use only 80% of allocated RAM per worker
    
    # Decision logic
    if num_files < preload_threshold:
        # Small dataset - always preload if it fits
        if estimated_total_mb < usable_mb:
            return True, None  # Preload all
        else:
            # Even small dataset too large - use cache
            cache_size = max(10, int(usable_mb / estimated_mb_per_file))
            return False, cache_size
    else:
        # Large dataset - use LRU cache
        # Calculate optimal cache size based on available RAM
        optimal_cache_size = int(usable_mb / estimated_mb_per_file)
        
        # CRITICAL: Cache should never exceed dataset size!
        # If we have 100 files, no point caching 5000 slots
        optimal_cache_size = min(optimal_cache_size, num_files)
        
        # Clamp between reasonable bounds (increased max for high-RAM systems)
        optimal_cache_size = max(50, min(optimal_cache_size, 5000))
        
        return False, optimal_cache_size


class CachedSLAKHStemDataset(Dataset, FrameProcessingMixin):
    """Load SLAKH stems from cached .pt files with intelligent memory management."""

    def __init__(self, cache_dir: Optional[str] = None, split: str = "train", max_tracks: Optional[int] = None,
                 cache_size: Optional[int] = None, preload_all: Optional[bool] = None, num_workers: int = 1):
        """
        Args:
            cache_dir: Directory with cached .pt files
            split: 'train', 'validation', or 'test'
            max_tracks: Limit number of tracks loaded
            cache_size: LRU cache size (default: auto-detect based on RAM). Only used if not preloading all.
            preload_all: Force preloading all data into memory (default: auto-detect based on size and RAM)
            num_workers: Number of DataLoader workers (for RAM allocation)
        """
        if cache_dir is None:
            cache_dir = f"/cluster/home/ahmedyz/yassine/MT3-pytorch/cache/slakh/{split}_fp16"
        
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.num_workers = num_workers  # Store for logging
        
        # Find all .pt files
        cache_files = sorted(self.cache_dir.glob("*.pt"))
        if max_tracks:
            cache_files = cache_files[:max_tracks]
        
        self.cache_files = [str(f) for f in cache_files]
        num_files = len(self.cache_files)
        
        # Auto-determine strategy based on available RAM
        # Estimate: SLAKH stems ~180 MB per file (includes all stems)
        auto_preload, auto_cache_size = auto_determine_strategy(
            num_files=num_files,
            estimated_mb_per_file=180,
            preload_threshold=PRELOAD_THRESHOLD,
            num_workers=num_workers
        )
        
        # Decide on memory strategy (user can override)
        # preload_all semantics:
        #   None -> auto-detect (use loader's auto_preload decision)
        #   True -> force preload all
        #   False -> force no-preload (use on-demand LRU)
        if preload_all is None:
            self.preload_all = auto_preload
        else:
            self.preload_all = bool(preload_all)
        
        # Determine cache size
        if cache_size is None:
            cache_size = auto_cache_size if auto_cache_size else DEFAULT_CACHE_SIZE
        
        if self.preload_all:
            available_gb = get_available_memory_gb()
            logger.info("[SLAKH][%s] Preloading %d cached files into memory (small dataset). Available RAM: %.1f GB",
                        self.split, num_files, available_gb)
            # Pre-load ALL data into memory
            self.cached_data = []
            self.stem_index = []
            
            for track_idx, cache_file in enumerate(self.cache_files):
                try:
                    data = torch.load(cache_file, map_location='cpu', weights_only=False)
                    self.cached_data.append(data)
                    num_stems = len(data['stems_audio'])
                    for stem_idx in range(num_stems):
                        self.stem_index.append((track_idx, stem_idx))
                except Exception as e:
                    print(f"Warning: Could not load {cache_file}: {e}")
            
            logger.info("[SLAKH][%s] ✓ Loaded %d tracks with %d stems into memory",
                        self.split, len(self.cached_data), len(self.stem_index))
        else:
            # Use LRU cache for large datasets
            available_gb = get_available_memory_gb()
            logger.info("[SLAKH][%s] Using LRU cache (size=%d) for %d files (large dataset). Available RAM: %.1f GB. DataLoader workers: %d",
                        self.split, cache_size, num_files, available_gb, self.num_workers)
            
            # Build stem index without loading full data (we still need to inspect files once)
            self.stem_index = []
            for track_idx, cache_file in enumerate(self.cache_files):
                try:
                    # Quick load to get stem count; we can't avoid reading a bit of each file
                    data = torch.load(cache_file, map_location='cpu', weights_only=False)
                    if 'stems_audio' in data and data['stems_audio']:
                        num_stems = len(data['stems_audio'])
                    elif 'stem_frames' in data and data['stem_frames']:
                        num_stems = len(data['stem_frames'])
                    else:
                        raise KeyError("Cached file missing stem data")
                    for stem_idx in range(num_stems):
                        self.stem_index.append((track_idx, stem_idx))
                except Exception as e:
                    logger.warning(f"Could not index {cache_file}: {e}")

            # Create a process-local LRU for on-demand stem loads
            self._file_cache = LRUCache(maxsize=cache_size, loader=_load_cached_pt)
            logger.info("[SLAKH][%s] ✓ Indexed %d stems; will load on-demand from disk with LRU (size=%d)",
                        self.split, len(self.stem_index), cache_size)

    def __len__(self):
        return len(self.stem_index)

    def __getitem__(self, idx):
        track_idx, stem_idx = self.stem_index[idx]
        
        # Get data based on strategy
        if self.preload_all:
            data = self.cached_data[track_idx]  # From preloaded memory
        else:
            # Load file on-demand via per-process LRU cache
            data = self._file_cache.get(self.cache_files[track_idx])
        
        tokens = data['stem_tokens'][stem_idx]

        if 'stem_frames' in data:
            frames_full = data['stem_frames'][stem_idx]
            if not isinstance(frames_full, torch.Tensor):
                frames_full = torch.tensor(frames_full)
            if frames_full.dtype == torch.float16:
                frames_full = frames_full.float()
            total_frames = frames_full.shape[0]
            if total_frames <= MEL_LENGTH:
                frames_chunk = frames_full
            else:
                start_frame = random.randint(0, total_frames - MEL_LENGTH)
                frames_chunk = frames_full[start_frame : start_frame + MEL_LENGTH]
            frames_padded = self._pad_frames(frames_chunk)
        else:
            audio = data['stems_audio'][stem_idx]
            if audio.dtype == torch.float16:
                audio = audio.float()
            audio_chunk = self._select_random_audio_chunk(audio)
            frames = self._split_audio_to_frames(audio_chunk)
            frames_padded = self._pad_frames(frames)

        tokens_padded = self._pad_tokens(tokens)
        
        return {
            "inputs": frames_padded,  # Frames, not mel-spectrograms
            "targets": tokens_padded,
        }
    def warm_cache_for_workers(self):
        if self.preload_all or not hasattr(self, "_file_cache"):
            return
        files_to_warm = self.cache_files[: self._file_cache.maxsize]
        logger.info(
            "[SLAKH][%s] Warming LRU cache with %d files before forking workers",
            self.split,
            len(files_to_warm),
        )
        for path in files_to_warm:
            try:
                self._file_cache.get(path)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "[SLAKH][%s] Warm cache failed for %s: %s",
                    self.split,
                    path,
                    exc,
                )

    def _select_random_audio_chunk(self, audio):
        """
        Randomly select a chunk of audio for training.
        Target chunk size: MEL_LENGTH frames = 256 * 128 hop = 32,768 samples
        """
        target_samples = MEL_LENGTH * DEFAULT_HOP_WIDTH  # 256 * 128 = 32,768 samples
        
        # If audio is shorter than target, return as-is (will be padded later)
        if len(audio) <= target_samples:
            return audio
        
        # Select random start position
        max_start = len(audio) - target_samples
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + target_samples
        
        return audio[start_idx:end_idx]
    
    # Note: _split_audio_to_frames, _pad_frames, _pad_tokens inherited from FrameProcessingMixin


class CachedSLAKHMixDataset(Dataset, FrameProcessingMixin):
    """Load SLAKH mix tracks from cached .pt files with intelligent memory management."""

    def __init__(self, cache_dir: Optional[str] = None, split: str = "train", max_tracks: Optional[int] = None,
                 cache_size: Optional[int] = None, preload_all: Optional[bool] = None, num_workers: int = 1):
        """
        Args:
            cache_dir: Directory with cached .pt files
            split: 'train', 'validation', or 'test'
            max_tracks: Limit number of tracks loaded
            cache_size: LRU cache size (default: auto-detect based on RAM)
            preload_all: Force preloading all data into memory (default: auto-detect based on size and RAM)
            num_workers: Number of DataLoader workers (for RAM allocation)
        """
        if cache_dir is None:
            cache_dir = f"/cluster/home/ahmedyz/yassine/MT3-pytorch/cache/slakh/{split}_fp16"
        
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.num_workers = num_workers
        
        cache_files = sorted(self.cache_dir.glob("*.pt"))
        if max_tracks:
            cache_files = cache_files[:max_tracks]
        
        self.cache_files = [str(f) for f in cache_files]
        num_files = len(self.cache_files)
        
        # Auto-determine strategy based on available RAM
        # Estimate: SLAKH mix ~175 MB per file (full mix with all stems)
        auto_preload, auto_cache_size = auto_determine_strategy(
            num_files=num_files,
            estimated_mb_per_file=175,
            preload_threshold=PRELOAD_THRESHOLD,
            num_workers=num_workers
        )
        
        # Decide on memory strategy (user can override)
        if preload_all is None:
            self.preload_all = auto_preload
        else:
            self.preload_all = bool(preload_all)
        
        # Determine cache size
        if cache_size is None:
            cache_size = auto_cache_size if auto_cache_size else DEFAULT_CACHE_SIZE
        
        if self.preload_all:
            available_gb = get_available_memory_gb()
            logger.info("[SLAKH-MIX][%s] Preloading %d mix cached files into memory (small dataset). Available RAM: %.1f GB",
                        self.split, num_files, available_gb)
            self.cached_data = []
            for cache_file in self.cache_files:
                try:
                    data = torch.load(cache_file, map_location='cpu', weights_only=False)
                    self.cached_data.append(data)
                except Exception as e:
                    print(f"Warning: Could not load {cache_file}: {e}")
            logger.info("[SLAKH-MIX][%s] ✓ Loaded %d mix tracks into memory",
                        self.split, len(self.cached_data))
        else:
            available_gb = get_available_memory_gb()
            logger.info("[SLAKH-MIX][%s] Using LRU cache (size=%d) for %d mix files. Available RAM: %.1f GB",
                        self.split, cache_size, num_files, available_gb)
            self._file_cache = LRUCache(maxsize=cache_size, loader=_load_cached_pt)
            logger.info("[SLAKH-MIX][%s] ✓ Ready to load %d files with LRU cache (size=%d)",
                        self.split, num_files, cache_size)
        
        self.melspectrogram = MelSpectrogram(
            DEFAULT_NUM_MEL_BINS,
            DEFAULT_SAMPLE_RATE,
            FFT_SIZE,
            DEFAULT_HOP_WIDTH,
            mel_fmin=MEL_FMIN,
            mel_fmax=MEL_FMAX,
        )

    def __len__(self):
        return len(self.cached_data) if self.preload_all else len(self.cache_files)

    def __getitem__(self, idx):
        # Get data based on strategy
        if self.preload_all:
            data = self.cached_data[idx]  # From preloaded memory
        else:
            data = self._file_cache.get(self.cache_files[idx])
        
        if 'mix_frames' in data:
            frames_full = data['mix_frames']
            if not isinstance(frames_full, torch.Tensor):
                frames_full = torch.tensor(frames_full)
            if frames_full.dtype == torch.float16:
                frames_full = frames_full.float()
            total_frames = frames_full.shape[0]
            if total_frames <= MEL_LENGTH:
                frames_chunk = frames_full
            else:
                start_frame = random.randint(0, total_frames - MEL_LENGTH)
                frames_chunk = frames_full[start_frame : start_frame + MEL_LENGTH]
            frames_padded = self._pad_frames(frames_chunk)
            if 'mix_tokens' in data:
                tokens = _ensure_token_tensor(data['mix_tokens'])
            else:
                tokens = _concat_token_list(data['stem_tokens'])
        else:
            audio = data['mix_audio']
            if audio.dtype == torch.float16:
                audio = audio.float()
            tokens = _concat_token_list(data['stem_tokens'])
            audio_chunk = self._select_random_audio_chunk(audio)
            frames = self._split_audio_to_frames(audio_chunk)
            frames_padded = self._pad_frames(frames)

        tokens_padded = self._pad_tokens(tokens)
        
        return {
            "inputs": frames_padded,  # Frames, not mel-spectrograms
            "targets": tokens_padded,
        }
    def warm_cache_for_workers(self):
        if self.preload_all or not hasattr(self, "_file_cache"):
            return
        files_to_warm = self.cache_files[: self._file_cache.maxsize]
        logger.info(
            "[SLAKH-MIX][%s] Warming LRU cache with %d files before forking workers",
            self.split,
            len(files_to_warm),
        )
        for path in files_to_warm:
            try:
                self._file_cache.get(path)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "[SLAKH-MIX][%s] Warm cache failed for %s: %s",
                    self.split,
                    path,
                    exc,
                )
    def _select_random_audio_chunk(self, audio):
        """
        Randomly select a chunk of audio for training.
        Target chunk size: MEL_LENGTH frames = 256 * 128 hop = 32,768 samples
        """
        target_samples = MEL_LENGTH * DEFAULT_HOP_WIDTH  # 256 * 128 = 32,768 samples
        
        # If audio is shorter than target, return as-is (will be padded later)
        if len(audio) <= target_samples:
            return audio
        
        # Select random start position
        max_start = len(audio) - target_samples
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + target_samples
        
        return audio[start_idx:end_idx]



class CachedSLAKHMixedDataset(Dataset, FrameProcessingMixin):
    """Load randomly mixed SLAKH stems from cached .pt files with intelligent memory management."""

    def __init__(self, cache_dir: Optional[str] = None, split: str = "train", max_tracks: Optional[int] = None,
                 min_stems: int = 1, max_stems: int = 4, cache_size: Optional[int] = None, preload_all: Optional[bool] = None):
        if cache_dir is None:
            cache_dir = f"/cluster/home/ahmedyz/yassine/MT3-pytorch/cache/slakh/{split}_fp16"
        
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.min_stems = min_stems
        self.max_stems = max_stems
        
        cache_files = sorted(self.cache_dir.glob("*.pt"))
        if max_tracks:
            cache_files = cache_files[:max_tracks]
        
        self.cache_files = [str(f) for f in cache_files]
        num_files = len(self.cache_files)
        
        # Decide on memory strategy
        # If preload_all is None, auto-detect based on small dataset threshold; otherwise respect explicit flag
        if preload_all is None:
            self.preload_all = (num_files <= PRELOAD_THRESHOLD)
        else:
            self.preload_all = bool(preload_all)
        
        if self.preload_all:
            logger.info("[SLAKH-MIXED][%s] Preloading %d mixed cached files into memory (small dataset)",
                        self.split, num_files)
            self.cached_data = []
            for cache_file in self.cache_files:
                try:
                    data = torch.load(cache_file, map_location='cpu', weights_only=False)
                    self.cached_data.append(data)
                except Exception as e:
                    print(f"Warning: Could not load {cache_file}: {e}")
            logger.info("[SLAKH-MIXED][%s] ✓ Loaded %d mixed tracks into memory",
                        self.split, len(self.cached_data))
        else:
            cache_size = cache_size or DEFAULT_CACHE_SIZE
            logger.info("[SLAKH-MIXED][%s] Using LRU cache (size=%d) for %d mixed files (large dataset)",
                        self.split, cache_size, num_files)
            self._file_cache = LRUCache(maxsize=cache_size, loader=_load_cached_pt)
            logger.info("[SLAKH-MIXED][%s] ✓ Ready to load %d files with LRU cache (size=%d)",
                        self.split, num_files, cache_size)
        
        self.melspectrogram = MelSpectrogram(
            DEFAULT_NUM_MEL_BINS,
            DEFAULT_SAMPLE_RATE,
            FFT_SIZE,
            DEFAULT_HOP_WIDTH,
            mel_fmin=MEL_FMIN,
            mel_fmax=MEL_FMAX,
        )

    def __len__(self):
        return len(self.cached_data) if self.preload_all else len(self.cache_files)

    def __getitem__(self, idx):
        # Get data based on strategy
        if self.preload_all:
            data = self.cached_data[idx]  # From preloaded memory
        else:
            data = self._file_cache.get(self.cache_files[idx])  # From LRU cache
        
        stems_audio = data['stems_audio']
        stem_tokens = data['stem_tokens']
        
        # Randomly select N stems to mix
        num_stems = random.randint(self.min_stems, min(self.max_stems, len(stems_audio)))
        selected_indices = random.sample(range(len(stems_audio)), num_stems)
        
        # Mix - keep as torch tensors
        max_len = max(len(stems_audio[i]) for i in selected_indices)
        mixed_audio = torch.zeros(max_len, dtype=torch.float32)
        for i in selected_indices:
            audio = stems_audio[i]
            # Auto-convert fp16 to float32 if needed
            if audio.dtype == torch.float16:
                audio = audio.float()
            mixed_audio[:len(audio)] += audio
        
        # Normalize
        max_val = torch.max(torch.abs(mixed_audio))
        if max_val > 0:
            mixed_audio = mixed_audio / max_val
        
        # Merge tokens
        tokens = _concat_token_list([stem_tokens[i] for i in selected_indices])
        
        # Select random chunk
        audio_chunk = self._select_random_audio_chunk(mixed_audio)
        
        # Split to frames (hybrid approach)
        frames = self._split_audio_to_frames(audio_chunk)
        frames_padded = self._pad_frames(frames)
        tokens_padded = self._pad_tokens(tokens)
        
        return {
            "inputs": frames_padded,  # Frames, not mel-spectrograms
            "targets": tokens_padded,
        }

    def _select_random_audio_chunk(self, audio):
        """
        Randomly select a chunk of audio for training.
        Target chunk size: MEL_LENGTH frames = 256 * 128 hop = 32,768 samples
        """
        target_samples = MEL_LENGTH * DEFAULT_HOP_WIDTH  # 256 * 128 = 32,768 samples
        
        # If audio is shorter than target, return as-is (will be padded later)
        if len(audio) <= target_samples:
            return audio
        
        # Select random start position
        max_start = len(audio) - target_samples
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + target_samples
        
        return audio[start_idx:end_idx]


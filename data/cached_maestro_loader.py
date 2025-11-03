"""
Cached MAESTRO v3 dataset loader using preprocessed audio.
Eliminates disk I/O bottleneck by loading from pre-cached .pt files.
Uses pure PyTorch for 77% performance improvement over numpy conversion.

HYBRID APPROACH (matching legacy MT3 pipeline):
Returns audio FRAMES instead of mel-spectrograms.
Mel-spectrogram computation happens on GPU in training loop for maximum speed.

Memory Management Strategies:
1. Small datasets (< 100 files): Load all into memory
2. Large datasets (> 100 files): Use LRU cache with configurable size

Runtime processing (following legacy MT3 pipeline):
1. Load cached audio + tokens
2. Select random chunk (256 frames = 32,768 samples)
3. Split chunk into frames (256 frames x 128 hop_width)
4. Extract tokens for chunk + add TIE events for sustained notes
5. Apply RLE to compress consecutive shifts
6. Return frames (mel-spectrogram computed on GPU later)
"""

import os
import torch
import random
import numpy as np
import psutil
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, List, Tuple, Set
from functools import lru_cache
from data.lru_cache import LRUCache
import logging

logger = logging.getLogger(__name__)

from data.constants import (
    DEFAULT_NUM_MEL_BINS, DEFAULT_SAMPLE_RATE, FFT_SIZE,
    DEFAULT_HOP_WIDTH, MEL_FMIN, MEL_FMAX, codec
)
from data import vocabularies
from data.dataset_utils import FrameProcessingMixin

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


def auto_determine_strategy(num_files, estimated_mb_per_file=30, preload_threshold=PRELOAD_THRESHOLD, num_workers=1):
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
        # If we have 200 files, no point caching 5000 slots
        optimal_cache_size = min(optimal_cache_size, num_files)
        
        # Clamp between reasonable bounds (increased max for high-RAM systems)
        optimal_cache_size = max(50, min(optimal_cache_size, 5000))
        
        return False, optimal_cache_size


class CachedMaestroDataset(Dataset, FrameProcessingMixin):
    """Load MAESTRO v3 tracks from cached .pt files with intelligent memory management."""

    def __init__(self, cache_dir: Optional[str] = None, max_tracks: Optional[int] = None,
                 cache_size: Optional[int] = None, preload_all: Optional[bool] = None, num_workers: int = 1):
        """
        Args:
            cache_dir: Directory with cached .pt files
            max_tracks: Limit number of tracks loaded
            cache_size: LRU cache size (default: auto-detect based on RAM). Only used if not preloading all.
            preload_all: Force preloading all data into memory (default: auto-detect based on size and RAM)
            num_workers: Number of DataLoader workers (for RAM allocation)
        """
        if cache_dir is None:
            cache_dir = "/cluster/home/ahmedyz/yassine/MT3-pytorch/cache/maestro/maestro_v3_fp16"
        
        self.cache_dir = Path(cache_dir)
        self.num_workers = num_workers  # Store for logging
        
        # Find all .pt files
        cache_files = sorted(self.cache_dir.glob("*.pt"))
        if max_tracks:
            cache_files = cache_files[:max_tracks]
        
        self.cache_files = [str(f) for f in cache_files]
        num_files = len(self.cache_files)
        
        # Auto-determine strategy based on available RAM
        # Estimate: MAESTRO ~30 MB per file
        auto_preload, auto_cache_size = auto_determine_strategy(
            num_files=num_files,
            estimated_mb_per_file=30,
            preload_threshold=PRELOAD_THRESHOLD,
            num_workers=num_workers
        )
        
        # Decide on memory strategy (user can override)
        # preload_all semantics:
        #   None -> auto-detect (use loader's auto_preload decision)
        #   True -> force preload all files into memory
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
            logger.info("[MAESTRO] Preloading %d cached files into memory (small dataset). Available RAM: %.1f GB",
                        num_files, available_gb)
            self.cached_data = []
            for cache_file in self.cache_files:
                try:
                    data = torch.load(cache_file, map_location='cpu', weights_only=False)
                    self.cached_data.append(data)
                except Exception as e:
                    logger.warning("[MAESTRO] Warning: Could not load %s: %s", cache_file, e)
            logger.info("[MAESTRO] ✓ Loaded %d MAESTRO tracks into memory", len(self.cached_data))
        else:
            # Load files on-demand without caching (avoids worker memory duplication)
            available_gb = get_available_memory_gb()
            logger.info("[MAESTRO] Loading %d MAESTRO files on-demand (no caching). Available RAM: %.1f GB. DataLoader workers: %d",
                        num_files, available_gb, self.num_workers)
            logger.info("[MAESTRO] ✓ Ready to load %d files on-demand from disk", num_files)
            # Create a process-local LRU for on-demand loads so repeated hits are fast.
            def _loader(path):
                return torch.load(path, map_location='cpu', weights_only=False)

            self._file_cache = LRUCache(maxsize=cache_size, loader=_loader)
            logger.info("[MAESTRO] LRU for on-demand loads initialized (size=%d)", cache_size)

    def __len__(self):
        return len(self.cached_data) if self.preload_all else len(self.cache_files)

    def __getitem__(self, idx):
        # Get data based on strategy
        if self.preload_all:
            data = self.cached_data[idx]  # From preloaded memory
        else:
            # Load file on-demand via per-process LRU cache
            data = self._file_cache.get(self.cache_files[idx])
        
        audio = data['audio']
        # Auto-convert fp16 to float32 if needed
        if audio.dtype == torch.float16:
            audio = audio.float()
        
        tokens = data['tokens']
        
        # Step 1: Select random audio chunk (256 frames = 32,768 samples)
        audio_chunk, chunk_start_sample = self._select_random_audio_chunk(audio)
        
        # Step 2: Split audio chunk into frames (legacy MT3 approach)
        # This matches: spectrograms.split_audio()
        frames = self._split_audio_to_frames(audio_chunk)
        
        # Step 3: Extract tokens for this chunk + add TIE events
        chunk_start_frame = chunk_start_sample // DEFAULT_HOP_WIDTH
        tokens_with_ties = self._extract_tokens_with_ties(
            tokens, 
            chunk_start_frame,
            num_frames=MEL_LENGTH
        )
        
        # Step 4: Apply RLE to compress consecutive shifts
        tokens_rle = self._apply_rle_encoding(tokens_with_ties)
        
        # Step 5: Pad frames and tokens
        frames_padded = self._pad_frames(frames)
        tokens_padded = self._pad_tokens(tokens_rle)
        
        # Return frames (NOT mel-spectrograms)
        # Mel-spectrogram will be computed on GPU in training loop
        return {
            "inputs": frames_padded,  # Shape: [MEL_LENGTH, DEFAULT_HOP_WIDTH] = [256, 128]
            "targets": tokens_padded,
        }

    def _select_random_audio_chunk(self, audio) -> Tuple[torch.Tensor, int]:
        """
        Randomly select a chunk of audio for training.
        Target chunk size: MEL_LENGTH frames = 256 * 128 hop = 32,768 samples
        
        Returns:
            (audio_chunk, start_sample_index)
        """
        target_samples = MEL_LENGTH * DEFAULT_HOP_WIDTH  # 256 * 128 = 32,768 samples
        
        # If audio is shorter than target, return from start
        if len(audio) <= target_samples:
            return audio, 0
        
        # Select random start position
        max_start = len(audio) - target_samples
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + target_samples
        
        return audio[start_idx:end_idx], start_idx
    
    def _extract_tokens_with_ties(
        self, 
        tokens: np.ndarray, 
        chunk_start_frame: int,
        num_frames: int
    ) -> List[int]:
        """
        Extract tokens for the selected chunk and add TIE events for sustained notes.
        
        Since we don't have event_start_indices in cache, we extract tokens by
        tracking time through SHIFT events.
        
        Args:
            tokens: Full token sequence for the track
            chunk_start_frame: Frame index where chunk starts  
            num_frames: Number of frames in chunk (typically 256)
            
        Returns:
            Token list with TIE events prepended for notes active at chunk start
        """
        # Convert to list for easier processing
        tokens_list = tokens.tolist() if isinstance(tokens, np.ndarray) else list(tokens)
        
        # Calculate time range for our chunk
        frames_per_second = DEFAULT_SAMPLE_RATE / DEFAULT_HOP_WIDTH  # ~125 frames/second
        chunk_start_time = chunk_start_frame / frames_per_second
        chunk_end_time = (chunk_start_frame + num_frames) / frames_per_second
        
        # Scan through tokens to find the time range
        tokens_before_chunk = []
        tokens_for_chunk = []
        tokens_after_chunk = []
        
        current_time = 0.0
        current_list = tokens_before_chunk  # Start with "before"
        
        for token in tokens_list:
            try:
                event = codec.decode_event_index(token)
            except:
                current_list.append(token)
                continue
            
            if event.type == 'shift':
                # Advance time
                shift_seconds = event.value / codec.steps_per_second
                current_time += shift_seconds
                
                # Check if we've crossed into the chunk
                if current_time >= chunk_start_time and current_list is tokens_before_chunk:
                    current_list = tokens_for_chunk
                elif current_time >= chunk_end_time and current_list is tokens_for_chunk:
                    current_list = tokens_after_chunk
                    # Can stop early if we have enough tokens
                    if len(tokens_for_chunk) > EVENT_LENGTH:
                        break
            
            current_list.append(token)
        
        # Extract TIE events from tokens before chunk
        tie_events = self._extract_tie_events(tokens_before_chunk)
        
        # Prepend TIE events to chunk tokens
        return tie_events + tokens_for_chunk
    
    def _extract_tie_events(self, tokens_before_chunk: List[int]) -> List[int]:
        """
        Extract TIE events for notes that are active (have NOTE_ON but no NOTE_OFF).
        
        This replicates legacy MT3's extract_state_events function.
        
        Args:
            tokens_before_chunk: All tokens that occurred before the selected chunk
            
        Returns:
            List of TIE event tokens: [PROGRAM, PITCH, TIE, PROGRAM, PITCH, TIE, ...]
        """
        # Track active notes: (pitch, program) -> velocity
        active_notes: dict[Tuple[int, int], int] = {}
        
        current_program = 0
        current_velocity = 0
        
        # Scan through tokens to find active notes
        i = 0
        while i < len(tokens_before_chunk):
            token = tokens_before_chunk[i]
            
            try:
                event = codec.decode_event_index(token)
            except:
                i += 1
                continue
            
            if event.type == 'program':
                current_program = event.value
            elif event.type == 'velocity':
                current_velocity = event.value
            elif event.type == 'pitch':
                # This is a NOTE_ON (velocity > 0) or NOTE_OFF (velocity == 1 in legacy)
                pitch = event.value
                if current_velocity > 1:  # NOTE_ON
                    active_notes[(pitch, current_program)] = current_velocity
                else:  # NOTE_OFF
                    active_notes.pop((pitch, current_program), None)
            elif event.type == 'drum':
                # Drums work similarly
                pitch = event.value
                if current_velocity > 1:  # NOTE_ON
                    active_notes[(pitch, current_program)] = current_velocity
                else:  # NOTE_OFF
                    active_notes.pop((pitch, current_program), None)
            
            i += 1
        
        # Create TIE events for active notes
        tie_tokens = []
        for (pitch, program), velocity in sorted(active_notes.items()):
            # Add: PROGRAM, PITCH, TIE
            tie_tokens.append(codec.encode_event(vocabularies.Event("program", program)))
            tie_tokens.append(codec.encode_event(vocabularies.Event("pitch", pitch)))
            tie_tokens.append(codec.encode_event(vocabularies.Event("tie", 0)))
        
        return tie_tokens
    
    def _apply_rle_encoding(self, tokens: List[int]) -> List[int]:
        """
        Apply run-length encoding to compress consecutive SHIFT events.
        
        This replicates legacy MT3's run_length_encode_shifts_fn.
        
        Args:
            tokens: Token list (with TIE events already added)
            
        Returns:
            Token list with consecutive shifts compressed
        """
        if not tokens:
            return tokens
        
        compressed = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            # Check if this is a shift token
            if codec.is_shift_event_index(token):
                # Decode to get shift value
                try:
                    event = codec.decode_event_index(token)
                    total_shift = event.value
                    
                    # Look ahead for more consecutive shifts
                    j = i + 1
                    while j < len(tokens) and codec.is_shift_event_index(tokens[j]):
                        next_event = codec.decode_event_index(tokens[j])
                        total_shift += next_event.value
                        j += 1
                    
                    # Encode the total shift (may need multiple tokens if > max_shift)
                    while total_shift > 0:
                        shift_val = min(total_shift, codec.max_shift_steps)
                        compressed.append(codec.encode_event(vocabularies.Event("shift", shift_val)))
                        total_shift -= shift_val
                    
                    i = j  # Skip the shifts we just compressed
                except:
                    # If decoding fails, just keep the token as-is
                    compressed.append(token)
                    i += 1
            else:
                # Not a shift, keep as-is
                compressed.append(token)
                i += 1
        
        return compressed
    
    # Note: _split_audio_to_frames, _pad_frames, and _pad_tokens are inherited from FrameProcessingMixin


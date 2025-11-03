#!/usr/bin/env python3
"""
Preprocess SLAKH2100 dataset into cached format.

FP32 (base format - no suffix on cache dir):
  Cache structure: /cache/slakh/[split]/[track_id].pt
  
Cache format (per sample):
  {
    'track_id': str,
    'mix_audio': torch.Tensor (float32 or float16),
    'stems_audio': list of torch.Tensor,
    'stem_tokens': list of np.ndarray,
    'stem_programs': np.ndarray,
    'stem_names': list,
    'metadata': dict with split, duration, sample_rate, etc.
  }
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import torch
import librosa
from tqdm import tqdm

import sys
sys.path.insert(0, '/cluster/home/ahmedyz/yassine/MT3-pytorch')

from data.constants import DEFAULT_SAMPLE_RATE, DEFAULT_HOP_WIDTH, codec
from data.slakh_loader import load_stem, parse_slakh_metadata
from data.utils import note_sequence_to_timed_events, timed_events_to_tokens

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_single_track(
    track_data: Tuple[Path, str],
    split_cache_dir: Path,
    skip_errors: bool = True,
    compression: str = 'none'
) -> Tuple[bool, str, float, str]:
    """
    Process single SLAKH track.
    
    Stores everything together: mix_audio, stems_audio, tokens, programs, metadata
    
    Args:
        track_data: (track_dir, split)
        compression: 'none' (fp32), 'fp16' (50% smaller)
    """
    try:
        track_dir, split = track_data
        track_id = track_dir.name
        
        stems_metadata = parse_slakh_metadata(str(track_dir))
        
        if not stems_metadata:
            return False, track_id, 0, "No stems found"
        
        stems_audio = []
        stem_tokens = []
        stem_programs = []
        stem_names = []
        
        # Load all stems for this track
        for stem_id, stem_info in stems_metadata.items():
            try:
                result = load_stem(str(track_dir), stem_id, stem_info)
                if result is None:
                    continue
                
                audio, note_seq = result
                
                events = note_sequence_to_timed_events(note_seq)
                num_frames = len(audio) // DEFAULT_HOP_WIDTH
                frame_times = np.arange(num_frames) / (DEFAULT_SAMPLE_RATE / DEFAULT_HOP_WIDTH)
                tokens, _, _ = timed_events_to_tokens(events, codec, frame_times)
                
                # Apply compression
                if compression == 'fp16':
                    audio = audio.astype(np.float16)
                # else: keep as float32
                
                stems_audio.append(audio)
                stem_tokens.append(np.array(tokens, dtype=np.int64))
                stem_programs.append(stem_info.get('program', 0))
                stem_names.append(stem_info.get('name', f'stem_{stem_id}'))
                
            except Exception as e:
                if not skip_errors:
                    raise
                continue
        
        if not stems_audio:
            return False, track_id, 0, "No valid stems"
        
        # Create mix from all stems
        max_len = max(len(a) for a in stems_audio)
        
        if compression == 'fp16':
            mix_audio = np.zeros(max_len, dtype=np.float16)
            # Convert stems to float16 if needed
            stems_audio_normalized = []
            for audio in stems_audio:
                if audio.dtype == np.float16:
                    stems_audio_normalized.append(audio)
                else:
                    stems_audio_normalized.append(audio.astype(np.float16))
                # Pad and add to mix
                padded = np.zeros(max_len, dtype=np.float16)
                padded[:len(audio)] = stems_audio_normalized[-1]
                mix_audio += padded
        else:
            mix_audio = np.zeros(max_len, dtype=np.float32)
            stems_audio_normalized = []
            for audio in stems_audio:
                if audio.dtype == np.float16:
                    audio = audio.astype(np.float32)
                stems_audio_normalized.append(audio)
                padded = np.zeros(max_len, dtype=np.float32)
                padded[:len(audio)] = audio
                mix_audio += padded
        
        # Normalize mix to prevent clipping
        mix_max = np.abs(mix_audio).max()
        if mix_max > 0:
            mix_audio = mix_audio / mix_max
        
        # Pad all stems to max length
        stems_audio_padded = []
        for audio in stems_audio:
            if len(audio) < max_len:
                if compression == 'fp16':
                    padded = np.zeros(max_len, dtype=np.float16)
                else:
                    padded = np.zeros(max_len, dtype=np.float32)
                padded[:len(audio)] = audio
                stems_audio_padded.append(padded)
            else:
                stems_audio_padded.append(audio)
        
        # Convert to torch tensors
        mix_tensor = torch.from_numpy(mix_audio)
        stems_tensors = [torch.from_numpy(a) for a in stems_audio_padded]
        
        # Save complete cache file with all data together
        cache_data = {
            'track_id': track_id,
            'mix_audio': mix_tensor,
            'stems_audio': stems_tensors,
            'stem_tokens': stem_tokens,
            'stem_programs': np.array(stem_programs, dtype=np.int64),
            'stem_names': stem_names,
            'metadata': {
                'sample_rate': DEFAULT_SAMPLE_RATE,
                'duration_seconds': max_len / DEFAULT_SAMPLE_RATE,
                'num_stems': len(stems_audio),
                'compression': compression,
                'source': 'slakh2100',
                'split': split,
            }
        }
        
        cache_file = split_cache_dir / f"{track_id}.pt"
        torch.save(cache_data, cache_file)
        
        cache_size_mb = cache_file.stat().st_size / (1024 * 1024)
        return True, track_id, cache_size_mb, ""
        
    except Exception as e:
        return False, track_data[0].name, 0, str(e)


def preprocess_slakh(
    root_dir: str,
    cache_dir: str,
    split: str = 'train',
    max_samples=None,
    skip_errors: bool = True,
    num_workers=None,
    compression: str = 'none'
) -> Dict:
    """Preprocess SLAKH2100 dataset with split organization"""
    
    root_path = Path(root_dir)
    cache_path = Path(cache_dir)
    
    # Get split directory
    split_dir = root_path / split
    if not split_dir.exists():
        logger.error(f"Split '{split}' not found in {root_path}")
        return {'failed': True}
    
    # Find all track directories
    track_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    
    if max_samples is not None:
        track_dirs = track_dirs[:max_samples]
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)
    
    logger.info(f"Found {len(track_dirs)} SLAKH tracks in '{split}' split")
    logger.info(f"Using {num_workers} parallel workers with {compression} compression")
    
    # Create cache directory structure
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # For FP32 (no suffix), for others use suffix
    if compression == 'none':
        base_cache_dir = cache_path / "slakh"
    else:
        base_cache_dir = cache_path / f"slakh_{compression}"
    
    split_cache_dir = base_cache_dir / split
    split_cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Cache directory: {split_cache_dir}")
    
    # Process all tracks
    stats = {
        'split': split,
        'total_samples': len(track_dirs),
        'successful': 0,
        'failed': 0,
        'total_cache_size_mb': 0,
        'errors': []
    }
    
    process_fn = partial(
        process_single_track,
        split_cache_dir=split_cache_dir,
        skip_errors=skip_errors,
        compression=compression
    )
    
    with Pool(num_workers) as pool:
        # Create tasks
        tasks = [(track_dir, split) for track_dir in track_dirs]
        
        # Process with progress bar
        with tqdm(total=len(tasks), desc=f"Processing {split}", unit="track") as pbar:
            for result in pool.imap_unordered(process_fn, tasks):
                success, track_id, cache_size_mb, error = result
                
                if success:
                    stats['successful'] += 1
                    stats['total_cache_size_mb'] += cache_size_mb
                else:
                    stats['failed'] += 1
                    if error:
                        stats['errors'].append(f"{track_id}: {error}")
                
                pbar.update(1)
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"SLAKH {split.upper()} Preprocessing Summary".center(70))
    logger.info(f"{'='*70}")
    logger.info(f"✅ Successful:      {stats['successful']}/{stats['total_samples']}")
    logger.info(f"❌ Failed:          {stats['failed']}/{stats['total_samples']}")
    logger.info(f"📦 Total cache:     {stats['total_cache_size_mb']:.1f} MB ({stats['total_cache_size_mb']/1024:.1f} GB)")
    logger.info(f"📂 Cache location:  {split_cache_dir}")
    logger.info(f"🔧 Compression:     {compression}")
    
    if stats['errors']:
        logger.info(f"\nFirst 5 errors:")
        for error in stats['errors'][:5]:
            logger.info(f"  - {error}")
    
    logger.info(f"{'='*70}\n")
    
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess SLAKH2100 dataset")
    parser.add_argument('--root', default='/cluster/home/ahmedyz/yassine/MT3-pytorch/datasets/slakh2100_flac_redux',
                        help='SLAKH dataset root directory')
    parser.add_argument('--cache', default='/cluster/home/ahmedyz/yassine/MT3-pytorch/cache',
                        help='Cache directory')
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='train',
                        help='Which split to process')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to process (for testing)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 2)')
    parser.add_argument('--compression', choices=['none', 'fp16'], default='none',
                        help='Compression format (default: none for FP32)')
    
    args = parser.parse_args()
    
    preprocess_slakh(
        root_dir=args.root,
        cache_dir=args.cache,
        split=args.split,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        compression=args.compression
    )

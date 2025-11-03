#!/usr/bin/env python3
"""
Preprocess MAESTRO v3 dataset into cached format.

FP32 (base format - no suffix on cache dir):
  Cache structure: /cache/maestro/[split]/[track_id].pt
  
Cache format (per sample):
  {
    'track_id': str,
    'audio': torch.Tensor (float32 or float16),
    'tokens': np.ndarray (int64),
    'program': int,
    'metadata': dict with split, duration, sample_rate, etc.
  }
"""

import os
import csv
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
from data.utils import note_sequence_to_timed_events, timed_events_to_tokens

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_maestro_split_assignments(maestro_root: Path) -> Dict[str, str]:
    """Read MAESTRO CSV to get split assignment for each track"""
    csv_file = maestro_root / "maestro-v3.0.0.csv"
    
    if not csv_file.exists():
        logger.error(f"MAESTRO CSV not found: {csv_file}")
        return {}
    
    assignments = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_filename = row['audio_filename']
            split = row['split']
            # Extract track ID from path (e.g., "2018/MIDI-...-1.wav" -> "MIDI-...-1")
            track_id = Path(audio_filename).stem
            assignments[track_id] = split
    
    return assignments


def process_single_track(
    track_data: Tuple[Path, Path],
    split_cache_dir: Path,
    skip_errors: bool = True,
    compression: str = 'none',
    split: str = 'train'
) -> Tuple[bool, str, float, str]:
    """
    Process single MAESTRO track (WAV + MIDI pair).
    
    Args:
        track_data: (wav_file, midi_file)
        compression: 'none' (fp32), 'fp16' (50% smaller)
    """
    try:
        wav_file, midi_file = track_data
        track_id = wav_file.stem
        
        # Load WAV
        try:
            audio, sr = librosa.load(str(wav_file), sr=DEFAULT_SAMPLE_RATE, mono=True)
            if sr != DEFAULT_SAMPLE_RATE:
                logger.warning(f"{track_id}: resampling from {sr} to {DEFAULT_SAMPLE_RATE}")
        except Exception as e:
            return False, track_id, 0, f"Failed to load WAV: {e}"
        
        # Load MIDI
        try:
            from note_seq import midi_file_to_note_sequence
            note_seq = midi_file_to_note_sequence(str(midi_file))
        except Exception as e:
            return False, track_id, 0, f"Failed to load MIDI: {e}"
        
        # Tokenize MIDI
        try:
            events = note_sequence_to_timed_events(note_seq)
            num_frames = len(audio) // DEFAULT_HOP_WIDTH
            frame_times = np.arange(num_frames) / (DEFAULT_SAMPLE_RATE / DEFAULT_HOP_WIDTH)
            tokens, _, _ = timed_events_to_tokens(events, codec, frame_times)
            tokens = np.array(tokens, dtype=np.int64)
        except Exception as e:
            return False, track_id, 0, f"Failed to tokenize: {e}"
        
        # Apply compression
        audio_tensor = torch.from_numpy(audio)
        if compression == 'fp16':
            audio_tensor = audio_tensor.half()
        # else: keep as float32 (default)
        
        # Save to cache with split subdirectory
        cache_data = {
            'track_id': track_id,
            'audio': audio_tensor,
            'tokens': tokens,
            'program': 0,  # Piano (default for MAESTRO)
            'metadata': {
                'sample_rate': DEFAULT_SAMPLE_RATE,
                'duration_seconds': len(audio) / DEFAULT_SAMPLE_RATE,
                'compression': compression,
                'source': 'maestro-v3',
                'split': split,
            }
        }
        
        cache_file = split_cache_dir / f"{track_id}.pt"
        torch.save(cache_data, cache_file)
        
        cache_size_mb = cache_file.stat().st_size / (1024 * 1024)
        return True, track_id, cache_size_mb, ""
        
    except Exception as e:
        return False, track_data[0].stem, 0, str(e)


def process_split(
    root_path: Path,
    cache_path: Path,
    split: str,
    split_assignments: Dict[str, str],
    num_workers: int,
    compression: str,
    skip_errors: bool,
    max_samples: Optional[int] = None
) -> Dict:
    """Process a single split with dedicated parallelism - MUCH BETTER SCALING"""
    
    logger.info(f"\n▶️  Processing MAESTRO {split.upper()}")
    
    # Find all WAV files for this split
    wav_files = sorted(root_path.rglob("*.wav"))
    
    track_pairs = []
    for wav_file in wav_files:
        track_id = wav_file.stem
        midi_file = wav_file.with_suffix('.midi')
        
        if midi_file.exists() and split_assignments.get(track_id) == split:
            track_pairs.append((wav_file, midi_file))
    
    if max_samples is not None:
        track_pairs = track_pairs[:max_samples]
    
    if not track_pairs:
        logger.info(f"No tracks found for '{split}' split")
        return {
            'split': split,
            'total': 0,
            'successful': 0,
            'failed': 0,
            'size_mb': 0
        }
    
    # Create cache directory for split
    if compression == 'none':
        base_cache_dir = cache_path / "maestro"
    else:
        base_cache_dir = cache_path / f"maestro_{compression}"
    
    split_cache_dir = base_cache_dir / split
    split_cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Found {len(track_pairs)} tracks for '{split}' split")
    
    # Process with parallel workers
    stats = {
        'split': split,
        'total': len(track_pairs),
        'successful': 0,
        'failed': 0,
        'total_size_mb': 0,
        'errors': []
    }
    
    process_fn = partial(
        process_single_track,
        split_cache_dir=split_cache_dir,
        skip_errors=skip_errors,
        compression=compression,
        split=split
    )
    
    with Pool(num_workers) as pool:
        with tqdm(total=len(track_pairs), desc=split, unit="track") as pbar:
            for result in pool.imap_unordered(process_fn, track_pairs):
                success, track_id, size_mb, error = result
                
                if success:
                    stats['successful'] += 1
                    stats['total_size_mb'] += size_mb
                else:
                    stats['failed'] += 1
                    if error:
                        stats['errors'].append(f"{track_id}: {error}")
                
                pbar.update(1)
    
    # Log split summary
    logger.info(f"  ✅ {stats['successful']}/{stats['total']} successful")
    logger.info(f"  📦 {stats['total_size_mb']:.1f} MB ({stats['total_size_mb']/1024:.2f} GB)")
    
    return stats


def preprocess_maestro(
    root_dir: str,
    cache_dir: str,
    max_samples: Optional[int] = None,
    skip_errors: bool = True,
    num_workers: Optional[int] = None,
    compression: str = 'none',
    splits: Optional[List] = None
) -> Dict:
    """
    Preprocess MAESTRO v3 dataset - OPTIMIZED for parallelism.
    
    Key optimization: Process each split independently with full parallelism.
    This avoids synchronization issues and allows better load balancing.
    """
    
    if splits is None:
        splits = ['train', 'validation', 'test']
    
    root_path = Path(root_dir)
    cache_path = Path(cache_dir)
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)
    
    # Get split assignments once
    logger.info("Reading MAESTRO splits from CSV...")
    split_assignments = get_maestro_split_assignments(root_path)
    logger.info(f"Found {len(split_assignments)} tracks with splits")
    logger.info(f"Using {num_workers} workers per split with {compression} compression")
    
    # Process each split independently
    all_stats = {
        'total_samples': 0,
        'total_successful': 0,
        'total_failed': 0,
        'total_cache_mb': 0,
        'splits': {}
    }
    
    for split in splits:
        split_stats = process_split(
            root_path=root_path,
            cache_path=cache_path,
            split=split,
            split_assignments=split_assignments,
            num_workers=num_workers,
            compression=compression,
            skip_errors=skip_errors,
            max_samples=max_samples
        )
        
        all_stats['total_samples'] += split_stats['total']
        all_stats['total_successful'] += split_stats['successful']
        all_stats['total_failed'] += split_stats['failed']
        all_stats['total_cache_mb'] += split_stats['total_size_mb']
        all_stats['splits'][split] = split_stats
    
    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("MAESTRO FINAL SUMMARY".center(70))
    logger.info(f"{'='*70}")
    logger.info(f"✅ Total successful:    {all_stats['total_successful']}/{all_stats['total_samples']}")
    logger.info(f"❌ Total failed:        {all_stats['total_failed']}/{all_stats['total_samples']}")
    logger.info(f"📦 Total cache:         {all_stats['total_cache_mb']:.1f} MB ({all_stats['total_cache_mb']/1024:.1f} GB)")
    logger.info(f"🔧 Compression:         {compression}")
    logger.info(f"{'='*70}\n")
    
    return all_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess MAESTRO v3 dataset")
    parser.add_argument('--root', default='/cluster/home/ahmedyz/yassine/MT3-pytorch/datasets/maestro-v3.0.0',
                        help='MAESTRO dataset root directory')
    parser.add_argument('--cache', default='/cluster/home/ahmedyz/yassine/MT3-pytorch/cache',
                        help='Cache directory')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to process (for testing)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 2)')
    parser.add_argument('--compression', choices=['none', 'fp16'], default='none',
                        help='Compression format (default: none for FP32)')
    
    args = parser.parse_args()
    
    preprocess_maestro(
        root_dir=args.root,
        cache_dir=args.cache,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        compression=args.compression
    )

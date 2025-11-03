#!/usr/bin/env python3
"""
Master cache orchestration script.

Preprocesses all MAESTRO and SLAKH data with proper split organization.

FP32 Structure (no suffix - base format):
  /cache/maestro/train/*.pt
  /cache/maestro/validation/*.pt
  /cache/maestro/test/*.pt
  
  /cache/slakh/train/*.pt
  /cache/slakh/validation/*.pt
  /cache/slakh/test/*.pt

Optional compression:
  /cache/maestro_fp16/train/*.pt  (if --compression fp16)
  /cache/slakh_fp16/train/*.pt    (if --compression fp16)
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status"""
    logger.info(f"\n{'='*70}")
    logger.info(f"▶️  {description}")
    logger.info(f"{'='*70}")
    logger.info(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        logger.info(f"\n✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"\n❌ {description} - FAILED (exit code: {e.returncode})")
        return False


def cache_maestro(
    cache_dir: str,
    max_samples: Optional[int] = None,
    num_workers: int = 32,
    compression: str = 'none'
) -> bool:
    """Cache MAESTRO dataset - uses optimized split-independent processing"""
    cmd = [
        sys.executable,
        'scripts/preprocess_maestro.py',
        '--cache', cache_dir,
        '--num-workers', str(num_workers),
        '--compression', compression,
    ]
    
    if max_samples:
        cmd.extend(['--max-samples', str(max_samples)])
    
    return run_command(cmd, "Preprocessing MAESTRO v3 (optimized split processing)")


def cache_slakh(
    cache_dir: str,
    splits: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    num_workers: int = 32,
    compression: str = 'none'
) -> bool:
    """Cache SLAKH dataset for specified splits"""
    if splits is None:
        splits = ['train', 'validation', 'test']
    
    all_success = True
    
    for split in splits:
        cmd = [
            sys.executable,
            'scripts/preprocess_slakh.py',
            '--cache', cache_dir,
            '--split', split,
            '--num-workers', str(num_workers),
            '--compression', compression,
        ]
        
        if max_samples:
            cmd.extend(['--max-samples', str(max_samples)])
        
        success = run_command(cmd, f"Preprocessing SLAKH {split} split")
        all_success = all_success and success
    
    return all_success


def verify_cache(cache_dir: str) -> None:
    """Verify cache structure"""
    cache_path = Path(cache_dir)
    
    logger.info(f"\n{'='*70}")
    logger.info("Cache Structure".center(70))
    logger.info(f"{'='*70}\n")
    
    # Check MAESTRO
    maestro_base = cache_path / "maestro"
    if maestro_base.exists():
        logger.info("📂 MAESTRO Cache:")
        for split in ['train', 'validation', 'test']:
            split_dir = maestro_base / split
            if split_dir.exists():
                files = list(split_dir.glob('*.pt'))
                size_gb = sum(f.stat().st_size for f in files) / (1024**3)
                logger.info(f"   ✅ {split:<15} {len(files):>6} files ({size_gb:>6.1f} GB)")
    
    # Check SLAKH
    slakh_base = cache_path / "slakh"
    if slakh_base.exists():
        logger.info("\n📂 SLAKH Cache:")
        for split in ['train', 'validation', 'test']:
            split_dir = slakh_base / split
            if split_dir.exists():
                files = list(split_dir.glob('*.pt'))
                size_gb = sum(f.stat().st_size for f in files) / (1024**3)
                logger.info(f"   ✅ {split:<15} {len(files):>6} files ({size_gb:>6.1f} GB)")
    
    logger.info(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Master cache orchestration for MAESTRO and SLAKH datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cache full MAESTRO + SLAKH with FP32 (default, no compression)
  python scripts/cache_all_datasets_v2.py --num-workers 32
  
  # Cache with FP16 compression (50% size reduction)
  python scripts/cache_all_datasets_v2.py --compression fp16 --num-workers 32
  
  # Test with small samples
  python scripts/cache_all_datasets_v2.py --max-samples 10 --num-workers 8
  
  # Cache only specific splits
  python scripts/cache_all_datasets_v2.py --slakh-splits train validation
        """
    )
    
    parser.add_argument('--cache', default='/cluster/home/ahmedyz/yassine/MT3-pytorch/cache',
                        help='Cache directory (default: /cluster/home/ahmedyz/yassine/MT3-pytorch/cache)')
    parser.add_argument('--maestro', action='store_true', default=True,
                        help='Cache MAESTRO dataset (default: True)')
    parser.add_argument('--no-maestro', action='store_true',
                        help='Skip MAESTRO caching')
    parser.add_argument('--slakh', action='store_true', default=True,
                        help='Cache SLAKH dataset (default: True)')
    parser.add_argument('--no-slakh', action='store_true',
                        help='Skip SLAKH caching')
    parser.add_argument('--slakh-splits', nargs='+', default=['train', 'validation', 'test'],
                        choices=['train', 'validation', 'test'],
                        help='Which SLAKH splits to cache (default: all)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples per split for testing')
    parser.add_argument('--num-workers', type=int, default=32,
                        help='Number of parallel workers (default: 32, recommended for A100)')
    parser.add_argument('--compression', choices=['none', 'fp16'], default='none',
                        help='Compression format (default: none for FP32)')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify cache structure, skip preprocessing')
    
    args = parser.parse_args()
    
    # Handle flags
    do_maestro = args.maestro and not args.no_maestro
    do_slakh = args.slakh and not args.no_slakh
    
    logger.info("="*70)
    logger.info("CACHE ORCHESTRATION".center(70))
    logger.info("="*70)
    logger.info(f"\n📊 Configuration:")
    logger.info(f"  Cache directory:  {args.cache}")
    logger.info(f"  Compression:      {args.compression}")
    logger.info(f"  Workers:          {args.num_workers}")
    logger.info(f"  Max samples:      {args.max_samples if args.max_samples else 'All'}")
    logger.info(f"  MAESTRO:          {'✅ Yes' if do_maestro else '❌ No'}")
    logger.info(f"  SLAKH:            {'✅ Yes' if do_slakh else '❌ No'}")
    if do_slakh:
        logger.info(f"  SLAKH splits:     {', '.join(args.slakh_splits)}")
    
    if args.verify_only:
        logger.info("\n🔍 Verify-only mode - skipping preprocessing\n")
        verify_cache(args.cache)
        return
    
    logger.info(f"\n{'='*70}\n")
    
    # Cache datasets
    all_success = True
    
    if do_maestro:
        success = cache_maestro(
            cache_dir=args.cache,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            compression=args.compression
        )
        all_success = all_success and success
    
    if do_slakh:
        success = cache_slakh(
            cache_dir=args.cache,
            splits=args.slakh_splits,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            compression=args.compression
        )
        all_success = all_success and success
    
    # Verify result
    verify_cache(args.cache)
    
    # Final status
    logger.info("="*70)
    if all_success:
        logger.info("✅ ALL CACHING COMPLETED SUCCESSFULLY".center(70))
    else:
        logger.info("❌ SOME OPERATIONS FAILED - CHECK LOGS ABOVE".center(70))
    logger.info("="*70)
    
    sys.exit(0 if all_success else 1)


if __name__ == '__main__':
    main()

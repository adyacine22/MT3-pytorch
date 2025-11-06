#!/usr/bin/env python3
"""
Quick end-to-end smoke test for cached datasets.

Examples:
    python scripts/test_cache_dataloader.py --dataset maestro --cache cache/maestro_opt_fp16/train
    python scripts/test_cache_dataloader.py --dataset slakh_stems --cache cache/slakh_opt_fp16/train
    python scripts/test_cache_dataloader.py --dataset slakh_mix --cache cache/slakh_opt_fp16/train --batch-size 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, '/cluster/home/ahmedyz/yassine/MT3-pytorch')

from data.cached_maestro_loader import CachedMaestroDataset  # noqa: E402
from data.cached_slakh_loader import (  # noqa: E402
    CachedSLAKHStemDataset,
    CachedSLAKHMixDataset,
)


def simple_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    inputs = torch.stack([sample["inputs"] for sample in batch])
    targets = torch.stack([sample["targets"] for sample in batch])
    return {"inputs": inputs, "targets": targets}


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test cached dataset DataLoader.")
    parser.add_argument(
        "--dataset",
        choices=["maestro", "slakh_stems", "slakh_mix"],
        required=True,
        help="Which cached dataset to test.",
    )
    parser.add_argument(
        "--cache",
        required=True,
        type=Path,
        help="Path to directory containing cached .pt files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for the DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers to use.",
    )
    args = parser.parse_args()

    cache_dir = str(args.cache.resolve())

    if args.dataset == "maestro":
        dataset = CachedMaestroDataset(
            cache_dir=cache_dir,
            preload_all=False,
            cache_size=2,
            num_workers=max(1, args.num_workers or 1),
        )
    elif args.dataset == "slakh_stems":
        dataset = CachedSLAKHStemDataset(
            cache_dir=cache_dir,
            preload_all=False,
            cache_size=2,
            num_workers=max(1, args.num_workers or 1),
        )
    else:
        dataset = CachedSLAKHMixDataset(
            cache_dir=cache_dir,
            preload_all=False,
            cache_size=2,
            num_workers=max(1, args.num_workers or 1),
        )

    print(f"Dataset length: {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=simple_collate,
        pin_memory=False,
    )

    batch = next(iter(loader))
    inputs, targets = batch["inputs"], batch["targets"]
    print(f"Batch inputs: shape={inputs.shape}, dtype={inputs.dtype}, device={inputs.device}")
    print(f"Batch targets: shape={targets.shape}, dtype={targets.dtype}, device={targets.device}")
    print("✓ DataLoader smoke test succeeded.")


if __name__ == "__main__":
    main()

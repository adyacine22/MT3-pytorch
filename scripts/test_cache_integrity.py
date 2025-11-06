#!/usr/bin/env python3
"""
Lightweight cache validation for MAESTRO and SLAKH optimized datasets.

Examples:
    python scripts/test_cache_integrity.py --dataset maestro --cache cache/maestro_opt_fp16/train --limit 5
    python scripts/test_cache_integrity.py --dataset slakh --cache cache/slakh_opt_fp16/train --format optimized
"""

import argparse
import sys
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, '/cluster/home/ahmedyz/yassine/MT3-pytorch')

from data.constants import DEFAULT_HOP_WIDTH  # noqa: E402


def _pick_files(cache_dir: Path, limit: int) -> List[Path]:
    files = sorted(cache_dir.glob("*.pt"))
    if limit is not None:
        files = files[:limit]
    return files


def _check_maestro(data: dict, fmt: str) -> List[str]:
    issues: List[str] = []
    if fmt == "optimized":
        if "frames" not in data:
            issues.append("missing 'frames'")
        else:
            frames = data["frames"]
            if not isinstance(frames, torch.Tensor):
                issues.append("'frames' is not a torch.Tensor")
            elif frames.dim() != 2 or frames.shape[1] != DEFAULT_HOP_WIDTH:
                issues.append(f"unexpected frames shape {tuple(frames.shape)}")
        tokens = data.get("tokens")
        if tokens is None:
            issues.append("missing 'tokens'")
        elif not isinstance(tokens, torch.Tensor):
            issues.append("'tokens' is not a torch.Tensor")
        metadata = data.get("metadata", {})
        if metadata.get("format_version") != "optimized_v1":
            issues.append("format_version missing/incorrect for optimized cache")
    else:
        if "audio" not in data:
            issues.append("missing 'audio' in legacy cache")
        tokens = data.get("tokens")
        if tokens is None:
            issues.append("missing 'tokens'")
    return issues


def _check_slakh(data: dict, fmt: str) -> List[str]:
    issues: List[str] = []
    metadata = data.get("metadata", {})
    if fmt == "optimized":
        if metadata.get("format_version") != "optimized_v1":
            issues.append("format_version missing/incorrect for optimized cache")
        if "stem_frames" not in data:
            issues.append("missing 'stem_frames'")
        else:
            stem_frames = data["stem_frames"]
            if not isinstance(stem_frames, list) or not stem_frames:
                issues.append("'stem_frames' should be a non-empty list")
            else:
                for idx, frame_tensor in enumerate(stem_frames[:2]):
                    if not isinstance(frame_tensor, torch.Tensor):
                        issues.append(f"stem_frames[{idx}] not torch.Tensor")
                    elif frame_tensor.dim() != 2 or frame_tensor.shape[1] != DEFAULT_HOP_WIDTH:
                        issues.append(f"stem_frames[{idx}] has bad shape {tuple(frame_tensor.shape)}")
        if "mix_frames" not in data:
            issues.append("missing 'mix_frames'")
    else:
        if metadata.get("format_version") != "legacy_v1":
            issues.append("format_version missing/incorrect for legacy cache")
        if "stems_audio" not in data:
            issues.append("missing 'stems_audio' in legacy cache")
    # Shared checks
    if "stem_tokens" not in data or not data["stem_tokens"]:
        issues.append("missing 'stem_tokens'")
    return issues


def validate_cache(dataset: str, cache_dir: Path, fmt: str, limit: int) -> int:
    issues_found = 0
    files = _pick_files(cache_dir, limit)
    if not files:
        print(f"[WARN] No .pt files found under {cache_dir}")
        return 1

    for path in files:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if dataset == "maestro":
            issues = _check_maestro(data, fmt)
        else:
            issues = _check_slakh(data, fmt)
        if issues:
            issues_found += 1
            print(f"[FAIL] {path.name}:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"[OK]   {path.name}")

    if issues_found == 0:
        print(f"\n✅ All {len(files)} files passed basic integrity checks.")
    else:
        print(f"\n❌ {issues_found}/{len(files)} files reported issues.")
    return issues_found


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate cached dataset structure")
    parser.add_argument("--dataset", choices=["maestro", "slakh"], required=True)
    parser.add_argument("--cache", type=Path, required=True, help="Directory containing .pt files")
    parser.add_argument(
        "--format",
        choices=["legacy", "optimized"],
        default="optimized",
        help="Expected cache layout (affects validation rules)",
    )
    parser.add_argument("--limit", type=int, default=5, help="How many cache files to inspect")
    args = parser.parse_args()

    exit_code = validate_cache(args.dataset, args.cache, args.format, args.limit)
    raise SystemExit(1 if exit_code else 0)


if __name__ == "__main__":
    main()

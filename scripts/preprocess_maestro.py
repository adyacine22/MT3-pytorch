#!/usr/bin/env python3
"""
Build cached MAESTRO v3 datasets (legacy or optimized layout).

Legacy layout (default MT3 behaviour):
    cache/maestro[/fp16]/<split>/<track>.pt  → raw audio waveform + numpy tokens

Optimized layout (default for this script):
    cache/maestro_opt[/fp16]/<split>/<track>.pt  → audio already split into frame tensors,
    token tensors, optional mel spectrograms, and rich metadata. This avoids costly audio
    decoding and framing work during training.
"""

from __future__ import annotations

import argparse
import csv
import logging
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torchaudio
from tqdm import tqdm

import sys

sys.path.insert(0, '/cluster/home/ahmedyz/yassine/MT3-pytorch')

from data.constants import (  # noqa: E402
    DEFAULT_HOP_WIDTH,
    DEFAULT_NUM_MEL_BINS,
    DEFAULT_SAMPLE_RATE,
    FFT_SIZE,
    MEL_FMAX,
    MEL_FMIN,
    codec,
)
from data.dataset_utils import FrameProcessingMixin  # noqa: E402
from data.spectrogram import MelSpectrogram  # noqa: E402
from data.utils import note_sequence_to_timed_events, timed_events_to_tokens  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _FrameProcessor(FrameProcessingMixin):
    """Minimal wrapper around frame utilities."""

    pass


def get_maestro_split_assignments(maestro_root: Path) -> Dict[str, str]:
    """Read MAESTRO CSV to get split assignment for each track."""
    csv_file = maestro_root / "maestro-v3.0.0.csv"

    if not csv_file.exists():
        logger.error("MAESTRO CSV not found: %s", csv_file)
        return {}

    assignments: Dict[str, str] = {}
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_filename = row["audio_filename"]
            split = row["split"]
            track_id = Path(audio_filename).stem
            assignments[track_id] = split

    return assignments


def _select_cache_base(cache_path: Path, fmt: Literal["legacy", "optimized"], compression: str) -> Path:
    suffix = "" if compression == "none" else f"_{compression}"
    if fmt == "optimized":
        return cache_path / f"maestro_opt{suffix}"
    return cache_path / f"maestro{suffix}"


def _maybe_compute_mels(audio: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mel_module = MelSpectrogram(
        DEFAULT_NUM_MEL_BINS,
        DEFAULT_SAMPLE_RATE,
        FFT_SIZE,
        DEFAULT_HOP_WIDTH,
        mel_fmin=MEL_FMIN,
        mel_fmax=MEL_FMAX,
    ).to(device)

    with torch.no_grad():
        mel = mel_module(audio.to(device).unsqueeze(0))
    mel = mel.squeeze(0).transpose(0, 1)  # (frames, mel_bins)
    return mel.to(dtype=dtype, copy=False).contiguous().cpu()


def process_single_track(
    track_data: Tuple[Path, Path],
    split_cache_dir: Path,
    *,
    skip_errors: bool,
    compression: str,
    split: str,
    fmt: Literal["legacy", "optimized"],
    store_mels: bool,
    mel_device: str,
) -> Tuple[bool, str, float, str]:
    """Worker routine for a single MAESTRO track."""
    wav_file, midi_file = track_data
    track_id = wav_file.stem

    try:
        audio_tensor, sr = torchaudio.load(str(wav_file), backend="soundfile")
        if audio_tensor.size(0) > 1:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
        audio_tensor = audio_tensor.squeeze(0)
        if sr != DEFAULT_SAMPLE_RATE:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, DEFAULT_SAMPLE_RATE)
    except Exception as exc:
        return False, track_id, 0.0, f"Failed to load WAV: {exc}"

    try:
        from note_seq import midi_file_to_note_sequence

        note_seq_obj = midi_file_to_note_sequence(str(midi_file))
    except Exception as exc:
        return False, track_id, 0.0, f"Failed to load MIDI: {exc}"

    try:
        events = note_sequence_to_timed_events(note_seq_obj)
        num_frames = int(audio_tensor.numel() // DEFAULT_HOP_WIDTH)
        frame_times = (torch.arange(num_frames, dtype=torch.float32) /
                       (DEFAULT_SAMPLE_RATE / DEFAULT_HOP_WIDTH)).tolist()
        token_list, _, _ = timed_events_to_tokens(events, codec, frame_times)
    except Exception as exc:
        return False, track_id, 0.0, f"Failed to tokenize: {exc}"

    compression_dtype = torch.float16 if compression == "fp16" else torch.float32
    token_tensor = torch.tensor(token_list, dtype=torch.int32)
    audio_tensor = audio_tensor.to(torch.float32)

    if fmt == "optimized":
        frame_processor = _FrameProcessor()
        frames = frame_processor._split_audio_to_frames(audio_tensor)
        frames = frames.to(dtype=compression_dtype, copy=False).contiguous()

        cache_data: Dict[str, object] = {
            "track_id": track_id,
            "frames": frames,
            "tokens": token_tensor,
            "metadata": {
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "duration_seconds": audio_tensor.numel() / DEFAULT_SAMPLE_RATE,
                "compression": compression,
                "source": "maestro-v3",
                "split": split,
                "format_version": "optimized_v1",
                "frame_hop": DEFAULT_HOP_WIDTH,
                "num_frames": int(frames.shape[0]),
            },
        }

        if store_mels:
            device = torch.device(
                "cuda" if mel_device == "cuda" and torch.cuda.is_available() else "cpu"
            )
            cache_data["mel_spectrogram"] = _maybe_compute_mels(audio_tensor, compression_dtype, device)
    else:
        audio_out = audio_tensor.to(dtype=compression_dtype, copy=False)
        cache_data = {
            "track_id": track_id,
            "audio": audio_out,
            "tokens": token_tensor,
            "program": 0,
            "metadata": {
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "duration_seconds": audio_tensor.numel() / DEFAULT_SAMPLE_RATE,
                "compression": compression,
                "source": "maestro-v3",
                "split": split,
                "format_version": "legacy_v1",
            },
        }

    cache_file = split_cache_dir / f"{track_id}.pt"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.save(cache_data, cache_file)
    except Exception as exc:
        if not skip_errors:
            raise
        return False, track_id, 0.0, f"torch.save failed: {exc}"

    cache_size_mb = cache_file.stat().st_size / (1024 * 1024)
    return True, track_id, cache_size_mb, ""


def process_split(
    root_path: Path,
    cache_path: Path,
    split: str,
    split_assignments: Dict[str, str],
    num_workers: int,
    *,
    compression: str,
    skip_errors: bool,
    max_samples: Optional[int],
    fmt: Literal["legacy", "optimized"],
    store_mels: bool,
    mel_device: str,
) -> Dict[str, object]:
    logger.info("▶️  Processing MAESTRO %s", split.upper())

    wav_files = sorted(root_path.rglob("*.wav"))
    track_pairs: List[Tuple[Path, Path]] = []
    for wav_file in wav_files:
        track_id = wav_file.stem
        midi_file = wav_file.with_suffix(".midi")
        if midi_file.exists() and split_assignments.get(track_id) == split:
            track_pairs.append((wav_file, midi_file))

    if max_samples is not None:
        track_pairs = track_pairs[:max_samples]

    if not track_pairs:
        logger.info("No tracks found for split '%s'", split)
        return {"split": split, "total": 0, "successful": 0, "failed": 0}

    base_cache_dir = _select_cache_base(cache_path, fmt, compression)
    split_cache_dir = base_cache_dir / split
    split_cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("  Found %d tracks → %s", len(track_pairs), split_cache_dir)

    stats = {
        "split": split,
        "total": len(track_pairs),
        "successful": 0,
        "failed": 0,
        "total_size_mb": 0.0,
        "errors": [],
    }

    worker_fn = partial(
        process_single_track,
        split_cache_dir=split_cache_dir,
        skip_errors=skip_errors,
        compression=compression,
        split=split,
        fmt=fmt,
        store_mels=store_mels,
        mel_device=mel_device,
    )

    with Pool(num_workers) as pool:
        with tqdm(total=len(track_pairs), desc=f"{split} split", unit="track") as pbar:
            for success, track_id, cache_size_mb, error in pool.imap_unordered(worker_fn, track_pairs):
                if success:
                    stats["successful"] += 1
                    stats["total_size_mb"] += cache_size_mb
                else:
                    stats["failed"] += 1
                    if error:
                        stats["errors"].append(f"{track_id}: {error}")
                pbar.update(1)

    logger.info(
        "  Summary: %s/%s ok | %.1f MB written",
        stats["successful"],
        stats["total"],
        stats["total_size_mb"],
    )
    if stats["errors"]:
        logger.info("  Errors (showing up to 5):")
        for err in stats["errors"][:5]:
            logger.info("    %s", err)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MAESTRO v3 dataset")
    parser.add_argument(
        "--root",
        default="/cluster/home/ahmedyz/yassine/MT3-pytorch/datasets/maestro-v3.0.0",
        help="MAESTRO dataset root directory",
    )
    parser.add_argument(
        "--cache",
        default="/cluster/home/ahmedyz/yassine/MT3-pytorch/cache",
        help="Cache directory",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of tracks (debug)")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel worker count (default: CPU count - 2)",
    )
    parser.add_argument("--compression", choices=["none", "fp16"], default="fp16")
    parser.add_argument("--format", choices=["legacy", "optimized"], default="optimized")
    parser.add_argument("--store-mels", action="store_true", help="Store mel spectrogram tensors")
    parser.add_argument("--mel-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--fail-on-error", action="store_true", help="Abort on first failure")

    args = parser.parse_args()

    num_workers = args.num_workers or max(1, cpu_count() - 2)
    split_assignments = get_maestro_split_assignments(Path(args.root))

    results = []
    for split in ["train", "validation", "test"]:
        result = process_split(
            Path(args.root),
            Path(args.cache),
            split,
            split_assignments,
            num_workers,
            compression=args.compression,
            skip_errors=not args.fail_on_error,
            max_samples=args.max_samples,
            fmt=args.format,
            store_mels=args.store_mels,
            mel_device=args.mel_device,
        )
        results.append(result)

    total_mb = sum(r.get("total_size_mb", 0.0) for r in results)
    logger.info("✅ Finished. Total cache size: %.1f MB (%.1f GB)", total_mb, total_mb / 1024)


if __name__ == "__main__":
    main()

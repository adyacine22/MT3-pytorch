#!/usr/bin/env python3
"""
Build cached SLAKH2100 datasets with optional optimized layout.

Legacy layout (default MT3)  : cache/slakh[/fp16]/split/TrackXXXX.pt
Optimized layout (pre-framed): cache/slakh_opt[/fp16]/split/TrackXXXX.pt

The optimized format stores frame tensors (per stem and mix) alongside tokens so
training can avoid expensive waveform processing. Legacy fields remain
available by default to preserve compatibility with existing loaders; they can
be dropped with --drop-audio once the pipeline migrates fully.
"""

import argparse
import logging
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
import warnings

import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm

import sys

sys.path.insert(0, '/cluster/home/ahmedyz/yassine/MT3-pytorch')

from data.constants import (
    DEFAULT_HOP_WIDTH,
    DEFAULT_NUM_MEL_BINS,
    DEFAULT_SAMPLE_RATE,
    FFT_SIZE,
    MEL_FMAX,
    MEL_FMIN,
    codec,
)
from data.dataset_utils import FrameProcessingMixin
from data.spectrogram import MelSpectrogram
from data.slakh_loader import parse_slakh_metadata, merge_note_sequences
from data.utils import note_sequence_to_timed_events, timed_events_to_tokens
import note_seq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    torchaudio.set_audio_backend("soundfile")
except (AttributeError, RuntimeError):
    pass

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class _FrameProcessor(FrameProcessingMixin):
    """Stateless helper to expose mixin methods."""

    pass


@dataclass
class SlakhStem:
    """Container for per-stem cached information."""

    stem_id: str
    audio: torch.Tensor
    tokens: torch.Tensor
    program: int
    name: str
    note_sequence: object  # note_seq.NoteSequence, kept generic to avoid import cycling


def _select_cache_base(cache_path: Path, compression: str, fmt: Literal["legacy", "optimized"]) -> Path:
    suffix = "" if compression == "none" else f"_{compression}"
    if fmt == "optimized":
        return cache_path / f"slakh_opt{suffix}"
    return cache_path / f"slakh{suffix}"


def _compute_tokens(note_seq_obj, audio_len: int) -> torch.Tensor:
    """Convert note sequence to integer tokens."""
    events = note_sequence_to_timed_events(note_seq_obj)
    num_frames = audio_len // DEFAULT_HOP_WIDTH
    if num_frames == 0:
        return torch.empty(0, dtype=torch.int32)
    frame_times = torch.arange(num_frames, dtype=torch.float32) / (
        DEFAULT_SAMPLE_RATE / DEFAULT_HOP_WIDTH
    )
    token_array, _, _ = timed_events_to_tokens(
        events, codec, frame_times.cpu().numpy()
    )
    return torch.tensor(token_array, dtype=torch.int32)


def _load_stem_audio(track_dir: Path, stem_id: str) -> Optional[torch.Tensor]:
    """Load a single stem waveform using torchaudio."""
    for extension in (".flac", ".wav"):
        audio_path = track_dir / "stems" / f"{stem_id}{extension}"
        if audio_path.exists():
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec",
                        category=UserWarning,
                    )
                    waveform, sample_rate = torchaudio.load(audio_path)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", audio_path, exc)
                return None

            waveform = waveform.to(torch.float32)
            if waveform.dim() == 2 and waveform.size(0) > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)

            if sample_rate != DEFAULT_SAMPLE_RATE and waveform.numel() > 0:
                waveform = torchaudio.functional.resample(
                    waveform.unsqueeze(0), sample_rate, DEFAULT_SAMPLE_RATE
                ).squeeze(0)

            return waveform.contiguous()

    return None


def _load_stem_note_sequence(
    track_dir: Path, stem_id: str, stem_info: dict, audio_len: int
) -> note_seq.NoteSequence:
    """Load MIDI for a single stem and ensure program metadata is set."""
    midi_path = track_dir / "MIDI" / f"{stem_id}.mid"
    if not midi_path.exists():
        ns = note_seq.NoteSequence()
        ns.total_time = audio_len / DEFAULT_SAMPLE_RATE if audio_len else 0.0
        return ns

    ns = note_seq.midi_file_to_note_sequence(str(midi_path))
    program = int(stem_info.get("program", 0))
    program = max(0, min(127, program))
    is_drum = bool(stem_info.get("is_drum", False))
    for note in ns.notes:
        note.program = program
        note.is_drum = is_drum
    return ns


def _maybe_compute_mels(audio: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Compute mel spectrogram for a mono audio tensor."""
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
    mel = mel.squeeze(0).transpose(0, 1)
    return mel.to(dtype=dtype, copy=False).contiguous().cpu()


def _load_track_stems(track_dir: Path) -> List[SlakhStem]:
    """Load every stem for the given track."""
    stems_metadata = parse_slakh_metadata(str(track_dir))
    stems: List[SlakhStem] = []

    for stem_id, stem_info in stems_metadata.items():
        audio = _load_stem_audio(track_dir, stem_id)
        if audio is None:
            continue
        note_seq_obj = _load_stem_note_sequence(track_dir, stem_id, stem_info, audio.numel())
        tokens = _compute_tokens(note_seq_obj, audio.numel())
        stems.append(
            SlakhStem(
                stem_id=stem_id,
                audio=audio.to(torch.float32),
                tokens=tokens,
                program=int(stem_info.get("program", 0)),
                name=stem_info.get("name", f"stem_{stem_id}"),
                note_sequence=note_seq_obj,
            )
        )

    return stems


def _build_mix_audio(stems: List[SlakhStem]) -> torch.Tensor:
    """Construct the mix audio by summing stems (matches legacy behaviour)."""
    if not stems:
        return torch.empty(0, dtype=torch.float32)

    max_len = max(stem.audio.numel() for stem in stems)
    padded = []
    for stem in stems:
        audio = stem.audio
        if audio.numel() < max_len:
            audio = F.pad(audio, (0, max_len - audio.numel()))
        padded.append(audio)

    mix = torch.stack(padded, dim=0).sum(dim=0)
    if mix.numel() > 0:
        max_val = mix.abs().max().item()
        if max_val > 0:
            mix = mix / max_val

    return mix.contiguous()


def process_single_track(
    track_info: Tuple[Path, str],
    *,
    cache_dir: Path,
    compression: str,
    fmt: Literal["legacy", "optimized"],
    drop_audio: bool,
    store_mels: bool,
    mel_device: str,
    skip_errors: bool,
) -> Tuple[bool, str, float, str]:
    """Process one SLAKH track (stems + metadata)."""
    track_dir, split = track_info
    track_id = track_dir.name

    try:
        stems = _load_track_stems(track_dir)
    except Exception as exc:
        if not skip_errors:
            raise
        return False, track_id, 0.0, f"Failed to load stems: {exc}"

    if not stems:
        return False, track_id, 0.0, "No stems found"

    compression_dtype = torch.float16 if compression == "fp16" else torch.float32
    frame_processor = _FrameProcessor()

    stem_frames: List[torch.Tensor] = []
    stem_audio_tensors: List[torch.Tensor] = []
    stem_tokens: List[torch.Tensor] = []
    stem_programs: List[int] = []
    stem_names: List[str] = []

    for stem in stems:
        audio_tensor = stem.audio.to(torch.float32)

        if fmt == "optimized":
            frames = frame_processor._split_audio_to_frames(audio_tensor)
            frames = frames.to(dtype=compression_dtype, copy=False).contiguous()
            stem_frames.append(frames)

        stem_tokens.append(stem.tokens.to(torch.int32))
        stem_programs.append(stem.program)
        stem_names.append(stem.name)

        if not drop_audio or fmt == "legacy":
            stem_audio_tensors.append(
                audio_tensor.to(dtype=compression_dtype, copy=False).contiguous()
            )

    mix_audio_tensor = _build_mix_audio(stems)
    mix_audio_tensor = mix_audio_tensor.to(torch.float32)

    mix_frames: Optional[torch.Tensor] = None
    mix_tokens: Optional[torch.Tensor] = None

    if fmt == "optimized":
        mix_frames = frame_processor._split_audio_to_frames(mix_audio_tensor)
        mix_frames = mix_frames.to(dtype=compression_dtype, copy=False).contiguous()
        mix_note_seq = merge_note_sequences([stem.note_sequence for stem in stems])
        mix_tokens = _compute_tokens(mix_note_seq, mix_audio_tensor.numel())

    cache_entry: Dict[str, object] = {
        "track_id": track_id,
        "stem_tokens": stem_tokens,
        "stem_programs": torch.tensor(stem_programs, dtype=torch.int16),
        "stem_names": stem_names,
        "metadata": {
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "duration_seconds": mix_audio_tensor.numel() / DEFAULT_SAMPLE_RATE,
            "compression": compression,
            "source": "slakh2100",
            "split": split,
            "num_stems": len(stems),
        },
    }

    if fmt == "optimized":
        cache_entry["stem_frames"] = stem_frames
        cache_entry["mix_frames"] = mix_frames
        cache_entry["mix_tokens"] = mix_tokens
        cache_entry["metadata"]["format_version"] = "optimized_v1"
        cache_entry["metadata"]["frame_hop"] = DEFAULT_HOP_WIDTH
        cache_entry["metadata"]["num_frames"] = int(mix_frames.shape[0])

        if store_mels:
            device = torch.device(
                "cuda" if mel_device == "cuda" and torch.cuda.is_available() else "cpu"
            )
            cache_entry["mix_mel"] = _maybe_compute_mels(mix_audio_tensor.float(), compression_dtype, device)
    else:
        cache_entry["metadata"]["format_version"] = "legacy_v1"

    if not drop_audio or fmt == "legacy":
        cache_entry["mix_audio"] = mix_audio_tensor.to(dtype=compression_dtype, copy=False).contiguous()
        cache_entry["stems_audio"] = stem_audio_tensors

    cache_path = cache_dir / f"{track_id}.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.save(cache_entry, cache_path)
    except Exception as exc:
        if not skip_errors:
            raise
        return False, track_id, 0.0, f"torch.save failed: {exc}"

    cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
    return True, track_id, cache_size_mb, ""


def preprocess_slakh(
    root_dir: str,
    cache_dir: str,
    *,
    split: str,
    max_samples: Optional[int],
    num_workers: Optional[int],
    compression: str,
    fmt: Literal["legacy", "optimized"],
    drop_audio: bool,
    store_mels: bool,
    mel_device: str,
    skip_errors: bool,
) -> Dict[str, object]:
    """Preprocess a single SLAKH split."""
    root_path = Path(root_dir)
    split_dir = root_path / split
    if not split_dir.exists():
        logger.error("Split '%s' not found in %s", split, root_path)
        return {"failed": True, "split": split}

    track_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    if max_samples is not None:
        track_dirs = track_dirs[:max_samples]

    worker_count = num_workers or 16
    logger.info(
        "Processing SLAKH %s (%d tracks) using %d workers [%s format, compression=%s]",
        split,
        len(track_dirs),
        worker_count,
        fmt,
        compression,
    )

    base_cache_dir = _select_cache_base(Path(cache_dir), compression, fmt)
    split_cache_dir = base_cache_dir / split
    split_cache_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "split": split,
        "total_samples": len(track_dirs),
        "successful": 0,
        "failed": 0,
        "total_cache_size_mb": 0.0,
        "errors": [],
        "cache_dir": str(split_cache_dir),
    }

    worker_fn = partial(
        process_single_track,
        cache_dir=split_cache_dir,
        compression=compression,
        fmt=fmt,
        drop_audio=drop_audio,
        store_mels=store_mels,
        mel_device=mel_device,
        skip_errors=skip_errors,
    )

    with Pool(worker_count) as pool:
        with tqdm(total=len(track_dirs), desc=f"{split} split", unit="track") as pbar:
            for result in pool.imap_unordered(worker_fn, [(track_dir, split) for track_dir in track_dirs]):
                success, track_id, cache_size_mb, error = result
                if success:
                    stats["successful"] += 1
                    stats["total_cache_size_mb"] += cache_size_mb
                else:
                    stats["failed"] += 1
                    if error:
                        stats["errors"].append(f"{track_id}: {error}")
                pbar.update(1)

    logger.info(
        "  Summary (%s): %s/%s ok | %.1f MB written → %s",
        split,
        stats["successful"],
        stats["total_samples"],
        stats["total_cache_size_mb"],
        split_cache_dir,
    )

    if stats["errors"]:
        logger.info("  Errors (up to 5 shown):")
        for err in stats["errors"][:5]:
            logger.info("    %s", err)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess SLAKH2100 dataset")
    parser.add_argument(
        "--root",
        default="/cluster/home/ahmedyz/yassine/MT3-pytorch/datasets/slakh2100_flac_redux",
        help="SLAKH dataset root directory",
    )
    parser.add_argument(
        "--cache",
        default="/cluster/home/ahmedyz/yassine/MT3-pytorch/cache",
        help="Cache directory",
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "validation", "test"],
        default="all",
        help="Which split to process (default: all)",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of tracks")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 2)",
    )
    parser.add_argument("--compression", choices=["none", "fp16"], default="fp16")
    parser.add_argument("--format", choices=["legacy", "optimized"], default="optimized")
    parser.add_argument(
        "--drop-audio",
        action="store_true",
        help="Remove raw audio tensors when generating optimized caches",
    )
    parser.add_argument(
        "--store-mels",
        action="store_true",
        help="Pre-compute mel spectrograms (stored as mix_mel)",
    )
    parser.add_argument(
        "--mel-device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device for mel computation (only used with --store-mels)",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Raise on preprocessing errors instead of skipping tracks",
    )

    args = parser.parse_args()

    if args.split == "all":
        overall = {"successful": 0, "failed": 0, "total_cache_size_mb": 0.0, "splits": {}}
        for sp in ["train", "validation", "test"]:
            stats = preprocess_slakh(
                root_dir=args.root,
                cache_dir=args.cache,
                split=sp,
                max_samples=args.max_samples,
                num_workers=args.num_workers,
                compression=args.compression,
                fmt=args.format,
                drop_audio=args.drop_audio,
                store_mels=args.store_mels,
                mel_device=args.mel_device,
                skip_errors=not args.fail_on_error,
            )
            overall["splits"][sp] = stats
            if not stats.get("failed") and not stats.get("failed") is None:
                pass
            overall["successful"] += stats.get("successful", 0)
            overall["failed"] += stats.get("failed", 0)
            overall["total_cache_size_mb"] += stats.get("total_cache_size_mb", 0.0)
        logger.info(
            "Overall: %d ok | %d failed | %.1f MB total",
            overall["successful"],
            overall["failed"],
            overall["total_cache_size_mb"],
        )
    else:
        preprocess_slakh(
            root_dir=args.root,
            cache_dir=args.cache,
            split=args.split,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            compression=args.compression,
            fmt=args.format,
            drop_audio=args.drop_audio,
            store_mels=args.store_mels,
            mel_device=args.mel_device,
            skip_errors=not args.fail_on_error,
        )


if __name__ == "__main__":
    main()

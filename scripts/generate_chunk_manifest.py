#!/usr/bin/env python3
"""Generate chunk-level manifest entries from the unified dataset index."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.preprocessing.options import PrecomputeOptions, load_precompute_options
from data.preprocessing.storage import slugify


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate chunk manifest from unified_index.json."
    )
    parser.add_argument(
        "--unified-index",
        type=Path,
        default=None,
        help="Path to unified_index.json (defaults to config path).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output manifest path (json/jsonl/parquet).",
    )
    parser.add_argument(
        "--chunk-duration-ms",
        type=float,
        default=None,
        help="Override chunk duration in milliseconds (defaults from profile).",
    )
    return parser.parse_args()


def load_unified_entries(unified_index_path: Path) -> List[Dict]:
    if not unified_index_path.exists():
        raise FileNotFoundError(
            f"Unified index not found at {unified_index_path}. "
            "Run scripts/create_unified_index.py first."
        )
    with unified_index_path.open() as fp:
        payload = json.load(fp)
    entries = payload.get("entries", [])
    logger.info("Loaded %d entries from unified index", len(entries))
    return entries


def compute_duration_ms(entry: Dict) -> float:
    if entry.get("duration_seconds"):
        return float(entry["duration_seconds"]) * 1000.0
    frames = entry.get("audio_num_frames")
    sample_rate = entry.get("audio_sample_rate")
    if frames and sample_rate:
        return (frames / sample_rate) * 1000.0
    return 0.0


def generate_jobs(
    entries: Iterable[Dict],
    chunk_duration_ms: float,
    chunk_frames: int,
    options: PrecomputeOptions,
    precomputed_dir: Path,
) -> List[Dict]:
    jobs: List[Dict] = []
    chunk_storage = options.chunk_storage
    augment_enabled = options.augment_enabled
    augment_profiles = options.augment_profiles if augment_enabled else ["none"]
    chunk_samples = options.chunk_samples
    hop_length = options.hop_length
    sample_rate = options.sample_rate
    allowed_datasets = set(options.datasets) if options.datasets else None
    logger.info("Allowed datasets: %s", allowed_datasets)
    i=0
    for entry in entries:
        duration_ms = compute_duration_ms(entry)
        if duration_ms <= 0:
            logger.debug("Skipping %s (no duration info)", entry["track_id"])
            continue
        dataset = entry["dataset"]
        if allowed_datasets and dataset not in allowed_datasets:
            logger.debug("Skipping %s (not in allowed datasets)", entry["track_id"])
            continue
        track_id = entry["track_id"]
        audio_path = entry.get("audio_path", "")
        track_slug = slugify(track_id, Path(audio_path).stem if audio_path else track_id)
        # Determine number of chunks (non-overlapping windows)
        num_chunks = max(1, math.ceil(duration_ms / chunk_duration_ms))
        if (i % 10000) == 0:
            logger.info("Generating %d chunks for track %s", num_chunks, track_id)
        for idx in range(num_chunks):
            start_ms = idx * chunk_duration_ms
            end_ms = min(duration_ms, start_ms + chunk_duration_ms)
            actual_duration = end_ms - start_ms

            chunk_id = f"{track_id}-chunk{idx:05d}"
            chunk_slug = f"chunk_{idx:05d}"
            stem_ids = []
            if dataset == "slakh_full_mix":
                stem_ids = entry.get("rendered_stem_ids", []) or []
            elif dataset == "slakh_stem":
                stem = entry.get("stem_id")
                if stem:
                    stem_ids = [stem]
            if chunk_storage == "per_track":
                storage_rel = Path(dataset) / f"{track_slug}.pt"
            else:
                storage_rel = Path(dataset) / track_slug / f"{chunk_slug}.pt"
            precomputed_path = (precomputed_dir / storage_rel).as_posix()

            job = {
                "chunk_id": chunk_id,
                "track_id": track_id,
                "dataset": dataset,
                "split": entry.get("split"),
                "audio_path": entry["audio_path"],
                "midi_path": entry["midi_path"],
                "chunk_index": idx,
                "chunk_start_s": start_ms / 1000.0,
                "chunk_end_s": end_ms / 1000.0,
                "chunk_start_ms": start_ms,
                "chunk_end_ms": end_ms,
                "chunk_duration_ms": actual_duration,
                "chunk_frames": chunk_frames,
                "chunk_samples": chunk_samples,
                "sample_rate": sample_rate,
                "hop_length": hop_length,
                "augmentation_profile": "none",
                "mix_group_id": None,
                "stem_ids": stem_ids,
                "instrument_programs": entry.get("instrument_programs", []),
                "instrument_names": entry.get("instrument_names", []),
                "instrument_classes": entry.get("instrument_classes", []),
                "is_drum": bool(entry.get("is_drum", False)),
                "metadata_hash": f"{track_id}-{start_ms:.0f}-{end_ms:.0f}",
                "precomputed_path": precomputed_path,
                "chunk_path": precomputed_path,
                "chunk_storage": chunk_storage,
                "chunk_shard_path": precomputed_path,
                "chunk_shard_index": idx,
                "precompute": {
                    "chunk_device": options.chunk_device or "auto",
                    "batch_size": options.batch_size,
                    "max_tokenize_workers": options.max_tokenize_workers,
                    "augment_enabled": augment_enabled,
                    "augment_profiles": augment_profiles,
                },
            }
            jobs.append(job)
            i += 1
    logger.info("Generated %d chunk jobs", len(jobs))
    return jobs


def save_manifest(jobs: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".jsonl":
        with output_path.open("w") as fp:
            for job in jobs:
                fp.write(json.dumps(job))
                fp.write("\n")
        logger.info("Wrote manifest as JSONL to %s", output_path)
    elif suffix == ".json":
        with output_path.open("w") as fp:
            json.dump({"jobs": jobs}, fp, indent=2)
        logger.info("Wrote manifest as JSON to %s", output_path)
    elif suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pandas (with pyarrow/fastparquet) is required to write parquet output"
            ) from exc
        df = pd.DataFrame(jobs)
        df.to_parquet(output_path, index=False)
        logger.info("Wrote manifest as Parquet to %s", output_path)
    else:
        raise ValueError(
            f"Unsupported manifest format '{suffix}'. Use .jsonl, .json, or .parquet."
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    config, options_dict, default_unified, output_root, default_manifest = load_precompute_options()
    options = PrecomputeOptions(**options_dict)
    chunk_duration_default = options.chunk_samples / options.sample_rate * 1000.0
    chunk_duration_ms = args.chunk_duration_ms or chunk_duration_default
    unified_index_path = (args.unified_index or default_unified).expanduser().resolve()
    output_path = (args.output or default_manifest).expanduser().resolve()
    precomputed_dir = output_root
    logger.info("Using preprocessing profile '%s' for manifest generation", options.profile_name)

    entries = load_unified_entries(unified_index_path)
    jobs = generate_jobs(
        entries=entries,
        chunk_duration_ms=chunk_duration_ms,
        chunk_frames=config["audio"]["features"]["chunk_frames"],
        options=options,
        precomputed_dir=precomputed_dir,
    )
    save_manifest(jobs, output_path)
    logger.info("Chunk manifest generation complete.")


if __name__ == "__main__":
    main()

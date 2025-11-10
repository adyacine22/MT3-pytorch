#!/usr/bin/env python3
"""Generate chunk-level manifest entries from the unified dataset index."""

from __future__ import annotations
import sys
import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs import load_project_config
from configs.project_config import PROJECT_ROOT


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    cfg = load_project_config()
    dataset_paths = cfg["paths"]["datasets"]
    cache_paths = cfg["paths"]["cache"]

    parser = argparse.ArgumentParser(
        description="Generate chunk manifest from unified_index.json."
    )
    parser.add_argument(
        "--unified-index",
        type=Path,
        default=PROJECT_ROOT / dataset_paths["unified_index"],
        help="Path to unified_index.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / cache_paths["chunk_manifest"],
        help="Output manifest path (json/jsonl/parquet)",
    )
    parser.add_argument(
        "--chunk-duration-ms",
        type=float,
        default=None,
        help="Override chunk duration in milliseconds (defaults from config).",
    )
    return parser.parse_args()


def load_unified_entries(unified_index_path: Path) -> List[Dict]:
    if not unified_index_path.exists():
        raise FileNotFoundError(
            f"Unified index not found at {unified_index_path}. "
            "Run data/create_unified_index.py first."
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
    chunk_samples: int,
    sample_rate: int,
    hop_length: int,
    compute_cfg: Dict,
    precomputed_dir: Path,
) -> List[Dict]:
    jobs: List[Dict] = []
    augment_enabled = bool(compute_cfg.get("augment_enabled", False))
    augment_profiles = compute_cfg.get("augment_profiles", ["none"])
    if isinstance(augment_profiles, str):
        augment_profiles = [augment_profiles]

    for entry in entries:
        duration_ms = compute_duration_ms(entry)
        if duration_ms <= 0:
            logger.debug("Skipping %s (no duration info)", entry["track_id"])
            continue

        dataset = entry["dataset"]
        track_id = entry["track_id"]

        # Determine number of chunks (non-overlapping windows)
        num_chunks = max(1, math.ceil(duration_ms / chunk_duration_ms))
        for idx in range(num_chunks):
            start_ms = idx * chunk_duration_ms
            end_ms = min(duration_ms, start_ms + chunk_duration_ms)
            actual_duration = end_ms - start_ms

            chunk_id = f"{track_id}-chunk{idx:05d}"
            stem_ids = []
            if dataset == "slakh_full_mix":
                stem_ids = entry.get("rendered_stem_ids", []) or []
            elif dataset == "slakh_stem":
                stem = entry.get("stem_id")
                if stem:
                    stem_ids = [stem]

            job = {
                "chunk_id": chunk_id,
                "track_id": track_id,
                "dataset": dataset,
                "split": entry.get("split"),
                "audio_path": entry["audio_path"],
                "midi_path": entry["midi_path"],
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
                "precomputed_path": str((precomputed_dir / f"{chunk_id}.pt").as_posix()),
                "precompute": {
                    "chunk_device": compute_cfg.get("chunk_device", "auto"),
                    "batch_size": compute_cfg.get("batch_size"),
                    "max_tokenize_workers": compute_cfg.get("max_tokenize_workers"),
                    "augment_enabled": augment_enabled,
                    "augment_profiles": augment_profiles if augment_enabled else ["none"],
                },
            }
            jobs.append(job)
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

    cfg = load_project_config()
    audio_cfg = cfg["audio"]
    sample_rate = audio_cfg["io"]["sample_rate"]
    feature_cfg = audio_cfg["features"]
    chunk_samples = feature_cfg["chunk_samples"]
    chunk_duration_default = chunk_samples / sample_rate * 1000.0
    chunk_duration_ms = args.chunk_duration_ms or chunk_duration_default
    compute_cfg = cfg.get("compute", {}).get("preprocessing", {})
    cache_paths = cfg["paths"]["cache"]
    precomputed_dir = PROJECT_ROOT / cache_paths.get("precomputed_chunks", Path(cache_paths["root"]) / "precomputed_chunks")

    entries = load_unified_entries(args.unified_index)
    jobs = generate_jobs(
        entries=entries,
        chunk_duration_ms=chunk_duration_ms,
        chunk_frames=feature_cfg["chunk_frames"],
        chunk_samples=chunk_samples,
        sample_rate=sample_rate,
        hop_length=feature_cfg["hop_length"],
        compute_cfg=compute_cfg,
        precomputed_dir=precomputed_dir,
    )
    save_manifest(jobs, args.output)
    logger.info("Chunk manifest generation complete.")


if __name__ == "__main__":
    main()

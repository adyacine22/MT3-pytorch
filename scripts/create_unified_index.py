"""
Utilities for constructing a unified JSON index spanning MAESTRO and Slakh datasets.

The generated index normalises each track (and optionally each Slakh stem) to a
common schema so that downstream PyTorch data loaders can treat all examples the
same way:

{
    "track_id": str,
    "dataset": Literal["maestro", "slakh_full_mix", "slakh_stem"],
    "audio_path": str,
    "midi_path": str,
    "instrument_programs": List[int],
    "instrument_names": List[str],
    "instrument_classes": List[str],
    "split": Literal["train", "validation", "test"]
}
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import soundfile as sf
import yaml

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from configs import load_project_config
from configs.project_config import PROJECT_ROOT as CONFIG_PROJECT_ROOT


MAESTRO_INSTRUMENT_PROGRAM = 0
MAESTRO_INSTRUMENT_NAME = "Acoustic Grand Piano"
MAESTRO_INSTRUMENT_CLASS = "Piano"


def _normalise_split(split: str) -> str:
    """Normalise split strings to `train`, `validation`, or `test`."""
    split = split.strip().lower()
    if split in {"val", "valid", "validation"}:
        return "validation"
    if split in {"evaluate", "eval"}:
        return "validation"
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported split '{split}'.")
    return split


def _make_track_id(prefix: str, *parts: str) -> str:
    """Create a stable track identifier."""
    safe_parts = []
    for part in parts:
        part = part.replace("\\", "/")
        safe_parts.append(part.replace("/", "_"))
    return f"{prefix}-{'_'.join(safe_parts)}"


def _relative_dataset_path(path: Path) -> str:
    """Return a project-relative dataset path, ensuring it starts with datasets/."""
    rel_path = path.resolve().relative_to(CONFIG_PROJECT_ROOT)
    rel_str = rel_path.as_posix()
    if not rel_str.startswith("datasets/"):
        raise ValueError(f"Expected dataset path to start with 'datasets/': {rel_str}")
    return rel_str


def _get_audio_stats(path: Path) -> Dict[str, float]:
    """Return audio metadata (frames, sample rate, duration) for a file."""
    info = sf.info(str(path))
    duration = info.frames / info.samplerate if info.samplerate else 0.0
    return {
        "audio_num_frames": info.frames,
        "audio_sample_rate": info.samplerate,
        "duration_seconds": duration,
    }


def build_maestro_index(maestro_root: Path) -> List[Dict]:
    """Create index entries for the MAESTRO dataset."""
    csv_path = maestro_root / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"MAESTRO metadata CSV not found at {csv_path}")

    logger.info("Building MAESTRO index from %s", csv_path)

    entries: List[Dict] = []
    with csv_path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            audio_rel = row["audio_filename"]
            midi_rel = row["midi_filename"]
            split = _normalise_split(row["split"])

            audio_path = (maestro_root / audio_rel).resolve()
            midi_path = (maestro_root / midi_rel).resolve()

            if not audio_path.exists() or not midi_path.exists():
                # Skip entries with missing files – dataset may be partial.
                continue

            audio_rel = _relative_dataset_path(audio_path)
            midi_rel = _relative_dataset_path(midi_path)
            audio_stats = _get_audio_stats(audio_path)
            duration = float(row.get("duration", audio_stats["duration_seconds"]))

            track_id = _make_track_id("maestro", midi_rel)
            entries.append(
                {
                    "track_id": track_id,
                    "dataset": "maestro",
                    "audio_path": audio_rel,
                    "midi_path": midi_rel,
                    "is_drum": False,
                    "duration_seconds": duration,
                    "audio_num_frames": audio_stats["audio_num_frames"],
                    "audio_sample_rate": audio_stats["audio_sample_rate"],
                    "instrument_programs": [MAESTRO_INSTRUMENT_PROGRAM],
                    "instrument_names": [MAESTRO_INSTRUMENT_NAME],
                    "instrument_classes": [MAESTRO_INSTRUMENT_CLASS],
                    "split": split,
                }
            )

    return entries


def _iter_slakh_tracks(slakh_root: Path) -> Iterable[Path]:
    """Yield every Slakh track directory across train/validation/test splits."""
    logger.debug("Iterating Slakh tracks under %s", slakh_root)
    for split_dir in ["train", "validation", "test"]:
        split_path = slakh_root / split_dir
        if not split_path.exists():
            continue
        for track_dir in sorted(split_path.iterdir()):
            if track_dir.is_dir():
                yield track_dir


def _load_slakh_metadata(track_dir: Path) -> Dict:
    metadata_path = track_dir / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing Slakh metadata file: {metadata_path}")
    with metadata_path.open("r") as fp:
        return yaml.safe_load(fp)


def build_slakh_full_mix_index(slakh_root: Path) -> List[Dict]:
    """Create index entries for Slakh full mixes."""
    logger.info("Building Slakh full mix index from %s", slakh_root)
    entries: List[Dict] = []
    for track_dir in _iter_slakh_tracks(slakh_root):
        metadata = _load_slakh_metadata(track_dir)
        split = _normalise_split(track_dir.parent.name)

        audio_path = (track_dir / "mix.flac").resolve()
        midi_path = (track_dir / "all_src.mid").resolve()

        if not audio_path.exists() or not midi_path.exists():
            continue

        audio_rel = _relative_dataset_path(audio_path)
        midi_rel = _relative_dataset_path(midi_path)
        audio_stats = _get_audio_stats(audio_path)

        programs: List[int] = []
        names: List[str] = []
        classes: List[str] = []
        stem_ids: List[str] = []

        for stem_id, stem_meta in sorted(metadata.get("stems", {}).items()):
            if not stem_meta.get("audio_rendered", False):
                continue
            if not stem_meta.get("midi_saved", False):
                continue

            program = int(stem_meta.get("program_num", 0))
            programs.append(program)
            names.append(stem_meta.get("midi_program_name", "Unknown"))
            classes.append(stem_meta.get("inst_class", "Unknown"))
            stem_ids.append(stem_id)

        track_id = _make_track_id("slakh_full_mix", track_dir.name)
        entries.append(
            {
                "track_id": track_id,
                "dataset": "slakh_full_mix",
                "audio_path": audio_rel,
                "midi_path": midi_rel,
                "duration_seconds": audio_stats["duration_seconds"],
                "audio_num_frames": audio_stats["audio_num_frames"],
                "audio_sample_rate": audio_stats["audio_sample_rate"],
                "instrument_programs": programs,
                "instrument_names": names,
                "instrument_classes": classes,
                "num_rendered_stems": len(programs),
                "num_total_stems": len(metadata.get("stems", {})),
                "rendered_stem_ids": stem_ids,
                "split": split,
            }
        )

    return entries


def build_slakh_stem_index(slakh_root: Path) -> List[Dict]:
    """Create index entries for individual Slakh stems."""
    logger.info("Building Slakh stem index from %s", slakh_root)
    entries: List[Dict] = []
    for track_dir in _iter_slakh_tracks(slakh_root):
        metadata = _load_slakh_metadata(track_dir)
        split = _normalise_split(track_dir.parent.name)
        stems_meta = metadata.get("stems", {})

        for stem_id, stem_meta in sorted(stems_meta.items()):
            if not stem_meta.get("audio_rendered", False):
                continue
            if not stem_meta.get("midi_saved", False):
                continue

            audio_path = (track_dir / "stems" / f"{stem_id}.flac").resolve()
            midi_path = (track_dir / "MIDI" / f"{stem_id}.mid").resolve()
            if not audio_path.exists() or not midi_path.exists():
                continue

            audio_rel = _relative_dataset_path(audio_path)
            midi_rel = _relative_dataset_path(midi_path)
            audio_stats = _get_audio_stats(audio_path)

            program = int(stem_meta.get("program_num", 0))
            names = [stem_meta.get("midi_program_name", "Unknown")]
            classes = [stem_meta.get("inst_class", "Unknown")]

            track_id = _make_track_id("slakh_stem", track_dir.name, stem_id)
            entries.append(
                {
                    "track_id": track_id,
                    "dataset": "slakh_stem",
                    "audio_path": audio_rel,
                    "midi_path": midi_rel,
                    "duration_seconds": audio_stats["duration_seconds"],
                    "audio_num_frames": audio_stats["audio_num_frames"],
                    "audio_sample_rate": audio_stats["audio_sample_rate"],
                    "instrument_programs": [program],
                    "instrument_names": names,
                    "instrument_classes": classes,
                    "stem_id": stem_id,
                    "is_drum": bool(stem_meta.get("is_drum", False)),
                    "split": split,
                }
            )

    return entries


def build_unified_index(maestro_root: Path, slakh_root: Path) -> List[Dict]:
    """Create the complete unified dataset index."""
    logger.info("Starting unified index build")
    entries = []
    entries.extend(build_maestro_index(maestro_root))
    entries.extend(build_slakh_full_mix_index(slakh_root))
    entries.extend(build_slakh_stem_index(slakh_root))
    logger.info("Unified index build complete: %d entries", len(entries))
    return sorted(entries, key=lambda item: (item["dataset"], item["track_id"]))


def save_index(entries: List[Dict], output_path: Path, root_path: Path = CONFIG_PROJECT_ROOT) -> None:
    """Write the unified index to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "root": str(root_path.resolve()),
        "entries": entries,
    }
    with output_path.open("w") as fp:
        json.dump(payload, fp, indent=2)


def parse_args() -> argparse.Namespace:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    cfg = load_project_config()
    dataset_paths = cfg["paths"]["datasets"]
    maestro_default = (PROJECT_ROOT / dataset_paths["maestro_root"]).resolve()
    slakh_default = (PROJECT_ROOT / dataset_paths["slakh_root"]).resolve()
    output_default = (PROJECT_ROOT / dataset_paths["unified_index"]).resolve()

    parser = argparse.ArgumentParser(description="Create unified dataset index for MT3-PyTorch.")
    parser.add_argument(
        "--maestro-root",
        type=Path,
        default=maestro_default,
        help=f"Path to the MAESTRO v3 root directory (default: {maestro_default})",
    )
    parser.add_argument(
        "--slakh-root",
        type=Path,
        default=slakh_default,
        help=f"Path to the Slakh2100 root directory (default: {slakh_default})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=output_default,
        help=f"Destination JSON path (default: {output_default})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maestro_root = args.maestro_root.resolve()
    slakh_root = args.slakh_root.resolve()
    output_path = args.output.resolve()

    entries = build_unified_index(maestro_root, slakh_root)
    save_index(entries, output_path)
    print(f"Wrote {len(entries)} entries to {output_path}")


if __name__ == "__main__":
    main()

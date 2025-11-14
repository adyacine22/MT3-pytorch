"""Configuration + environment resolution helpers for preprocessing scripts."""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

from configs import load_project_config
from configs.project_config import PROJECT_ROOT


def resolve_dataset_path(path_str: str | Path) -> Path:
    """Resolve a dataset-relative path into an absolute path under the repo root."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_limited_int(value: str | None) -> int | None:
    if value is None or value.strip().lower() == "all":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("Expected a positive integer override")
    return parsed


def _resolve_preprocess_profile(
    section: Dict[str, Any], profile_override: str | None
) -> Tuple[Dict[str, Any], str]:
    if "profiles" not in section:
        return dict(section), profile_override or "default"
    profiles = section.get("profiles") or {}
    if not profiles:
        return dict(section), profile_override or "default"
    default_profile = section.get("default_profile")
    profile_name = profile_override or default_profile
    if not profile_name or profile_name not in profiles:
        profile_name = default_profile if default_profile in profiles else next(iter(profiles))
    return dict(profiles[profile_name]), profile_name



def _normalize_dataset_list(dataset_pref: Any) -> list[str]:
    if isinstance(dataset_pref, str):
        return [dataset_pref]
    if isinstance(dataset_pref, Iterable):
        return [str(name) for name in dataset_pref if str(name)]
    return []


@dataclass(frozen=True)
class PrecomputeOptions:
    chunk_device: str | None
    augment_enabled: bool
    augment_profiles: list[str]
    batch_size: int
    max_tokenize_workers: int
    store_tokens: bool
    track_workers: int
    chunk_samples: int
    hop_length: int
    num_mels: int
    sample_rate: int
    pad_value: float
    datasets: list[str]
    max_tracks_per_dataset: int | None
    max_chunks_per_track: int | None
    dry_run: bool
    overwrite: bool
    store_spectrogram: bool
    write_manifest: bool
    skip_existing: bool
    chunk_storage: str
    profile_name: str

    def manifest_config(self) -> Dict[str, Any]:
        return {
            "chunk_samples": self.chunk_samples,
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "batch_size": self.batch_size,
            "max_tokenize_workers": self.max_tokenize_workers,
        }

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_precompute_options(
    env: Mapping[str, str] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Path, Path, Path]:
    """Return (config, options, unified_index, output_root, manifest_path)."""
    env = env or os.environ
    config = load_project_config()
    audio_cfg = config["audio"]
    feature_cfg = audio_cfg["features"]
    io_cfg = audio_cfg["io"]
    preprocess_cfg = audio_cfg["preprocess"]
    preprocess_section = config.get("compute", {}).get("preprocessing", {})
    profile_override = env.get("PRECOMPUTE_PROFILE")
    compute_cfg, profile_name = _resolve_preprocess_profile(preprocess_section, profile_override)

    chunk_device_pref = str(compute_cfg.get("chunk_device", "auto")).lower()
    chunk_device = None if chunk_device_pref in {"auto", ""} else chunk_device_pref
    storage_strategy = str(compute_cfg.get("chunk_storage", "per_chunk")).lower()
    augment_profiles_cfg = compute_cfg.get("augment_profiles", ["none"])
    augment_profiles = (
        [augment_profiles_cfg]
        if isinstance(augment_profiles_cfg, str)
        else [str(name) for name in augment_profiles_cfg] or ["none"]
    )

    datasets = _normalize_dataset_list(compute_cfg.get("datasets"))

    options: Dict[str, Any] = {
        "chunk_device": chunk_device,
        "augment_enabled": bool(compute_cfg.get("augment_enabled", False)),
        "augment_profiles": augment_profiles,
        "batch_size": max(1, int(compute_cfg.get("batch_size", 1))),
        "max_tokenize_workers": max(
            0, int(compute_cfg.get("max_tokenize_workers", os.cpu_count() or 1))
        ),
        "store_tokens": bool(compute_cfg.get("store_tokens", True)),
        "track_workers": max(1, int(compute_cfg.get("track_workers", 1))),
        "chunk_samples": int(feature_cfg["chunk_samples"]),
        "hop_length": int(feature_cfg["hop_length"]),
        "num_mels": int(feature_cfg["num_mels"]),
        "sample_rate": int(io_cfg["sample_rate"]),
        "pad_value": float(preprocess_cfg["pad_value"]),
        "datasets": datasets,
        "max_tracks_per_dataset": (
            None
            if compute_cfg.get("max_tracks_per_dataset") in {None, "all"}
            else max(1, int(compute_cfg.get("max_tracks_per_dataset", 1)))
        ),
        "max_chunks_per_track": (
            None
            if compute_cfg.get("max_chunks_per_track") in {None, "all"}
            else max(1, int(compute_cfg.get("max_chunks_per_track", 4)))
        ),
        "dry_run": bool(compute_cfg.get("dry_run", False)),
        "overwrite": bool(compute_cfg.get("overwrite", False)),
        "store_spectrogram": bool(compute_cfg.get("store_spectrogram", True)),
        "write_manifest": bool(compute_cfg.get("write_manifest", True)),
        "skip_existing": bool(compute_cfg.get("skip_existing", True)),
        "chunk_storage": storage_strategy if storage_strategy in {"per_chunk", "per_track"} else "per_chunk",
        "profile_name": profile_name,
    }

    dataset_paths = config["paths"]["datasets"]
    cache_paths = config["paths"]["cache"]
    unified_index_path = resolve_dataset_path(dataset_paths["unified_index"])
    output_root = resolve_dataset_path(cache_paths.get("precomputed_chunks", "cache/precomputed_chunks"))
    manifest_rel = cache_paths.get("chunk_manifest")
    manifest_path = (
        resolve_dataset_path(manifest_rel)
        if manifest_rel
        else output_root / "chunk_manifest.parquet"
    )

    # Environment overrides
    if env.get("PRECOMPUTE_DATASETS"):
        options["datasets"] = [part.strip() for part in env["PRECOMPUTE_DATASETS"].split(",") if part.strip()]
    if env.get("PRECOMPUTE_MAX_TRACKS"):
        options["max_tracks_per_dataset"] = _parse_limited_int(env["PRECOMPUTE_MAX_TRACKS"])
    if env.get("PRECOMPUTE_MAX_CHUNKS"):
        options["max_chunks_per_track"] = _parse_limited_int(env["PRECOMPUTE_MAX_CHUNKS"])
    if env.get("PRECOMPUTE_MAX_TOKENIZE_WORKERS"):
        options["max_tokenize_workers"] = max(0, int(env["PRECOMPUTE_MAX_TOKENIZE_WORKERS"]))
    if env.get("PRECOMPUTE_STORE_TOKENS"):
        options["store_tokens"] = _parse_bool(env.get("PRECOMPUTE_STORE_TOKENS"), options["store_tokens"])
    if env.get("PRECOMPUTE_TRACK_WORKERS"):
        options["track_workers"] = max(1, int(env["PRECOMPUTE_TRACK_WORKERS"]))
    if env.get("PRECOMPUTE_CHUNK_STORAGE"):
        storage_override = env["PRECOMPUTE_CHUNK_STORAGE"].strip().lower()
        if storage_override in {"per_chunk", "per_track"}:
            options["chunk_storage"] = storage_override
    options["dry_run"] = _parse_bool(env.get("PRECOMPUTE_DRY_RUN"), options["dry_run"])
    options["overwrite"] = _parse_bool(env.get("PRECOMPUTE_OVERWRITE"), options["overwrite"])
    options["write_manifest"] = _parse_bool(env.get("PRECOMPUTE_WRITE_MANIFEST"), options["write_manifest"])
    options["skip_existing"] = _parse_bool(env.get("PRECOMPUTE_SKIP_EXISTING"), options["skip_existing"])

    if env.get("PRECOMPUTE_OUTPUT_ROOT"):
        output_root = Path(env["PRECOMPUTE_OUTPUT_ROOT"]).expanduser().resolve()
    if env.get("PRECOMPUTE_MANIFEST_PATH"):
        manifest_path = Path(env["PRECOMPUTE_MANIFEST_PATH"]).expanduser().resolve()

    return config, options, unified_index_path, output_root, manifest_path


__all__ = ["load_precompute_options", "resolve_dataset_path", "PrecomputeOptions"]

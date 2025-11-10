#!/usr/bin/env python
"""Process the unified index and emit precomputed chunk tensors + manifest."""

from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, Future
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"pydub\.utils",
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"pretty_midi\.pretty_midi",
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import note_seq  
import pandas as pd  
import torch
from tqdm import tqdm

from configs import load_project_config
from configs.project_config import PROJECT_ROOT
from data.audio import io as audio_io
from data.audio import spectrogram
from data.audio.augment import apply_augmentation
from data.symbolic import tokenizer


LOG_PREFIX = "[precompute]"


def _log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}")


def _time_step(name: str, fn):
    start = time.perf_counter()
    result = fn()
    duration = time.perf_counter() - start
    _log(f"[time] {name}: {duration:0.3f}s")
    return result


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

def _load_config() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config = load_project_config()
    audio_cfg = config["audio"]
    feature_cfg = audio_cfg["features"]
    io_cfg = audio_cfg["io"]
    preprocess_cfg = audio_cfg["preprocess"]
    compute_cfg = config.get("compute", {}).get("preprocessing", {})
    chunk_device_pref = str(compute_cfg.get("chunk_device", "auto")).lower()
    chunk_device = None if chunk_device_pref in {"auto", ""} else chunk_device_pref
    profiles_value = compute_cfg.get("augment_profiles", ["none"])
    if isinstance(profiles_value, str):
        augment_profiles = [profiles_value]
    else:
        augment_profiles = [str(name) for name in profiles_value] or ["none"]
    dataset_pref = compute_cfg.get("datasets")
    if isinstance(dataset_pref, str):
        datasets = [dataset_pref]
    elif isinstance(dataset_pref, Iterable):
        datasets = [str(name) for name in dataset_pref if str(name)]
    else:
        datasets = []
    options = {
        "chunk_device": chunk_device,
        "augment_enabled": bool(compute_cfg.get("augment_enabled", False)),
        "augment_profiles": augment_profiles,
        "batch_size": max(1, int(compute_cfg.get("batch_size", 1))),
        "max_tokenize_workers": max(
            0, int(compute_cfg.get("max_tokenize_workers", os.cpu_count() or 1))
        ),
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
    }
    return config, options


def _load_unified_entries(unified_index_path: Path) -> List[Dict[str, Any]]:
    if not unified_index_path.exists():
        from data.create_unified_index import build_unified_index, save_index

        _log(f"Unified index missing, building {unified_index_path}")
        config = load_project_config()
        dataset_paths = config["paths"]["datasets"]
        maestro_root = _resolve_dataset_path(dataset_paths["maestro_root"])
        slakh_root = _resolve_dataset_path(dataset_paths["slakh_root"])
        entries = build_unified_index(maestro_root, slakh_root)
        unified_index_path.parent.mkdir(parents=True, exist_ok=True)
        save_index(entries, unified_index_path, root_path=PROJECT_ROOT)
        return entries

    with unified_index_path.open() as fp:
        payload = json.load(fp)
    if isinstance(payload, dict):
        entries = payload.get("entries", [])
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError("Unified index must be a list or contain an 'entries' field.")
    return entries


def _select_cases(
    entries: List[Dict[str, Any]],
    dataset: str,
    max_tracks: int | None,
) -> List[Dict[str, Any]]:
    dataset_entries = [entry for entry in entries if entry.get("dataset") == dataset]
    cases: List[Dict[str, Any]] = []
    selected = dataset_entries if max_tracks is None else dataset_entries[:max_tracks]
    for entry in selected:
        audio_path = _resolve_dataset_path(entry["audio_path"])
        midi_path = _resolve_dataset_path(entry["midi_path"])
        if not audio_path.exists() or not midi_path.exists():
            continue
        cases.append(
            {
                "dataset": dataset,
                "audio_path": audio_path,
                "midi_path": midi_path,
                "metadata": entry,
            }
        )
    return cases


def _dataset_order(entries: List[Dict[str, Any]], requested: Sequence[str]) -> List[str]:
    discovered = sorted(
        {str(entry.get("dataset")) for entry in entries if entry.get("dataset")}
    )
    if not requested:
        return discovered
    requested_set = []
    available = set(discovered)
    for name in requested:
        if name in available:
            requested_set.append(name)
    return requested_set


def _resolve_dataset_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path_str).resolve()


def _plan_chunk_positions(
    total_samples: int, sample_rate: int, chunk_samples: int
) -> List[Dict[str, float]]:
    if total_samples <= 0 or chunk_samples <= 0:
        return [{"start_sample": 0, "start_s": 0.0}]
    positions: List[Dict[str, float]] = []
    start = 0
    while start <= total_samples - chunk_samples:
        positions.append({"start_sample": start, "start_s": start / sample_rate})
        start += chunk_samples
    if not positions or positions[-1]["start_sample"] < total_samples - chunk_samples:
        last_start = max(0, total_samples - chunk_samples)
        positions.append({"start_sample": last_start, "start_s": last_start / sample_rate})
    return positions


TOKENIZE_WORKER_THRESHOLDS: List[Tuple[float, int]] = [
    (500.0, 1),
    (1500.0, 4),
    (3500.0, 16),
    (float("inf"), 32),
]


def estimate_work_units(note_sequence: note_seq.NoteSequence, chunk_count: int) -> float:
    duration = max(float(note_sequence.total_time), 1e-6)
    density = len(note_sequence.notes) / duration if duration > 0 else 0.0
    return density * max(1, chunk_count)


def select_tokenize_workers(
    note_sequence: note_seq.NoteSequence,
    chunk_count: int,
    max_tokenize_workers: int,
) -> int:
    """Select an adaptive worker count based on event density and chunk count."""
    if max_tokenize_workers <= 0:
        return 0
    work_units = estimate_work_units(note_sequence, chunk_count)
    for threshold, worker_target in TOKENIZE_WORKER_THRESHOLDS:
        if work_units < threshold:
            return min(worker_target, max_tokenize_workers)
    return min(TOKENIZE_WORKER_THRESHOLDS[-1][1], max_tokenize_workers)


def _batched(plans: List[Dict[str, float]], batch_size: int) -> Iterable[List[Dict[str, float]]]:
    if batch_size <= 1:
        for plan in plans:
            yield [plan]
        return
    for idx in range(0, len(plans), batch_size):
        yield plans[idx : idx + batch_size]


def _slugify(text: str | None, fallback: str) -> str:
    if not text:
        text = fallback
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in text)
    return safe.strip("_") or fallback


def _tokenize_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    import note_seq
    from data.symbolic import tokenizer as tokenizer_module

    ns = note_seq.NoteSequence()
    ns.ParseFromString(payload["note_sequence_bytes"])
    start = time.perf_counter()
    output = tokenizer_module.tokenize_note_sequence(
        ns,
        chunk_start_s=payload["chunk_start_s"],
        chunk_end_s=payload["chunk_end_s"],
        frame_times=payload["frame_times"],
    )
    duration = time.perf_counter() - start
    return {
        "chunk_index": payload["chunk_index"],
        "tokens": output.tokens,
        "tokenize_time": duration,
    }


def _save_chunk(
    dest_dir: Path,
    chunk_slug: str,
    log_mel: torch.Tensor | None,
    waveform: torch.Tensor | None,
    frame_times: Sequence[float],
    tokens: Sequence[int],
    meta: Dict[str, Any],
    dry_run: bool,
    overwrite: bool,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = dest_dir / f"{chunk_slug}.pt"
    if not overwrite and chunk_path.exists():
        return chunk_path
    if dry_run:
        return chunk_path
    payload = {
        "frame_times": torch.tensor(frame_times, dtype=torch.float32),
        "chunk_start_s": float(meta["chunk_start_s"]),
        "chunk_end_s": float(meta["chunk_end_s"]),
        "tokens": torch.tensor(tokens, dtype=torch.int32),
        "spectrogram_device": meta["spectrogram_device"],
        "dataset": meta["dataset"],
        "track_id": meta["track_id"],
        "audio_path": meta["audio_path"],
        "midi_path": meta["midi_path"],
        "augment_profiles": meta["augment_profiles"],
    }
    if log_mel is not None:
        payload["log_mel"] = log_mel.cpu()
    if waveform is not None:
        payload["waveform"] = waveform.detach().cpu().to(dtype=torch.float16)
    torch.save(payload, chunk_path)
    return chunk_path


def _process_case(
    case: Dict[str, Any],
    options: Dict[str, Any],
    output_root: Path,
    max_chunks: int | None,
    dry_run: bool,
    overwrite: bool,
) -> Tuple[List[Dict[str, Any]], int]:
    waveform, sample_rate = audio_io.prepare_waveform(str(case["audio_path"]))
    note_sequence = note_seq.midi_file_to_note_sequence(str(case["midi_path"]))
    chunk_samples = options["chunk_samples"]
    plans = _plan_chunk_positions(len(waveform), sample_rate, chunk_samples)
    chunk_duration_s = chunk_samples / sample_rate

    planned_chunk_count = len(plans)
    adaptive_workers = select_tokenize_workers(
        note_sequence, planned_chunk_count, options["max_tokenize_workers"]
    )
    token_pool: ProcessPoolExecutor | None = (
        ProcessPoolExecutor(max_workers=adaptive_workers)
        if adaptive_workers > 0
        else None
    )
    token_futures: List[Tuple[Future, Dict[str, Any], Dict[str, Any]]] = []
    serialized_ns = note_sequence.SerializeToString()
    hop_seconds = options["hop_length"] / sample_rate
    manifest_rows: List[Dict[str, Any]] = []

    metadata = case["metadata"]
    track_slug = _slugify(metadata.get("track_id"), case["audio_path"].stem)
    dest_dir = output_root / case["dataset"] / track_slug

    chunk_counter = 0
    total_chunks = 0
    written_chunks = 0

    for batch in _batched(plans, options["batch_size"]):
        if max_chunks and chunk_counter >= max_chunks:
            break
        batch_data: List[Dict[str, Any]] = []
        for plan in batch:
            if max_chunks and chunk_counter >= max_chunks:
                break
            chunk_idx = total_chunks
            total_chunks += 1
            chunk_start_s = plan["start_s"]
            chunk_end_s = min(chunk_start_s + chunk_duration_s, len(waveform) / sample_rate)
            if chunk_end_s <= chunk_start_s:
                chunk_end_s = chunk_start_s + 1e-6
            chunk_slug = f"chunk_{chunk_idx:05d}"
            chunk_path = dest_dir / f"{chunk_slug}.pt"
            if (
                options["skip_existing"]
                and not overwrite
                and chunk_path.exists()
                and not options["write_manifest"]
            ):
                # Skip processing entirely; chunk already on disk and manifest won't be updated.
                chunk_counter += 1
                continue
            chunk_entry = {
                "chunk_index": chunk_idx,
                "chunk_start_s": chunk_start_s,
                "chunk_end_s": chunk_end_s,
                "chunk_slug": chunk_slug,
                "chunk_path": chunk_path,
            }
            chunk = audio_io.slice_chunk(
                waveform,
                start_sample=int(plan["start_sample"]),
                chunk_samples=chunk_samples,
                pad_value=options["pad_value"],
            )
            chunk_tensor = torch.as_tensor(
                chunk,
                dtype=torch.float32,
                device=options["chunk_device"] or "cpu",
            )
            if options["augment_enabled"]:
                chunk_tensor = apply_augmentation(
                    chunk_tensor, sample_rate=sample_rate, profiles=options["augment_profiles"]
                )
            if options["store_spectrogram"]:
                chunk_entry["tensor"] = chunk_tensor
            else:
                chunk_entry["waveform_cpu"] = chunk_tensor.detach().to("cpu")
            batch_data.append(chunk_entry)
            chunk_counter += 1

        if not batch_data:
            continue

        if options["store_spectrogram"]:
            stacked = torch.stack([item["tensor"] for item in batch_data], dim=0)
            if options["chunk_device"]:
                stacked = stacked.to(device=options["chunk_device"])
            log_mel_batch = spectrogram.waveform_to_logmel(
                stacked,
                sample_rate=sample_rate,
                device=options["chunk_device"],
            )
            if log_mel_batch.ndim == 2:
                log_mel_batch = log_mel_batch.unsqueeze(0)
            batch_log_mels: List[torch.Tensor | None] = [tensor for tensor in log_mel_batch]
        else:
            log_mel_batch = None
            batch_log_mels = [None] * len(batch_data)

        for chunk_data, log_mel in zip(batch_data, batch_log_mels):
            if log_mel is not None:
                frames = log_mel.shape[-1]
                frame_times = (torch.arange(frames, dtype=torch.float32) * hop_seconds).tolist()
                spectrogram_device = str(log_mel.device)
                stored_log_mel = log_mel.detach().cpu()
            else:
                duration = chunk_data["chunk_end_s"] - chunk_data["chunk_start_s"]
                frames = max(1, int(math.ceil(duration / hop_seconds)))
                frame_times = [idx * hop_seconds for idx in range(frames)]
                spectrogram_device = "none"
                stored_log_mel = None
            stored_waveform = None
            if not options["store_spectrogram"]:
                stored_waveform = chunk_data.pop("waveform_cpu", None)
            chunk_meta = {
                "chunk_index": chunk_data["chunk_index"],
                "chunk_start_s": chunk_data["chunk_start_s"],
                "chunk_end_s": chunk_data["chunk_end_s"],
                "frames": frames,
                "spectrogram_device": spectrogram_device,
                "dataset": case["dataset"],
                "track_id": track_slug,
                "audio_path": metadata.get("audio_path"),
                "midi_path": metadata.get("midi_path"),
                "augment_profiles": options["augment_profiles"] if options["augment_enabled"] else [],
                "tokenize_workers": adaptive_workers,
                "max_tokenize_workers": options["max_tokenize_workers"],
            }
            chunk_slug = chunk_data["chunk_slug"]
            chunk_payload = {
                "log_mel": stored_log_mel,
                "waveform": stored_waveform,
                "frame_times": frame_times,
                "meta": chunk_meta,
                "slug": chunk_slug,
            }
            if token_pool:
                payload = {
                    "chunk_index": chunk_meta["chunk_index"],
                    "chunk_start_s": chunk_meta["chunk_start_s"],
                    "chunk_end_s": chunk_meta["chunk_end_s"],
                    "frame_times": frame_times,
                    "note_sequence_bytes": serialized_ns,
                }
                future = token_pool.submit(_tokenize_job, payload)
                token_futures.append((future, chunk_payload, chunk_meta))
            else:
                token_output = tokenizer.tokenize_note_sequence(
                    note_sequence,
                    chunk_start_s=chunk_meta["chunk_start_s"],
                    chunk_end_s=chunk_meta["chunk_end_s"],
                    frame_times=frame_times,
                )
                chunk_path = _save_chunk(
                    dest_dir,
                    chunk_slug,
                    chunk_payload["log_mel"],
                    chunk_payload["waveform"],
                    frame_times,
                    token_output.tokens,
                    chunk_meta,
                    dry_run,
                    overwrite,
                )
                written_chunks += 1
                if options["write_manifest"]:
                    manifest_rows.append(
                        _manifest_row(chunk_meta, chunk_path, token_output.tokens, options)
                    )

        if options["store_spectrogram"]:
            del stacked, log_mel_batch

    if token_pool:
        for future, chunk_payload, chunk_meta in token_futures:
            result = future.result()
            tokens = result["tokens"]
            chunk_path = _save_chunk(
                dest_dir,
                chunk_payload["slug"],
                chunk_payload["log_mel"],
                chunk_payload["waveform"],
                chunk_payload["frame_times"],
                tokens,
                chunk_meta,
                dry_run,
                overwrite,
            )
            written_chunks += 1
            if options["write_manifest"]:
                manifest_rows.append(_manifest_row(chunk_meta, chunk_path, tokens, options))
        token_pool.shutdown()

    return manifest_rows if options["write_manifest"] else [], written_chunks


def _manifest_row(
    chunk_meta: Dict[str, Any],
    chunk_path: Path,
    tokens: Sequence[int],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "dataset": chunk_meta["dataset"],
        "track_id": chunk_meta["track_id"],
        "chunk_index": chunk_meta["chunk_index"],
        "chunk_start_s": chunk_meta["chunk_start_s"],
        "chunk_end_s": chunk_meta["chunk_end_s"],
        "frames": chunk_meta["frames"],
        "tokens": len(tokens),
        "chunk_path": chunk_path.resolve().as_posix(),
        "spectrogram_device": chunk_meta["spectrogram_device"],
        "augment_profiles": ",".join(chunk_meta["augment_profiles"]),
        "batch_size": options["batch_size"],
        "tokenize_workers_used": chunk_meta["tokenize_workers"],
        "max_tokenize_workers": options["max_tokenize_workers"],
        "chunk_samples": options["chunk_samples"],
        "sample_rate": options["sample_rate"],
    }


def process_unified_index() -> Path:
    config, options = _load_config()
    dataset_paths = config["paths"]["datasets"]
    cache_paths = config["paths"]["cache"]
    unified_index_path = _resolve_dataset_path(dataset_paths["unified_index"])
    output_root = _resolve_dataset_path(cache_paths.get("precomputed_chunks", "cache/precomputed_chunks"))
    manifest_rel = cache_paths.get("chunk_manifest")
    manifest_path = (
        _resolve_dataset_path(manifest_rel)
        if manifest_rel
        else output_root / "chunk_manifest.parquet"
    )

    env_output = os.environ.get("PRECOMPUTE_OUTPUT_ROOT")
    if env_output:
        output_root = Path(env_output).expanduser().resolve()
    env_manifest = os.environ.get("PRECOMPUTE_MANIFEST_PATH")
    if env_manifest:
        manifest_path = Path(env_manifest).expanduser().resolve()
    env_datasets = os.environ.get("PRECOMPUTE_DATASETS")
    if env_datasets:
        options["datasets"] = [part.strip() for part in env_datasets.split(",") if part.strip()]
    env_max_tracks = os.environ.get("PRECOMPUTE_MAX_TRACKS")
    if env_max_tracks:
        options["max_tracks_per_dataset"] = _parse_limited_int(env_max_tracks)
    env_max_chunks = os.environ.get("PRECOMPUTE_MAX_CHUNKS")
    if env_max_chunks:
        options["max_chunks_per_track"] = _parse_limited_int(env_max_chunks)
    env_max_token_workers = os.environ.get("PRECOMPUTE_MAX_TOKENIZE_WORKERS")
    if env_max_token_workers:
        options["max_tokenize_workers"] = max(0, int(env_max_token_workers))
    options["dry_run"] = _parse_bool(os.environ.get("PRECOMPUTE_DRY_RUN"), options["dry_run"])
    options["overwrite"] = _parse_bool(
        os.environ.get("PRECOMPUTE_OVERWRITE"), options["overwrite"]
    )
    options["write_manifest"] = _parse_bool(
        os.environ.get("PRECOMPUTE_WRITE_MANIFEST"), options["write_manifest"]
    )
    options["skip_existing"] = _parse_bool(
        os.environ.get("PRECOMPUTE_SKIP_EXISTING"), options["skip_existing"]
    )

    output_root.mkdir(parents=True, exist_ok=True)
    if options["write_manifest"]:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _log(
        "Starting unified-index processing "
        f"(datasets={options['datasets'] or 'all'} "
        f"max_tracks={options['max_tracks_per_dataset'] or 'all'} "
        f"max_chunks={options['max_chunks_per_track'] or 'all'} "
        f"dry_run={options['dry_run']} overwrite={options['overwrite']} "
        f"max_tokenize_workers={options['max_tokenize_workers']} "
        f"write_manifest={options['write_manifest']})"
    )

    entries = _time_step("load_unified_index", lambda: _load_unified_entries(unified_index_path))
    dataset_order = _dataset_order(entries, options["datasets"])
    if not dataset_order:
        _log("No datasets available in unified index.")
        return manifest_path

    records: List[Dict[str, Any]] = []
    total_start = time.perf_counter()
    total_chunks = 0

    for dataset in dataset_order:
        dataset_start = time.perf_counter()
        cases = _select_cases(entries, dataset, options["max_tracks_per_dataset"])
        if not cases:
            _log(f"[dataset] {dataset}: no eligible audio/midi pairs found, skipping.")
            continue
        _log(f"[dataset] {dataset}: processing {len(cases)} track(s)")
        progress = tqdm(
            cases,
            desc=f"{dataset} tracks",
            unit="track",
            leave=False,
        )
        dataset_chunk_count = 0
        for case in progress:
            progress.set_postfix_str(case["metadata"].get("track_id") or "")
            rows, chunk_count = _process_case(
                case,
                options,
                output_root,
                max_chunks=options["max_chunks_per_track"],
                dry_run=options["dry_run"],
                overwrite=options["overwrite"],
            )
            dataset_chunk_count += chunk_count
            total_chunks += chunk_count
            if options["write_manifest"]:
                records.extend(rows)
        progress.close()
        dataset_elapsed = time.perf_counter() - dataset_start
        _log(f"[dataset] {dataset}: {dataset_chunk_count} chunk(s) in {dataset_elapsed:0.3f}s")

    if total_chunks == 0:
        _log("No chunks were generated.")
        return manifest_path

    if options["write_manifest"]:
        df = _time_step("build_manifest", lambda: pd.DataFrame.from_records(records))
        _time_step("write_manifest", lambda: df.to_parquet(manifest_path, index=False))
        _log(f"Wrote {len(records)} chunk rows → {manifest_path}")
    else:
        _log("Manifest writing disabled; run data/generate_chunk_manifest.py later if needed.")
    total_elapsed = time.perf_counter() - total_start
    _log(f"Total processing time: {total_elapsed:0.3f}s")
    return manifest_path


if __name__ == "__main__":
    process_unified_index()

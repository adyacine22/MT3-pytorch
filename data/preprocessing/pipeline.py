#!/usr/bin/env python
"""Process the unified index and emit precomputed chunk tensors + manifest."""

from __future__ import annotations

import json
import math
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import note_seq  
import pandas as pd  
import torch
from tqdm import tqdm

from configs.project_config import PROJECT_ROOT
from data.audio import io as audio_io
from data.audio import spectrogram
from data.audio.augment import apply_augmentation
from data.preprocessing.manifest import build_manifest_row
from data.preprocessing.options import load_precompute_options, resolve_dataset_path
from data.preprocessing.storage import (
    ChunkShardWriter,
    batched,
    build_chunk_payload,
    plan_chunk_positions,
    slugify,
)
from data.preprocessing.tokenization import select_tokenize_workers, tokenize_chunk
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


 


def _load_unified_entries(unified_index_path: Path, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not unified_index_path.exists():
        from scripts.create_unified_index import build_unified_index, save_index

        _log(f"Unified index missing, building {unified_index_path}")
        dataset_paths = config["paths"]["datasets"]
        maestro_root = resolve_dataset_path(dataset_paths["maestro_root"])
        slakh_root = resolve_dataset_path(dataset_paths["slakh_root"])
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
        audio_path = resolve_dataset_path(entry["audio_path"])
        midi_path = resolve_dataset_path(entry["midi_path"])
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



def _process_case(
    case: Dict[str, Any],
    options: Dict[str, Any],
    output_root: Path,
    max_chunks: int | None,
    dry_run: bool,
    overwrite: bool,
) -> Tuple[List[Dict[str, Any]], int]:
    waveform, sample_rate = audio_io.prepare_waveform(str(case["audio_path"]))
    store_tokens = options.get("store_tokens", True)
    note_sequence: note_seq.NoteSequence | None = None
    if store_tokens:
        note_sequence = note_seq.midi_file_to_note_sequence(str(case["midi_path"]))
    chunk_samples = options["chunk_samples"]
    plans = plan_chunk_positions(len(waveform), sample_rate, chunk_samples)
    chunk_duration_s = chunk_samples / sample_rate

    planned_chunk_count = len(plans)
    adaptive_workers = 0
    token_pool: ProcessPoolExecutor | None = None
    token_futures: List[Tuple[Future, Dict[str, Any], Dict[str, Any], str]] = []
    serialized_ns: bytes | None = None
    if store_tokens and note_sequence is not None:
        adaptive_workers = select_tokenize_workers(
            note_sequence, planned_chunk_count, options["max_tokenize_workers"]
        )
        token_pool = (
            ProcessPoolExecutor(max_workers=adaptive_workers)
            if adaptive_workers > 0
            else None
        )
        serialized_ns = note_sequence.SerializeToString()
    hop_seconds = options["hop_length"] / sample_rate
    manifest_rows: List[Dict[str, Any]] = []

    metadata = case["metadata"]
    track_id = metadata.get("track_id") or slugify(None, case["audio_path"].stem)
    track_slug = slugify(metadata.get("track_id"), case["audio_path"].stem)
    storage = ChunkShardWriter(
        output_root=output_root,
        dataset=case["dataset"],
        track_slug=track_slug,
        track_id=track_id,
        chunk_storage=options.get("chunk_storage", "per_chunk"),
        dry_run=dry_run,
        overwrite=overwrite,
    )
    if storage.should_skip_track(options["skip_existing"], options["write_manifest"]):
        return [], 0

    chunk_counter = 0
    total_chunks = 0
    written_chunks = 0

    for batch in batched(plans, options["batch_size"]):
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
            if (
                storage.strategy == "per_chunk"
                and options["skip_existing"]
                and not overwrite
                and storage.chunk_exists(chunk_slug)
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
                "track_id": track_id,
                "track_slug": track_slug,
                "split": metadata.get("split"),
                "audio_path": metadata.get("audio_path"),
                "midi_path": metadata.get("midi_path"),
                "instrument_programs": metadata.get("instrument_programs", []),
                "instrument_names": metadata.get("instrument_names", []),
                "instrument_classes": metadata.get("instrument_classes", []),
                "rendered_stem_ids": metadata.get("rendered_stem_ids", []),
                "stem_id": metadata.get("stem_id"),
                "is_drum": bool(metadata.get("is_drum", False)),
                "mix_group_id": metadata.get("mix_group_id"),
                "augment_profiles": options["augment_profiles"] if options["augment_enabled"] else ["none"],
                "tokenize_workers": adaptive_workers,
                "max_tokenize_workers": options["max_tokenize_workers"],
                "tokens_included": store_tokens,
                "chunk_storage": storage.strategy,
                "chunk_shard_index": chunk_data["chunk_index"],
            }
            chunk_slug = chunk_data["chunk_slug"]
            chunk_payload = {
                "log_mel": stored_log_mel,
                "waveform": stored_waveform,
                "frame_times": frame_times,
            }
            if store_tokens and note_sequence is not None:
                if token_pool and serialized_ns is not None:
                    payload = {
                        "chunk_index": chunk_meta["chunk_index"],
                        "chunk_start_s": chunk_meta["chunk_start_s"],
                        "chunk_end_s": chunk_meta["chunk_end_s"],
                        "frame_times": frame_times,
                        "note_sequence_bytes": serialized_ns,
                    }
                    future = token_pool.submit(tokenize_chunk, payload)
                    token_futures.append((future, chunk_payload, chunk_meta, chunk_slug))
                else:
                    token_output = tokenizer.tokenize_note_sequence(
                        note_sequence,
                        chunk_start_s=chunk_meta["chunk_start_s"],
                        chunk_end_s=chunk_meta["chunk_end_s"],
                        frame_times=frame_times,
                    )
                    payload = build_chunk_payload(
                        chunk_payload["log_mel"],
                        chunk_payload["waveform"],
                        frame_times,
                        token_output.tokens,
                        chunk_meta,
                    )
                    chunk_path = storage.write_chunk(chunk_slug, payload)
                    chunk_meta["chunk_shard_path"] = chunk_path
                    written_chunks += 1
                    if options["write_manifest"]:
                        manifest_rows.append(
                            build_manifest_row(chunk_meta, chunk_path, token_output.tokens, options)
                        )
            else:
                empty_tokens: List[int] = []
                payload = build_chunk_payload(
                    chunk_payload["log_mel"],
                    chunk_payload["waveform"],
                    frame_times,
                    empty_tokens,
                    chunk_meta,
                )
                chunk_path = storage.write_chunk(chunk_slug, payload)
                chunk_meta["chunk_shard_path"] = chunk_path
                written_chunks += 1
                if options["write_manifest"]:
                    manifest_rows.append(
                        build_manifest_row(chunk_meta, chunk_path, empty_tokens, options)
                    )

        if options["store_spectrogram"]:
            del stacked, log_mel_batch

    if store_tokens and token_pool:
        for future, chunk_payload, chunk_meta, chunk_slug in token_futures:
            result = future.result()
            tokens = result["tokens"]
            payload = build_chunk_payload(
                chunk_payload["log_mel"],
                chunk_payload["waveform"],
                chunk_payload["frame_times"],
                tokens,
                chunk_meta,
            )
            chunk_path = storage.write_chunk(chunk_slug, payload)
            chunk_meta["chunk_shard_path"] = chunk_path
            written_chunks += 1
            if options["write_manifest"]:
                manifest_rows.append(build_manifest_row(chunk_meta, chunk_path, tokens, options))
        token_pool.shutdown()

    storage.finalize()

    return manifest_rows if options["write_manifest"] else [], written_chunks


def process_unified_index() -> Path:
    config, options, unified_index_path, output_root, manifest_path = load_precompute_options()

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
        f"track_workers={options['track_workers']} "
        f"store_tokens={options['store_tokens']} "
        f"chunk_storage={options['chunk_storage']} "
        f"profile={options['profile_name']} "
        f"write_manifest={options['write_manifest']})"
    )

    entries = _time_step("load_unified_index", lambda: _load_unified_entries(unified_index_path, config))
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
        dataset_chunk_count = 0
        track_workers = options.get("track_workers", 1)
        use_parallel = track_workers > 1 and not options["store_spectrogram"]
        if use_parallel:
            progress = tqdm(
                total=len(cases),
                desc=f"{dataset} tracks",
                unit="track",
                leave=False,
            )
            with ProcessPoolExecutor(max_workers=track_workers) as pool:
                future_to_case = {
                    pool.submit(
                        _process_case,
                        case,
                        options,
                        output_root,
                        options["max_chunks_per_track"],
                        options["dry_run"],
                        options["overwrite"],
                    ): case
                    for case in cases
                }
                for future in as_completed(future_to_case):
                    case = future_to_case[future]
                    rows, chunk_count = future.result()
                    dataset_chunk_count += chunk_count
                    total_chunks += chunk_count
                    if options["write_manifest"]:
                        records.extend(rows)
                    progress.set_postfix_str(case["metadata"].get("track_id") or "")
                    progress.update(1)
            progress.close()
        else:
            progress = tqdm(
                cases,
                desc=f"{dataset} tracks",
                unit="track",
                leave=False,
            )
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
        _log("Manifest writing disabled; run scripts/generate_chunk_manifest.py later if needed.")
    total_elapsed = time.perf_counter() - total_start
    _log(f"Total processing time: {total_elapsed:0.3f}s")
    return manifest_path


if __name__ == "__main__":
    process_unified_index()

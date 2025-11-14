"""Rudimentary timing monitor for end-to-end preprocessing on sample files."""

from __future__ import annotations

import sys
import json
import os
import random
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, TypeVar

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
warnings.filterwarnings(
    "ignore",
    message="At least one mel filterbank has all zero values",
    category=UserWarning,
)

import note_seq  # type: ignore
import pytest
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

from configs import load_project_config
from configs.project_config import PROJECT_ROOT
from data.audio import io as audio_io
from data.audio import spectrogram
from data.audio.augment import apply_augmentation
from data.symbolic import tokenizer
from scripts.precompute_chunks import select_tokenize_workers

CONFIG = load_project_config()
AUDIO_CFG = CONFIG["audio"]
FEATURE_CFG = AUDIO_CFG["features"]
IO_CFG = AUDIO_CFG["io"]
PREPROCESS_CFG = AUDIO_CFG["preprocess"]
PATH_CFG = CONFIG["paths"]["datasets"]
PREPROCESS_SECTION = CONFIG.get("compute", {}).get("preprocessing", {})


def _resolve_preprocess_profile(section: Dict[str, Any], env_value: str | None) -> Tuple[Dict[str, Any], str]:
    if "profiles" not in section:
        return dict(section), env_value or "default"
    profiles = section.get("profiles") or {}
    if not profiles:
        return dict(section), env_value or "default"
    default_profile = section.get("default_profile")
    profile_name = env_value or default_profile
    if not profile_name or profile_name not in profiles:
        profile_name = default_profile if default_profile in profiles else next(iter(profiles))
    return dict(profiles[profile_name]), profile_name


PROFILE_OVERRIDE = os.environ.get("PREPROCESS_PROFILE") or os.environ.get("PRECOMPUTE_PROFILE")
COMPUTE_CFG, COMPUTE_PROFILE = _resolve_preprocess_profile(PREPROCESS_SECTION, PROFILE_OVERRIDE)
OVERRIDES = os.environ.get("PREPROCESS_TIMING_OVERRIDES")
if OVERRIDES:
    try:
        override_cfg = json.loads(OVERRIDES)
        if not isinstance(override_cfg, dict):
            raise ValueError
        COMPUTE_CFG.update(override_cfg)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to parse PREPROCESS_TIMING_OVERRIDES (expected JSON object)"
        ) from exc
CHUNK_DEVICE_PREF = str(COMPUTE_CFG.get("chunk_device", "auto")).lower()
AUGMENT_ENABLED = bool(COMPUTE_CFG.get("augment_enabled", False))
_PROFILE_VALUE = COMPUTE_CFG.get("augment_profiles", ["none"])
if isinstance(_PROFILE_VALUE, str):
    AUGMENT_PROFILES = [_PROFILE_VALUE]
else:
    AUGMENT_PROFILES = [str(name) for name in _PROFILE_VALUE] or ["none"]
BATCH_SIZE = max(1, int(COMPUTE_CFG.get("batch_size", 1)))
MAX_TOKENIZE_WORKERS = max(
    0, int(COMPUTE_CFG.get("max_tokenize_workers", os.cpu_count() or 1))
)
DATASETS_ROOT = PROJECT_ROOT / "datasets"
LOG_DIR = PROJECT_ROOT / "test_files" / "test_preprocessing_benchmark"
LOG_LINES: List[str] = []
T = TypeVar("T")

DATASET_TYPES = ["maestro", "slakh_full_mix", "slakh_stem"]
PROCESS = psutil.Process(os.getpid()) if psutil else None


def _resolve_dataset_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


UNIFIED_INDEX_PATH = _resolve_dataset_path(PATH_CFG["unified_index"])

STEP_ORDER = [
    "prepare_waveform",
    "load_midi",
]

CHUNK_STEP_ORDER = ["slice_chunk", "to_tensor", "augment", "spectrogram", "tokenize"]


def _chunk_device() -> str | None:
    if CHUNK_DEVICE_PREF in {"auto", ""}:
        return None
    return CHUNK_DEVICE_PREF


def _log(message: str) -> None:
    LOG_LINES.append(message)
    print(message)


def _write_log_file(device_label: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    aug_label = "on" if AUGMENT_ENABLED else "off"
    device_slug = (device_label or "auto").replace(":", "_")
    profiles_slug = "-".join(AUGMENT_PROFILES) if AUGMENT_PROFILES else "none"
    filename = (
        f"preprocessing_batch{BATCH_SIZE}_max_tokenizers{MAX_TOKENIZE_WORKERS}"
        f"_aug{aug_label}_profiles{profiles_slug}_device{device_slug}.log"
    )
    log_path = LOG_DIR / filename
    with log_path.open("w") as fp:
        fp.write("\n".join(LOG_LINES))
    return log_path


def _relative_dataset_path(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(DATASETS_ROOT.resolve()).as_posix()
        return f"datasets/{rel}"
    except ValueError:
        try:
            rel = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
            return rel
        except ValueError:
            return path.as_posix()


def _relative_project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _device_label_from_results(results: Dict[str, Dict[str, Any]]) -> str:
    devices = set()
    for payload in results.values():
        info = payload.get("info", {})
        for chunk in info.get("chunk_results", []):
            device = str(chunk.get("spectrogram_device") or "").strip()
            if device:
                devices.add(device)
    if not devices:
        return CHUNK_DEVICE_PREF if CHUNK_DEVICE_PREF else "auto"
    if len(devices) == 1:
        return next(iter(devices))
    return "mixed"


def _format_bytes(num: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num) < 1024.0:
            return f"{num:0.2f}{unit}"
        num /= 1024.0
    return f"{num:.2f}PB"


def _log_memory(tag: str) -> None:
    ram_str = "unknown"
    vram_str = "n/a"
    if PROCESS:
        try:
            mem = PROCESS.memory_info().rss
            ram_str = _format_bytes(mem)
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            vram = torch.cuda.memory_allocated()
            vram_str = _format_bytes(vram)
        except Exception:
            pass
    _log(f"[mem] {tag}: RAM={ram_str} VRAM={vram_str}")


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
        "frames": payload["frames"],
        "token_count": len(output.tokens),
        "spectrogram_device": payload["spectrogram_device"],
        "tokenize_time": duration,
    }


@dataclass
class SampleCase:
    name: str
    audio_path: Path
    midi_path: Path
    metadata: Dict[str, Any]


def _time_step(timings: Dict[str, float], name: str, func: Callable[[], T]) -> T:
    start = time.perf_counter()
    result = func()
    timings[name] = time.perf_counter() - start
    return result


def _load_unified_entries() -> List[Dict[str, Any]]:
    if not UNIFIED_INDEX_PATH.exists():
        raise FileNotFoundError(f"Unified index not found at {UNIFIED_INDEX_PATH}")
    with UNIFIED_INDEX_PATH.open() as fp:
        payload = json.load(fp)
    if isinstance(payload, dict):
        entries = payload.get("entries", [])
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError("Unified index must be a list or contain an 'entries' field.")
    if not entries:
        raise ValueError("Unified index contains no entries.")
    return entries


def _rng() -> random.Random:
    seed_text = os.environ.get("PREPROCESS_TIMING_SEED")
    if seed_text is None or seed_text == "":
        return random.Random(0)
    try:
        seed_value = int(seed_text)
    except ValueError:
        seed_value = hash(seed_text)
    return random.Random(seed_value)


def _gather_cases() -> Iterable[SampleCase]:
    entries = _load_unified_entries()
    rng = _rng()
    cases: list[SampleCase] = []
    for dataset_name in DATASET_TYPES:
        dataset_entries = [entry for entry in entries if entry.get("dataset") == dataset_name]
        if not dataset_entries:
            continue
        entry = rng.choice(dataset_entries)
        audio_path = _resolve_dataset_path(entry["audio_path"])
        midi_path = _resolve_dataset_path(entry["midi_path"])
        cases.append(
            SampleCase(
                name=dataset_name,
                audio_path=audio_path,
                midi_path=midi_path,
                metadata=entry,
            )
        )
    return cases


def _plan_chunk_positions(
    total_samples: int, sample_rate: int, chunk_samples: int
) -> List[Dict[str, float]]:
    if total_samples <= 0 or chunk_samples <= 0:
        return [{"start_sample": 0, "start_s": 0.0}]
    step = chunk_samples
    positions: List[Dict[str, float]] = []
    start = 0
    while start <= total_samples - chunk_samples:
        positions.append({"start_sample": start, "start_s": start / sample_rate})
        start += step
    if not positions or positions[-1]["start_sample"] < total_samples - chunk_samples:
        last_start = max(0, total_samples - chunk_samples)
        positions.append({"start_sample": last_start, "start_s": last_start / sample_rate})
    return positions


def _batched(plans: List[Dict[str, float]], batch_size: int) -> List[List[Dict[str, float]]]:
    if batch_size <= 1:
        return [[plan] for plan in plans]
    return [plans[idx : idx + batch_size] for idx in range(0, len(plans), batch_size)]


def _run_pipeline(case: SampleCase) -> Tuple[Dict[str, float], Dict[str, Any]]:
    timings: Dict[str, float] = {}
    chunk_samples = FEATURE_CFG["chunk_samples"]
    chunk_duration_s = chunk_samples / IO_CFG["sample_rate"]
    audio_rel = _relative_dataset_path(case.audio_path)
    midi_rel = _relative_dataset_path(case.midi_path)
    info: Dict[str, Any] = {
        "dataset": case.name,
        "track_id": case.metadata.get("track_id"),
        "split": case.metadata.get("split"),
        "audio_path": audio_rel,
        "midi_path": midi_rel,
        "chunk_samples": chunk_samples,
        "hop_length": FEATURE_CFG["hop_length"],
        "num_mels": FEATURE_CFG["num_mels"],
        "chunk_device_pref": CHUNK_DEVICE_PREF,
        "augment_enabled": AUGMENT_ENABLED,
        "augment_profiles": AUGMENT_PROFILES,
        "max_tokenize_workers": MAX_TOKENIZE_WORKERS,
    }

    _log(
        "[prepare_waveform.config] "
        f"convert_to_mono={IO_CFG['convert_to_mono']} "
        f"target_sr={IO_CFG['sample_rate']} "
        f"dc_block={PREPROCESS_CFG['dc_block']} "
        f"high_pass_hz={PREPROCESS_CFG['high_pass_hz']} "
        f"high_pass_order={PREPROCESS_CFG['high_pass_order']} "
        f"loudness_target_lufs={PREPROCESS_CFG['loudness_target_lufs']} "
        f"pre_emphasis={PREPROCESS_CFG['pre_emphasis']} "
        f"dither_amplitude={PREPROCESS_CFG['dither_amplitude']} "
        f"pad_value={PREPROCESS_CFG['pad_value']}"
    )
    waveform, sample_rate = _time_step(
        timings, "prepare_waveform", lambda: audio_io.prepare_waveform(str(case.audio_path))
    )
    _log_memory("after_prepare_waveform")
    info["sample_rate"] = sample_rate
    total_samples = len(waveform)
    duration_seconds = case.metadata.get("duration_seconds") or total_samples / sample_rate
    info["duration_seconds"] = duration_seconds
    info["chunk_duration_s"] = chunk_samples / sample_rate

    note_sequence = _time_step(
        timings,
        "load_midi",
        lambda: note_seq.midi_file_to_note_sequence(str(case.midi_path)),
    )
    _log_memory("after_load_midi")

    plans = _plan_chunk_positions(total_samples, sample_rate, chunk_samples)
    info["planned_chunks"] = len(plans)
    info["batch_size"] = BATCH_SIZE
    adaptive_workers = select_tokenize_workers(note_sequence, len(plans), MAX_TOKENIZE_WORKERS)
    info["tokenize_workers_used"] = adaptive_workers

    chunk_device = _chunk_device()
    info["chunk_device_pref"] = chunk_device or "auto"
    chunk_results: List[Dict[str, Any]] = []
    batch_id = 0
    token_futures: List[Tuple[Any, Dict[str, Any], Dict[str, Any]]] = []
    token_pool = ProcessPoolExecutor(max_workers=adaptive_workers) if adaptive_workers > 0 else None
    serialized_ns = note_sequence.SerializeToString()
    chunk_counter = 0

    for batch in _batched(plans, BATCH_SIZE):
        batch_data: List[Dict[str, Any]] = []
        for plan in batch:
            chunk_idx = chunk_counter
            chunk_counter += 1
            chunk_timings: Dict[str, float] = {}

            def _capture(name: str, fn: Callable[[], Any]) -> Any:
                start_time = time.perf_counter()
                result = fn()
                chunk_timings[name] = chunk_timings.get(name, 0.0) + (
                    time.perf_counter() - start_time
                )
                return result

            start_sample = int(plan["start_sample"])
            chunk_start_s = plan["start_s"]
            chunk_end_s = min(chunk_start_s + chunk_duration_s, duration_seconds)
            if chunk_end_s <= chunk_start_s:
                chunk_end_s = chunk_start_s + 1e-6

            chunk = _capture(
                "slice_chunk",
                lambda: audio_io.slice_chunk(
                    waveform,
                    start_sample=start_sample,
                    chunk_samples=chunk_samples,
                    pad_value=PREPROCESS_CFG["pad_value"],
                ),
            )

            chunk_tensor = _capture(
                "to_tensor",
                lambda: torch.as_tensor(
                    chunk,
                    dtype=torch.float32,
                    device=chunk_device if chunk_device else "cpu",
                ),
            )

            if AUGMENT_ENABLED:
                chunk_tensor = _capture(
                    "augment",
                    lambda tensor=chunk_tensor: apply_augmentation(
                        tensor, sample_rate=sample_rate, profiles=AUGMENT_PROFILES
                    ),
                )

            batch_data.append(
                {
                    "chunk_index": chunk_idx,
                    "chunk_start_s": chunk_start_s,
                    "chunk_end_s": chunk_end_s,
                    "tensor": chunk_tensor,
                    "frame_times": None,
                    "timings": chunk_timings,
                }
            )

        if not batch_data:
            continue

        stacked = torch.stack([item["tensor"] for item in batch_data], dim=0)
        if chunk_device:
            stacked = stacked.to(device=chunk_device)

        spec_start = time.perf_counter()
        log_mel_batch = spectrogram.waveform_to_logmel(
            stacked,
            sample_rate=sample_rate,
            device=chunk_device,
        )
        spec_elapsed = time.perf_counter() - spec_start
        _log_memory(f"after_spectrogram_batch{batch_id}")
        batch_id += 1

        if log_mel_batch.ndim == 2:
            log_mel_batch = log_mel_batch.unsqueeze(0)

        hop_seconds = FEATURE_CFG["hop_length"] / sample_rate

        for chunk_data, log_mel in zip(batch_data, log_mel_batch):
            chunk_timings = chunk_data["timings"]
            chunk_timings["spectrogram"] = spec_elapsed / len(batch_data)
            frames = log_mel.shape[-1]
            assert frames > 0, "Log-mel generation produced zero frames"
            frame_times = (torch.arange(frames, dtype=torch.float32) * hop_seconds).tolist()

            chunk_meta = {
                "chunk_index": chunk_data["chunk_index"],
                "chunk_start_s": chunk_data["chunk_start_s"],
                "chunk_end_s": chunk_data["chunk_end_s"],
                "frames": frames,
                "spectrogram_device": str(log_mel.device),
            }

            if token_pool:
                payload = {
                    "chunk_index": chunk_meta["chunk_index"],
                    "chunk_start_s": chunk_meta["chunk_start_s"],
                    "chunk_end_s": chunk_meta["chunk_end_s"],
                    "frames": frames,
                    "frame_times": frame_times,
                    "note_sequence_bytes": serialized_ns,
                    "spectrogram_device": chunk_meta["spectrogram_device"],
                }
                future = token_pool.submit(_tokenize_job, payload)
                token_futures.append((future, chunk_meta, chunk_timings))
            else:
                token_output = _time_step(
                    chunk_timings,
                    "tokenize",
                    lambda ns=note_sequence, start=chunk_meta["chunk_start_s"], end=chunk_meta["chunk_end_s"], ft=frame_times: tokenizer.tokenize_note_sequence(
                        ns,
                        chunk_start_s=start,
                        chunk_end_s=end,
                        frame_times=ft,
                    ),
                )

                chunk_results.append(
                    {
                        "chunk_index": chunk_meta["chunk_index"],
                        "chunk_start_s": chunk_meta["chunk_start_s"],
                        "chunk_end_s": chunk_meta["chunk_end_s"],
                        "frames": frames,
                        "token_count": len(token_output.tokens),
                        "timings": chunk_timings,
                        "spectrogram_device": chunk_meta["spectrogram_device"],
                    }
                )

            chunk_data.pop("tensor", None)

        del stacked, log_mel_batch

    if token_pool:
        for future, chunk_meta, chunk_timings in token_futures:
            result = future.result()
            chunk_timings["tokenize"] = result["tokenize_time"]
            chunk_results.append(
                {
                    "chunk_index": chunk_meta["chunk_index"],
                    "chunk_start_s": chunk_meta["chunk_start_s"],
                    "chunk_end_s": chunk_meta["chunk_end_s"],
                    "frames": chunk_meta["frames"],
                    "token_count": result["token_count"],
                    "timings": chunk_timings,
                    "spectrogram_device": chunk_meta["spectrogram_device"],
                }
            )
        token_pool.shutdown()

    info["chunk_results"] = sorted(chunk_results, key=lambda item: item["chunk_index"])
    info["num_chunks"] = len(chunk_results)

    chunk_step_totals: Dict[str, float] = {
        step: sum(result["timings"].get(step, 0.0) for result in chunk_results)
        for step in CHUNK_STEP_ORDER
    }
    info["chunk_step_totals"] = chunk_step_totals
    info["chunk_step_means"] = {
        step: (chunk_step_totals[step] / info["num_chunks"]) if info["num_chunks"] else 0.0
        for step in CHUNK_STEP_ORDER
    }

    return timings, info


def _collect_timings() -> Dict[str, Dict[str, Any]]:
    cases = list(_gather_cases())
    if not cases:
        raise FileNotFoundError("Sample dataset files not found for timing test.")
    results: Dict[str, Dict[str, Any]] = {}
    _log(f"[runner] Starting preprocessing timing for {len(cases)} case(s)")
    progress = tqdm(
        list(enumerate(cases, start=1)),
        total=len(cases),
        desc="datasets",
        unit="dataset",
        leave=False,
    )
    for idx, case in progress:
        progress.set_postfix_str(case.name)
        if case.audio_path.exists() and case.midi_path.exists():
            timings, info = _run_pipeline(case)
            progress.write(
                f"[runner]  completed {case.name}: "
                f"{info.get('num_chunks')} chunks "
                f"over {info.get('duration_seconds'):.2f}s"
            )
            results[case.name] = {"timings": timings, "info": info}
        else:
            progress.write(f"[runner]  missing files for {case.name}, skipping")
    progress.close()
    return results


def _run_benchmark() -> Tuple[Dict[str, Dict[str, Any]], float]:
    start = time.perf_counter()
    results = _collect_timings()
    elapsed = time.perf_counter() - start
    return results, elapsed


def _print_report(results: Dict[str, Dict[str, Any]], elapsed_time: float) -> None:
    _log("Global config:")
    _log(
        f"  sample_rate={IO_CFG['sample_rate']} | "
        f"chunk_samples={FEATURE_CFG['chunk_samples']} | "
        f"chunk_frames={FEATURE_CFG['chunk_frames']} | "
        f"hop_length={FEATURE_CFG['hop_length']} | "
        f"num_mels={FEATURE_CFG['num_mels']}"
    )
    _log(
        f"Compute config: inferred chunks per track | "
        f"batch_size={BATCH_SIZE} | "
        f"max_tokenize_workers={MAX_TOKENIZE_WORKERS} | "
        f"chunk_device_pref={CHUNK_DEVICE_PREF} | "
        f"profile={COMPUTE_PROFILE} | "
        f"augment_profiles={AUGMENT_PROFILES}"
    )
    _log("Preprocessing timings per dataset (seconds):")
    for name, payload in results.items():
        timings = payload["timings"]
        info = payload["info"]
        base_total = sum(timings.get(step, 0.0) for step in STEP_ORDER)
        chunk_total = sum(info.get("chunk_step_totals", {}).values())
        total = base_total + chunk_total
        duration_val = float(info.get("duration_seconds") or 0.0)
        sample_rate_val = int(info.get("sample_rate") or IO_CFG["sample_rate"])
        chunk_duration_val = float(info.get("chunk_duration_s") or 0.0)
        planned_chunks = int(info.get("planned_chunks") or 0)
        actual_chunks = int(info.get("num_chunks") or 0)
        _log(f"- {name}:")
        track_desc = info.get("track_id", "unknown")
        split_desc = info.get("split", "unknown")
        _log(f"    track_id={track_desc} | split={split_desc}")
        _log(f"    audio={info.get('audio_path')} | midi={info.get('midi_path')}")
        aug_state = "on" if info.get("augment_enabled") else "off"
        profiles_desc = ", ".join(info.get("augment_profiles", ["none"]))
        _log(f"    augment={aug_state} (profiles=[{profiles_desc}])")
        _log(
            "    tokenize_workers_used={used} (max={max_cap})".format(
                used=info.get("tokenize_workers_used", 0),
                max_cap=info.get("max_tokenize_workers", MAX_TOKENIZE_WORKERS),
            )
        )
        _log(
            "    duration={duration:.2f}s | sample_rate={sample_rate} Hz | "
            "chunk_duration={chunk_duration:.3f}s | planned_chunks={planned} | actual_chunks={actual}".format(
                duration=duration_val,
                sample_rate=sample_rate_val,
                chunk_duration=chunk_duration_val,
                planned=planned_chunks,
                actual=actual_chunks,
            )
        )
        for step in STEP_ORDER:
            if step in timings:
                _log(f"    {step:>15}: {timings[step]:.4f}")
        _log("    Chunk step aggregates:")
        for step in CHUNK_STEP_ORDER:
            total_step = info.get("chunk_step_totals", {}).get(step, 0.0)
            mean_step = info.get("chunk_step_means", {}).get(step, 0.0)
            step_times = [chunk["timings"].get(step, 0.0) for chunk in info.get("chunk_results", [])]
            min_step = min(step_times) if step_times else 0.0
            max_step = max(step_times) if step_times else 0.0
            _log(
                f"        {step:>12}: total={total_step:.4f}s | mean={mean_step:.4f}s "
                f"| min={min_step:.4f}s | max={max_step:.4f}s"
            )
        _log(f"    {'total':>15}: {total:.4f}")
        assert total >= 0.0
    _log(f"[runner] Total processing time: {elapsed_time:.3f}s")
    device_label = _device_label_from_results(results)
    log_path = _write_log_file(device_label)
    log_rel = _relative_project_path(log_path)
    _log(f"[runner] Writing timing log to {log_rel}")


@pytest.mark.slow
def test_full_preprocessing_timings():
    try:
        results, elapsed = _run_benchmark()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
        return
    if not results:
        pytest.skip("No preprocessing cases were executed.")
    _print_report(results, elapsed)


if __name__ == "__main__":
    try:
        timing_results, total_elapsed = _run_benchmark()
    except FileNotFoundError as exc:
        raise SystemExit(f"Cannot run timing script: {exc}") from exc
    if not timing_results:
        raise SystemExit("No preprocessing cases were executed.")
    _print_report(timing_results, total_elapsed)

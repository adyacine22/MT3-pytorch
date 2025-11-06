#!/usr/bin/env python3
"""
Lightweight dataloader benchmark for this repo.
Runs a small fetch benchmark for a set of data-loading configurations derived from
`A100_pretraining_test` and records timings + parent/worker memory usage.

Usage:
    conda activate mt3-pytorch
    python scripts/dataloader_benchmark.py --config A100_pretraining_test --num_batches 5 --out results.json

This script:
 - imports `get_config` from `config.training_config`
 - reuses `setup_dataloaders` from `training.train_multitrack` to construct loaders
 - for each variant: builds DataLoader, does a few fetches, records timings and memory
 - writes JSON results to disk

Notes:
 - Keep `num_batches` small (3-10) when running on real training machines.
 - Preloading many files into RAM can take time and memory; this script will run the dataset constructor with the chosen `preload_all` flag.
"""

import argparse
import copy
import gc
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover - fallback for minimal environments
    psutil = None
import torch
import torch.multiprocessing as mp

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

# Match the training script’s worker behaviour (fork + copy-on-write) when possible.
if mp.get_start_method(allow_none=True) != "fork":
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        # Already initialised with a different method; continue without changing it.
        pass

# Suppress repeated ffmpeg warnings emitted by pydub in cache-only paths.
warnings.filterwarnings(
    "ignore",
    message="Couldn't find ffmpeg or avconv",
    category=RuntimeWarning,
    module="pydub.utils",
)

from config.training_config import get_config, update_config
from data.gpu_spectrogram import GPUSpectrogramComputer
from training.train_multitrack import setup_dataloaders


def bytes_to_mb(x):
    if x is None:
        return None
    return round(x / (1024 ** 2), 2)


def system_memory_snapshot():
    """Capture system-wide memory similar to `free -h`. Returns MB values."""
    if psutil is None:
        return None
    vm = psutil.virtual_memory()
    return {
        "total_mb": bytes_to_mb(vm.total),
        "available_mb": bytes_to_mb(vm.available),
        "used_mb": bytes_to_mb(vm.total - vm.available),
        "percent": round(vm.percent, 2),
    }


DATALOADER_KEYS = [
    "num_workers",
    "pin_memory",
    "prefetch_factor",
    "persistent_workers",
    "cache_size",
    "preload_all",
]


def snapshot_dataloader_settings(config):
    """Return a shallow snapshot of the knobs we are actually sweeping."""
    return {k: config.get(k) for k in DATALOADER_KEYS if k in config}


def snapshot_workers(proc):
    """Capture a lightweight view of worker processes for memory reporting."""
    if psutil is None or proc is None:
        return []
    info = []
    for child in proc.children(recursive=False):
        try:
            info.append(
                {
                    "pid": child.pid,
                    "rss_mb": bytes_to_mb(child.memory_info().rss),
                    "cmdline": " ".join(child.cmdline()),
                }
            )
        except psutil.NoSuchProcess:
            continue
    return info


def _worker_measure(queue, config, num_batches, warmup):
    try:
        result = measure_config(config, num_batches=num_batches, warmup=warmup)
    except Exception as exc:  # pragma: no cover
        result = {
            "error": str(exc),
            "config_snapshot": snapshot_dataloader_settings(config),
        }
    queue.put(result)


def measure_config(config, num_batches=5, warmup=1):
    """Construct dataloaders with `config`, iterate `num_batches` batches and
    collect timing + memory info. Returns a dict with measurements.
    """
    import logging

    config = copy.deepcopy(config)
    logger = logging.getLogger("dataloader_benchmark")
    if psutil is not None:
        proc = psutil.Process()
        parent_before = proc.memory_info().rss
        system_before = system_memory_snapshot()
    else:
        proc = None
        parent_before = None
        system_before = None

    start_time = time.time()
    # Build dataloaders (this will construct datasets and possibly preload)
    try:
        train_loader, val_loader = setup_dataloaders(config)
    except Exception as e:
        return {
            "error": f"Failed to setup dataloaders: {e}",
            "config_snapshot": snapshot_dataloader_settings(config),
        }
    setup_time = time.time() - start_time

    # Give OS a moment to settle and to start workers
    time.sleep(0.5)

    # Identify current child processes (likely DataLoader workers)
    workers_info_before = snapshot_workers(proc)
    workers_rss_before = (
        sum(w["rss_mb"] for w in workers_info_before) if workers_info_before else None
    )

    # Prepare iterator and spectrogram computer (mimic training loop)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec_computer = None  # Lazy init: must happen *after* workers are forked.

    # Make an infinite iterator (reshuffles each epoch) so we can time arbitrary
    # numbers of batches even when a single epoch is shorter than num_batches.
    def infinite_dataloader(loader):
        while True:
            for batch in loader:
                yield batch

    train_iter = infinite_dataloader(train_loader)

    # Warm-up fetches (may trigger worker init or compilation)
    fetch_times = []
    fetch_to_device_times = []
    spec_compute_times = []
    first_fetch_time = None
    for i in range(warmup + num_batches):
        # fetch timing
        t0 = time.time()
        try:
            batch = next(train_iter)
        except StopIteration:
            break
        t1 = time.time()
        fetch_elapsed = t1 - t0

        logger.info(f"     [{i}] Fetching batch from DataLoader...")
        logger.info(f"     [{i}] ✓ Batch fetched in {fetch_elapsed:.3f}s")

        # Lazy init of spec_computer after the first fetch, which is when the
        # DataLoader workers are actually forked.
        if spec_computer is None:
            spec_computer = GPUSpectrogramComputer(device=device)

        # Move to device timing (frames and targets)
        t2 = time.time()
        try:
            frames = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
        except Exception:
            # fall back if keys differ
            if isinstance(batch, (list, tuple)):
                item = batch[0]
            else:
                item = batch
            frames = item["inputs"].to(device)
            targets = item["targets"].to(device)
        t3 = time.time()
        to_device_elapsed = t3 - t2
        logger.info(f"     [{i}] Moving data to device {device}...")
        logger.info(f"     [{i}] ✓ Data moved to device in {to_device_elapsed:.3f}s (frames: {frames.shape}, targets: {targets.shape})")

        # Spectrogram compute timing (hybrid approach)
        t4 = time.time()
        try:
            mel = spec_computer(frames)
            spec_elapsed = time.time() - t4
            logger.info(f"     [{i}] ✓ Mel-spectrograms computed on GPU in {spec_elapsed:.3f}s (shape: {mel.shape})")
        except Exception as e:
            spec_elapsed = time.time() - t4
            logger.info(f"     [{i}] Mel-spectrogram compute failed: {e}")

        # record only after warmup phase
        if i >= warmup:
            fetch_times.append(fetch_elapsed)
            fetch_to_device_times.append(to_device_elapsed)
        spec_compute_times.append(spec_elapsed)
    if i == 0:
        first_fetch_time = fetch_elapsed

    # Memory after fetches
    system_after = system_memory_snapshot() if psutil is not None else None
    parent_after = proc.memory_info().rss if proc is not None else None
    workers_info_after = snapshot_workers(proc)
    workers_rss_after = (
        sum(w["rss_mb"] for w in workers_info_after) if workers_info_after else None
    )

    # Basic stats
    avg_fetch = sum(fetch_times) / len(fetch_times) if fetch_times else None
    max_fetch = max(fetch_times) if fetch_times else None

    def dataset_sample_counts(loader):
        if loader is None:
            return None
        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            return None

        counts = {}
        if hasattr(dataset, "dataset_names") and hasattr(dataset, "datasets"):
            for name, ds in zip(dataset.dataset_names, dataset.datasets):
                try:
                    counts[name] = len(ds)
                except Exception:
                    counts[name] = None
        else:
            try:
                counts[dataset.__class__.__name__] = len(dataset)
            except Exception:
                counts[dataset.__class__.__name__] = None
        return counts

    def dataset_cache_stats(loader):
        if loader is None:
            return None
        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            return None

        def inspect_dataset(ds):
            entry = {"class": ds.__class__.__name__}
            if hasattr(ds, "preload_all"):
                try:
                    entry["preload_all"] = bool(getattr(ds, "preload_all"))
                except Exception:
                    entry["preload_all"] = None
            if hasattr(ds, "cache_files"):
                try:
                    entry["num_cache_files"] = len(getattr(ds, "cache_files"))
                except Exception:
                    entry["num_cache_files"] = None
            cache_obj = getattr(ds, "_file_cache", None)
            if cache_obj is not None and hasattr(cache_obj, "info"):
                try:
                    entry["lru_cache"] = cache_obj.info()
                except Exception:
                    entry["lru_cache"] = None
            return entry

        if hasattr(dataset, "dataset_names") and hasattr(dataset, "datasets"):
            stats = {}
            for name, ds in zip(dataset.dataset_names, dataset.datasets):
                stats[name] = inspect_dataset(ds)
            return stats

        return {"dataset": inspect_dataset(dataset)}

    train_counts = dataset_sample_counts(locals().get("train_loader"))
    val_counts = dataset_sample_counts(val_loader)
    train_cache_stats = dataset_cache_stats(locals().get("train_loader"))
    val_cache_stats = dataset_cache_stats(val_loader)

    # Clean up loaders and force GC (this should terminate persistent workers if loader is deleted)
    try:
        del train_loader
        del train_iter
        del val_loader
    except Exception:
        pass
    gc.collect()
    time.sleep(0.5)

    return {
        "config_snapshot": snapshot_dataloader_settings(config),
        "setup_time_s": setup_time,
        "first_fetch_time_s": first_fetch_time,
        "fetch_times_s": fetch_times,
        "avg_fetch_time_s": avg_fetch,
        "max_fetch_time_s": max_fetch,
        "train_dataset_samples": train_counts,
        "val_dataset_samples": val_counts,
        "train_cache_stats": train_cache_stats,
        "val_cache_stats": val_cache_stats,
        "parent_rss_before_mb": bytes_to_mb(parent_before),
        "parent_rss_after_mb": bytes_to_mb(parent_after),
        "worker_rss_before_mb": workers_rss_before,
        "worker_rss_after_mb": workers_rss_after,
        "system_memory_before": system_before,
        "system_memory_after": system_after,
        "workers_before": workers_info_before,
        "workers_after": workers_info_after,
    }


def build_tests_from_base(base_config):
    """Construct a compact, human-readable grid of loader scenarios.

    Rather than sweeping every knob independently, we assemble named bundles that
    mirror the patterns we typically experiment with during training.
    """

    def scenario(name, **overrides):
        cfg = copy.deepcopy(base_config)
        cfg.update(overrides)
        return (name, cfg)

    scenarios = [
        scenario("baseline"),

        # Worker scaling ladder (persistent_workers disabled for parity checks)
        scenario("workers_0_cpu_only", num_workers=0, persistent_workers=False),
        scenario("workers_2_fastlane", num_workers=2, persistent_workers=False),
        scenario("workers_4_balanced", num_workers=4, persistent_workers=False),
        scenario("workers_8_overdrive", num_workers=8, persistent_workers=False),

        # Preload vs on-demand comparisons
        scenario("preload_true_small_cache", preload_all=True, cache_size=100, num_workers=4, persistent_workers=False),
        scenario("preload_false_large_cache", preload_all=False, cache_size=800, num_workers=4, persistent_workers=False),

        # Prefetch experiments (only meaningful when workers > 0)
        scenario("prefetch_anaemic", num_workers=4, prefetch_factor=2, persistent_workers=False),
        scenario("prefetch_aggressive", num_workers=4, prefetch_factor=8, persistent_workers=False),

        # Targeted worker/prefetch comparisons requested by users
        scenario("workers8_prefetch2_persist_true", num_workers=8, prefetch_factor=2, persistent_workers=True),
        scenario("workers8_prefetch4_persist_true", num_workers=8, prefetch_factor=4, persistent_workers=True),
        scenario("workers8_prefetch2_persist_false", num_workers=8, prefetch_factor=2, persistent_workers=False),
        scenario("workers8_prefetch4_persist_false", num_workers=8, prefetch_factor=4, persistent_workers=False),

        # Pin-memory toggles
        scenario("pin_memory_off", num_workers=4, pin_memory=False, persistent_workers=False),

        # High-throughput CUDA profile (persistent workers on for long runs)
        scenario(
            "cuda_pipeline_full",
            num_workers=8,
            preload_all=True,
            cache_size=400,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
        ),

        # Memory-constrained fallback (minimal cache & workers)
        scenario(
            "memory_lightweight",
            num_workers=1,
            preload_all=False,
            cache_size=50,
            persistent_workers=False,
            pin_memory=False,
            prefetch_factor=2,
        ),
    ]

    for cache_size in [20, 200]:
        for workers in [2, 4]:
            for prefetch in [2, 4]:
                for persist in [False, True]:
                    name = (
                        f"cache{cache_size}_workers{workers}_pref{prefetch}_"
                        f"{'persist_true' if persist else 'persist_false'}"
                    )
                    scenarios.append(
                        scenario(
                            name,
                            cache_size=cache_size,
                            preload_all=False,
                            num_workers=workers,
                            prefetch_factor=prefetch,
                            persistent_workers=persist,
                        )
                    )

    return scenarios


def log_result_summary(name, result):
    """Emit a concise summary so we can eyeball impact per configuration."""
    if "error" in result:
        print(f"✗ {name}: {result['error']}")
        return

    first = result.get("first_fetch_time_s")
    avg = result.get("avg_fetch_time_s")
    peak = result.get("max_fetch_time_s")
    parent_before = result.get("parent_rss_before_mb")
    parent_after = result.get("parent_rss_after_mb")
    mem_delta = None
    if parent_before is not None and parent_after is not None:
        mem_delta = round(parent_after - parent_before, 2)

    summary = [f"✓ {name}"]
    if first is not None:
        summary.append(f"first_fetch={first:.3f}s")
    if avg is not None:
        summary.append(f"avg_fetch={avg:.3f}s")
    if peak is not None:
        summary.append(f"max_fetch={peak:.3f}s")
    if mem_delta is not None:
        summary.append(f"ΔRAM={mem_delta:.2f} MB")

    sys_before = result.get("system_memory_before")
    sys_after = result.get("system_memory_after")
    if sys_before and sys_after and "used_mb" in sys_before and "used_mb" in sys_after:
        sys_delta = round(sys_after["used_mb"] - sys_before["used_mb"], 2)
        summary.append(f"ΔSysUsed={sys_delta:.2f} MB")

    caches = result.get("train_cache_stats") or {}
    cache_fragments = []
    for name, info in caches.items():
        cache_info = info.get("lru_cache") if isinstance(info, dict) else None
        if not cache_info:
            continue
        current = cache_info.get("current_size")
        maxsize = cache_info.get("maxsize")
        hits = cache_info.get("hits")
        cache_fragments.append(f"{name}:size{current}/{maxsize},hits={hits}")
    if cache_fragments:
        summary.append("cache[" + "; ".join(cache_fragments) + "]")

    print(" | ".join(summary))


def main():
    print(f"CUDA initialized at start of main: {torch.cuda.is_initialized()}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="A100_pretraining_test")
    parser.add_argument("--only", type=str, default=None, help="Run only a single test by name (e.g. 'baseline' or 'num_workers_0')")
    parser.add_argument("--num_batches", type=int, default=20, help="Number of batches to time per test")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup fetches before timing")
    parser.add_argument("--out", type=str, default=None, help="Filename for JSON results (placed under benchmark/)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tests to run (for quick debug)")
    parser.add_argument("--maestro_max_samples", type=int, default=None, help="Override maestro_max_samples in the base config")
    parser.add_argument("--slakh_max_tracks", type=int, default=None, help="Override slakh_max_tracks in the base config")
    parser.add_argument("--maestro-cache-dir", type=str, default=None, help="Override MAESTRO cache directory")
    parser.add_argument("--maestro-val-cache-dir", type=str, default=None, help="Override MAESTRO validation cache directory")
    parser.add_argument("--slakh-cache-dir", type=str, default=None, help="Override SLAKH cache directory")
    parser.add_argument("--slakh-val-cache-dir", type=str, default=None, help="Override SLAKH validation cache directory")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.resolve()

    base = get_config(args.config)
    # Allow overriding sample sizes for faster test sweeps
    if args.maestro_max_samples is not None:
        base["maestro_max_samples"] = args.maestro_max_samples
    if args.slakh_max_tracks is not None:
        base["slakh_max_tracks"] = args.slakh_max_tracks

    # Determine cache directories (support optimized layout overrides)
    # CLI overrides take precedence
    if args.maestro_cache_dir:
        base["maestro_cache_dir"] = args.maestro_cache_dir
    if args.maestro_val_cache_dir:
        base["maestro_val_cache_dir"] = args.maestro_val_cache_dir
    if args.slakh_cache_dir:
        base["slakh_cache_dir"] = args.slakh_cache_dir
    if args.slakh_val_cache_dir:
        base["slakh_val_cache_dir"] = args.slakh_val_cache_dir

    for key_name in ["maestro_cache_dir", "maestro_val_cache_dir", "slakh_cache_dir", "slakh_val_cache_dir"]:
        path_str = base.get(key_name)
        if path_str:
            cache_path = Path(path_str)
            if not cache_path.exists():
                print(f"[WARN] cache path for {key_name} not found: {cache_path}")

    # Ensure cached data is enabled when cache paths are provided/forced.
    cache_keys = [
        base.get("maestro_cache_dir"),
        base.get("slakh_cache_dir"),
        base.get("maestro_val_cache_dir"),
        base.get("slakh_val_cache_dir"),
    ]
    if any(cache_keys):
        base["use_cached_data"] = True

    tests = build_tests_from_base(base)
    results_dir = Path("benchmark")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def resolve_out_path(base_name):
        if args.out:
            return results_dir / args.out
        return results_dir / f"{base_name}_{timestamp}.json"

    def run_in_isolated_process(conf, num_batches, warmup):
        print(f"CUDA initialized before fork: {torch.cuda.is_initialized()}")
        ctx = mp.get_context("fork")
        queue = ctx.Queue()
        proc = ctx.Process(
            target=_worker_measure,
            args=(queue, copy.deepcopy(conf), num_batches, warmup),
        )
        proc.start()
        result = queue.get()
        proc.join()
        return result

    # If --only is used, just run that one test
    if args.only:
        tests = [t for t in tests if t[0] == args.only]
        if not tests:
            raise ValueError(f"Requested --only '{args.only}' but no matching test was found.")
        
        name, conf = tests[0]
        print(f"\n=== Running test: {name} ===")
        print(json.dumps({k: conf[k] for k in ["num_workers", "pin_memory", "prefetch_factor", "persistent_workers", "cache_size", "preload_all"] if k in conf}, indent=2))
        
        out_path = resolve_out_path(f"{args.config}_{name}")
        try:
            res = run_in_isolated_process(conf, num_batches=args.num_batches, warmup=args.warmup)
        except Exception as e:
            res = {
                "error": str(e),
                "config_snapshot": snapshot_dataloader_settings(conf),
            }
        
        results = {"meta": {"config_name": args.config, "timestamp": time.time()}, "tests": [{"name": name, "result": res}]}
        log_result_summary(name, res)

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTest finished. Results written to {out_path}")

    else:
        # If --only is NOT used, run all tests in separate processes
        print("Running all benchmark tests in isolated processes...")
        
        if args.limit is not None:
            tests = tests[: args.limit]

        all_args = sys.argv[1:]
        base_command = [sys.executable, __file__]

        # Filter out --only and --limit from the base command
        i = 0
        while i < len(all_args):
            if all_args[i] == '--only' or all_args[i] == '--limit':
                i += 2  # Skip the flag and its value
            else:
                base_command.append(all_args[i])
                i += 1
        
        for name, conf in tests:
            command = base_command + ["--only", name]
            
            print(f"\n--- Spawning process for test: {name} ---")
            print(f"  Command: {' '.join(command)}")
            subprocess.run(command, check=True)

        print("\nAll tests finished.")


if __name__ == "__main__":
    main()

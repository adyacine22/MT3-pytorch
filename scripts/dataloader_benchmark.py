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
import time
from pathlib import Path

import psutil
import torch

# Ensure repo root is on sys.path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from config.training_config import get_config, update_config
from training.train_multitrack import setup_dataloaders
from data.gpu_spectrogram import GPUSpectrogramComputer


def bytes_to_mb(x):
    return round(x / (1024 ** 2), 2)


def measure_config(config, num_batches=5, warmup=1):
    """Construct dataloaders with `config`, iterate `num_batches` batches and
    collect timing + memory info. Returns a dict with measurements.
    """
    proc = psutil.Process()
    parent_before = proc.memory_info().rss

    start_time = time.time()
    # Build dataloaders (this will construct datasets and possibly preload)
    try:
        train_loader, val_loader = setup_dataloaders(config)
    except Exception as e:
        return {
            "error": f"Failed to setup dataloaders: {e}",
            "config": config,
        }
    setup_time = time.time() - start_time

    # Give OS a moment to settle and to start workers
    time.sleep(0.5)

    # Identify current child processes (likely DataLoader workers)
    children = proc.children(recursive=False)
    workers_info_before = []
    for c in children:
        try:
            workers_info_before.append({
                "pid": c.pid,
                "rss_mb": bytes_to_mb(c.memory_info().rss),
                "cmdline": " ".join(c.cmdline()),
            })
        except psutil.NoSuchProcess:
            pass

    # Prepare iterator and spectrogram computer (mimic training loop)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec_computer = GPUSpectrogramComputer(device=device)

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

        # Log fetch like training script
        import logging
        logger = logging.getLogger("dataloader_benchmark")
        logger.info(f"     [{i}] Fetching batch from DataLoader...")
        logger.info(f"     [{i}] ✓ Batch fetched in {fetch_elapsed:.3f}s")

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
    parent_after = proc.memory_info().rss
    children = proc.children(recursive=False)
    workers_info_after = []
    for c in children:
        try:
            workers_info_after.append({
                "pid": c.pid,
                "rss_mb": bytes_to_mb(c.memory_info().rss),
                "cmdline": " ".join(c.cmdline()),
            })
        except psutil.NoSuchProcess:
            pass

    # Basic stats
    avg_fetch = sum(fetch_times) / len(fetch_times) if fetch_times else None
    max_fetch = max(fetch_times) if fetch_times else None

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
        "config": config,
        "setup_time_s": setup_time,
        "first_fetch_time_s": first_fetch_time,
        "fetch_times_s": fetch_times,
        "avg_fetch_time_s": avg_fetch,
        "max_fetch_time_s": max_fetch,
        "parent_rss_before_mb": bytes_to_mb(parent_before),
        "parent_rss_after_mb": bytes_to_mb(parent_after),
        "workers_before": workers_info_before,
        "workers_after": workers_info_after,
    }


def build_tests_from_base(base_config):
    """Create a concise list of test configs by varying only the parameters the user requested.
    We'll vary each parameter individually from the base, plus a couple combined safe/unsafe combos.
    """
    tests = []

    # Baseline
    tests.append(("baseline", {}))

    # num_workers variants
    # Include 1 to allow an isolated single-worker test (useful to detect concurrent IO stalls)
    for w in [0, 1, 2, 4, 8]:
        tests.append((f"num_workers_{w}", {"num_workers": w}))

    # pin_memory toggle
    tests.append(("pin_memory_False", {"pin_memory": False}))

    # prefetch_factor variants (only meaningful when num_workers>0)
    tests.append(("prefetch_2", {"prefetch_factor": 2}))
    tests.append(("prefetch_8", {"prefetch_factor": 8}))

    # persistent workers toggle
    tests.append(("persistent_workers_False", {"persistent_workers": False}))
    tests.append(("persistent_workers_True", {"persistent_workers": True}))

    # preload_all toggle
    tests.append(("preload_all_True", {"preload_all": True}))
    tests.append(("preload_all_False", {"preload_all": False}))

    # cache_size variants
    # More cache_size variants to explore LRU behavior
    for c in [50, 100, 200, 400, 800]:
        tests.append((f"cache_{c}", {"cache_size": c}))

    # Also test cache sizes when not preloading (LRU active)
    for c in [50, 100, 200, 400, 800]:
        tests.append((f"cache_{c}_no_preload", {"cache_size": c, "preload_all": False}))

    # Combined scenarios (safe and aggressive)
    tests.append(("aggressive_workers_preload", {"num_workers": 8, "preload_all": True, "cache_size": 400, "persistent_workers": True}))
    tests.append(("safe_single_worker_no_preload", {"num_workers": 0, "preload_all": False, "cache_size": 100, "persistent_workers": False}))

    # Build actual configs
    built = []
    for name, changes in tests:
        conf = base_config.copy()
        conf.update(changes)
        built.append((name, conf))
    return built


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="A100_pretraining_test")
    parser.add_argument("--only", type=str, default=None, help="Run only a single test by name (e.g. 'baseline' or 'num_workers_0')")
    parser.add_argument("--num_batches", type=int, default=5, help="Number of batches to time per test (small) ")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup fetches before timing")
    parser.add_argument("--out", type=str, default="dataloader_benchmark_results.json")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tests to run (for quick debug)")
    parser.add_argument("--maestro_max_samples", type=int, default=None, help="Override maestro_max_samples in the base config")
    parser.add_argument("--slakh_max_tracks", type=int, default=None, help="Override slakh_max_tracks in the base config")
    args = parser.parse_args()

    base = get_config(args.config)
    # Allow overriding sample sizes for faster test sweeps
    if args.maestro_max_samples is not None:
        base["maestro_max_samples"] = args.maestro_max_samples
    if args.slakh_max_tracks is not None:
        base["slakh_max_tracks"] = args.slakh_max_tracks
    tests = build_tests_from_base(base)

    # Optionally run only a single named test (useful for isolated runs)
    if args.only:
        tests = [t for t in tests if t[0] == args.only]
        if not tests:
            raise ValueError(f"Requested --only '{args.only}' but no matching test was found. Available: {[t[0] for t in build_tests_from_base(base)]}")
    results = {"meta": {"config_name": args.config, "timestamp": time.time()}, "tests": []}

    if args.limit is not None:
        tests = tests[: args.limit]

    for name, conf in tests:
        print(f"\n=== Running test: {name} ===")
        print(json.dumps({k: conf[k] for k in ["num_workers", "pin_memory", "prefetch_factor", "persistent_workers", "cache_size", "preload_all"] if k in conf}, indent=2))
        try:
            res = measure_config(conf, num_batches=args.num_batches, warmup=args.warmup)
        except Exception as e:
            res = {"config": conf, "error": str(e)}
        results["tests"].append({"name": name, "result": res})

        # Save intermediate results in case of interruption
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nAll tests finished. Results written to {args.out}")


if __name__ == "__main__":
    main()

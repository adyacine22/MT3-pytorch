#!/usr/bin/env python3
"""Run preprocessing benchmark across preset compute configurations."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_SCRIPT = ROOT / "tests" / "test_preprocessing_timing.py"

# Define the sweep grids here.
CHUNK_DEVICES = ["cpu", "cuda:0"]
BATCH_SIZES = [128, 256]
TOKENIZE_WORKERS_OPTIONS = [16, 30]
AUGMENT_ENABLED_OPTIONS = [False, True]
AUGMENT_PROFILE_COMBOS = [["noise"], ["noise", "eq", "rir", "gain", "clip"]]


def main() -> None:
    combos = []
    for chunk_device in CHUNK_DEVICES:
        for batch_size in BATCH_SIZES:
            for tokenize_workers in TOKENIZE_WORKERS_OPTIONS:
                for augment_enabled in AUGMENT_ENABLED_OPTIONS:
                    profile_list = AUGMENT_PROFILE_COMBOS if augment_enabled else [["none"]]
                    for profiles in profile_list:
                        combos.append(
                            {
                                "chunk_device": chunk_device,
                                "batch_size": batch_size,
                                "tokenize_workers": tokenize_workers,
                                "augment_enabled": augment_enabled,
                                "augment_profiles": profiles,
                            }
                        )

    for idx, overrides in enumerate(combos, start=1):
        env = os.environ.copy()
        env["PREPROCESS_TIMING_OVERRIDES"] = json.dumps(overrides)
        label = f"[{idx}/{len(combos)}] {overrides}"
        print(f"Running {label}")
        result = subprocess.run([sys.executable, str(TEST_SCRIPT)], env=env, cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

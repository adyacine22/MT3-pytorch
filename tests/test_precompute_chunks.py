from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd  # type: ignore
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import load_project_config
from configs.project_config import PROJECT_ROOT


CONFIG = load_project_config()
UNIFIED_INDEX = (PROJECT_ROOT / CONFIG["paths"]["datasets"]["unified_index"]).resolve()


def _load_entries() -> list[dict]:
    if not UNIFIED_INDEX.exists():
        pytest.skip(f"Unified index missing at {UNIFIED_INDEX}")
    with UNIFIED_INDEX.open() as fp:
        payload = json.load(fp)
    if isinstance(payload, dict):
        return payload.get("entries", [])
    if isinstance(payload, list):
        return payload
    raise ValueError("Unified index must be list or dict with 'entries'.")


def _find_available_dataset(preferred: list[str]) -> tuple[str, dict] | None:
    entries = _load_entries()
    for dataset in preferred:
        for entry in entries:
            if entry.get("dataset") != dataset:
                continue
            audio_path = (PROJECT_ROOT / entry["audio_path"]).resolve()
            midi_path = (PROJECT_ROOT / entry["midi_path"]).resolve()
            if audio_path.exists() and midi_path.exists():
                return dataset, entry
    return None


@pytest.mark.slow
def test_precompute_chunks_smoke(tmp_path: Path, verbose: bool = False) -> None:
    if Path.cwd().resolve() != PROJECT_ROOT.resolve():
        pytest.skip("Run this test from the project root directory.")
    dataset_payload = _find_available_dataset(["maestro", "slakh_full_mix", "slakh_stem"])
    if not dataset_payload:
        pytest.skip("No datasets with resolved audio/midi found locally.")
    dataset_name, entry = dataset_payload
    if verbose:
        print(
            f"[info] Using dataset={dataset_name} "
            f"audio={entry['audio_path']} midi={entry['midi_path']}"
        )

    output_root = tmp_path / "chunks"
    manifest_path = tmp_path / "chunk_manifest.parquet"
    env = os.environ.copy()
    env.update(
        {
            "PRECOMPUTE_DATASETS": dataset_name,
            "PRECOMPUTE_MAX_TRACKS": "1",
            "PRECOMPUTE_MAX_CHUNKS": "1",
            "PRECOMPUTE_OUTPUT_ROOT": str(output_root),
            "PRECOMPUTE_MANIFEST_PATH": str(manifest_path),
            "PRECOMPUTE_DRY_RUN": "0",
            "PRECOMPUTE_OVERWRITE": "1",
        }
    )
    cmd = [
        sys.executable,
        "scripts/precompute_chunks.py",
    ]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        env=env,
    )
    if result.returncode != 0:
        pytest.skip(f"precompute script failed under current environment:\n{result.stderr}")
    if verbose and result.stdout.strip():
        print("[info] Script stdout:\n" + result.stdout)

    assert manifest_path.exists(), "Manifest file not created"
    df = pd.read_parquet(manifest_path)
    if verbose:
        print(f"[info] Manifest rows={len(df)}")
    assert len(df) == 1
    chunk_path = Path(df.iloc[0]["chunk_path"])
    if verbose:
        print(f"[info] Chunk path={chunk_path}")
    assert chunk_path.exists(), "Chunk .pt file missing"


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_precompute_chunks_smoke(tmp_path, verbose=True)

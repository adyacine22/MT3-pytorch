from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import load_project_config
from configs.project_config import PROJECT_ROOT
from data.preprocessing.manifest import MANIFEST_REQUIRED_COLUMNS


CONFIG = load_project_config()
MANIFEST_PATH = (PROJECT_ROOT / CONFIG["paths"]["cache"]["chunk_manifest"]).resolve()
UNIFIED_INDEX_PATH = (PROJECT_ROOT / CONFIG["paths"]["datasets"]["unified_index"]).resolve()


def test_chunk_manifest_parquet_schema():
    assert MANIFEST_PATH.exists(), f"Manifest not found at {MANIFEST_PATH}"
    df = pd.read_parquet(MANIFEST_PATH)
    print(f"Loaded chunk manifest with {len(df)} rows from {MANIFEST_PATH}")

    for col in MANIFEST_REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column '{col}' in manifest"

    # Basic sanity checks
    assert (df["chunk_duration_ms"] > 0).all()
    assert (df["chunk_end_ms"] >= df["chunk_start_ms"]).all()
    print("Unique datasets in manifest:", df["dataset"].unique())
    print("Total duration hours:", df["chunk_duration_ms"].sum() / (1000 * 60 * 60))


def test_manifest_covers_all_tracks():
    with UNIFIED_INDEX_PATH.open() as fp:
        unified_entries = json.load(fp)["entries"]
    track_ids = {entry["track_id"] for entry in unified_entries}

    df = pd.read_parquet(MANIFEST_PATH, columns=["track_id"])
    manifest_track_ids = set(df["track_id"].unique())

    assert track_ids.issubset(manifest_track_ids), "Some tracks missing from chunk manifest"
    print(f"Manifest covers {len(manifest_track_ids)} tracks")

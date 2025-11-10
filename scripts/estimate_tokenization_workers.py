from __future__ import annotations
import json, math
from pathlib import Path
import statistics as stats
import note_seq
from configs import load_project_config
from configs.project_config import PROJECT_ROOT
CONFIG = load_project_config()
paths = CONFIG["paths"]["datasets"]
unified_index = (PROJECT_ROOT / paths["unified_index"]).resolve()
with unified_index.open() as fp:
    payload = json.load(fp)
entries = payload.get("entries", []) if isinstance(payload, dict) else payload
sample_rate = CONFIG["audio"]["io"]["sample_rate"]
chunk_samples = CONFIG["audio"]["features"]["chunk_samples"]
chunk_duration = chunk_samples / sample_rate
MAX_PER_DATASET = 50
stats_by_ds: dict[str, list[tuple[float,float,float]]] = {}
counts: dict[str, int] = {}
for entry in entries:
    ds = entry.get("dataset")
    if not ds:
        continue
    counts.setdefault(ds, 0)
    if counts[ds] >= MAX_PER_DATASET:
        continue
    midi_path = (PROJECT_ROOT / entry["midi_path"]).resolve()
    if not midi_path.exists():
        continue
    try:
        ns = note_seq.midi_file_to_note_sequence(str(midi_path))
    except Exception as exc:
        print(f"[warn] failed to load {midi_path}: {exc}")
        continue
    duration = max(ns.total_time, 1e-6)
    note_count = len(ns.notes)
    density = note_count / duration
    chunk_count = max(1, math.ceil(duration / chunk_duration))
    work = density * chunk_count
    stats_by_ds.setdefault(ds, []).append((density, chunk_count, work))
    counts[ds] += 1
quantiles = (0.1, 0.5, 0.9)
for ds, rows in stats_by_ds.items():
    densities = [r[0] for r in rows]
    chunks = [r[1] for r in rows]
    work_units = [r[2] for r in rows]
    print(f"Dataset: {ds} (samples={len(rows)})")
    for label, data in (("density", densities), ("chunks", chunks), ("work", work_units)):
        if not data:
            continue
        data_sorted = sorted(data)
        q_vals = []
        for q in quantiles:
            idx = int(round((len(data_sorted) - 1) * q))
            q_vals.append(data_sorted[idx])
        avg = stats.mean(data)
        print(f"  {label}: mean={avg:.2f}, q10={q_vals[0]:.2f}, q50={q_vals[1]:.2f}, q90={q_vals[2]:.2f}")
    print()
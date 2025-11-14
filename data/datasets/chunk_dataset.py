"""PyTorch Dataset that loads cached MT3 chunks on demand."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional

import math
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info

from configs import load_project_config
from data.audio import spectrogram
from data.audio.mixing import AudioMixer
from data.datasets.tokenizer import OnTheFlyTokenizer, TokenizerConfig
from data.preprocessing.shard_reader import ChunkShard

PROJECT_CONFIG = load_project_config()
DEFAULT_SAMPLE_RATE = PROJECT_CONFIG["audio"]["io"]["sample_rate"]
DEFAULT_HOP_LENGTH = PROJECT_CONFIG["audio"]["features"]["hop_length"]
TOKENIZER_MAX_LENGTH = int(
    PROJECT_CONFIG.get("symbolic", {})
    .get("tokenizer", {})
    .get("max_token_length", 0)
    or 0
)


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return [value]


def _pad_waveform(waveform: torch.Tensor, length: int) -> torch.Tensor:
    if waveform.shape[-1] == length:
        return waveform
    if waveform.shape[-1] > length:
        return waveform[..., :length]
    pad = length - waveform.shape[-1]
    return torch.nn.functional.pad(waveform, (0, pad))


@dataclass
class ChunkDatasetConfig:
    feature_type: str = "waveform"  # "waveform", "log_mel", or "both"
    load_tokens: bool = True
    shard_cache_size: int = 8
    max_examples_per_mix: int = 1


class ChunkDataset(Dataset):
    """Loads waveform chunks (and optional tokens) from cached shards."""

    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        *,
        config: ChunkDatasetConfig | None = None,
        tokenizer: OnTheFlyTokenizer | None = None,
        audio_mixer: AudioMixer | None = None,
        split_filter: str | Iterable[str] | None = None,
    ) -> None:
        super().__init__()
        self.manifest = self._load_manifest(manifest)
        self.manifest = self._filter_split(self.manifest, split_filter)
        self.config = config or ChunkDatasetConfig()
        self.tokenizer = tokenizer or OnTheFlyTokenizer()
        self.audio_mixer = audio_mixer
        self._worker_states: Dict[int, Dict[str, Any]] = {}
        self._slakh_indices = [
            idx for idx, value in self.manifest["dataset"].items() if value == "slakh_stem"
        ]

    @staticmethod
    def _load_manifest(manifest: pd.DataFrame | str | Path) -> pd.DataFrame:
        if isinstance(manifest, pd.DataFrame):
            df = manifest.copy()
        else:
            path = Path(manifest)
            if path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path)
            elif path.suffix.lower() in {".json", ".jsonl"}:
                df = pd.read_json(path, lines=(path.suffix.lower() == ".jsonl"))
            else:
                raise ValueError(f"Unsupported manifest format: {path}")
        return df.reset_index(drop=True)

    @staticmethod
    def _filter_split(
        manifest: pd.DataFrame,
        split_filter: str | Iterable[str] | None,
    ) -> pd.DataFrame:
        if split_filter is None:
            return manifest
        if "split" not in manifest.columns:
            raise KeyError("Manifest missing 'split' column required for filtering.")
        if isinstance(split_filter, str):
            allowed = {split_filter}
        else:
            allowed = {str(item) for item in split_filter}
        filtered = manifest[manifest["split"].isin(allowed)]
        return filtered.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        state = self._ensure_worker_state()
        primary_row = self.manifest.iloc[index]

        mix_rows = [primary_row]
        max_mix = max(1, self.config.max_examples_per_mix)
        allow_mixing = (
            self.audio_mixer is not None
            and max_mix > 1
            and primary_row.get("dataset") == "slakh_stem"
            and self._slakh_indices
        )
        if allow_mixing:
            max_possible = min(max_mix, len(self._slakh_indices))
            mix_count = int(torch.randint(1, max_possible + 1, (1,)).item())
        else:
            mix_count = 1
        if allow_mixing and mix_count > 1:
            extra_indices = torch.randint(len(self._slakh_indices), (mix_count - 1,))
            for idx_val in extra_indices.tolist():
                manifest_idx = self._slakh_indices[idx_val]
                mix_rows.append(self.manifest.iloc[manifest_idx])

        for col in ("chunk_index", "chunk_start_s", "chunk_end_s"):
            if col not in primary_row:
                raise KeyError(f"Manifest missing required column '{col}'.")
        chunk_index = int(primary_row["chunk_index"])
        chunk_start_s = float(primary_row["chunk_start_s"])
        chunk_end_s = float(primary_row["chunk_end_s"])
        duration_s = max(0.0, chunk_end_s - chunk_start_s)

        waveforms: List[torch.Tensor] = []
        chunk_payloads: List[Dict[str, Any]] = []
        for mix_row in mix_rows:
            chunk = self._load_chunk(mix_row, state)
            chunk_payloads.append(chunk)
            waveform = chunk.get("waveform")
            if waveform is None:
                raise ValueError("Chunk payload missing waveform; regenerate cache with waveform storage enabled.")
            waveforms.append(waveform.to(dtype=torch.float32))

        if self.audio_mixer and len(waveforms) > 1:
            waveform = self.audio_mixer.mix(waveforms)
        else:
            waveform = waveforms[0]

        features: Dict[str, torch.Tensor] = {}
        if self.config.feature_type in {"waveform", "both"}:
            features["waveform"] = waveform

        log_mel = None
        if self.config.feature_type in {"log_mel", "both"}:
            log_mel = self._compute_log_mel(waveform, primary_row)
            features["log_mel"] = log_mel

        frame_times = self._frame_times(primary_row, log_mel, duration_s)

        token_length: int | None = None
        if self.config.load_tokens:
            if len(mix_rows) == 1:
                chunk = chunk_payloads[0]
                stored = chunk.get("tokens")
                if stored is not None and stored.numel() > 0:
                    tokens = stored.to(dtype=torch.int32)
                else:
                    tokens = self.tokenizer.tokens_for(
                        midi_path=primary_row["midi_path"],
                        chunk_start_s=chunk_start_s,
                        chunk_end_s=chunk_end_s,
                        frame_times=frame_times,
                    )
            else:
                segments = []
                for row in mix_rows:
                    seg_start, seg_end = self._chunk_window(row)
                    segments.append(
                        self.tokenizer.segment_sequence(
                            midi_path=row["midi_path"],
                            chunk_start_s=seg_start,
                            chunk_end_s=seg_end,
                        )
                    )
                tokens = self.tokenizer.tokens_for_segments(
                    segments=segments,
                    chunk_duration_s=duration_s,
                    frame_times=frame_times,
                )
            if tokens is not None:
                raw_length = tokens.shape[-1]
                token_length = raw_length
                if TOKENIZER_MAX_LENGTH:
                    if raw_length > TOKENIZER_MAX_LENGTH:
                        tokens = tokens[:TOKENIZER_MAX_LENGTH]
                        token_length = TOKENIZER_MAX_LENGTH
                    elif raw_length < TOKENIZER_MAX_LENGTH:
                        tokens = torch.nn.functional.pad(
                            tokens,
                            (0, TOKENIZER_MAX_LENGTH - raw_length),
                        )
                features["tokens"] = tokens

        instrument_programs: List[int] = []
        instrument_classes: List[str] = []
        for row in mix_rows:
            instrument_programs.extend(_ensure_list(row.get("instrument_programs")))
            instrument_classes.extend(_ensure_list(row.get("instrument_classes")))
        deduped_programs: List[int] = []
        for program in instrument_programs:
            if program not in deduped_programs:
                deduped_programs.append(program)
        deduped_classes: List[str] = []
        for cls in instrument_classes:
            if cls not in deduped_classes:
                deduped_classes.append(cls)

        sample = {
            **features,
            "frame_times": torch.as_tensor(frame_times, dtype=torch.float32),
            "metadata": {
                "chunk_id": primary_row["chunk_id"],
                "chunk_index": chunk_index,
                "dataset": primary_row.get("dataset"),
                "split": primary_row.get("split"),
                "chunk_start_s": chunk_start_s,
                "chunk_end_s": chunk_end_s,
                "sample_rate": self._sample_rate(primary_row),
                "hop_length": self._hop_length(primary_row),
                "mix_count": len(mix_rows),
                "instrument_programs": deduped_programs,
                "instrument_classes": deduped_classes,
            },
        }
        if token_length is not None:
            sample["token_length"] = token_length
        return sample

    def _compute_log_mel(self, waveform: torch.Tensor, row: pd.Series) -> torch.Tensor:
        sample_rate = self._sample_rate(row)
        log_mel = spectrogram.waveform_to_logmel(
            waveform.unsqueeze(0),
            sample_rate=sample_rate,
        )
        return log_mel.squeeze(0)

    def _frame_times(
        self,
        row: pd.Series,
        log_mel: Optional[torch.Tensor],
        duration_s: float,
    ) -> List[float]:
        hop_seconds = self._hop_length(row) / self._sample_rate(row)

        if log_mel is not None:
            frames = log_mel.shape[-1]
        else:
            frames = max(1, int(torch.ceil(torch.tensor(duration_s / hop_seconds)).item()))
        return [i * hop_seconds for i in range(frames)]

    def _load_chunk(self, row: pd.Series, state: Dict[str, Any]) -> Dict[str, Any]:
        storage = str(row.get("chunk_storage", "per_chunk"))
        shard_path = Path(row["chunk_shard_path"])
        if storage == "per_track":
            shard = self._get_cached_shard(shard_path, state)
            chunk_slug = row.get("chunk_slug") or f"chunk_{int(row['chunk_index']):05d}"
            return shard.get_by_slug(chunk_slug)
        chunk_path = Path(row.get("chunk_path") or shard_path)
        return torch.load(chunk_path, map_location="cpu")

    def _get_cached_shard(self, shard_path: Path, state: Dict[str, Any]) -> ChunkShard:
        cache: OrderedDict[Path, ChunkShard] = state["shards"]
        shard = cache.get(shard_path)
        if shard is None:
            shard = ChunkShard(shard_path)
            cache[shard_path] = shard
            if len(cache) > self.config.shard_cache_size:
                cache.popitem(last=False)
        else:
            cache.move_to_end(shard_path)
        return shard

    def _ensure_worker_state(self) -> Dict[str, Any]:
        info = get_worker_info()
        worker_id = info.id if info else 0
        state = self._worker_states.get(worker_id)
        if state is None:
            state = {"shards": OrderedDict()}
            self._worker_states[worker_id] = state
        return state

    @staticmethod
    def _chunk_index(row: pd.Series) -> int:
        value = row.get("chunk_index")
        if pd.notna(value):
            return int(value)
        shard_index = row.get("chunk_shard_index")
        if pd.notna(shard_index):
            return int(shard_index)
        chunk_id = row.get("chunk_id") or ""
        if "chunk" in chunk_id:
            suffix = chunk_id.split("chunk")[-1]
            try:
                return int(suffix)
            except ValueError:
                pass
        return 0

    @staticmethod
    def _chunk_window(row: pd.Series) -> tuple[float, float]:
        return float(row["chunk_start_s"]), float(row["chunk_end_s"])

    @staticmethod
    def _sample_rate(row: pd.Series) -> int:
        value = row.get("sample_rate")
        if pd.notna(value):
            return int(value)
        return DEFAULT_SAMPLE_RATE

    @staticmethod
    def _hop_length(row: pd.Series) -> int:
        value = row.get("hop_length")
        if pd.notna(value):
            return int(value)
        return DEFAULT_HOP_LENGTH

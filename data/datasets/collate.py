"""Batch collation utilities for chunk datasets."""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from data.audio import spectrogram


def _pad_waveforms(samples: Sequence[torch.Tensor]) -> torch.Tensor:
    lengths = [s.shape[-1] for s in samples]
    max_len = max(lengths)
    padded = [torch.nn.functional.pad(s, (0, max_len - s.shape[-1])) for s in samples]
    return torch.stack(padded), torch.tensor(lengths, dtype=torch.int32)


def chunk_collate_fn(
    batch: List[Dict[str, Any]],
    *,
    compute_log_mel: bool = False,
    log_mel_on_device: str | torch.device | None = None,
) -> Dict[str, Any]:
    """Collate a list of chunk samples into padded tensors."""
    waveforms = [sample["waveform"] for sample in batch if "waveform" in sample]
    if not waveforms:
        raise ValueError("Each sample must include a waveform tensor for collation.")
    stacked_waveform, waveform_lengths = _pad_waveforms(waveforms)

    log_mel = None
    log_mel_mask = None
    if compute_log_mel or any("log_mel" in sample for sample in batch):
        if compute_log_mel:
            device = log_mel_on_device or "cpu"
            log_mel = spectrogram.waveform_to_logmel(
                stacked_waveform.to(device=device, non_blocking=False),
                sample_rate=int(batch[0]["metadata"]["sample_rate"]),
                device=device,
            )
        else:
            log_mel = pad_sequence(
                [sample["log_mel"].transpose(0, 1) for sample in batch],
                batch_first=True,
            ).permute(0, 2, 1)
    tokens = None
    token_mask = None
    if any("tokens" in sample for sample in batch):
        padded_tokens = pad_sequence(
            [sample["tokens"] for sample in batch],
            batch_first=True,
            padding_value=0,
        )
        token_lengths = torch.tensor(
            [
                sample.get("token_length", sample["tokens"].shape[-1])
                for sample in batch
            ],
            dtype=torch.int32,
        )
        token_mask = torch.arange(padded_tokens.shape[-1]).unsqueeze(0) < token_lengths.unsqueeze(1)
        tokens = padded_tokens

    metadata = {
        "chunk_id": [sample["metadata"]["chunk_id"] for sample in batch],
        "dataset": [sample["metadata"]["dataset"] for sample in batch],
        "split": [sample["metadata"]["split"] for sample in batch],
        "chunk_start_s": [sample["metadata"]["chunk_start_s"] for sample in batch],
        "chunk_end_s": [sample["metadata"]["chunk_end_s"] for sample in batch],
        "mix_count": [sample["metadata"]["mix_count"] for sample in batch],
        "instrument_programs": [
            sample["metadata"].get("instrument_programs", []) for sample in batch
        ],
        "instrument_classes": [
            sample["metadata"].get("instrument_classes", []) for sample in batch
        ],
    }

    result: Dict[str, Any] = {
        "waveform": stacked_waveform,
        "metadata": metadata,
    }
    if log_mel is not None:
        result["log_mel"] = log_mel
        hop_length = int(batch[0]["metadata"]["hop_length"])
        hop_length = hop_length if hop_length > 0 else 1
        frame_lengths = torch.div(waveform_lengths, hop_length, rounding_mode="ceil")
        log_mel_mask = (
            torch.arange(log_mel.shape[-1])
            .unsqueeze(0)
            .to(frame_lengths.device)
            < frame_lengths.unsqueeze(1)
        )
        result["log_mel_mask"] = log_mel_mask
    if tokens is not None:
        result["tokens"] = tokens
        result["token_mask"] = token_mask
    return result

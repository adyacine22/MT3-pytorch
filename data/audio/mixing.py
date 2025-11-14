"""Lightweight audio mixing utilities used during dataset loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass
class AudioMixerConfig:
    target_rms: float = 0.1
    noise_snr_db: float | None = None


class AudioMixer:
    """Combine multiple waveform tensors in a numerically stable way."""

    def __init__(self, config: AudioMixerConfig | None = None) -> None:
        self.config = config or AudioMixerConfig()

    def mix(self, waveforms: Sequence[torch.Tensor]) -> torch.Tensor:
        """Sum the provided mono waveforms and normalize RMS."""
        if not waveforms:
            raise ValueError("AudioMixer.mix requires at least one waveform tensor")
        stacked = torch.stack([self._match_length(w, waveforms[0].shape[-1]) for w in waveforms])
        mix = stacked.sum(dim=0)
        rms = torch.sqrt(torch.mean(mix**2) + 1e-8)
        target = self.config.target_rms
        if rms > 0:
            mix = mix * (target / rms)
        if self.config.noise_snr_db is not None:
            mix = self._add_noise(mix, self.config.noise_snr_db)
        return mix

    def _add_noise(self, waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
        signal_power = torch.mean(waveform**2) + 1e-8
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise

    @staticmethod
    def _match_length(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
        if waveform.shape[-1] == target_len:
            return waveform
        if waveform.shape[-1] > target_len:
            return waveform[..., :target_len]
        pad = target_len - waveform.shape[-1]
        return torch.nn.functional.pad(waveform, (0, pad))

"""Spectrogram utilities."""

from __future__ import annotations

import torch
import torchaudio

from configs import load_project_config

CONFIG = load_project_config()
AUDIO_CFG = CONFIG["audio"]
FEATURE_CFG = AUDIO_CFG["features"]
IO_CFG = AUDIO_CFG["io"]


def waveform_to_logmel(
    waveform: torch.Tensor,
    sample_rate: int | None = None,
    device: str | None = None,
) -> torch.Tensor:
    """Convert a waveform tensor to log-mel spectrogram using PyTorch/torchaudio."""
    sr = sample_rate or IO_CFG["sample_rate"]
    return _torch_logmel(waveform, sr, device=device)


def compute_features(
    waveform: torch.Tensor,
    sample_rate: int | None = None,
    store_logmel: bool = True,
    store_waveform: bool = True,
    device: str | None = None,
) -> dict:
    """Compute configured audio features from waveform chunk."""
    sr = sample_rate or IO_CFG["sample_rate"]
    result: dict[str, torch.Tensor] = {}

    if store_waveform:
        result["waveform"] = waveform.clone()

    if store_logmel:
        log_mel = _torch_logmel(waveform, sr, device=device)
        result["log_mel"] = log_mel

    result["sample_rate"] = torch.tensor(sr)
    return result


def _torch_logmel(waveform: torch.Tensor, sample_rate: int, device: str | None = None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tensor = waveform.to(device=device, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    window_fn = torch.hann_window if FEATURE_CFG["window"] == "hann" else torch.hann_window

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=FEATURE_CFG["n_fft"],
        hop_length=FEATURE_CFG["hop_length"],
        win_length=FEATURE_CFG["win_length"],
        window_fn=window_fn,
        power=FEATURE_CFG["mel_power"],
        center=FEATURE_CFG["center"],
        f_min=FEATURE_CFG["mel_fmin"],
        f_max=FEATURE_CFG["mel_fmax"],
        n_mels=FEATURE_CFG["num_mels"],
    ).to(device)

    mel = mel_transform(tensor)
    log_mel = torch.log(mel + FEATURE_CFG["log_epsilon"])
    return log_mel.squeeze(0)

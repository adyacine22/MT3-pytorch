"""Waveform augmentation utilities."""
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torchaudio
import pyroomacoustics as pra
import numpy as np

from configs import load_project_config

CONFIG = load_project_config()
AUG_CFG = CONFIG["audio"].get("augment", {})
IO_CFG = CONFIG["audio"]["io"]


def apply_augmentation(
    waveform: torch.Tensor,
    sample_rate: int,
    profiles: Iterable[str] | str = "none",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Apply one or more augmentation profiles sequentially."""
    if not AUG_CFG:
        return waveform
    if isinstance(profiles, str):
        profile_list = [profiles]
    else:
        profile_list = list(profiles)
    if not profile_list:
        return waveform
    waveform = waveform.to(dtype=torch.float32)
    target_len = waveform.shape[-1]

    rng = torch.Generator(device=waveform.device)
    if seed is not None:
        rng.manual_seed(seed)
    augmented = waveform
    for profile in profile_list:
        name = (profile or "").strip().lower()
        if name in {"", "none"}:
            continue
        if name == "rir":
            rir_params = AUG_CFG.get("rir_params", {})
            augmented = _apply_rir(augmented, sample_rate, rir_params, rng)
        elif name == "noise":
            snr_range = AUG_CFG.get("noise_snr_db", [-5, 5])
            augmented = _apply_noise(augmented, snr_range, rng)
        elif name == "eq":
            gains = AUG_CFG.get("eq_gain_db", [-3, 3])
            augmented = _apply_eq(augmented, gains, rng)
        elif name == "clip":
            threshold = AUG_CFG.get("clip_dbfs", -3.0)
            augmented = _apply_clipping(augmented, threshold)
        elif name == "gain":
            jitter_db = AUG_CFG.get("gain_jitter_db", [-3, 3])
            augmented = _apply_gain_jitter(augmented, jitter_db, rng)
        augmented = _match_length(augmented, target_len)
    return augmented


def _apply_rir(waveform: torch.Tensor, sample_rate: int, params: dict, rng: torch.Generator) -> torch.Tensor:
    rir = _generate_rir(sample_rate, params, rng, waveform.device)
    waveform_ = waveform.unsqueeze(0).unsqueeze(0)
    rir_ = rir.unsqueeze(0).unsqueeze(0)
    audio = torch.nn.functional.conv1d(
        waveform_,
        rir_,
        padding=rir.numel() - 1,
    )
    return audio.squeeze(0).squeeze(0)


def _generate_rir(sample_rate: int, params: dict, rng: torch.Generator, device: torch.device) -> torch.Tensor:
    room_dim_base = np.array(params.get("room_dim", [6.0, 5.0, 3.0]), dtype=np.float32)
    jitter = params.get("position_jitter", 0.3)
    room_dim = room_dim_base + (np.random.rand(3) - 0.5) * jitter
    absorption = params.get("absorption", 0.4)
    max_order = params.get("max_order", 3)

    room = pra.ShoeBox(
        room_dim,
        fs=sample_rate,
        absorption=absorption,
        max_order=max_order,
    )

    source = [
        room_dim[0] * 0.5 + (np.random.rand() - 0.5) * jitter,
        room_dim[1] * 0.5 + (np.random.rand() - 0.5) * jitter,
        params.get("source_height", 1.5),
    ]
    mic = [
        room_dim[0] * 0.3 + (np.random.rand() - 0.5) * jitter,
        room_dim[1] * 0.3 + (np.random.rand() - 0.5) * jitter,
        params.get("mic_height", 1.5),
    ]

    room.add_source(source)
    room.add_microphone(mic)
    room.compute_rir()

    rir = torch.tensor(room.rir[0][0], dtype=torch.float32, device=device) # type: ignore
    return rir / torch.norm(rir, p=2)


def _apply_noise(waveform: torch.Tensor, snr_range: list[float], rng: torch.Generator) -> torch.Tensor:
    snr_db_tensor = torch.empty(1, dtype=waveform.dtype, device=waveform.device)
    snr_db_tensor.uniform_(snr_range[0], snr_range[1])
    snr_db = snr_db_tensor.item()
    signal_power = waveform.pow(2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
    return waveform + noise


def _apply_eq(waveform: torch.Tensor, gains_db: list[float], rng: torch.Generator) -> torch.Tensor:
    low_gain_tensor = torch.empty(1, dtype=waveform.dtype, device=waveform.device)
    low_gain_tensor.uniform_(gains_db[0], gains_db[1])
    low_gain = low_gain_tensor.item()
    high_gain_tensor = torch.empty(1, dtype=waveform.dtype, device=waveform.device)
    high_gain_tensor.uniform_(gains_db[0], gains_db[1])
    high_gain = high_gain_tensor.item()
    eq = torchaudio.functional.equalizer_biquad
    audio = eq(waveform.unsqueeze(0), IO_CFG["sample_rate"], center_freq=100.0, Q=0.707, gain=low_gain)
    audio = eq(audio, IO_CFG["sample_rate"], center_freq=3000.0, Q=0.707, gain=high_gain)
    return audio.squeeze(0)


def _apply_clipping(waveform: torch.Tensor, threshold_dbfs: float) -> torch.Tensor:
    threshold = 10 ** (threshold_dbfs / 20)
    return torch.clamp(waveform, -threshold, threshold)


def _apply_gain_jitter(waveform: torch.Tensor, jitter_db: list[float], rng: torch.Generator) -> torch.Tensor:
    gain_tensor = torch.empty(1, dtype=waveform.dtype, device=waveform.device)
    gain_tensor.uniform_(jitter_db[0], jitter_db[1])
    gain_db = gain_tensor.item()
    gain = 10 ** (gain_db / 20)
    return waveform * gain


def _match_length(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    if waveform.shape[-1] == target_len:
        return waveform
    if waveform.shape[-1] > target_len:
        return waveform[..., :target_len]
    pad_len = target_len - waveform.shape[-1]
    pad = torch.zeros(pad_len, device=waveform.device, dtype=waveform.dtype)
    return torch.cat([waveform, pad], dim=-1)
import numpy as np

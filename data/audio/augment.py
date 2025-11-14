"""Waveform augmentation utilities."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pyroomacoustics as pra
import torch
import torchaudio

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
    if not AUG_CFG or waveform.numel() == 0:
        return waveform
    if isinstance(profiles, str):
        profile_list = [profiles]
    else:
        profile_list = list(profiles)
    if not profile_list:
        return waveform

    orig_dtype = waveform.dtype
    tensor = waveform.to(dtype=torch.float32)
    added_batch_dim = False
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
        added_batch_dim = True
    elif tensor.ndim != 2:
        raise ValueError("Augmentation expects waveform with shape [T] or [B, T].")

    target_len = tensor.shape[-1]
    devices = None
    if tensor.is_cuda:
        devices = [tensor.device]
    with torch.random.fork_rng(devices=devices, enabled=seed is not None):
        if seed is not None:
            torch.manual_seed(seed)
            if tensor.is_cuda:
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed % (2**32))
        augmented = tensor
        for profile in profile_list:
            name = (profile or "").strip().lower()
            if name in {"", "none"}:
                continue
            if name == "rir":
                rir_params = AUG_CFG.get("rir_params", {})
                augmented = _apply_rir_batch(augmented, sample_rate, rir_params)
            elif name == "noise":
                snr_range = AUG_CFG.get("noise_snr_db", [-5, 5])
                augmented = _apply_noise(augmented, snr_range)
            elif name == "eq":
                gains = AUG_CFG.get("eq_gain_db", [-3, 3])
                augmented = _apply_eq(augmented, sample_rate, gains)
            elif name == "clip":
                threshold = AUG_CFG.get("clip_dbfs", -3.0)
                augmented = _apply_clipping(augmented, threshold)
            elif name == "gain":
                jitter_db = AUG_CFG.get("gain_jitter_db", [-3, 3])
                augmented = _apply_gain_jitter(augmented, jitter_db)
            augmented = _match_length(augmented, target_len)

    if added_batch_dim:
        augmented = augmented.squeeze(0)
    return augmented.to(dtype=orig_dtype)


def _apply_rir_batch(
    waveforms: torch.Tensor,
    sample_rate: int,
    params: dict,
) -> torch.Tensor:
    outputs = []
    for sample in waveforms:
        outputs.append(_apply_rir_single(sample, sample_rate, params))
    return torch.stack(outputs, dim=0)


def _apply_rir_single(
    waveform: torch.Tensor,
    sample_rate: int,
    params: dict,
) -> torch.Tensor:
    rir = _generate_rir(sample_rate, params, waveform.device)
    waveform_ = waveform.unsqueeze(0).unsqueeze(0)
    rir_ = rir.unsqueeze(0).unsqueeze(0)
    audio = torch.nn.functional.conv1d(
        waveform_,
        rir_,
        padding=rir.numel() - 1,
    )
    return audio.squeeze(0).squeeze(0)


def _generate_rir(sample_rate: int, params: dict, device: torch.device) -> torch.Tensor:
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


def _apply_noise(
    waveforms: torch.Tensor,
    snr_range: list[float],
) -> torch.Tensor:
    batch = waveforms.size(0)
    snr_db = torch.rand(batch, dtype=waveforms.dtype, device=waveforms.device)
    snr_db = snr_db * (snr_range[1] - snr_range[0]) + snr_range[0]
    signal_power = waveforms.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8)
    ten = torch.tensor(10.0, dtype=waveforms.dtype, device=waveforms.device)
    snr_ratio = torch.pow(ten, snr_db / 10.0).unsqueeze(1)
    noise_power = signal_power / snr_ratio
    noise = torch.randn_like(waveforms) * torch.sqrt(noise_power)
    return waveforms + noise


def _apply_eq(
    waveforms: torch.Tensor,
    sample_rate: int,
    gains_db: list[float],
) -> torch.Tensor:
    batch = waveforms.size(0)
    low_gains = torch.rand(batch, dtype=waveforms.dtype, device=waveforms.device)
    high_gains = torch.rand(batch, dtype=waveforms.dtype, device=waveforms.device)
    low_gains = low_gains * (gains_db[1] - gains_db[0]) + gains_db[0]
    high_gains = high_gains * (gains_db[1] - gains_db[0]) + gains_db[0]

    outputs = []
    for idx in range(batch):
        audio = torchaudio.functional.equalizer_biquad(
            waveforms[idx].unsqueeze(0),
            sample_rate,
            center_freq=100.0,
            Q=0.707,
            gain=float(low_gains[idx].item()),
        )
        audio = torchaudio.functional.equalizer_biquad(
            audio,
            sample_rate,
            center_freq=3000.0,
            Q=0.707,
            gain=float(high_gains[idx].item()),
        )
        outputs.append(audio.squeeze(0))
    return torch.stack(outputs, dim=0)


def _apply_clipping(waveforms: torch.Tensor, threshold_dbfs: float) -> torch.Tensor:
    threshold = 10 ** (threshold_dbfs / 20)
    return torch.clamp(waveforms, -threshold, threshold)


def _apply_gain_jitter(
    waveforms: torch.Tensor,
    jitter_db: list[float],
) -> torch.Tensor:
    batch = waveforms.size(0)
    gain_db = torch.rand(batch, dtype=waveforms.dtype, device=waveforms.device)
    gain_db = gain_db * (jitter_db[1] - jitter_db[0]) + jitter_db[0]
    ten = torch.tensor(10.0, dtype=waveforms.dtype, device=waveforms.device)
    gains = torch.pow(ten, gain_db / 20.0).unsqueeze(1)
    return waveforms * gains


def _match_length(waveforms: torch.Tensor, target_len: int) -> torch.Tensor:
    current = waveforms.shape[-1]
    if current == target_len:
        return waveforms
    if current > target_len:
        return waveforms[..., :target_len]
    pad_len = target_len - current
    pad_shape = list(waveforms.shape)
    pad_shape[-1] = pad_len
    pad = torch.zeros(pad_shape, device=waveforms.device, dtype=waveforms.dtype)
    return torch.cat([waveforms, pad], dim=-1)

"""Audio loading and normalization utilities."""

from __future__ import annotations

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt

from configs import load_project_config

CONFIG = load_project_config()
AUDIO_CFG = CONFIG["audio"]
AUDIO_IO = AUDIO_CFG["io"]
AUDIO_PRE = AUDIO_CFG["preprocess"]


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio as float32 numpy array at native rate; no resampling yet."""
    audio, sr = sf.read(path, always_2d=False)
    audio = audio.astype(AUDIO_IO["dtype"])
    return audio, sr


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio (samples x channels) to mono."""
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio, axis=1)
    raise ValueError("Audio array must be 1D or 2D.")


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def high_pass_filter(audio: np.ndarray, sr: int, cutoff_hz: float, order: int) -> np.ndarray:
    """Apply a Butterworth high-pass filter to remove DC/rumble."""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normal_cutoff, btype="highpass", analog=False, output="ba")  # type: ignore[arg-type]
    return filtfilt(b, a, audio)


def normalize_loudness(audio: np.ndarray, target_lufs: float) -> np.ndarray:
    """Approximate loudness normalization using RMS."""
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return audio
    target_rms = 10 ** (target_lufs / 20)
    gain = target_rms / rms
    return audio * gain


def apply_pre_emphasis(audio: np.ndarray, alpha: float) -> np.ndarray:
    """Apply pre-emphasis filter."""
    if audio.size == 0:
        return audio
    emphasized = np.append(audio[0], audio[1:] - alpha * audio[:-1])
    return emphasized


def apply_dither(audio: np.ndarray, amplitude: float) -> np.ndarray:
    if amplitude <= 0:
        return audio
    noise = np.random.uniform(-amplitude, amplitude, size=audio.shape)
    return audio + noise


def slice_chunk(audio: np.ndarray, start_sample: int, chunk_samples: int, pad_value: float) -> np.ndarray:
    """Slice a chunk from the waveform, padding if needed."""
    end_sample = start_sample + chunk_samples
    chunk = audio[start_sample:end_sample]
    if len(chunk) < chunk_samples:
        chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode="constant", constant_values=pad_value)
    return chunk


def prepare_waveform(path: str) -> tuple[np.ndarray, int]:
    """Load audio file and apply standard preprocessing (mono, resample, filters)."""
    audio, sr = load_audio(path)
    if AUDIO_IO["convert_to_mono"]:
        audio = to_mono(audio)
    target_sr = AUDIO_IO["sample_rate"]
    audio = resample_audio(audio, sr, target_sr)

    if AUDIO_PRE["dc_block"]:
        audio = high_pass_filter(audio, target_sr, AUDIO_PRE["high_pass_hz"], AUDIO_PRE["high_pass_order"])
    audio = normalize_loudness(audio, AUDIO_PRE["loudness_target_lufs"])
    audio = apply_pre_emphasis(audio, AUDIO_PRE["pre_emphasis"])
    audio = apply_dither(audio, AUDIO_PRE["dither_amplitude"])
    return audio, target_sr

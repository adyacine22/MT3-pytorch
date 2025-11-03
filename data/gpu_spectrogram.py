"""
GPU-accelerated mel-spectrogram computation.
Matches legacy MT3 pipeline: frames → flatten → compute mel-spectrogram.
"""

import torch
from data.spectrogram import MelSpectrogram
from data.constants import (
    DEFAULT_NUM_MEL_BINS, DEFAULT_SAMPLE_RATE, FFT_SIZE,
    DEFAULT_HOP_WIDTH, MEL_FMIN, MEL_FMAX
)


class GPUSpectrogramComputer(torch.nn.Module):
    """
    GPU-accelerated mel-spectrogram computation following legacy MT3 pipeline.
    
    This matches the legacy code flow:
    1. Input: frames of shape [B, num_frames, hop_width] (e.g., [B, 256, 128])
    2. Flatten frames to samples: [B, num_frames * hop_width] (e.g., [B, 32768])
    3. Compute mel-spectrogram: [B, num_frames, n_mels] (e.g., [B, 256, 512])
    
    The key is that we receive PRE-CHUNKED frames from the DataLoader,
    then flatten and compute mel on GPU.
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        # Handle both string and torch.device
        if isinstance(device, torch.device):
            device = str(device)
        self.device = device
        
        # Create mel-spectrogram computer (same as used in cached loaders)
        self.mel_computer = MelSpectrogram(
            n_mels=DEFAULT_NUM_MEL_BINS,
            sample_rate=DEFAULT_SAMPLE_RATE,
            filter_length=FFT_SIZE,
            hop_length=DEFAULT_HOP_WIDTH,
            mel_fmin=MEL_FMIN,
            mel_fmax=MEL_FMAX,
        ).to(device)
        
        # Set to eval mode (no training needed)
        self.mel_computer.eval()
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Convert frames to mel-spectrograms on GPU.
        
        Args:
            frames: Tensor of shape [B, num_frames, hop_width]
                    e.g., [128, 256, 128] for batch_size=128, 256 frames, hop=128
        
        Returns:
            mel: Tensor of shape [B, num_frames, n_mels]
                 e.g., [128, 256, 512] for batch_size=128, 256 frames, 512 mel bins
                 Note: Shape is [B, T, F] to match model expectations
        """
        batch_size, num_frames, hop_width = frames.shape
        
        # Flatten frames to samples: [B, num_frames, hop_width] -> [B, num_frames * hop_width]
        samples = frames.reshape(batch_size, num_frames * hop_width)
        
        # Compute mel-spectrogram: [B, num_samples] -> [B, n_mels, num_frames_computed]
        with torch.no_grad():  # No gradients needed for spectrogram computation
            mel = self.mel_computer(samples)
        
        # Trim to exactly num_frames (STFT may produce num_frames+1 due to reflection padding)
        if mel.shape[2] > num_frames:
            mel = mel[:, :, :num_frames]
        
        # Transpose to [B, T, F] format expected by model: [B, n_mels, num_frames] -> [B, num_frames, n_mels]
        mel = mel.transpose(1, 2)
        
        return mel


def split_audio_to_frames(audio: torch.Tensor, hop_width: int = DEFAULT_HOP_WIDTH) -> torch.Tensor:
    """
    Split audio samples into non-overlapping frames.
    Matches legacy MT3: spectrograms.split_audio()
    
    Args:
        audio: Tensor of shape [num_samples] or [B, num_samples]
        hop_width: Frame size (default: 128 samples)
    
    Returns:
        frames: Tensor of shape [num_frames, hop_width] or [B, num_frames, hop_width]
    """
    if audio.dim() == 1:
        # Single audio: [num_samples] -> [num_frames, hop_width]
        num_samples = audio.shape[0]
        # Pad to multiple of hop_width
        pad_amount = (hop_width - num_samples % hop_width) % hop_width
        if pad_amount > 0:
            audio = torch.nn.functional.pad(audio, (0, pad_amount), mode='constant', value=0)
        
        # Reshape to frames
        num_frames = audio.shape[0] // hop_width
        frames = audio.reshape(num_frames, hop_width)
        
    else:
        # Batched audio: [B, num_samples] -> [B, num_frames, hop_width]
        batch_size, num_samples = audio.shape
        # Pad to multiple of hop_width
        pad_amount = (hop_width - num_samples % hop_width) % hop_width
        if pad_amount > 0:
            audio = torch.nn.functional.pad(audio, (0, pad_amount), mode='constant', value=0)
        
        # Reshape to frames
        num_frames = audio.shape[1] // hop_width
        frames = audio.reshape(batch_size, num_frames, hop_width)
    
    return frames

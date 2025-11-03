"""
Shared utility methods for dataset classes.
Used by all cached loaders to avoid code duplication.
"""

import torch
from data.constants import DEFAULT_HOP_WIDTH, TOKEN_PAD

# Constants (also defined in loaders but centralized here)
MEL_LENGTH = 256  # Number of frames per chunk
EVENT_LENGTH = 1024  # Token sequence length


class FrameProcessingMixin:
    """
    Mixin providing frame-based processing methods for hybrid approach.
    Returns frames instead of mel-spectrograms for GPU computation.
    """
    
    def _split_audio_to_frames(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Split audio samples into non-overlapping frames.
        Matches legacy MT3: spectrograms.split_audio()
        
        Args:
            audio: Audio samples [num_samples]
        
        Returns:
            frames: [num_frames, hop_width] where hop_width=128
        """
        num_samples = len(audio)
        hop_width = DEFAULT_HOP_WIDTH
        
        # Pad to multiple of hop_width
        pad_amount = (hop_width - num_samples % hop_width) % hop_width
        if pad_amount > 0:
            audio = torch.nn.functional.pad(audio, (0, pad_amount), mode='constant', value=0)
        
        # Reshape to frames: [num_samples] -> [num_frames, hop_width]
        num_frames = audio.shape[0] // hop_width
        frames = audio.reshape(num_frames, hop_width)
        
        return frames

    def _pad_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Pad or trim frames to exactly MEL_LENGTH frames.
        
        Args:
            frames: [num_frames, hop_width]
        
        Returns:
            padded_frames: [MEL_LENGTH, hop_width] = [256, 128]
        """
        num_frames = frames.shape[0]
        
        if num_frames < MEL_LENGTH:
            # Pad with zeros
            pad_length = MEL_LENGTH - num_frames
            frames = torch.nn.functional.pad(frames, (0, 0, 0, pad_length), mode='constant', value=0)
        else:
            # Trim to MEL_LENGTH
            frames = frames[:MEL_LENGTH]
        
        return frames

    def _pad_tokens(self, tokens):
        """Pad or trim token sequence to EVENT_LENGTH."""
        if len(tokens) < EVENT_LENGTH:
            tokens = list(tokens) + [TOKEN_PAD] * (EVENT_LENGTH - len(tokens))
        else:
            tokens = tokens[:EVENT_LENGTH]
        return torch.LongTensor(tokens)

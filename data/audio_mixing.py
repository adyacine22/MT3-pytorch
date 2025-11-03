"""
Audio mixing dataset wrapper for multi-example mixing during training.
Implements the CORRECT audio mixing strategy from legacy MT3 for pretraining.
CRITICAL: 
1. Mixes raw audio BEFORE computing mel-spectrogram (not after)
2. Merges MIDI events chronologically (not simple concatenation)
"""

import torch
import random
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, List
from data.spectrogram import MelSpectrogram
from data.constants import (
    DEFAULT_NUM_MEL_BINS, DEFAULT_SAMPLE_RATE, FFT_SIZE,
    DEFAULT_HOP_WIDTH, MEL_FMIN, MEL_FMAX, TOKEN_PAD
)
from data import vocabularies, utils


class AudioMixingDataset(Dataset):
    """
    Wraps a dataset to randomly mix audio examples during training.
    
    This implements the legacy MT3 audio mixing strategy where:
    1. Sample N examples (1 to max_examples_per_mix)
    2. Mix their audio waveforms
    3. Normalize by infinity norm
    4. Merge their token sequences
    
    Args:
        base_dataset: Base dataset to sample from
        max_examples_per_mix: Maximum number of examples to mix (default: 8)
        mix_probability: Probability of mixing vs single example (default: 1.0)
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        num_examples: int = 4,
        max_examples: int = 8,
        mixing_prob: float = 0.5,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        num_mel_bins: int = DEFAULT_NUM_MEL_BINS,
        codec: Optional[vocabularies.Codec] = None,
    ):
        """
        Args:
            base_dataset: Underlying dataset to wrap
            num_examples: Number of examples to mix per item
            max_examples: Maximum number of examples to mix
            mixing_prob: Probability of actually mixing (vs returning single example)
            sample_rate: Audio sample rate for mel computation
            num_mel_bins: Number of mel frequency bins
            codec: Vocabulary codec for token encoding/decoding (needed for event merging)
        """
        self.base_dataset = base_dataset
        self.num_examples = num_examples
        self.max_examples = max_examples
        self.mixing_prob = mixing_prob
        
        # Initialize mel-spectrogram computation for mixing
        self.melspectrogram = MelSpectrogram(
            n_mels=num_mel_bins,
            sample_rate=sample_rate,
            filter_length=FFT_SIZE,
            hop_length=DEFAULT_HOP_WIDTH,
            mel_fmin=MEL_FMIN,
            mel_fmax=MEL_FMAX,
        )
        
        # Initialize codec for proper event merging
        if codec is None:
            # Use default codec
            self.codec = vocabularies.build_codec(
                vocab_config=vocabularies.VocabularyConfig()
            )
        else:
            self.codec = codec
        
        # Audio parameters for frame time calculation
        self.sample_rate = sample_rate
        self.hop_length = DEFAULT_HOP_WIDTH

    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """
        Get item with optional mixing.
        
        Returns:
            dict with keys:
                - 'inputs': Mixed mel-spectrogram [MEL_LENGTH, num_mel_bins]
                - 'targets': Merged token sequence [EVENT_LENGTH]
        """
        # Decide whether to mix
        should_mix = random.random() < self.mixing_prob
        
        if not should_mix or self.max_examples == 1:
            # Return single example without mixing
            return self.base_dataset[idx]
        
        # Sample number of examples to mix (2 to max_examples)
        num_examples = random.randint(2, self.max_examples)
        
        # Sample random indices (can include the current idx)
        indices = random.sample(range(len(self.base_dataset)), num_examples)
        
        # Load all examples
        examples = [self.base_dataset[i] for i in indices]
        
        # Mix the examples
        return self._mix_examples(examples)
    
    def _mix_examples(self, examples):
        """
        Mix multiple examples together.
        CRITICAL: 
        1. Mixes raw audio BEFORE computing mel-spectrogram
        2. Merges MIDI events chronologically (proper polyphonic representation)
        
        Args:
            examples: List of example dicts with 'audio_chunk', 'inputs', and 'targets'
        
        Returns:
            Mixed example dict
        """
        # Extract raw audio chunks and tokens
        audio_chunks = [ex['audio_chunk'] for ex in examples]  # Each: [num_samples]
        token_lists = [ex['targets'] for ex in examples]  # Each: [EVENT_LENGTH]
        
        # ============================================================
        # PART 1: Mix raw audio waveforms (sum and normalize)
        # ============================================================
        # Legacy MT3: mixes audio, then computes mel (mel(a+b) != mel(a) + mel(b))
        audio_chunks = [torch.from_numpy(a) if isinstance(a, np.ndarray) else a for a in audio_chunks]
        mixed_audio = torch.stack(audio_chunks).sum(dim=0)  # Sum all audio
        
        # Normalize by L-infinity norm (max absolute value)
        max_abs = torch.max(torch.abs(mixed_audio))
        if max_abs > 1e-8:  # Avoid division by zero
            mixed_audio = mixed_audio / max_abs
        
        # Ensure audio is in valid range [-1, 1]
        mixed_audio = torch.clamp(mixed_audio, -1.0, 1.0)
        
        # Compute mel-spectrogram on mixed audio
        # Add batch dimension: [1, num_samples]
        mixed_audio_batch = mixed_audio.unsqueeze(0)
        mixed_mel = self.melspectrogram(mixed_audio_batch)  # [1, n_mels, frames]
        
        # Remove batch dimension and transpose to [frames, n_mels]
        mixed_mel = mixed_mel.squeeze(0).transpose(0, 1)
        
        # ============================================================
        # PART 2: Merge MIDI events chronologically
        # ============================================================
        # Compute frame times for the audio chunk
        num_frames = mixed_mel.shape[0]
        frame_times = np.arange(num_frames) / (self.sample_rate / self.hop_length)
        
        # Merge tokens properly (decode → merge events → re-encode)
        merged_tokens = self._merge_tokens_chronologically(token_lists, frame_times)
        
        return {
            'inputs': mixed_mel,
            'targets': merged_tokens
        }
    
    def _merge_tokens_chronologically(self, token_lists: List[torch.Tensor], frame_times: np.ndarray) -> torch.Tensor:
        """
        Properly merge multiple token sequences by decoding to events, 
        merging chronologically, and re-encoding.
        
        This ensures the ground truth MIDI accurately represents the mixed audio.
        
        Args:
            token_lists: List of token tensors to merge
            frame_times: Frame times for the audio chunk
            
        Returns:
            Merged token tensor
        """
        try:
            # Decode each token sequence to timed events
            event_lists = []
            for tokens in token_lists:
                # Convert to list of ints
                if isinstance(tokens, torch.Tensor):
                    token_ids = tokens.tolist()
                else:
                    token_ids = list(tokens)
                
                # Decode tokens to timed events
                # Note: This is approximate - we reconstruct events from tokens
                events = self._tokens_to_events(token_ids)
                event_lists.append(events)
            
            # Merge events chronologically
            merged_events = utils.merge_events(event_lists)
            
            # Re-encode events to tokens
            merged_tokens, _, _ = utils.timed_events_to_tokens(
                merged_events, self.codec, frame_times
            )
            
            # Truncate or pad to EVENT_LENGTH
            EVENT_LENGTH = len(token_lists[0])
            if len(merged_tokens) > EVENT_LENGTH:
                merged_tokens = merged_tokens[:EVENT_LENGTH]
            else:
                merged_tokens = list(merged_tokens) + [TOKEN_PAD] * (EVENT_LENGTH - len(merged_tokens))
            
            return torch.LongTensor(merged_tokens)
        
        except Exception as e:
            # Fallback to simple concatenation if event merging fails
            # This ensures training doesn't crash
            print(f"Warning: Event merging failed ({e}), falling back to concatenation")
            return self._merge_tokens_simple(token_lists)
    
    def _tokens_to_events(self, token_ids: List[int]) -> List[utils.TimedEvent]:
        """
        Convert token sequence to timed events.
        
        This reconstructs the event timeline from tokens by tracking:
        - Current time (accumulated from SHIFT events)
        - Instrument/program
        - Note on/off events
        """
        events = []
        current_time = 0.0
        
        for token_id in token_ids:
            # Skip padding tokens
            if token_id == TOKEN_PAD or token_id == 0:
                continue
            
            # Decode the token
            try:
                event = self.codec.decode_event_index(token_id)
                
                # Update time for shift events
                if event.type == 'shift':
                    current_time += event.value / self.codec.steps_per_second
                else:
                    # Create timed event for non-shift events
                    # TimedEvent(time, type, value)
                    timed_event = utils.TimedEvent(
                        time=current_time,
                        type=event.type,
                        value=event.value
                    )
                    events.append(timed_event)
            except:
                # Skip invalid tokens
                continue
        
        return events
    
    def _merge_tokens_simple(self, token_lists: List[torch.Tensor]) -> torch.Tensor:
        """
        Simple concatenation fallback (used if event merging fails).
        
        This is the old approach - concatenates tokens sequentially.
        Less accurate but guaranteed to work.
        """
        # Concatenate all tokens
        all_tokens = []
        for tokens in token_lists:
            # Convert to list if tensor
            if isinstance(tokens, torch.Tensor):
                all_tokens.extend(tokens.tolist())
            else:
                all_tokens.extend(tokens)
        
        # Get EVENT_LENGTH from first example
        EVENT_LENGTH = len(token_lists[0])
        
        # Truncate or pad to EVENT_LENGTH
        if len(all_tokens) > EVENT_LENGTH:
            merged = all_tokens[:EVENT_LENGTH]
        else:
            merged = all_tokens + [TOKEN_PAD] * (EVENT_LENGTH - len(all_tokens))
        
        return torch.LongTensor(merged)



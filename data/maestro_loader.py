from pathlib import Path
import torch
from torch.utils.data import IterableDataset
from data.utils import note_sequence_to_timed_events, timed_events_to_tokens
import librosa
import numpy as np
import warnings

# Suppress Xing header warnings from MP3 files
warnings.filterwarnings("ignore", message=".*Xing stream.*")

from itertools import cycle
import json
import os

import random
from data.constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_HOP_WIDTH,
    DEFAULT_NUM_MEL_BINS,
    FFT_SIZE,
    MEL_FMIN,
    MEL_FMAX,
    TOKEN_END,
    TOKEN_PAD,
    codec,
)
from data.spectrogram import MelSpectrogram
from config.data_config import data_config
import note_seq
from typing import Optional, Dict, Any
from data.lru_cache import LRUCache


_DEFAULT_RAW_CACHE_SIZE = 200


class MIDIDataset(IterableDataset):
    def __init__(
        self,
        root_dir=None,
        dataset_file="maestro-v3.0.0.json",
        split="train",
        cache_size: Optional[int] = _DEFAULT_RAW_CACHE_SIZE,
        preload_all: bool = False,
        num_workers: int = 1,
    ):
        """
        Initialize the MIDI dataset for MAESTRO v3.0.0.

        Args:
            root_dir: Root directory containing the MAESTRO dataset (defaults to data_config.maestro_root)
            dataset_file: JSON file containing dataset metadata
            split: Dataset split ('train', 'validation', 'test')
            cache_size: Number of tracks to keep hot in LRU cache (None to disable)
            preload_all: If True, load all tracks into memory during initialization
            num_workers: DataLoader worker count (for logging/estimation)
        """
        # Use config path if not provided
        self.root_dir = root_dir if root_dir is not None else data_config.maestro_root
        self.split = split
        self.type = split  # For compatibility with _preprocess method
        self.config = data_config  # Store config reference
        self.num_workers = max(1, num_workers)
        self.preload_all = bool(preload_all)
        self.cache_size = int(cache_size) if cache_size else None

        # Load dataset metadata (MAESTRO v3 uses columnar format)
        json_path = os.path.join(self.root_dir, dataset_file)
        with open(json_path, "r") as f:
            metadata = json.load(f)

        # MAESTRO v3 format: dict with keys like 'split', 'midi_filename', etc.
        # Each key maps to a dict with indices as keys
        # Filter entries by split
        self.midi_paths = []
        self.audio_paths = []
        
        num_entries = len(metadata['split'])
        for idx in range(num_entries):
            idx_str = str(idx)
            if metadata['split'][idx_str] == split:
                midi_filename = metadata['midi_filename'][idx_str]
                audio_filename = metadata['audio_filename'][idx_str]
                
                self.midi_paths.append(os.path.join(self.root_dir, midi_filename))
                self.audio_paths.append(os.path.join(self.root_dir, audio_filename))

        # Create MelSpectrogram once for efficiency (reused across all samples)
        self.melspectrogram = MelSpectrogram(
            DEFAULT_NUM_MEL_BINS,
            DEFAULT_SAMPLE_RATE,
            FFT_SIZE,
            DEFAULT_HOP_WIDTH,
            mel_fmin=MEL_FMIN,
            mel_fmax=MEL_FMAX,
        )

        self._track_indices = list(range(len(self.midi_paths)))
        self._track_cache: Optional[LRUCache] = None
        self._preloaded_tracks: Optional[list[Optional[Dict[str, Any]]]] = None
        if self.preload_all:
            self._preloaded_tracks = []
            for idx in self._track_indices:
                try:
                    self._preloaded_tracks.append(self._load_track_data(idx))
                except Exception as exc:
                    print(f"[MAESTRO][{self.split}] Failed to preload index {idx}: {exc}")
                    self._preloaded_tracks.append(None)
        elif self.cache_size:
            self._track_cache = LRUCache(maxsize=self.cache_size, loader=self._load_track_data)

    def _load_audio(self, audio_path, sample_rate):
        """Load and normalize audio to [-1, 1] range (legacy MT3 behavior)."""
        audio, sr = librosa.load(audio_path, sr=None)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=DEFAULT_SAMPLE_RATE)

        # Normalize to [-1, 1] by max absolute value (matching legacy MT3)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        return audio

    def _frame(self, signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
        signal_length = signal.shape[axis]
        if pad_end:
            frames_overlap = frame_length - frame_step
            rest_samples = np.abs(signal_length - frames_overlap) % np.abs(
                frame_length - frames_overlap
            )
            pad_size = int(frame_length - rest_samples)
            if pad_size != 0:
                pad_axis = pad_axis = tuple([0 for i in range(2)])
                signal = torch.nn.functional.pad(
                    torch.Tensor(signal), pad_axis, "constant", pad_value
                )
        frames = signal.unfold(axis, frame_length, frame_step)

        return frames

    def _audio_to_frames(self, samples):
        frame_size = DEFAULT_HOP_WIDTH
        samples = np.pad(samples, [0, frame_size - len(samples) % frame_size], mode="constant")
        frames = self._frame(
            samples, frame_length=DEFAULT_HOP_WIDTH, frame_step=DEFAULT_HOP_WIDTH, pad_end=True
        )
        num_frames = len(samples) // frame_size
        times = np.arange(num_frames) / (DEFAULT_SAMPLE_RATE / DEFAULT_HOP_WIDTH)

        return frames, times

    def _tokenize(self, midi_path):
        """Convert MIDI file to tokens using note_seq and codec."""
        # Load MIDI file
        ns = note_seq.midi_file_to_note_sequence(midi_path)

        # Convert to timed events
        events = note_sequence_to_timed_events(ns)

        # Convert events to tokens (we'll need frame_times, so return both ns and events)
        return ns, events

    def _load_track_data(self, idx: int) -> Dict[str, Any]:
        midi_path = str(self.midi_paths[idx])
        audio_path = str(self.audio_paths[idx])
        audio = self._load_audio(audio_path=audio_path, sample_rate=DEFAULT_SAMPLE_RATE)
        note_seq_obj, events = self._tokenize(str(midi_path))
        return {
            "audio": audio,
            "note_sequence": note_seq_obj,
            "events": events,
            "midi_path": midi_path,
            "audio_path": audio_path,
        }

    def _get_track(self, idx: int) -> Dict[str, Any]:
        if self.preload_all:
            if self._preloaded_tracks is None:
                raise RuntimeError("Preloaded tracks missing")
            data = self._preloaded_tracks[idx]
            if data is None:
                raise FileNotFoundError(f"Preloaded data missing for index {idx}")
            return data
        if self._track_cache is not None:
            return self._track_cache.get(idx)
        return self._load_track_data(idx)

    def warm_cache_for_workers(self):
        if self.preload_all or self._track_cache is None:
            return
        indices = self._track_indices[: self._track_cache.maxsize]
        print(f"[MAESTRO][{self.split}] Warming raw cache with {len(indices)} tracks before forking workers")
        for idx in indices:
            try:
                self._track_cache.get(idx)
            except Exception as exc:
                print(f"[MAESTRO][{self.split}] Warning: failed to warm track {idx}: {exc}")

    def _get_random_length_segment(self, row):
        new_row = {}
        input_length = row["inputs"].shape[0]
        sample_length = random.randint(self.config.mel_length, input_length)
        start_length = random.randint(0, input_length - self.config.mel_length)

        for k in row.keys():
            if k in ["inputs", "input_times"]:
                new_row[k] = row[k][start_length : start_length + sample_length]
            else:
                new_row[k] = row[k]

        return new_row

    def _slice_segment(self, row):
        """
        Split long audio into multiple segments of mel_length frames.
        Matches legacy MT3 behavior: creates MULTIPLE examples per file.
        
        Each segment is exactly mel_length (256) frames, padding the last one if needed.
        """
        rows = []
        input_length = row["inputs"].shape[0]
        segment_length = self.config.mel_length  # 256 frames
        
        # Calculate number of segments (including partial last segment)
        num_segments = (input_length + segment_length - 1) // segment_length
        
        for i in range(num_segments):
            split_start = i * segment_length
            split_end = min(split_start + segment_length, input_length)
            
            new_row = {}
            for k in row.keys():
                if k in ["inputs", "input_times"]:
                    segment = row[k][split_start:split_end]
                    
                    # Pad last segment if needed
                    if k == "inputs" and segment.shape[0] < segment_length:
                        pad_size = segment_length - segment.shape[0]
                        if isinstance(segment, torch.Tensor):
                            pad = torch.zeros(pad_size, segment.shape[1], dtype=segment.dtype)
                            segment = torch.cat([segment, pad], dim=0)
                        else:
                            pad = np.zeros((pad_size, segment.shape[1]), dtype=segment.dtype)
                            segment = np.concatenate([segment, pad], axis=0)
                    elif k == "input_times" and len(segment) < segment_length:
                        # Pad times by extending the last time value
                        pad_size = segment_length - len(segment)
                        if isinstance(segment, torch.Tensor):
                            last_time = segment[-1] if len(segment) > 0 else 0
                            pad = torch.full((pad_size,), last_time, dtype=segment.dtype)
                            segment = torch.cat([segment, pad])
                        else:
                            last_time = segment[-1] if len(segment) > 0 else 0
                            pad = np.full(pad_size, last_time, dtype=segment.dtype)
                            segment = np.concatenate([segment, pad])
                    
                    new_row[k] = segment
                else:
                    new_row[k] = row[k]
            rows.append(new_row)

        if len(rows) == 0:
            return [row]
        return rows

    def _extract_target_sequence_with_indices(self, row):
        """Extract target sequence corresponding to audio token segment."""
        # row['targets'] is now (note_sequence, events) tuple from _tokenize
        note_sequence, events = row["targets"]

        # Convert events to tokens using frame times
        tokens, event_start_indices, event_end_indices = timed_events_to_tokens(
            events, codec, row["input_times"]
        )

        # Update row with tokens
        row["targets"] = tokens

        return row

    def _target_to_int(self, row):
        # Tokens are already integers from timed_events_to_tokens
        # Just ensure they are in the right format
        if not isinstance(row["targets"], list):
            row["targets"] = list(row["targets"])
        return row

    def _compute_spectrogram(self, ex):
        samples = torch.flatten(ex["inputs"])
        # Use pre-initialized MelSpectrogram (avoids recreation overhead)
        ex["inputs"] = (
            self.melspectrogram(samples.reshape(-1, samples.shape[-1])[:, :-1])
            .transpose(-1, -2)
            .squeeze(0)
        )

        return ex

    def _pad_length(self, row):
        inputs = row["inputs"]
        end = row["end"]
        targets = (
            torch.from_numpy(np.array(row["targets"][: self.config.event_length]))
            .to(torch.long)
            .to("cpu")
        )

        if inputs.shape[0] < self.config.mel_length:
            pad = torch.zeros(
                self.config.mel_length - inputs.shape[0],
                inputs.shape[1],
                dtype=inputs.dtype,
                device=inputs.device,
            )
            inputs = torch.cat([inputs, pad], dim=0)

        if targets.shape[0] < self.config.event_length:
            eos_value = TOKEN_END if TOKEN_END is not None else 1
            pad_value = TOKEN_PAD if TOKEN_PAD is not None else 0

            eos = torch.ones(1, dtype=targets.dtype, device=targets.device) * eos_value
            if self.config.event_length - targets.shape[0] - 1 > 0:
                pad = (
                    torch.ones(
                        self.config.event_length - targets.shape[0] - 1,
                        dtype=targets.dtype,
                        device=targets.device,
                    )
                    * pad_value
                )
                targets = torch.cat([targets, eos, pad], dim=0)
            else:
                targets = torch.cat([targets, eos], dim=0)

        result = {"inputs": inputs, "targets": targets}

        # Pass through metadata if available
        if "source_file" in row:
            result["source_file"] = row["source_file"]
        if "segment_idx" in row:
            result["segment_idx"] = row["segment_idx"]
        if self.type != "train" and "end" in row:
            result["end"] = row["end"]

        return result

    def _preprocess(self):
        for idx in range(len(self.midi_paths)):
            try:
                track = self._get_track(idx)
            except Exception as exc:
                print(f"[MAESTRO][{self.split}] Warning: skipping track {idx}: {exc}")
                continue

            audio = track["audio"]
            frames, frame_times = self._audio_to_frames(audio)
            encoded_midi = (track["note_sequence"], track["events"])
            midi_path = track["midi_path"]

            row = {
                "inputs": frames,
                "input_times": frame_times,
                "targets": encoded_midi,
                "end": False,
                "source_file": midi_path,  # Add source file path
            }
            if self.type == "train":
                row = self._get_random_length_segment(row)
            rows = self._slice_segment(row)

            for i, row in enumerate(rows):
                # Mark last segment with end flag
                if i == len(rows) - 1:
                    row["end"] = True
                row["segment_idx"] = i  # Add segment index
                row = self._extract_target_sequence_with_indices(row)
                row = self._target_to_int(row)
                row = self._compute_spectrogram(row)
                row = self._pad_length(row)

                yield row

    def __len__(self):
        return len(self.midi_paths)

    def __iter__(self):
        if self.type == "train":
            return cycle(self._preprocess())
        else:
            return self._preprocess()

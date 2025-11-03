from typing import Any, Dict, List, Union

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from . import vocabularies
from . import utils
from .spectrogram import MelSpectrogram
from config.data_config import data_config
import note_seq

import tfrecord.torch.dataset as tf_dataset


class TranscriptionDataset(Dataset):
    """Dataset for transcription tasks."""

    def __init__(
        self,
        data_path,
        spectrogram_config,
        codec,
        is_training,
        onsets_only,
        include_ties,
        segment_length=2048,
        mixing_rate=0.5,
    ):
        self.data_path = data_path
        self.spectrogram_config = spectrogram_config
        self.codec = codec
        self.is_training = is_training
        self.onsets_only = onsets_only
        self.include_ties = include_ties
        self.segment_length = segment_length
        self.mixing_rate = mixing_rate

        self.examples = self._load_tfrecord_data()

    def _mix_examples(self, example1, example2):
        """Mixes two examples."""
        # Mix audio
        audio1 = example1["audio"]
        audio2 = example2["audio"]
        min_len = min(len(audio1), len(audio2))
        mixed_audio = audio1[:min_len] + audio2[:min_len]

        # Mix NoteSequences
        ns1 = note_seq.NoteSequence.FromString(example1["sequence"])
        ns2 = note_seq.NoteSequence.FromString(example2["sequence"])
        mixed_ns = note_seq.concatenate_sequences([ns1, ns2])

        return {
            "sequence": mixed_ns.SerializeToString(),
            "audio": mixed_audio,
            "sample_rate": example1["sample_rate"],
        }

    def _load_tfrecord_data(self):
        """Loads data from TFRecord files."""
        dataset = tf_dataset.TFRecordDataset(
            self.data_path,
            index_path=None,
            description={"sequence": "byte", "audio": "byte", "sample_rate": "int"},
        )
        return list(dataset)

    def __len__(self):
        return len(self.examples)

    def _load_dummy_data(self):
        """Loads dummy data for demonstration purposes."""
        # Create a dummy multi-instrument NoteSequence
        ns = note_seq.NoteSequence()
        ns.notes.add(start_time=0.1, end_time=0.2, pitch=60, velocity=100, program=0)
        ns.notes.add(start_time=0.3, end_time=0.4, pitch=62, velocity=100, program=0)
        ns.notes.add(start_time=0.1, end_time=0.2, pitch=72, velocity=80, program=1)
        ns.notes.add(start_time=0.3, end_time=0.4, pitch=74, velocity=80, program=1)
        ns.total_time = 0.4

        # Create dummy audio
        audio = np.sin(np.linspace(0, 440 * 2 * np.pi, 16000 * 1))

        return [
            {"sequence": ns.SerializeToString(), "audio": audio, "sample_rate": 16000}
        ] * 10  # Repeat the example for a larger dataset

    def __getitem__(self, idx):
        example = self.examples[idx]

        if self.is_training and np.random.rand() < self.mixing_rate:
            other_idx = np.random.randint(0, len(self.examples))
            other_example = self.examples[other_idx]
            example = self._mix_examples(example, other_example)

        # Decode audio from bytes
        audio = np.frombuffer(example["audio"], dtype=np.float32)

        # Extract frames from audio
        frame_size = self.spectrogram_config.hop_width
        audio = np.pad(audio, [0, frame_size - len(audio) % frame_size], mode="constant")
        num_frames = len(audio) // frame_size
        frame_times = np.arange(num_frames) / (
            self.spectrogram_config.sample_rate / self.spectrogram_config.hop_width
        )

        # Reshape audio into frames for spectrogram computation
        frames = audio.reshape(-1, frame_size)

        # Tokenize the NoteSequence
        ns = note_seq.NoteSequence.FromString(example["sequence"])

        # Extract events for each instrument
        instrument_events = []
        for instrument in set(note.instrument for note in ns.notes):
            instrument_ns = note_seq.NoteSequence()
            instrument_ns.notes.extend([n for n in ns.notes if n.instrument == instrument])
            instrument_events.append(utils.note_sequence_to_timed_events(instrument_ns))

        # Merge and tokenize
        merged_events = utils.merge_events(instrument_events)
        tokens, event_start_indices, event_end_indices = utils.timed_events_to_tokens(
            merged_events, self.codec, frame_times
        )

        # Compute spectrogram
        mel_spectrogram = MelSpectrogram(
            self.spectrogram_config.num_mel_bins,
            self.spectrogram_config.sample_rate,
            self.spectrogram_config.fft_size,
            self.spectrogram_config.hop_width,
            mel_fmin=self.spectrogram_config.mel_fmin,
            mel_fmax=self.spectrogram_config.mel_fmax,
        )
        spectrogram = mel_spectrogram(torch.from_numpy(frames).float()).numpy()

        if self.is_training:
            # Random chunking
            if len(spectrogram) > self.segment_length:
                start_frame = np.random.randint(0, len(spectrogram) - self.segment_length)
                end_frame = start_frame + self.segment_length
                spectrogram = spectrogram[start_frame:end_frame]

                start_token_idx = event_start_indices[start_frame]
                end_token_idx = event_end_indices[end_frame - 1]
                tokens = tokens[start_token_idx:end_token_idx]
            return {"inputs": spectrogram, "targets": np.array(tokens, dtype=np.int32)}
        else:
            # Splitting into non-overlapping segments
            segments = []
            for start_frame in range(0, len(spectrogram), self.segment_length):
                end_frame = start_frame + self.segment_length
                if end_frame > len(spectrogram):
                    continue

                segment_spectrogram = spectrogram[start_frame:end_frame]

                start_token_idx = event_start_indices[start_frame]
                end_token_idx = event_end_indices[end_frame - 1]
                segment_tokens = tokens[start_token_idx:end_token_idx]

                segments.append(
                    {
                        "inputs": segment_spectrogram,
                        "targets": np.array(segment_tokens, dtype=np.int32),
                    }
                )
            return segments


def collate_fn(batch):
    """Custom collate function for the DataLoader."""
    # If the batch is a list of lists (from evaluation), flatten it
    if isinstance(batch[0], list):
        batch = [item for sublist in batch for item in sublist]

    inputs = [item["inputs"] for item in batch]
    targets = [item["targets"] for item in batch]

    # Pad inputs and targets to the max length in the batch
    max_input_len = max(len(x) for x in inputs)
    max_target_len = max(len(x) for x in targets)

    padded_inputs = np.zeros((len(inputs), max_input_len, inputs[0].shape[-1]))
    padded_targets = np.zeros((len(targets), max_target_len), dtype=np.int32)

    for i, (x, y) in enumerate(zip(inputs, targets)):
        padded_inputs[i, : len(x)] = x
        padded_targets[i, : len(y)] = y

    result: Dict[str, Union[torch.Tensor, List[Any]]] = {
        "inputs": torch.from_numpy(padded_inputs).float(),
        "targets": torch.from_numpy(padded_targets).long(),
    }

    # Pass through metadata fields if available (check all samples have the field)
    if batch and "source_file" in batch[0]:
        if all("source_file" in item for item in batch):
            result["source_file"] = [item["source_file"] for item in batch]
    if batch and "segment_idx" in batch[0]:
        if all("segment_idx" in item for item in batch):
            result["segment_idx"] = [item["segment_idx"] for item in batch]

    return result

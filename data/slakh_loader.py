"""
SLAKH2100 Dataset Loader for MT3-PyTorch.
Loads multi-track audio and MIDI from SLAKH2100 dataset.
Supports both individual stems and full mixes.
"""

from pathlib import Path
import torch
from torch.utils.data import IterableDataset, Dataset
from data.utils import note_sequence_to_timed_events, timed_events_to_tokens
import librosa
import numpy as np
import yaml
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


def parse_slakh_metadata(track_dir):
    """Parse metadata.yaml to get stem info.

    Args:
        track_dir: Path to track directory

    Returns:
        dict: Stem information with keys like 'S00', 'S01', etc.
    """
    metadata_path = os.path.join(track_dir, "metadata.yaml")
    if not os.path.exists(metadata_path):
        return {}

    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)

    stems = {}
    if "stems" in metadata:
        for stem_id, info in metadata["stems"].items():
            stems[stem_id] = {
                "program": info.get("program_num", 0),
                "is_drum": info.get("is_drum", False),
                "name": info.get("inst_class", "Unknown"),
                "midi_program_name": info.get("midi_program_name", "Unknown"),
            }
    return stems


def load_stem(track_dir, stem_id, stem_info):
    """Load single stem audio + MIDI.

    Args:
        track_dir: Path to track directory
        stem_id: Stem identifier (e.g., 'S00')
        stem_info: Stem metadata dict

    Returns:
        tuple: (audio_array, note_sequence) or None if loading fails
    """
    # Find audio file - try both .flac and .wav
    audio_path = os.path.join(track_dir, "stems", f"{stem_id}.flac")
    if not os.path.exists(audio_path):
        audio_path = os.path.join(track_dir, "stems", f"{stem_id}.wav")

    # If neither file exists, this stem is not present in this track (variable stems)
    if not os.path.exists(audio_path):
        return None

    try:
        audio, sr = librosa.load(audio_path, sr=DEFAULT_SAMPLE_RATE)
    except Exception as e:
        # Only print warning for unexpected errors (not missing files)
        if "No such file" not in str(e):
            print(f"Warning: Could not load audio from {audio_path}: {e}")
        return None

    # Load MIDI
    midi_path = os.path.join(track_dir, "MIDI", f"{stem_id}.mid")
    if not os.path.exists(midi_path):
        # Create empty note sequence if MIDI doesn't exist
        ns = note_seq.NoteSequence()
        ns.total_time = len(audio) / DEFAULT_SAMPLE_RATE
        return audio, ns

    ns = note_seq.midi_file_to_note_sequence(midi_path)

    # Set program number on all notes (clamp to valid MIDI range 0-127)
    program = min(127, max(0, stem_info["program"]))
    for note in ns.notes:
        note.program = program
        note.is_drum = stem_info["is_drum"]

    return audio, ns


def merge_note_sequences(note_seqs):
    """Merge multiple NoteSequences (preserves program numbers).

    Args:
        note_seqs: List of NoteSequence objects

    Returns:
        NoteSequence: Merged sequence
    """
    merged = note_seq.NoteSequence()

    for ns in note_seqs:
        for note in ns.notes:
            new_note = merged.notes.add()
            new_note.CopyFrom(note)  # Keeps note.program intact

    # Sort by start time
    merged.notes.sort(key=lambda n: n.start_time)

    # Update total time
    if merged.notes:
        merged.total_time = max(note.end_time for note in merged.notes)

    return merged


def mix_audio_stems(audio_list, weights=None):
    """Mix multiple audio arrays.

    Args:
        audio_list: List of audio arrays
        weights: Optional list of mixing weights

    Returns:
        np.ndarray: Mixed audio
    """
    if not audio_list:
        return np.array([], dtype=np.float32)

    if weights is None:
        weights = [1.0] * len(audio_list)

    # Ensure same length
    max_len = max(len(a) for a in audio_list)
    padded = [np.pad(a, (0, max_len - len(a))) for a in audio_list]

    # Mix with weights
    mixed = sum(w * a for w, a in zip(weights, padded))

    # Normalize
    max_val = np.abs(mixed).max()
    if max_val > 0:
        mixed = mixed / max_val * 0.9

    return np.asarray(mixed)


class SLAKHDataset(IterableDataset):
    """
    SLAKH2100 Dataset loader for multi-track music transcription.

    Loads audio (mix.wav) and MIDI (all_src.mid or individual stems) from SLAKH2100.
    """

    def __init__(self, root_dir=None, split="train", use_stems=False, max_tracks=None):
        """
        Initialize the SLAKH dataset.

        Args:
            root_dir: Root directory for SLAKH split (defaults to data_config.slakh_train_path)
            split: Dataset split ('train', 'validation', 'test')
            use_stems: If True, use individual stem MIDI files; if False, use all_src.mid
            max_tracks: Maximum number of tracks to load (None = all tracks)
        """
        # Use config path if not provided
        if root_dir is None:
            if split == "train":
                self.root_dir = data_config.slakh_train_path
            elif split == "validation":
                self.root_dir = data_config.slakh_valid_path
            elif split == "test":
                self.root_dir = data_config.slakh_test_path
            else:
                raise ValueError(f"Unknown split: {split}")
        else:
            self.root_dir = root_dir

        self.split = split
        self.type = split  # For compatibility
        self.use_stems = use_stems
        self.config = data_config

        # Find all track directories
        track_dirs = sorted(Path(self.root_dir).glob("Track*"))

        if max_tracks is not None:
            track_dirs = track_dirs[:max_tracks]

        self.track_paths = [str(d) for d in track_dirs]

        if len(self.track_paths) == 0:
            raise ValueError(f"No tracks found in {self.root_dir}")

    def _load_audio(self, audio_path, sample_rate):
        """Load and resample audio file."""
        audio, sr = librosa.load(audio_path, sr=None)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        return audio

    def _frame(self, signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
        """Convert audio signal to frames."""
        signal_length = signal.shape[axis]
        if pad_end:
            frames_overlap = frame_length - frame_step
            rest_samples = np.abs(signal_length - frames_overlap) % np.abs(
                frame_length - frames_overlap
            )
            pad_size = int(frame_length - rest_samples)
            if pad_size != 0:
                pad_axis = tuple([0 for i in range(2)])
                signal = torch.nn.functional.pad(
                    torch.Tensor(signal), pad_axis, "constant", pad_value
                )
        frames = signal.unfold(axis, frame_length, frame_step)
        return frames

    def _audio_to_frames(self, samples):
        """Convert audio samples to frames with timestamps."""
        frame_size = DEFAULT_HOP_WIDTH
        samples = np.pad(samples, [0, frame_size - len(samples) % frame_size], mode="constant")
        frames = self._frame(
            samples, frame_length=DEFAULT_HOP_WIDTH, frame_step=DEFAULT_HOP_WIDTH, pad_end=True
        )
        num_frames = len(samples) // frame_size
        times = np.arange(num_frames) / (DEFAULT_SAMPLE_RATE / DEFAULT_HOP_WIDTH)
        return frames, times

    def _load_note_sequence(self, midi_path):
        """Load MIDI file and convert to NoteSequence with events."""
        # Load MIDI using note_seq
        ns = note_seq.midi_file_to_note_sequence(midi_path)

        # Convert to timed events
        events = note_sequence_to_timed_events(ns)

        # Return both note sequence and events (tokens will be created later with frame_times)
        return ns, events

    def _get_random_length_segment(self, row):
        """Extract random length segment from data (for training)."""
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
        """Slice data into fixed-length segments."""
        rows = []
        input_length = row["inputs"].shape[0]
        for split in range(0, input_length, self.config.mel_length):
            if split + self.config.mel_length >= input_length:
                continue
            new_row = {}
            for k in row.keys():
                if k in ["inputs", "input_times"]:
                    new_row[k] = row[k][split : split + self.config.mel_length]
                else:
                    new_row[k] = row[k]
            rows.append(new_row)

        if len(rows) == 0:
            return [row]
        return rows

    def _extract_target_sequence_with_indices(self, row):
        """Extract target sequence corresponding to audio token segment."""
        # row['targets'] is now (note_sequence, events) tuple from _load_note_sequence
        note_sequence, events = row["targets"]

        # Convert events to tokens using frame times
        tokens, event_start_indices, event_end_indices = timed_events_to_tokens(
            events, codec, row["input_times"]
        )

        # Update row with tokens
        row["targets"] = tokens

        return row

    def _target_to_int(self, row):
        """Ensure targets are in the right format."""
        # Tokens are already integers from timed_events_to_tokens
        # Just ensure they are in the right format
        if not isinstance(row["targets"], list):
            row["targets"] = list(row["targets"])
        return row

    def _compute_spectrogram(self, ex):
        """Compute mel spectrogram from audio frames."""
        samples = torch.flatten(ex["inputs"])
        melspectrogram = MelSpectrogram(
            DEFAULT_NUM_MEL_BINS,
            DEFAULT_SAMPLE_RATE,
            FFT_SIZE,
            DEFAULT_HOP_WIDTH,
            mel_fmin=MEL_FMIN,
            mel_fmax=MEL_FMAX,
        )

        ex["inputs"] = (
            melspectrogram(samples.reshape(-1, samples.shape[-1])[:, :-1])
            .transpose(-1, -2)
            .squeeze(0)
        )

        return ex

    def _pad_length(self, row):
        """Pad inputs and targets to fixed lengths."""
        inputs = row["inputs"]
        end = row["end"]
        targets = (
            torch.from_numpy(np.array(row["targets"][: self.config.event_length]))
            .to(torch.long)
            .to("cpu")
        )

        # Pad inputs (mel spectrogram)
        if inputs.shape[0] < self.config.mel_length:
            pad = torch.zeros(
                self.config.mel_length - inputs.shape[0],
                inputs.shape[1],
                dtype=inputs.dtype,
                device=inputs.device,
            )
            inputs = torch.cat([inputs, pad], dim=0)

        # Pad targets (token sequence)
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

        if self.type == "train":
            return {"inputs": inputs, "targets": targets}
        else:
            return {"inputs": inputs, "targets": targets, "end": end}

    def _preprocess(self):
        """Preprocess and yield data samples."""
        for track_path in self.track_paths:
            try:
                # Load audio (mix)
                audio_path = os.path.join(track_path, "mix.wav")
                if not os.path.exists(audio_path):
                    # Try .flac extension
                    audio_path = os.path.join(track_path, "mix.flac")
                    if not os.path.exists(audio_path):
                        print(f"Warning: No audio file found for {track_path}")
                        continue

                audio = self._load_audio(audio_path, DEFAULT_SAMPLE_RATE)
                frames, frame_times = self._audio_to_frames(audio)

                # Load MIDI
                if self.use_stems:
                    # TODO: Implement stem-based loading for multi-track
                    # For now, use all_src.mid
                    midi_path = os.path.join(track_path, "all_src.mid")
                else:
                    midi_path = os.path.join(track_path, "all_src.mid")

                if not os.path.exists(midi_path):
                    print(f"Warning: MIDI file not found: {midi_path}")
                    continue

                encoded_midi = self._load_note_sequence(midi_path)

                # Create data row
                row = {
                    "inputs": frames,
                    "input_times": frame_times,
                    "targets": encoded_midi,
                    "end": False,
                }

                # Apply random segment for training
                if self.type == "train":
                    row = self._get_random_length_segment(row)

                # Slice into segments
                rows = self._slice_segment(row)

                # Process each segment
                for i, row in enumerate(rows):
                    if i == len(rows) - 1:
                        row["end"] = True

                    row = self._extract_target_sequence_with_indices(row)
                    row = self._target_to_int(row)
                    row = self._compute_spectrogram(row)
                    row = self._pad_length(row)

                    yield row

            except Exception as e:
                print(f"Error processing track {track_path}: {e}")
                continue

    def __len__(self):
        """Return number of tracks."""
        return len(self.track_paths)

    def __iter__(self):
        """Iterate over preprocessed samples."""
        return self._preprocess()


class SLAKHStemDataset(Dataset):
    """Load individual SLAKH stems (one instrument per sample)."""

    def __init__(self, root_dir=None, split="train", max_tracks=None):
        """
        Initialize the SLAKH stem dataset.

        Args:
            root_dir: Root directory for SLAKH split
            split: Dataset split ('train', 'validation', 'test')
            max_tracks: Maximum number of tracks to load
        """
        # Use config path if not provided
        if root_dir is None:
            if split == "train":
                self.root_dir = data_config.slakh_train_path
            elif split == "validation":
                self.root_dir = data_config.slakh_valid_path
            elif split == "test":
                self.root_dir = data_config.slakh_test_path
            else:
                raise ValueError(f"Unknown split: {split}")
        else:
            self.root_dir = root_dir

        self.split = split
        self.config = data_config

        # Create MelSpectrogram object once (reused for all samples)
        self.melspectrogram = MelSpectrogram(
            DEFAULT_NUM_MEL_BINS,
            DEFAULT_SAMPLE_RATE,
            FFT_SIZE,
            DEFAULT_HOP_WIDTH,
            mel_fmin=MEL_FMIN,
            mel_fmax=MEL_FMAX,
        )

        # Find all track directories
        track_dirs = sorted(Path(self.root_dir).glob("Track*"))

        if max_tracks is not None:
            track_dirs = track_dirs[:max_tracks]

        self.track_paths = [str(d) for d in track_dirs]
        self.stem_index = self._build_stem_index()  # List of (track, stem_id)

    def _build_stem_index(self):
        """Create flat list of all stems across all tracks."""
        index = []
        for track_dir in self.track_paths:
            stems = parse_slakh_metadata(track_dir)
            for stem_id in stems.keys():
                index.append((track_dir, stem_id))
        return index

    def __getitem__(self, idx):
        """Get a single stem sample."""
        track_dir, stem_id = self.stem_index[idx]
        stems = parse_slakh_metadata(track_dir)

        # Load stem
        result = load_stem(track_dir, stem_id, stems[stem_id])
        if result is None:
            # Skip to next valid sample if this one fails
            return self.__getitem__((idx + 1) % len(self))

        audio, note_seq = result

        # Convert audio to mel spectrogram
        mel = self._compute_mel_spectrogram(audio)

        # Convert note sequence to tokens
        events = note_sequence_to_timed_events(note_seq)

        # Create frame times for tokenization
        num_frames = mel.shape[0]
        frame_times = np.arange(num_frames) / (DEFAULT_SAMPLE_RATE / DEFAULT_HOP_WIDTH)

        tokens, _, _ = timed_events_to_tokens(events, codec, frame_times)

        # Pad to fixed lengths
        mel_padded = self._pad_mel(mel)
        tokens_padded = self._pad_tokens(tokens)

        return {"inputs": mel_padded, "targets": tokens_padded}

    def _compute_mel_spectrogram(self, audio):
        """Compute mel spectrogram from audio."""
        # Convert to frames
        frame_size = DEFAULT_HOP_WIDTH
        samples = np.pad(audio, [0, frame_size - len(audio) % frame_size], mode="constant")

        # Use pre-initialized MelSpectrogram (avoids recreation overhead)
        samples_tensor = torch.from_numpy(samples).float()
        mel = self.melspectrogram(samples_tensor.unsqueeze(0)).squeeze(0).transpose(0, 1)

        return mel.numpy()

    def _pad_mel(self, mel):
        """Pad mel spectrogram to fixed length."""
        if mel.shape[0] < self.config.mel_length:
            pad_length = self.config.mel_length - mel.shape[0]
            mel = np.pad(mel, ((0, pad_length), (0, 0)), mode="constant")
        else:
            mel = mel[: self.config.mel_length]

        return torch.from_numpy(mel).float()

    def _pad_tokens(self, tokens):
        """Pad token sequence to fixed length."""
        if len(tokens) < self.config.event_length:
            tokens = tokens + [TOKEN_PAD] * (self.config.event_length - len(tokens))
        else:
            tokens = tokens[: self.config.event_length]

        return torch.LongTensor(tokens)

    def __len__(self):
        """Return number of stems."""
        return len(self.stem_index)


class SLAKHMixDataset(Dataset):
    """Load full SLAKH mixes (all instruments per sample)."""

    def __init__(self, root_dir=None, split="train", max_tracks=None):
        """
        Initialize the SLAKH mix dataset.

        Args:
            root_dir: Root directory for SLAKH split
            split: Dataset split ('train', 'validation', 'test')
            max_tracks: Maximum number of tracks to load
        """
        # Use config path if not provided
        if root_dir is None:
            if split == "train":
                self.root_dir = data_config.slakh_train_path
            elif split == "validation":
                self.root_dir = data_config.slakh_valid_path
            elif split == "test":
                self.root_dir = data_config.slakh_test_path
            else:
                raise ValueError(f"Unknown split: {split}")
        else:
            self.root_dir = root_dir

        self.split = split
        self.config = data_config

        # Find all track directories
        track_dirs = sorted(Path(self.root_dir).glob("Track*"))

        if max_tracks is not None:
            track_dirs = track_dirs[:max_tracks]

        self.track_paths = [str(d) for d in track_dirs]

        # Create MelSpectrogram once for efficiency (reused across all samples)
        self.melspectrogram = MelSpectrogram(
            DEFAULT_NUM_MEL_BINS,
            DEFAULT_SAMPLE_RATE,
            FFT_SIZE,
            DEFAULT_HOP_WIDTH,
            mel_fmin=MEL_FMIN,
            mel_fmax=MEL_FMAX,
        )

    def __getitem__(self, idx):
        """Get a full mix sample."""
        track_dir = self.track_paths[idx]
        stems = parse_slakh_metadata(track_dir)

        # Load all stems
        note_seqs = []
        for stem_id, stem_info in stems.items():
            try:
                result = load_stem(track_dir, stem_id, stem_info)
                if result is None:
                    continue
                _, note_seq = result
                note_seqs.append(note_seq)
            except Exception as e:
                print(f"Warning: Could not load stem {stem_id} from {track_dir}: {e}")
                continue

        # Load mixed audio
        mix_path = os.path.join(track_dir, "mix.flac")
        if not os.path.exists(mix_path):
            mix_path = os.path.join(track_dir, "mix.wav")

        mix_audio, _ = librosa.load(mix_path, sr=DEFAULT_SAMPLE_RATE)

        # Merge all MIDI with program numbers
        merged_ns = merge_note_sequences(note_seqs)

        # Convert audio to mel spectrogram
        mel = self._compute_mel_spectrogram(mix_audio)

        # Convert note sequence to tokens (with program changes)
        events = note_sequence_to_timed_events(merged_ns)

        # Create frame times for tokenization
        num_frames = mel.shape[0]
        frame_times = np.arange(num_frames) / (DEFAULT_SAMPLE_RATE / DEFAULT_HOP_WIDTH)

        tokens, _, _ = timed_events_to_tokens(events, codec, frame_times)

        # Pad to fixed lengths
        mel_padded = self._pad_mel(mel)
        tokens_padded = self._pad_tokens(tokens)

        return {"inputs": mel_padded, "targets": tokens_padded}

    def _compute_mel_spectrogram(self, audio):
        """Compute mel spectrogram from audio."""
        # Convert to frames
        frame_size = DEFAULT_HOP_WIDTH
        samples = np.pad(audio, [0, frame_size - len(audio) % frame_size], mode="constant")

        # Use pre-initialized MelSpectrogram (avoids recreation overhead)
        samples_tensor = torch.from_numpy(samples).float()
        mel = self.melspectrogram(samples_tensor.unsqueeze(0)).squeeze(0).transpose(0, 1)

        return mel.numpy()

    def _pad_mel(self, mel):
        """Pad mel spectrogram to fixed length."""
        if mel.shape[0] < self.config.mel_length:
            pad_length = self.config.mel_length - mel.shape[0]
            mel = np.pad(mel, ((0, pad_length), (0, 0)), mode="constant")
        else:
            mel = mel[: self.config.mel_length]

        return torch.from_numpy(mel).float()

    def _pad_tokens(self, tokens):
        """Pad token sequence to fixed length."""
        if len(tokens) < self.config.event_length:
            tokens = tokens + [TOKEN_PAD] * (self.config.event_length - len(tokens))
        else:
            tokens = tokens[: self.config.event_length]

        return torch.LongTensor(tokens)

    def __len__(self):
        """Return number of tracks."""
        return len(self.track_paths)


class SLAKHMixedDataset(Dataset):
    """Randomly mix 1-4 stems during training."""

    def __init__(self, root_dir=None, split="train", max_tracks=None, min_stems=1, max_stems=4):
        """
        Initialize the SLAKH mixed dataset.

        Args:
            root_dir: Root directory for SLAKH split
            split: Dataset split ('train', 'validation', 'test')
            max_tracks: Maximum number of tracks to load
            min_stems: Minimum number of stems to mix
            max_stems: Maximum number of stems to mix
        """
        # Use config path if not provided
        if root_dir is None:
            if split == "train":
                self.root_dir = data_config.slakh_train_path
            elif split == "validation":
                self.root_dir = data_config.slakh_valid_path
            elif split == "test":
                self.root_dir = data_config.slakh_test_path
            else:
                raise ValueError(f"Unknown split: {split}")
        else:
            self.root_dir = root_dir

        self.split = split
        self.min_stems = min_stems
        self.max_stems = max_stems
        self.config = data_config

        # Find all track directories
        track_dirs = sorted(Path(self.root_dir).glob("Track*"))

        if max_tracks is not None:
            track_dirs = track_dirs[:max_tracks]

        self.track_paths = [str(d) for d in track_dirs]

        # Create MelSpectrogram once for efficiency (reused across all samples)
        self.melspectrogram = MelSpectrogram(
            DEFAULT_NUM_MEL_BINS,
            DEFAULT_SAMPLE_RATE,
            FFT_SIZE,
            DEFAULT_HOP_WIDTH,
            mel_fmin=MEL_FMIN,
            mel_fmax=MEL_FMAX,
        )

    def __getitem__(self, idx):
        """Get a randomly mixed sample."""
        track_dir = self.track_paths[idx]
        stems = parse_slakh_metadata(track_dir)

        # Randomly select N stems
        num_stems = random.randint(self.min_stems, min(self.max_stems, len(stems)))
        selected_stems = random.sample(list(stems.items()), num_stems)

        # Load selected stems
        audios = []
        note_seqs = []
        for stem_id, stem_info in selected_stems:
            try:
                result = load_stem(track_dir, stem_id, stem_info)
                if result is None:
                    continue
                audio, note_seq = result
                audios.append(audio)
                note_seqs.append(note_seq)
            except Exception as e:
                print(f"Warning: Could not load stem {stem_id} from {track_dir}: {e}")
                continue

        if not audios:
            # Fallback to mix.flac if no stems loaded
            mix_path = os.path.join(track_dir, "mix.flac")
            if not os.path.exists(mix_path):
                mix_path = os.path.join(track_dir, "mix.wav")
            mixed_audio, _ = librosa.load(mix_path, sr=DEFAULT_SAMPLE_RATE)
        else:
            # Mix audio
            mixed_audio = mix_audio_stems(audios)

        # Merge MIDI
        if note_seqs:
            merged_ns = merge_note_sequences(note_seqs)
        else:
            # Create empty note sequence using merge function
            merged_ns = merge_note_sequences([])

        # Convert audio to mel spectrogram
        mel = self._compute_mel_spectrogram(mixed_audio)

        # Convert note sequence to tokens
        events = note_sequence_to_timed_events(merged_ns)

        # Create frame times for tokenization
        num_frames = mel.shape[0]
        frame_times = np.arange(num_frames) / (DEFAULT_SAMPLE_RATE / DEFAULT_HOP_WIDTH)

        tokens, _, _ = timed_events_to_tokens(events, codec, frame_times)

        # Pad to fixed lengths
        mel_padded = self._pad_mel(mel)
        tokens_padded = self._pad_tokens(tokens)

        return {"inputs": mel_padded, "targets": tokens_padded}

    def _compute_mel_spectrogram(self, audio):
        """Compute mel spectrogram from audio."""
        # Convert to frames
        frame_size = DEFAULT_HOP_WIDTH
        samples = np.pad(audio, [0, frame_size - len(audio) % frame_size], mode="constant")

        # Use pre-initialized MelSpectrogram (avoids recreation overhead)
        samples_tensor = torch.from_numpy(samples).float()
        mel = self.melspectrogram(samples_tensor.unsqueeze(0)).squeeze(0).transpose(0, 1)

        return mel.numpy()

    def _pad_mel(self, mel):
        """Pad mel spectrogram to fixed length."""
        if mel.shape[0] < self.config.mel_length:
            pad_length = self.config.mel_length - mel.shape[0]
            mel = np.pad(mel, ((0, pad_length), (0, 0)), mode="constant")
        else:
            mel = mel[: self.config.mel_length]

        return torch.from_numpy(mel).float()

    def _pad_tokens(self, tokens):
        """Pad token sequence to fixed length."""
        if len(tokens) < self.config.event_length:
            tokens = tokens + [TOKEN_PAD] * (self.config.event_length - len(tokens))
        else:
            tokens = tokens[: self.config.event_length]

        return torch.LongTensor(tokens)

    def __len__(self):
        """Return number of tracks."""
        return len(self.track_paths)

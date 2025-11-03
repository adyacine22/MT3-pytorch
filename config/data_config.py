"""
Data configuration for MT3-pytorch.
Contains dataset paths and audio processing parameters.
"""

import os
from pathlib import Path


class SpectrogramConfig:
    """Audio spectrogram configuration."""

    def __init__(
        self,
        sample_rate=16000,
        hop_width=128,
        num_mel_bins=512,
        fft_size=2048,
        mel_fmin=20.0,
        mel_fmax=8000.0,  # Set to Nyquist frequency (sample_rate / 2)
    ):
        self.sample_rate = sample_rate
        self.hop_width = hop_width
        self.num_mel_bins = num_mel_bins
        self.fft_size = fft_size
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax


class DataConfig:
    """Main data configuration class."""

    def __init__(
        self,
        project_root=None,
        datasets_dir="datasets",
        slakh_dir_name="slakh2100_flac_redux",
        maestro_dir_name="maestro-v3.0.0",
        maestro_version="v3.0.0",
        checkpoint_dir="cpt",
    ):
        """
        Initialize data configuration.

        Args:
            project_root: Project root path (defaults to parent of this file's directory)
            datasets_dir: Name of datasets directory
            slakh_dir_name: SLAKH dataset folder name
            maestro_dir_name: MAESTRO dataset folder name
            maestro_version: MAESTRO dataset version string
            checkpoint_dir: Checkpoint directory name
        """
        # Get project root
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)

        # Dataset base directory
        self.datasets_root = self.project_root / datasets_dir

        # SLAKH2100 Dataset paths
        self.slakh_root = self.datasets_root / slakh_dir_name
        self.slakh_train_path = self.slakh_root / "train"
        self.slakh_test_path = self.slakh_root / "test"
        self.slakh_valid_path = self.slakh_root / "validation"

        # MAESTRO Dataset paths
        self.maestro_root = self.datasets_root / maestro_dir_name
        self.maestro_json = self.maestro_root / f"maestro-{maestro_version}.json"
        self.maestro_csv = self.maestro_root / f"maestro-{maestro_version}.csv"

        # Legacy aliases for backwards compatibility (as strings)
        self.train_path = str(self.slakh_train_path)
        self.test_path = str(self.slakh_test_path)
        self.valid_path = str(self.slakh_valid_path)

        # SLAKH file naming conventions
        self.slakh_midi_folder = "MIDI"
        self.slakh_inst_filename = "inst_names.json"
        self.slakh_audio_filename = "mix.flac"

        # Checkpoint path
        self.cpt_path = self.project_root / checkpoint_dir

        # Sequence lengths (aligned with legacy MT3)
        self.mel_length = 256  # Changed from 512 to match legacy (256 frames ≈ 2 seconds)
        self.event_length = 1024  # Max target sequence length

        # Model configuration
        self.include_ties = True

        # Spectrogram configuration
        self.spectrogram_config = SpectrogramConfig()

        # Batch and training parameters
        self.batch_size = 8
        self.num_workers = 4

        # Create checkpoint directory if it doesn't exist
        self.cpt_path.mkdir(exist_ok=True, parents=True)

        # Convert Path objects to strings for backward compatibility
        self._convert_paths_to_strings()

    def _convert_paths_to_strings(self):
        """Convert Path objects to strings for backward compatibility."""
        self.slakh_root = str(self.slakh_root)
        self.slakh_train_path = str(self.slakh_train_path)
        self.slakh_test_path = str(self.slakh_test_path)
        self.slakh_valid_path = str(self.slakh_valid_path)
        self.maestro_root = str(self.maestro_root)
        self.maestro_json = str(self.maestro_json)
        self.maestro_csv = str(self.maestro_csv)
        self.cpt_path = str(self.cpt_path)
        self.datasets_root = str(self.datasets_root)

    def get_dataset_path(self, dataset_name="slakh", split="train"):
        """Get dataset path for a specific dataset and split.

        Args:
            dataset_name: 'slakh' or 'maestro'
            split: 'train', 'validation', or 'test'

        Returns:
            Path to the dataset split (as string)
        """
        if dataset_name.lower() == "slakh":
            if split == "train":
                return self.slakh_train_path
            elif split == "validation" or split == "valid":
                return self.slakh_valid_path
            elif split == "test":
                return self.slakh_test_path
        elif dataset_name.lower() == "maestro":
            return self.maestro_root

        raise ValueError(f"Unknown dataset: {dataset_name} or split: {split}")

    def get_slakh_track_path(self, track_id):
        """Get path to a specific SLAKH track directory.

        Args:
            track_id: Track ID (e.g., 'Track00001' or just 1)

        Returns:
            Path to the track directory (as string)
        """
        if isinstance(track_id, int):
            track_id = f"Track{track_id:05d}"
        return str(Path(self.slakh_train_path) / track_id)

    def get_maestro_metadata_path(self, format="json"):
        """Get path to MAESTRO metadata file.

        Args:
            format: 'json' or 'csv'

        Returns:
            Path to metadata file (as string)
        """
        if format.lower() == "json":
            return self.maestro_json
        elif format.lower() == "csv":
            return self.maestro_csv
        else:
            raise ValueError(f"Unknown format: {format}. Use 'json' or 'csv'.")


# Legacy compatibility: Create instance as module-level variable
data_config = DataConfig()


# Legacy compatibility: Create instance as module-level variable
data_config = DataConfig()

import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
import random


class MultiTaskDataset(Dataset):
    """Sample from multiple datasets with temperature sampling.

    Supports both list of datasets and dict of named datasets with per-dataset temperatures.
    """

    def __init__(self, datasets, temperature=1.0, temperatures=None):
        """
        Args:
            datasets: List of datasets OR Dict[name, dataset]
                Examples:
                - [maestro_ds, slakh_stem_ds]
                - {'maestro': maestro_ds, 'slakh_stems': slakh_stem_ds}
            temperature: Global temperature (used if temperatures not provided)
            temperatures: Dict[name, temperature] for per-dataset control
                Higher temp = sample more frequently
                Example: {'maestro': 1.0, 'slakh_stems': 1.0, 'slakh_mixed': 0.5}
        """
        # Handle both list and dict inputs
        if isinstance(datasets, dict):
            self.dataset_names = list(datasets.keys())
            self.datasets = list(datasets.values())
            self.temperatures = temperatures or {name: temperature for name in self.dataset_names}
        else:
            self.datasets = datasets
            self.dataset_names = [f"dataset_{i}" for i in range(len(datasets))]
            self.temperatures = {name: temperature for name in self.dataset_names}

        self.lengths = [len(d) for d in self.datasets]
        self.total_length = sum(self.lengths)
        self.probabilities = self._compute_probs()

    def _compute_probs(self):
        """Compute sampling probabilities for each dataset.

        Uses formula: p_i ∝ (n_i)^(1/T_i)
        """
        sizes = {name: self.lengths[i] for i, name in enumerate(self.dataset_names)}

        # Filter out empty datasets
        non_empty_names = [name for name in self.dataset_names if sizes[name] > 0]
        
        if not non_empty_names:
            raise ValueError("All datasets are empty!")

        # Adjust sizes by temperature: (n_i)^(1/T_i)
        adjusted = {}
        for name in non_empty_names:
            temp = self.temperatures.get(name, 1.0)
            adjusted[name] = sizes[name] ** (1.0 / temp)

        # Normalize to probabilities
        total = sum(adjusted.values())
        
        # Create probability list (0 for empty datasets)
        probs = []
        for name in self.dataset_names:
            if name in adjusted:
                probs.append(adjusted[name] / total)
            else:
                probs.append(0.0)

        return probs

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        """Sample from datasets according to temperature-adjusted probabilities."""
        # Sample dataset based on probabilities
        dataset_idx = random.choices(range(len(self.datasets)), weights=self.probabilities)[0]
        dataset = self.datasets[dataset_idx]

        # Sample random item from chosen dataset
        item_idx = random.randint(0, len(dataset) - 1)
        return dataset[item_idx]


class CombinedDataset(Dataset):
    """Simple concatenation of datasets without temperature sampling."""

    def __init__(self, datasets):
        """
        Args:
            datasets: List of datasets to concatenate
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        """Get item by global index."""
        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side="right")
        # Get local index within that dataset
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        return self.datasets[dataset_idx][local_idx]

"""
Training utilities for MT3-PyTorch.
Contains helper classes and functions for training multitrack models.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Union


class DummyDataset(Dataset):
    """
    Dummy dataset that returns placeholder data.
    Used when cache exists - actual data comes from cache, not this dataset.
    """

    def __init__(self, size: int):
        """
        Initialize dummy dataset.

        Args:
            size: Number of samples in the dataset
        """
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Return empty dict - cache will override this
        return {"inputs": torch.empty(0), "targets": torch.empty(0)}


class ListDataset(Dataset):
    """
    Wrapper to convert a list of samples into a PyTorch Dataset.
    Useful for converting IterableDataset samples to map-style dataset.
    """

    def __init__(self, samples: List[Dict[str, Any]]):
        """
        Initialize dataset from list of samples.

        Args:
            samples: List of sample dictionaries with 'inputs' and 'targets' keys
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class MaestroListDataset(Dataset):
    """
    Wrapper for MAESTRO samples converted from IterableDataset.
    Specialized version for MAESTRO training data.
    """

    def __init__(self, samples: List[Dict[str, Any]]):
        """
        Initialize MAESTRO dataset from list of samples.

        Args:
            samples: List of MAESTRO sample dictionaries
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ValListDataset(Dataset):
    """
    Wrapper for validation samples converted from IterableDataset.
    Specialized version for validation data.
    """

    def __init__(self, samples: List[Dict[str, Any]]):
        """
        Initialize validation dataset from list of samples.

        Args:
            samples: List of validation sample dictionaries
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Union[torch.Tensor, List[Any]]]:
    """
    Collate function for batching samples.

    Args:
        batch: List of sample dictionaries

    Returns:
        Dictionary with batched tensors and optional metadata lists
    """
    inputs = torch.stack([sample["inputs"] for sample in batch])
    targets = torch.stack([sample["targets"] for sample in batch])

    result: Dict[str, Union[torch.Tensor, List[Any]]] = {"inputs": inputs, "targets": targets}

    # Pass through metadata fields if available (check all samples have the field)
    if batch and "source_file" in batch[0]:
        if all("source_file" in sample for sample in batch):
            result["source_file"] = [sample["source_file"] for sample in batch]
    if batch and "segment_idx" in batch[0]:
        if all("segment_idx" in sample for sample in batch):
            result["segment_idx"] = [sample["segment_idx"] for sample in batch]

    return result


def pad_batch(batch: List[Dict[str, torch.Tensor]], pad_value: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate function with padding for variable-length sequences.

    Args:
        batch: List of sample dictionaries
        pad_value: Value to use for padding

    Returns:
        Dictionary with padded and batched tensors
    """
    # Get max lengths
    max_input_len = max(sample["inputs"].shape[0] for sample in batch)
    max_target_len = max(sample["targets"].shape[0] for sample in batch)

    # Pad inputs
    padded_inputs = []
    for sample in batch:
        input_tensor = sample["inputs"]
        pad_len = max_input_len - input_tensor.shape[0]
        if pad_len > 0:
            padding = torch.zeros((pad_len, *input_tensor.shape[1:]), dtype=input_tensor.dtype)
            padded_input = torch.cat([input_tensor, padding], dim=0)
        else:
            padded_input = input_tensor
        padded_inputs.append(padded_input)

    # Pad targets
    padded_targets = []
    for sample in batch:
        target_tensor = sample["targets"]
        pad_len = max_target_len - target_tensor.shape[0]
        if pad_len > 0:
            padding = torch.full((pad_len,), pad_value, dtype=target_tensor.dtype)
            padded_target = torch.cat([target_tensor, padding], dim=0)
        else:
            padded_target = target_tensor
        padded_targets.append(padded_target)

    return {
        "inputs": torch.stack(padded_inputs),
        "targets": torch.stack(padded_targets),
    }

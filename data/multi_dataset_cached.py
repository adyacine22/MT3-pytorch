"""
Unified cached multi-dataset loader for SLAKH + MAESTRO v3.
Enables training on combined datasets with cached fast loading.
"""

import torch
from torch.utils.data import ConcatDataset, DataLoader
from typing import Optional, Tuple

from data.cached_slakh_loader import (
    CachedSLAKHStemDataset,
    CachedSLAKHMixDataset,
    CachedSLAKHMixedDataset
)
from data.cached_maestro_loader import CachedMaestroDataset


class MultiDatasetCached:
    """Unified loader for multiple cached datasets."""
    
    def __init__(
        self,
        include_slakh: bool = True,
        include_maestro: bool = True,
        slakh_mode: str = "mixed",  # "stem", "mix", "mixed"
        maestro_cache_dir: Optional[str] = None,
        slakh_cache_dir: Optional[str] = None,
        max_tracks: Optional[int] = None
    ):
        """
        Args:
            include_slakh: Include SLAKH dataset
            include_maestro: Include MAESTRO v3 dataset
            slakh_mode: "stem" (individual stems), "mix" (full mix), "mixed" (random mixture)
            maestro_cache_dir: Custom MAESTRO cache directory
            slakh_cache_dir: Custom SLAKH cache directory
            max_tracks: Limit tracks per dataset
        """
        self.datasets = []
        self.dataset_names = []
        
        if include_slakh:
            if slakh_mode == "stem":
                slakh_dataset = CachedSLAKHStemDataset(
                    cache_dir=slakh_cache_dir,
                    split="train",
                    max_tracks=max_tracks
                )
            elif slakh_mode == "mix":
                slakh_dataset = CachedSLAKHMixDataset(
                    cache_dir=slakh_cache_dir,
                    split="train",
                    max_tracks=max_tracks
                )
            else:  # mixed
                slakh_dataset = CachedSLAKHMixedDataset(
                    cache_dir=slakh_cache_dir,
                    split="train",
                    max_tracks=max_tracks
                )
            
            self.datasets.append(slakh_dataset)
            self.dataset_names.append(f"SLAKH-{slakh_mode}")
        
        if include_maestro:
            maestro_dataset = CachedMaestroDataset(
                cache_dir=maestro_cache_dir,
                max_tracks=max_tracks
            )
            self.datasets.append(maestro_dataset)
            self.dataset_names.append("MAESTRO-v3")
        
        # Combine datasets
        self.combined = ConcatDataset(self.datasets)
    
    def __len__(self):
        return len(self.combined)
    
    def __getitem__(self, idx):
        return self.combined[idx]
    
    def get_dataloader(
        self,
        batch_size: int = 32,
        num_workers: int = 6,
        shuffle: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2
    ) -> DataLoader:
        """Create DataLoader for combined dataset."""
        
        # prefetch_factor only works with num_workers > 0
        kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'shuffle': shuffle,
            'pin_memory': True
        }
        
        if num_workers > 0:
            kwargs['persistent_workers'] = persistent_workers
            kwargs['prefetch_factor'] = prefetch_factor
        
        return DataLoader(self.combined, **kwargs)
    
    def print_stats(self):
        """Print dataset statistics."""
        print("\n" + "="*70)
        print("MULTI-DATASET STATISTICS")
        print("="*70)
        
        for name, dataset in zip(self.dataset_names, self.datasets):
            print(f"  {name:20s}: {len(dataset):6d} samples")
        
        print(f"  {'─'*68}")
        print(f"  {'TOTAL':20s}: {len(self.combined):6d} samples")
        print("="*70 + "\n")


def get_cached_train_loader(
    batch_size: int = 32,
    num_workers: int = 6,
    dataset_mix: str = "combined",  # "slakh-only", "maestro-only", "combined"
    slakh_mode: str = "mixed",  # For SLAKH: "stem", "mix", "mixed"
    max_tracks: Optional[int] = None
) -> DataLoader:
    """
    Convenience function to get a cached multi-dataset DataLoader.
    
    Args:
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        dataset_mix: Which datasets to include
        slakh_mode: SLAKH processing mode
        max_tracks: Max tracks per dataset
    
    Returns:
        DataLoader ready for training
    """
    
    if dataset_mix == "slakh-only":
        include_slakh = True
        include_maestro = False
    elif dataset_mix == "maestro-only":
        include_slakh = False
        include_maestro = True
    else:  # combined
        include_slakh = True
        include_maestro = True
    
    multi_dataset = MultiDatasetCached(
        include_slakh=include_slakh,
        include_maestro=include_maestro,
        slakh_mode=slakh_mode,
        max_tracks=max_tracks
    )
    
    multi_dataset.print_stats()
    
    return multi_dataset.get_dataloader(
        batch_size=batch_size,
        num_workers=num_workers
    )


if __name__ == '__main__':
    # Example usage
    print("Creating cached multi-dataset loader...")
    
    loader = get_cached_train_loader(
        batch_size=32,
        num_workers=6,
        dataset_mix="combined",
        slakh_mode="mixed"
    )
    
    print(f"DataLoader created with {len(loader.dataset)} total samples")
    print(f"Batch size: 32, Total batches: {len(loader)}")
    
    # Test loading a single batch
    print("\nLoading first batch...")
    for batch in loader:
        print(f"Batch inputs shape: {batch['inputs'].shape}")
        print(f"Batch targets shape: {batch['targets'].shape}")
        break
    
    print("✅ Success!")

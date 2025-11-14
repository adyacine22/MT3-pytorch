"""Sampling utilities for chunk datasets."""

from __future__ import annotations

import bisect
import random
from typing import Dict, Iterable, Iterator, List, Sequence

import numpy as np
from torch.utils.data import Sampler


class TemperatureSampler(Sampler[int]):
    """Sample dataset indices using temperature-weighted probabilities."""

    def __init__(
        self,
        dataset,
        *,
        temperature: float = 0.3,
        epoch_size: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.temperature = temperature
        self.epoch_size = epoch_size or len(dataset)
        self.random = random.Random(seed)

        manifest = dataset.manifest
        self.groups: Dict[str, List[int]] = {
            name: manifest.index[manifest["dataset"] == name].tolist()
            for name in manifest["dataset"].unique()
        }
        counts = np.array([len(idxs) for idxs in self.groups.values()], dtype=np.float64)
        weights = np.power(counts, temperature)
        weights /= weights.sum()
        self.labels = list(self.groups.keys())
        self.cumulative = np.cumsum(weights).tolist()

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.epoch_size):
            u = self.random.random()
            idx = bisect.bisect_left(self.cumulative, u)
            label = self.labels[idx]
            pool = self.groups[label]
            yield self.random.choice(pool)

    def __len__(self) -> int:
        return self.epoch_size

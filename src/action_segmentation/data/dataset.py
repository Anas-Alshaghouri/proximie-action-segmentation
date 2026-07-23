from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader, Dataset

from action_segmentation.config import AppConfig
from action_segmentation.data.synthetic import (
    generate_phase_prototypes,
    generate_synthetic_sample,
)

DatasetSplit = Literal["train", "validation", "test"]

_SPLIT_SEED_OFFSETS: Mapping[DatasetSplit, int] = {
    "train": 10_000,
    "validation": 20_000,
    "test": 30_000,
}


class SyntheticTemporalDataset(Dataset[dict[str, Any]]):
    """Deterministic synthetic multi-view workflow sequences."""

    def __init__(self, config: AppConfig, split: DatasetSplit) -> None:
        if split not in _SPLIT_SEED_OFFSETS:
            raise ValueError(f"Unsupported dataset split: {split}")

        self.config = config
        self.split = split
        self.prototypes = generate_phase_prototypes(config)
        self._split_seed = config.project.seed + _SPLIT_SEED_OFFSETS[split]
        self._sample_count = {
            "train": config.data.train_samples,
            "validation": config.data.validation_samples,
            "test": config.data.test_samples,
        }[split]

    def __len__(self) -> int:
        return self._sample_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(f"Dataset index out of range: {index}")

        return generate_synthetic_sample(
            config=self.config,
            prototypes=self.prototypes,
            seed=self._split_seed + index,
            sample_id=f"{self.split}_{index:04d}",
        )


def create_dataloader(
    config: AppConfig,
    split: DatasetSplit,
    *,
    shuffle: bool | None = None,
) -> DataLoader[dict[str, Any]]:
    """Create a deterministic DataLoader for one synthetic split."""
    dataset = SyntheticTemporalDataset(config=config, split=split)
    should_shuffle = split == "train" if shuffle is None else shuffle
    generator = torch.Generator().manual_seed(
        config.project.seed + _SPLIT_SEED_OFFSETS[split]
    )

    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=should_shuffle,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )


def create_dataloaders(
    config: AppConfig,
) -> dict[DatasetSplit, DataLoader[dict[str, Any]]]:
    """Create train, validation, and test DataLoaders."""
    return {
        split: create_dataloader(config, split)
        for split in ("train", "validation", "test")
    }

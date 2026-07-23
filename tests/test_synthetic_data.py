from pathlib import Path

import torch

from action_segmentation.config import load_config
from action_segmentation.data.dataset import (
    SyntheticTemporalDataset,
    create_dataloader,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "default.yaml"


def test_synthetic_sample_matches_tensor_contract() -> None:
    config = load_config(CONFIG_PATH)
    dataset = SyntheticTemporalDataset(config=config, split="validation")
    sample = dataset[0]

    assert sample["features"].shape == (
        config.data.max_views,
        config.data.sequence_length,
        config.data.feature_dim,
    )
    assert sample["labels"].shape == (config.data.sequence_length,)
    assert sample["timestamps"].shape == (config.data.sequence_length,)
    assert sample["view_mask"].shape == (
        config.data.max_views,
        config.data.sequence_length,
    )
    assert sample["time_mask"].shape == (config.data.sequence_length,)

    assert sample["features"].dtype == torch.float32
    assert sample["labels"].dtype == torch.int64
    assert sample["view_mask"].dtype == torch.bool
    assert sample["time_mask"].dtype == torch.bool


def test_phase_sequence_is_ordered_contiguous_and_complete() -> None:
    config = load_config(CONFIG_PATH)
    sample = SyntheticTemporalDataset(config, "validation")[0]

    ordered_unique_labels = torch.unique_consecutive(sample["labels"])
    expected = torch.arange(len(config.phases.names), dtype=torch.int64)

    assert torch.equal(ordered_unique_labels, expected)
    assert int(sample["phase_durations"].sum()) == config.data.sequence_length
    assert sample["phase_boundaries"][0].item() == 0
    assert sample["phase_boundaries"][-1].item() == config.data.sequence_length


def test_generation_is_reproducible_per_split_and_index() -> None:
    config = load_config(CONFIG_PATH)
    first_dataset = SyntheticTemporalDataset(config, "validation")
    second_dataset = SyntheticTemporalDataset(config, "validation")

    first = first_dataset[3]
    second = second_dataset[3]

    for key in (
        "features",
        "labels",
        "view_mask",
        "time_mask",
        "occlusion_mask",
        "patient_present_disturbance_mask",
        "phase_durations",
    ):
        assert torch.equal(first[key], second[key])


def test_masks_and_zero_filled_invalid_features_are_consistent() -> None:
    config = load_config(CONFIG_PATH)
    sample = SyntheticTemporalDataset(config, "validation")[0]

    assert torch.equal(sample["time_mask"], sample["view_mask"].any(dim=0))
    assert torch.all(sample["features"][~sample["view_mask"]] == 0)
    assert not torch.any(sample["occlusion_mask"] & sample["view_mask"])
    assert config.data.min_views <= sample["num_views"].item() <= config.data.max_views


def test_dataloader_stacks_a_complete_batch() -> None:
    config = load_config(CONFIG_PATH)
    loader = create_dataloader(config, "validation", shuffle=False)
    batch = next(iter(loader))

    assert batch["features"].shape == (
        config.training.batch_size,
        config.data.max_views,
        config.data.sequence_length,
        config.data.feature_dim,
    )
    assert batch["labels"].shape == (
        config.training.batch_size,
        config.data.sequence_length,
    )
    assert len(batch["sample_id"]) == config.training.batch_size

from pathlib import Path

import pytest
import torch

from action_segmentation.config import load_config
from action_segmentation.data.dataset import create_dataloader
from action_segmentation.data.ingestion import ingest_precomputed_feature_batch
from action_segmentation.models.fusion import MaskedMeanFusion, build_fusion_layer


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "default.yaml"


def test_masked_mean_fusion_handles_many_one_and_zero_views() -> None:
    features = torch.tensor(
        [
            [
                [[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]],
                [[3.0, 4.0], [30.0, 40.0], [300.0, 400.0]],
                [[5.0, 6.0], [50.0, 60.0], [500.0, 600.0]],
            ]
        ]
    )
    view_mask = torch.tensor(
        [[[True, True, False], [True, False, False], [True, False, False]]]
    )

    output = MaskedMeanFusion()(features, view_mask)

    expected = torch.tensor([[[3.0, 4.0], [10.0, 20.0], [0.0, 0.0]]])
    assert torch.allclose(output.fused_features, expected)
    assert torch.equal(output.available_view_count, torch.tensor([[3, 1, 0]]))
    assert torch.equal(output.time_mask, torch.tensor([[True, True, False]]))


def test_masked_values_never_affect_the_average() -> None:
    features = torch.tensor(
        [[[[2.0]], [[1_000_000.0]], [[-1_000_000.0]]]]
    )
    view_mask = torch.tensor([[[True], [False], [False]]])

    output = MaskedMeanFusion()(features, view_mask)

    assert output.fused_features.item() == pytest.approx(2.0)


def test_fusion_rejects_incompatible_shapes() -> None:
    features = torch.zeros(2, 3, 10, 4)
    view_mask = torch.ones(2, 2, 10, dtype=torch.bool)

    with pytest.raises(ValueError, match="dimensions must match"):
        MaskedMeanFusion()(features, view_mask)


def test_configured_fusion_runs_on_the_synthetic_batch() -> None:
    config = load_config(CONFIG_PATH)
    raw_batch = next(
        iter(create_dataloader(config, "validation", shuffle=False))
    )
    ingested = ingest_precomputed_feature_batch(raw_batch, config)
    fusion = build_fusion_layer(config)

    output = fusion(ingested.features, ingested.view_mask)

    assert output.fused_features.shape == (
        config.training.batch_size,
        config.data.sequence_length,
        config.data.feature_dim,
    )
    assert torch.equal(output.time_mask, ingested.time_mask)
    assert torch.all(output.fused_features[~output.time_mask] == 0)

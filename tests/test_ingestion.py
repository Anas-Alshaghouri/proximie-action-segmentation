from pathlib import Path

import pytest
import torch

from action_segmentation.config import load_config
from action_segmentation.data.dataset import create_dataloader
from action_segmentation.data.ingestion import (
    FeatureIngestionError,
    ingest_precomputed_feature_batch,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "default.yaml"


def _validation_batch() -> tuple[object, dict[str, object]]:
    config = load_config(CONFIG_PATH)
    loader = create_dataloader(config, "validation", shuffle=False)
    return config, next(iter(loader))


def test_ingestion_preserves_the_validated_batch_contract() -> None:
    config, raw_batch = _validation_batch()
    ingested = ingest_precomputed_feature_batch(raw_batch, config)

    assert ingested.sample_ids == tuple(raw_batch["sample_id"])
    assert ingested.features is raw_batch["features"]
    assert ingested.labels is raw_batch["labels"]
    assert ingested.timestamps is raw_batch["timestamps"]
    assert ingested.view_mask is raw_batch["view_mask"]
    assert ingested.time_mask is raw_batch["time_mask"]
    assert ingested.batch_size == config.training.batch_size
    assert ingested.num_view_slots == config.data.max_views
    assert "occlusion_mask" in ingested.metadata


def test_ingestion_rejects_an_inconsistent_time_mask() -> None:
    config, raw_batch = _validation_batch()
    invalid_batch = dict(raw_batch)
    invalid_batch["time_mask"] = raw_batch["time_mask"].clone()
    invalid_batch["time_mask"][0, 0] = ~invalid_batch["time_mask"][0, 0]

    with pytest.raises(FeatureIngestionError, match="time_mask"):
        ingest_precomputed_feature_batch(invalid_batch, config)


def test_ingestion_rejects_non_increasing_timestamps() -> None:
    config, raw_batch = _validation_batch()
    invalid_batch = dict(raw_batch)
    invalid_batch["timestamps"] = raw_batch["timestamps"].clone()
    invalid_batch["timestamps"][0, 2] = invalid_batch["timestamps"][0, 1]

    with pytest.raises(FeatureIngestionError, match="strictly increasing"):
        ingest_precomputed_feature_batch(invalid_batch, config)

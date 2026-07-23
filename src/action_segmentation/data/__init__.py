"""Synthetic data generation and precomputed-feature ingestion."""

from action_segmentation.data.dataset import (
    SyntheticTemporalDataset,
    create_dataloader,
    create_dataloaders,
)
from action_segmentation.data.ingestion import (
    FeatureIngestionError,
    MultiViewFeatureBatch,
    ingest_precomputed_feature_batch,
)

__all__ = [
    "FeatureIngestionError",
    "MultiViewFeatureBatch",
    "SyntheticTemporalDataset",
    "create_dataloader",
    "create_dataloaders",
    "ingest_precomputed_feature_batch",
]

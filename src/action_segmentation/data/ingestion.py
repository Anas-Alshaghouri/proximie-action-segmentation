from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from action_segmentation.config import AppConfig


class FeatureIngestionError(ValueError):
    """Raised when an incoming precomputed-feature batch breaks the contract."""


@dataclass(frozen=True)
class MultiViewFeatureBatch:
    """Validated, synchronized precomputed features ready for view fusion."""

    sample_ids: tuple[str, ...]
    features: torch.Tensor
    labels: torch.Tensor
    timestamps: torch.Tensor
    view_mask: torch.Tensor
    time_mask: torch.Tensor
    metadata: Mapping[str, Any]

    @property
    def batch_size(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_view_slots(self) -> int:
        return int(self.features.shape[1])

    @property
    def sequence_length(self) -> int:
        return int(self.features.shape[2])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[3])


def _require_tensor(batch: Mapping[str, Any], key: str) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise FeatureIngestionError(f"Batch field '{key}' must be a torch.Tensor.")
    return value


def ingest_precomputed_feature_batch(
    batch: Mapping[str, Any],
    config: AppConfig,
) -> MultiViewFeatureBatch:
    """Validate a synchronized batch of precomputed multi-camera features.

    The synthetic DataLoader is the prototype source. In production, an adapter
    at this boundary would receive timestamp-aligned embeddings from the live
    feature extraction and stream-assembly services.
    """
    features = _require_tensor(batch, "features")
    labels = _require_tensor(batch, "labels")
    timestamps = _require_tensor(batch, "timestamps")
    view_mask = _require_tensor(batch, "view_mask")
    time_mask = _require_tensor(batch, "time_mask")

    if features.ndim != 4:
        raise FeatureIngestionError(
            "'features' must have shape [batch, views, time, feature_dim]."
        )

    batch_size, views, sequence_length, feature_dim = features.shape
    expected_feature_shape = (
        batch_size,
        config.data.max_views,
        config.data.sequence_length,
        config.data.feature_dim,
    )
    if tuple(features.shape) != expected_feature_shape:
        raise FeatureIngestionError(
            f"Unexpected feature shape {tuple(features.shape)}; "
            f"expected {expected_feature_shape}."
        )
    if features.dtype != torch.float32:
        raise FeatureIngestionError("'features' must use torch.float32.")

    expected_sequence_shape = (batch_size, sequence_length)
    expected_view_mask_shape = (batch_size, views, sequence_length)
    if tuple(labels.shape) != expected_sequence_shape or labels.dtype != torch.int64:
        raise FeatureIngestionError(
            "'labels' must have shape [batch, time] and dtype torch.int64."
        )
    if tuple(timestamps.shape) != expected_sequence_shape:
        raise FeatureIngestionError(
            "'timestamps' must have shape [batch, time]."
        )
    if tuple(view_mask.shape) != expected_view_mask_shape:
        raise FeatureIngestionError(
            "'view_mask' must have shape [batch, views, time]."
        )
    if view_mask.dtype != torch.bool:
        raise FeatureIngestionError("'view_mask' must use torch.bool.")
    if tuple(time_mask.shape) != expected_sequence_shape:
        raise FeatureIngestionError("'time_mask' must have shape [batch, time].")
    if time_mask.dtype != torch.bool:
        raise FeatureIngestionError("'time_mask' must use torch.bool.")

    expected_time_mask = view_mask.any(dim=1)
    if not torch.equal(time_mask, expected_time_mask):
        raise FeatureIngestionError(
            "'time_mask' must be true exactly when at least one view is valid."
        )

    if sequence_length > 1 and torch.any(torch.diff(timestamps, dim=1) <= 0):
        raise FeatureIngestionError(
            "Timestamps must be strictly increasing within every sample."
        )

    sample_ids_raw = batch.get("sample_id")
    if not isinstance(sample_ids_raw, (list, tuple)):
        raise FeatureIngestionError("'sample_id' must be a list or tuple of strings.")
    sample_ids = tuple(str(value) for value in sample_ids_raw)
    if len(sample_ids) != batch_size:
        raise FeatureIngestionError(
            "The number of sample IDs must equal the feature batch size."
        )

    metadata = {
        key: value
        for key, value in batch.items()
        if key
        not in {
            "sample_id",
            "features",
            "labels",
            "timestamps",
            "view_mask",
            "time_mask",
        }
    }

    return MultiViewFeatureBatch(
        sample_ids=sample_ids,
        features=features,
        labels=labels,
        timestamps=timestamps,
        view_mask=view_mask,
        time_mask=time_mask,
        metadata=metadata,
    )

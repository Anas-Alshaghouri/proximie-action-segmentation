from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from action_segmentation.config import AppConfig


@dataclass(frozen=True)
class FusionOutput:
    """Feature sequence and availability information after view fusion."""

    fused_features: torch.Tensor
    time_mask: torch.Tensor
    available_view_count: torch.Tensor


class MaskedMeanFusion(nn.Module):
    """Average only the camera views marked valid at each timestamp.

    This parameter-free baseline supports one to three cameras, temporary
    occlusions, and complete stream dropout without inventing replacement data.
    """

    def forward(
        self,
        features: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> FusionOutput:
        if features.ndim != 4:
            raise ValueError(
                "'features' must have shape [batch, views, time, feature_dim]."
            )
        if view_mask.ndim != 3:
            raise ValueError(
                "'view_mask' must have shape [batch, views, time]."
            )
        if tuple(features.shape[:3]) != tuple(view_mask.shape):
            raise ValueError(
                "Feature and view-mask batch/view/time dimensions must match."
            )
        if not features.is_floating_point():
            raise TypeError("'features' must be a floating-point tensor.")
        if view_mask.dtype != torch.bool:
            raise TypeError("'view_mask' must use torch.bool.")
        if features.device != view_mask.device:
            raise ValueError("'features' and 'view_mask' must use the same device.")

        weights = view_mask.unsqueeze(-1).to(dtype=features.dtype)
        available_view_count = view_mask.sum(dim=1)
        denominator = available_view_count.clamp_min(1).unsqueeze(-1)

        fused_features = (features * weights).sum(dim=1) / denominator
        time_mask = available_view_count > 0
        fused_features = fused_features.masked_fill(
            ~time_mask.unsqueeze(-1),
            0.0,
        )

        return FusionOutput(
            fused_features=fused_features,
            time_mask=time_mask,
            available_view_count=available_view_count,
        )


def build_fusion_layer(config: AppConfig) -> nn.Module:
    """Construct the configured multi-view fusion layer."""
    if config.fusion.type == "masked_mean":
        return MaskedMeanFusion()
    raise ValueError(f"Unsupported fusion type: {config.fusion.type}")

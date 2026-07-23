from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from action_segmentation.config import AppConfig


@dataclass(frozen=True)
class TemporalModelOutput:
    """Frame-level workflow logits and their valid-timestamp mask."""

    logits: torch.Tensor
    time_mask: torch.Tensor


class CausalConv1d(nn.Conv1d):
    """One-dimensional convolution padded only on the past side."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias,
        )
        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        padded = F.pad(inputs, (self.left_padding, 0))
        return super().forward(padded)


class ResidualTemporalBlock(nn.Module):
    """Two causal dilated convolutions with a residual connection."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.conv2 = CausalConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.conv1(inputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        outputs = self.conv2(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return residual + outputs


class CausalTemporalConvNet(nn.Module):
    """Lightweight causal TCN for frame-level workflow phase prediction.

    Input shape:  [batch, time, feature_dim]
    Output shape: [batch, time, num_classes]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        kernel_size: int,
        dilations: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.dilations = dilations

        self.input_projection = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=1,
        )
        self.temporal_blocks = nn.ModuleList(
            ResidualTemporalBlock(
                channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            for dilation in dilations
        )
        self.classifier = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=num_classes,
            kernel_size=1,
        )

    @property
    def receptive_field_steps(self) -> int:
        """Number of current-and-past feature steps visible to each output."""
        convolutions_per_block = 2
        return 1 + convolutions_per_block * (self.kernel_size - 1) * sum(
            self.dilations
        )

    def forward(
        self,
        fused_features: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> TemporalModelOutput:
        if fused_features.ndim != 3:
            raise ValueError(
                "'fused_features' must have shape [batch, time, feature_dim]."
            )
        if time_mask.ndim != 2:
            raise ValueError("'time_mask' must have shape [batch, time].")
        if fused_features.shape[:2] != time_mask.shape:
            raise ValueError(
                "Feature and time-mask batch/time dimensions must match."
            )
        if fused_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected feature_dim={self.input_dim}, received "
                f"{fused_features.shape[-1]}."
            )
        if not fused_features.is_floating_point():
            raise TypeError("'fused_features' must be floating point.")
        if time_mask.dtype != torch.bool:
            raise TypeError("'time_mask' must use torch.bool.")
        if fused_features.device != time_mask.device:
            raise ValueError(
                "'fused_features' and 'time_mask' must use the same device."
            )

        masked_features = fused_features.masked_fill(
            ~time_mask.unsqueeze(-1),
            0.0,
        )
        outputs = masked_features.transpose(1, 2)
        outputs = self.input_projection(outputs)

        for block in self.temporal_blocks:
            outputs = block(outputs)

        logits = self.classifier(outputs).transpose(1, 2)
        logits = logits.masked_fill(~time_mask.unsqueeze(-1), 0.0)

        return TemporalModelOutput(logits=logits, time_mask=time_mask)


def build_temporal_model(config: AppConfig) -> CausalTemporalConvNet:
    """Construct the configured temporal classification model."""
    if config.model.type != "causal_tcn":
        raise ValueError(f"Unsupported temporal model: {config.model.type}")

    return CausalTemporalConvNet(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        num_classes=config.model.num_classes,
        kernel_size=config.model.kernel_size,
        dilations=config.model.dilations,
        dropout=config.model.dropout,
    )

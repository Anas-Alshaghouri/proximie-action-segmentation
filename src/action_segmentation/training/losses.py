from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    time_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy over timestamps with at least one available camera."""
    if logits.ndim != 3:
        raise ValueError("'logits' must have shape [batch, time, classes].")
    if labels.shape != logits.shape[:2]:
        raise ValueError("'labels' must match the logits batch/time dimensions.")
    if time_mask.shape != labels.shape:
        raise ValueError("'time_mask' must have the same shape as 'labels'.")
    if labels.dtype != torch.int64:
        raise TypeError("'labels' must use torch.int64.")
    if time_mask.dtype != torch.bool:
        raise TypeError("'time_mask' must use torch.bool.")
    if not time_mask.any():
        raise ValueError("Masked loss requires at least one valid timestamp.")

    return F.cross_entropy(logits[time_mask], labels[time_mask])

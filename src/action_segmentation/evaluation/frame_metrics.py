from __future__ import annotations

from typing import Any, Sequence

import torch

from action_segmentation.postprocessing.timeline import UNAVAILABLE_LABEL_ID


def _validate_inputs(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    time_mask: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    if targets.ndim != 1 or predictions.ndim != 1 or time_mask.ndim != 1:
        raise ValueError("Targets, predictions, and time_mask must have shape [time].")
    if targets.shape != predictions.shape or targets.shape != time_mask.shape:
        raise ValueError("Targets, predictions, and time_mask must share shape.")
    if targets.dtype != torch.int64 or predictions.dtype != torch.int64:
        raise TypeError("Targets and predictions must use torch.int64.")
    if time_mask.dtype != torch.bool:
        raise TypeError("time_mask must use torch.bool.")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    valid = time_mask & (predictions != UNAVAILABLE_LABEL_ID)
    if valid.any():
        if torch.any((targets[valid] < 0) | (targets[valid] >= num_classes)):
            raise ValueError("Targets contain an invalid class ID.")
        if torch.any((predictions[valid] < 0) | (predictions[valid] >= num_classes)):
            raise ValueError("Predictions contain an invalid class ID.")
    return valid


def confusion_matrix(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    time_mask: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Return rows=true classes and columns=predicted classes."""
    valid = _validate_inputs(targets, predictions, time_mask, num_classes)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    if not valid.any():
        return matrix

    flat_indices = targets[valid] * num_classes + predictions[valid]
    counts = torch.bincount(flat_indices, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).cpu()


def frame_classification_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    time_mask: torch.Tensor,
    phase_names: Sequence[str],
) -> dict[str, Any]:
    """Calculate accuracy and per-class precision, recall, and F1."""
    num_classes = len(phase_names)
    if num_classes == 0:
        raise ValueError("phase_names must not be empty.")

    valid = _validate_inputs(targets, predictions, time_mask, num_classes)
    matrix = confusion_matrix(targets, predictions, time_mask, num_classes)
    valid_count = int(valid.sum().item())
    correct_count = int((targets[valid] == predictions[valid]).sum().item())
    accuracy = correct_count / valid_count if valid_count else 0.0

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for class_id, phase_name in enumerate(phase_names):
        true_positive = int(matrix[class_id, class_id].item())
        false_positive = int(matrix[:, class_id].sum().item()) - true_positive
        false_negative = int(matrix[class_id, :].sum().item()) - true_positive
        support = int(matrix[class_id, :].sum().item())

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_class[str(phase_name)] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }

    return {
        "valid_timestamps": valid_count,
        "correct_timestamps": correct_count,
        "accuracy": round(accuracy, 6),
        "macro_f1": round(sum(f1_values) / len(f1_values), 6),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_rows": list(phase_names),
        "confusion_matrix_columns": list(phase_names),
    }

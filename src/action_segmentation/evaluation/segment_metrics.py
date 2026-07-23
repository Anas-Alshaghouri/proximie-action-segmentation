from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from action_segmentation.postprocessing.timeline import UNAVAILABLE_LABEL_ID


@dataclass(frozen=True)
class TemporalSegment:
    label_id: int
    start_index: int
    end_index: int

    @property
    def duration_steps(self) -> int:
        return self.end_index - self.start_index


def extract_segments(
    labels: torch.Tensor,
    time_mask: torch.Tensor | None = None,
) -> tuple[TemporalSegment, ...]:
    """Collapse frame labels into segments; unavailable gaps split segments."""
    if labels.ndim != 1:
        raise ValueError("labels must have shape [time].")
    if labels.dtype != torch.int64:
        raise TypeError("labels must use torch.int64.")
    if time_mask is None:
        time_mask = labels != UNAVAILABLE_LABEL_ID
    if time_mask.ndim != 1 or time_mask.shape != labels.shape:
        raise ValueError("time_mask must match labels.")
    if time_mask.dtype != torch.bool:
        raise TypeError("time_mask must use torch.bool.")

    segments: list[TemporalSegment] = []
    current_label: int | None = None
    start_index: int | None = None

    for index in range(labels.numel()):
        available = bool(time_mask[index]) and int(labels[index].item()) != UNAVAILABLE_LABEL_ID
        if not available:
            if current_label is not None and start_index is not None:
                segments.append(TemporalSegment(current_label, start_index, index))
            current_label = None
            start_index = None
            continue

        label_id = int(labels[index].item())
        if current_label is None:
            current_label = label_id
            start_index = index
        elif label_id != current_label:
            assert start_index is not None
            segments.append(TemporalSegment(current_label, start_index, index))
            current_label = label_id
            start_index = index

    if current_label is not None and start_index is not None:
        segments.append(TemporalSegment(current_label, start_index, labels.numel()))
    return tuple(segments)


def temporal_iou(first: TemporalSegment, second: TemporalSegment) -> float:
    intersection = max(
        0,
        min(first.end_index, second.end_index)
        - max(first.start_index, second.start_index),
    )
    union = max(first.end_index, second.end_index) - min(
        first.start_index,
        second.start_index,
    )
    return intersection / union if union else 0.0


def segmental_f1(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    time_mask: torch.Tensor,
    iou_threshold: float,
) -> dict[str, float | int]:
    """Calculate one-to-one segment F1 at a temporal IoU threshold."""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1].")

    ground_truth = extract_segments(targets, time_mask)
    predicted = extract_segments(predictions, time_mask)
    matched_ground_truth: set[int] = set()
    true_positive = 0

    for predicted_segment in predicted:
        best_index: int | None = None
        best_iou = 0.0
        for index, ground_truth_segment in enumerate(ground_truth):
            if index in matched_ground_truth:
                continue
            if ground_truth_segment.label_id != predicted_segment.label_id:
                continue
            overlap = temporal_iou(predicted_segment, ground_truth_segment)
            if overlap > best_iou:
                best_iou = overlap
                best_index = index

        if best_index is not None and best_iou >= iou_threshold:
            matched_ground_truth.add(best_index)
            true_positive += 1

    false_positive = len(predicted) - true_positive
    false_negative = len(ground_truth) - true_positive
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
    return {
        "iou_threshold": iou_threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _levenshtein_distance(first: Sequence[int], second: Sequence[int]) -> int:
    previous = list(range(len(second) + 1))
    for first_index, first_value in enumerate(first, start=1):
        current = [first_index]
        for second_index, second_value in enumerate(second, start=1):
            insertion = current[second_index - 1] + 1
            deletion = previous[second_index] + 1
            substitution = previous[second_index - 1] + (first_value != second_value)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def segment_edit_score(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    time_mask: torch.Tensor,
) -> dict[str, Any]:
    """Compare collapsed phase order using normalized Levenshtein score [0, 100]."""
    target_sequence = [segment.label_id for segment in extract_segments(targets, time_mask)]
    prediction_sequence = [
        segment.label_id for segment in extract_segments(predictions, time_mask)
    ]
    denominator = max(len(target_sequence), len(prediction_sequence))
    distance = _levenshtein_distance(target_sequence, prediction_sequence)
    score = 100.0 if denominator == 0 else (1.0 - distance / denominator) * 100.0
    return {
        "score": round(max(0.0, score), 6),
        "distance": distance,
        "ground_truth_sequence": target_sequence,
        "predicted_sequence": prediction_sequence,
    }


def _greedy_class_matches(
    ground_truth: tuple[TemporalSegment, ...],
    predicted: tuple[TemporalSegment, ...],
) -> list[tuple[TemporalSegment, TemporalSegment]]:
    candidates: list[tuple[float, int, int]] = []
    for ground_truth_index, ground_truth_segment in enumerate(ground_truth):
        for predicted_index, predicted_segment in enumerate(predicted):
            if ground_truth_segment.label_id != predicted_segment.label_id:
                continue
            overlap = temporal_iou(ground_truth_segment, predicted_segment)
            boundary_distance = abs(
                ground_truth_segment.start_index - predicted_segment.start_index
            ) + abs(ground_truth_segment.end_index - predicted_segment.end_index)
            # Prefer overlap, then the closest temporal boundaries.
            candidates.append((overlap - boundary_distance * 1e-9, ground_truth_index, predicted_index))

    matches: list[tuple[TemporalSegment, TemporalSegment]] = []
    used_ground_truth: set[int] = set()
    used_predicted: set[int] = set()
    for _, ground_truth_index, predicted_index in sorted(candidates, reverse=True):
        if ground_truth_index in used_ground_truth or predicted_index in used_predicted:
            continue
        used_ground_truth.add(ground_truth_index)
        used_predicted.add(predicted_index)
        matches.append((ground_truth[ground_truth_index], predicted[predicted_index]))
    return matches


def boundary_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    time_mask: torch.Tensor,
    sampling_rate_hz: float,
    tolerance_seconds: float,
) -> dict[str, Any]:
    """Measure signed and absolute start/end errors for matched phase segments."""
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds cannot be negative.")

    ground_truth = extract_segments(targets, time_mask)
    predicted = extract_segments(predictions, time_mask)
    matches = _greedy_class_matches(ground_truth, predicted)
    if not matches:
        return {
            "matched_segments": 0,
            "mean_start_error_seconds": None,
            "mean_end_error_seconds": None,
            "mean_absolute_start_error_seconds": None,
            "mean_absolute_end_error_seconds": None,
            "start_within_tolerance_ratio": 0.0,
            "end_within_tolerance_ratio": 0.0,
        }

    start_errors = [
        (predicted_segment.start_index - ground_truth_segment.start_index)
        / sampling_rate_hz
        for ground_truth_segment, predicted_segment in matches
    ]
    end_errors = [
        (predicted_segment.end_index - ground_truth_segment.end_index)
        / sampling_rate_hz
        for ground_truth_segment, predicted_segment in matches
    ]

    return {
        "matched_segments": len(matches),
        "mean_start_error_seconds": round(sum(start_errors) / len(start_errors), 6),
        "mean_end_error_seconds": round(sum(end_errors) / len(end_errors), 6),
        "mean_absolute_start_error_seconds": round(
            sum(abs(value) for value in start_errors) / len(start_errors), 6
        ),
        "mean_absolute_end_error_seconds": round(
            sum(abs(value) for value in end_errors) / len(end_errors), 6
        ),
        "start_within_tolerance_ratio": round(
            sum(abs(value) <= tolerance_seconds for value in start_errors)
            / len(start_errors),
            6,
        ),
        "end_within_tolerance_ratio": round(
            sum(abs(value) <= tolerance_seconds for value in end_errors)
            / len(end_errors),
            6,
        ),
    }


def segment_metric_stack(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    time_mask: torch.Tensor,
    iou_thresholds: Sequence[float],
    sampling_rate_hz: float,
    boundary_tolerance_seconds: float,
) -> dict[str, Any]:
    """Return edit, segmental F1, and boundary metrics together."""
    return {
        "ground_truth_segments": len(extract_segments(targets, time_mask)),
        "predicted_segments": len(extract_segments(predictions, time_mask)),
        "edit_score": segment_edit_score(targets, predictions, time_mask),
        "segmental_f1": {
            f"f1_at_{int(round(threshold * 100))}": segmental_f1(
                targets,
                predictions,
                time_mask,
                threshold,
            )
            for threshold in iou_thresholds
        },
        "boundaries": boundary_metrics(
            targets,
            predictions,
            time_mask,
            sampling_rate_hz,
            boundary_tolerance_seconds,
        ),
    }

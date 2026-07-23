from __future__ import annotations

from typing import Any, Sequence

import torch

from action_segmentation.evaluation.segment_metrics import extract_segments
from action_segmentation.postprocessing.timeline import UNAVAILABLE_LABEL_ID


def _phase_id(phase_names: Sequence[str], phase_name: str) -> int:
    if phase_name not in phase_names:
        raise ValueError(f"Required phase is missing: {phase_name}")
    return phase_names.index(phase_name)


def _duration_seconds(mask: torch.Tensor, sampling_rate_hz: float) -> float:
    return float(mask.sum().item()) / sampling_rate_hz


def _phase_segments(
    labels: torch.Tensor,
    time_mask: torch.Tensor,
    phase_id: int,
) -> list[Any]:
    return [
        segment
        for segment in extract_segments(labels, time_mask)
        if segment.label_id == phase_id
    ]


def _phase_boundary_delays(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    time_mask: torch.Tensor,
    phase_id: int,
    sampling_rate_hz: float,
) -> tuple[list[float], list[float]]:
    """Measure first activation and final deactivation for each true phase.

    Multiple overlapping predicted fragments are treated as one product episode here;
    fragmentation itself is measured separately.
    """
    ground_truth = _phase_segments(targets, time_mask, phase_id)
    predicted = _phase_segments(predictions, time_mask, phase_id)
    start_delays: list[float] = []
    end_delays: list[float] = []

    for ground_truth_segment in ground_truth:
        overlapping = [
            segment
            for segment in predicted
            if min(segment.end_index, ground_truth_segment.end_index)
            > max(segment.start_index, ground_truth_segment.start_index)
        ]
        if overlapping:
            first_start = min(segment.start_index for segment in overlapping)
            final_end = max(segment.end_index for segment in overlapping)
        elif predicted:
            closest = min(
                predicted,
                key=lambda segment: abs(
                    segment.start_index - ground_truth_segment.start_index
                )
                + abs(segment.end_index - ground_truth_segment.end_index),
            )
            first_start = closest.start_index
            final_end = closest.end_index
        else:
            continue

        start_delays.append(
            (first_start - ground_truth_segment.start_index) / sampling_rate_hz
        )
        end_delays.append(
            (final_end - ground_truth_segment.end_index) / sampling_rate_hz
        )
    return start_delays, end_delays


def _count_prediction_switches_inside_mask(
    predictions: torch.Tensor,
    mask: torch.Tensor,
) -> int:
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    switches = 0
    previous: int | None = None
    previous_index: int | None = None
    for tensor_index in indices:
        index = int(tensor_index.item())
        label = int(predictions[index].item())
        if label == UNAVAILABLE_LABEL_ID:
            previous = None
            previous_index = None
            continue
        if previous is not None and previous_index == index - 1 and label != previous:
            switches += 1
        previous = label
        previous_index = index
    return switches


def _contiguous_true_ranges(mask: torch.Tensor) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask.tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            ranges.append((start, index))
            start = None
    if start is not None:
        ranges.append((start, mask.numel()))
    return ranges


def _occlusion_recovery_seconds(
    predictions: torch.Tensor,
    operation_mask: torch.Tensor,
    occlusion_mask: torch.Tensor,
    time_mask: torch.Tensor,
    operation_id: int,
    sampling_rate_hz: float,
) -> list[float]:
    relevant = operation_mask & occlusion_mask & time_mask
    recoveries: list[float] = []
    for _, end in _contiguous_true_ranges(relevant):
        if end >= predictions.numel() or not bool(operation_mask[end]):
            continue
        if int(predictions[end - 1].item()) == operation_id:
            recoveries.append(0.0)
            continue
        recovery_index: int | None = None
        for index in range(end, predictions.numel()):
            if not bool(operation_mask[index]):
                break
            if bool(time_mask[index]) and int(predictions[index].item()) == operation_id:
                recovery_index = index
                break
        if recovery_index is not None:
            recoveries.append((recovery_index - end) / sampling_rate_hz)
    return recoveries


def product_quality_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    time_mask: torch.Tensor,
    phase_names: Sequence[str],
    sampling_rate_hz: float,
    *,
    occlusion_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Metrics tied to Patient Present stability and Operation utility."""
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")
    if targets.shape != predictions.shape or targets.shape != time_mask.shape:
        raise ValueError("targets, predictions, and time_mask must share shape.")

    patient_id = _phase_id(phase_names, "patient_present")
    operation_id = _phase_id(phase_names, "operation")
    valid = time_mask & (predictions != UNAVAILABLE_LABEL_ID)

    patient_ground_truth = valid & (targets == patient_id)
    patient_prediction = valid & (predictions == patient_id)
    patient_ground_truth_segments = _phase_segments(targets, time_mask, patient_id)
    patient_predicted_segments = _phase_segments(predictions, time_mask, patient_id)
    patient_duration_hours = _duration_seconds(
        patient_ground_truth,
        sampling_rate_hz,
    ) / 3600.0
    patient_switches = _count_prediction_switches_inside_mask(
        predictions,
        patient_ground_truth,
    )
    patient_start_delays, patient_end_delays = _phase_boundary_delays(
        targets,
        predictions,
        time_mask,
        patient_id,
        sampling_rate_hz,
    )

    operation_ground_truth = valid & (targets == operation_id)
    operation_prediction = valid & (predictions == operation_id)
    operation_start_delays, operation_end_delays = _phase_boundary_delays(
        targets,
        predictions,
        time_mask,
        operation_id,
        sampling_rate_hz,
    )
    operation_correct = operation_ground_truth & operation_prediction
    operation_gt_seconds = _duration_seconds(operation_ground_truth, sampling_rate_hz)

    if occlusion_mask is None:
        occluded_timestamps = torch.zeros_like(time_mask)
    else:
        if occlusion_mask.ndim == 2:
            occluded_timestamps = occlusion_mask.any(dim=0)
        elif occlusion_mask.ndim == 1:
            occluded_timestamps = occlusion_mask
        else:
            raise ValueError("occlusion_mask must have shape [views, time] or [time].")
        if occluded_timestamps.shape != time_mask.shape:
            raise ValueError("occlusion_mask time length must match predictions.")

    operation_occluded = operation_ground_truth & occluded_timestamps
    operation_occluded_seconds = _duration_seconds(
        operation_occluded,
        sampling_rate_hz,
    )
    operation_occluded_correct = operation_occluded & operation_prediction
    recoveries = _occlusion_recovery_seconds(
        predictions,
        targets == operation_id,
        occluded_timestamps,
        time_mask,
        operation_id,
        sampling_rate_hz,
    )

    return {
        "patient_present": {
            "ground_truth_segments": len(patient_ground_truth_segments),
            "predicted_segments": len(patient_predicted_segments),
            "extra_fragments": max(
                0,
                len(patient_predicted_segments) - len(patient_ground_truth_segments),
            ),
            "fragmentation_ratio": round(
                len(patient_predicted_segments) / len(patient_ground_truth_segments),
                6,
            )
            if patient_ground_truth_segments
            else None,
            "false_positive_duration_seconds": round(
                _duration_seconds(patient_prediction & ~patient_ground_truth, sampling_rate_hz),
                6,
            ),
            "missed_duration_seconds": round(
                _duration_seconds(patient_ground_truth & ~patient_prediction, sampling_rate_hz),
                6,
            ),
            "state_switches_inside_phase": patient_switches,
            "state_switches_per_hour": round(
                patient_switches / patient_duration_hours,
                6,
            )
            if patient_duration_hours > 0
            else 0.0,
            "mean_start_confirmation_delay_seconds": round(
                sum(patient_start_delays) / len(patient_start_delays), 6
            )
            if patient_start_delays
            else None,
            "mean_end_delay_seconds": round(
                sum(patient_end_delays) / len(patient_end_delays), 6
            )
            if patient_end_delays
            else None,
        },
        "operation": {
            "ground_truth_segments": len(
                _phase_segments(targets, time_mask, operation_id)
            ),
            "predicted_segments": len(
                _phase_segments(predictions, time_mask, operation_id)
            ),
            "start_delay_seconds": round(
                sum(operation_start_delays) / len(operation_start_delays), 6
            )
            if operation_start_delays
            else None,
            "end_delay_seconds": round(
                sum(operation_end_delays) / len(operation_end_delays), 6
            )
            if operation_end_delays
            else None,
            "coverage_ratio": round(
                _duration_seconds(operation_correct, sampling_rate_hz)
                / operation_gt_seconds,
                6,
            )
            if operation_gt_seconds > 0
            else 0.0,
            "missed_duration_seconds": round(
                _duration_seconds(operation_ground_truth & ~operation_prediction, sampling_rate_hz),
                6,
            ),
            "false_positive_duration_seconds": round(
                _duration_seconds(operation_prediction & ~operation_ground_truth, sampling_rate_hz),
                6,
            ),
            "occluded_operation_duration_seconds": round(
                operation_occluded_seconds,
                6,
            ),
            "coverage_during_occlusion_ratio": round(
                _duration_seconds(operation_occluded_correct, sampling_rate_hz)
                / operation_occluded_seconds,
                6,
            )
            if operation_occluded_seconds > 0
            else None,
            "mean_occlusion_recovery_seconds": round(
                sum(recoveries) / len(recoveries), 6
            )
            if recoveries
            else None,
            "occlusion_windows_evaluated": len(recoveries),
        },
    }

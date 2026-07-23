from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from action_segmentation.evaluation.product_metrics import product_quality_metrics
from action_segmentation.evaluation.segment_metrics import extract_segments
from action_segmentation.postprocessing.timeline import UNAVAILABLE_LABEL_ID


@dataclass(frozen=True)
class ErrorInterval:
    """One contiguous temporal prediction failure with an attributed cause."""

    sample_id: str
    target_phase: str
    predicted_phase: str
    start_index: int
    end_index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    cause: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "target_phase": self.target_phase,
            "predicted_phase": self.predicted_phase,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_seconds": round(self.start_seconds, 6),
            "end_seconds": round(self.end_seconds, 6),
            "duration_seconds": round(self.duration_seconds, 6),
            "cause": self.cause,
        }


def _phase_id(phase_names: Sequence[str], name: str) -> int:
    try:
        return tuple(phase_names).index(name)
    except ValueError as exc:
        raise ValueError(f"Required phase '{name}' is not configured.") from exc


def _contiguous_true_ranges(mask: torch.Tensor) -> list[tuple[int, int]]:
    if mask.ndim != 1 or mask.dtype != torch.bool:
        raise TypeError("mask must be a one-dimensional boolean tensor.")

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


def _set_false_class(
    logits: torch.Tensor,
    mask: torch.Tensor,
    false_class_id: int,
) -> None:
    if not mask.any():
        return
    logits[mask] = -4.0
    logits[mask, false_class_id] = 7.0


def build_mock_noisy_logits(
    *,
    labels: torch.Tensor,
    time_mask: torch.Tensor,
    view_mask: torch.Tensor,
    occlusion_mask: torch.Tensor,
    patient_disturbance_mask: torch.Tensor,
    phase_names: Sequence[str],
    boundary_noise_steps: int,
) -> torch.Tensor:
    """Create deterministic, explainable errors for validation analysis.

    The base prediction is correct. Controlled failures are then added where the
    synthetic dataset already marks difficult Patient Present and Operation
    conditions. This keeps the error analysis linked to known causes rather than
    arbitrary random mistakes.
    """
    if labels.ndim != 1 or time_mask.ndim != 1:
        raise ValueError("labels and time_mask must have shape [time].")
    if labels.shape != time_mask.shape:
        raise ValueError("labels and time_mask must share shape.")
    if view_mask.ndim != 2 or occlusion_mask.shape != view_mask.shape:
        raise ValueError("view_mask and occlusion_mask must have shape [views, time].")
    if view_mask.shape[1] != labels.numel():
        raise ValueError("view masks must match the label time length.")
    if patient_disturbance_mask.shape != labels.shape:
        raise ValueError("patient_disturbance_mask must match labels.")
    if boundary_noise_steps < 0:
        raise ValueError("boundary_noise_steps cannot be negative.")

    names = tuple(phase_names)
    num_classes = len(names)
    logits = torch.full(
        (labels.numel(), num_classes),
        -4.0,
        dtype=torch.float32,
        device=labels.device,
    )
    logits.scatter_(1, labels.unsqueeze(-1), 6.0)

    patient_id = _phase_id(names, "patient_present")
    preparation_id = _phase_id(names, "preparation")
    operation_id = _phase_id(names, "operation")
    closing_id = _phase_id(names, "closing")
    empty_id = _phase_id(names, "empty")

    # Background activity temporarily makes Patient Present resemble Preparation.
    patient_disturbance = patient_disturbance_mask & (labels == patient_id) & time_mask
    _set_false_class(logits, patient_disturbance, preparation_id)

    # A short premature Patient Present activation before the patient truly arrives.
    empty_indices = torch.nonzero((labels == empty_id) & time_mask, as_tuple=False).flatten()
    if empty_indices.numel() >= 8:
        empty_start = int(empty_indices[0].item())
        empty_end = int(empty_indices[-1].item()) + 1
        false_duration = min(4, max(1, empty_end - empty_start - 1))
        start = empty_start + int((empty_end - empty_start - false_duration) * 0.65)
        false_mask = torch.zeros_like(time_mask)
        false_mask[start : start + false_duration] = True
        _set_false_class(logits, false_mask & time_mask, patient_id)

    # Operation boundaries are deliberately ambiguous for a few seconds.
    operation_indices = torch.nonzero(labels == operation_id, as_tuple=False).flatten()
    if operation_indices.numel() > 0 and boundary_noise_steps > 0:
        operation_start = int(operation_indices[0].item())
        operation_end = int(operation_indices[-1].item()) + 1
        start_end = min(operation_end, operation_start + boundary_noise_steps)
        start_mask = torch.zeros_like(time_mask)
        start_mask[operation_start:start_end] = True
        _set_false_class(logits, start_mask & time_mask, preparation_id)

        closing_indices = torch.nonzero(labels == closing_id, as_tuple=False).flatten()
        if closing_indices.numel() > 0:
            closing_start = int(closing_indices[0].item())
            late_end = min(labels.numel(), closing_start + boundary_noise_steps)
            end_mask = torch.zeros_like(time_mask)
            end_mask[closing_start:late_end] = True
            _set_false_class(logits, end_mask & time_mask, operation_id)

    # Camera occlusion causes stronger confusion when only one view remains.
    occluded_operation = occlusion_mask.any(dim=0) & (labels == operation_id) & time_mask
    available_view_count = view_mask.sum(dim=0)
    severe_occlusion = occluded_operation & (available_view_count <= 1)
    _set_false_class(logits, severe_occlusion, closing_id)

    # Even when multiple views remain, the beginning of each occlusion window can
    # cause a brief unstable prediction before temporal context stabilizes it.
    moderate_occlusion = occluded_operation & ~severe_occlusion
    for start, end in _contiguous_true_ranges(moderate_occlusion):
        brief_end = min(end, start + 2)
        brief_mask = torch.zeros_like(time_mask)
        brief_mask[start:brief_end] = True
        _set_false_class(logits, brief_mask, preparation_id)

    return logits.masked_fill(~time_mask.unsqueeze(-1), 0.0)


def _near_phase_boundary(
    labels: torch.Tensor,
    phase_id: int,
    start: int,
    end: int,
    tolerance_steps: int,
) -> bool:
    indices = torch.nonzero(labels == phase_id, as_tuple=False).flatten()
    if indices.numel() == 0:
        return False
    phase_start = int(indices[0].item())
    phase_end = int(indices[-1].item()) + 1
    return (
        abs(start - phase_start) <= tolerance_steps
        or abs(end - phase_end) <= tolerance_steps
        or (start < phase_start < end)
        or (start < phase_end < end)
    )


def _dominant_prediction_name(
    predictions: torch.Tensor,
    start: int,
    end: int,
    phase_names: Sequence[str],
) -> str:
    values = predictions[start:end]
    if values.numel() == 0:
        return "unknown"
    unavailable = values == UNAVAILABLE_LABEL_ID
    if unavailable.all():
        return "unavailable"
    valid_values = values[~unavailable]
    counts = torch.bincount(valid_values, minlength=len(phase_names))
    return str(phase_names[int(counts.argmax().item())])


def _attribute_cause(
    *,
    phase_name: str,
    labels: torch.Tensor,
    phase_id: int,
    predictions: torch.Tensor,
    phase_names: Sequence[str],
    start: int,
    end: int,
    time_mask: torch.Tensor,
    view_mask: torch.Tensor,
    occlusion_mask: torch.Tensor,
    patient_disturbance_mask: torch.Tensor,
    boundary_tolerance_steps: int,
    false_positive: bool,
) -> str:
    if (~time_mask[start:end]).any():
        return "all_views_unavailable"

    if phase_name == "patient_present":
        empty_id = _phase_id(phase_names, "empty")
        if false_positive and (labels[start:end] == empty_id).any():
            return "premature_false_activation"
        if patient_disturbance_mask[start:end].any():
            return "background_disturbance"
        if _near_phase_boundary(
            labels,
            phase_id,
            start,
            end,
            boundary_tolerance_steps,
        ):
            return "boundary_transition"

    if phase_name == "operation":
        occluded = occlusion_mask[:, start:end].any()
        if occluded:
            minimum_views = int(view_mask[:, start:end].sum(dim=0).min().item())
            return "severe_view_occlusion" if minimum_views <= 1 else "view_occlusion"
        if _near_phase_boundary(
            labels,
            phase_id,
            start,
            end,
            boundary_tolerance_steps,
        ):
            return "boundary_transition"

    return "unattributed_model_confusion"


def _phase_error_intervals(
    *,
    sample_id: str,
    phase_name: str,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    time_mask: torch.Tensor,
    view_mask: torch.Tensor,
    occlusion_mask: torch.Tensor,
    patient_disturbance_mask: torch.Tensor,
    phase_names: Sequence[str],
    sampling_rate_hz: float,
    boundary_tolerance_steps: int,
) -> tuple[list[ErrorInterval], list[ErrorInterval]]:
    phase_id = _phase_id(phase_names, phase_name)
    missed_mask = (targets == phase_id) & (predictions != phase_id)
    false_positive_mask = (targets != phase_id) & (predictions == phase_id) & time_mask

    def build(mask: torch.Tensor, false_positive: bool) -> list[ErrorInterval]:
        intervals: list[ErrorInterval] = []
        for start, end in _contiguous_true_ranges(mask):
            target_name = (
                _dominant_prediction_name(targets, start, end, phase_names)
                if false_positive
                else phase_name
            )
            predicted_name = (
                phase_name
                if false_positive
                else _dominant_prediction_name(predictions, start, end, phase_names)
            )
            intervals.append(
                ErrorInterval(
                    sample_id=sample_id,
                    target_phase=target_name,
                    predicted_phase=predicted_name,
                    start_index=start,
                    end_index=end,
                    start_seconds=start / sampling_rate_hz,
                    end_seconds=end / sampling_rate_hz,
                    duration_seconds=(end - start) / sampling_rate_hz,
                    cause=_attribute_cause(
                        phase_name=phase_name,
                        labels=targets,
                        phase_id=phase_id,
                        predictions=predictions,
                        phase_names=phase_names,
                        start=start,
                        end=end,
                        time_mask=time_mask,
                        view_mask=view_mask,
                        occlusion_mask=occlusion_mask,
                        patient_disturbance_mask=patient_disturbance_mask,
                        boundary_tolerance_steps=boundary_tolerance_steps,
                        false_positive=false_positive,
                    ),
                )
            )
        return intervals

    return build(missed_mask, False), build(false_positive_mask, True)


def _phase_segment_count(
    predictions: torch.Tensor,
    time_mask: torch.Tensor,
    phase_id: int,
) -> int:
    return sum(
        segment.label_id == phase_id
        for segment in extract_segments(predictions, time_mask)
    )


def _empty_phase_accumulator() -> dict[str, Any]:
    return {
        "ground_truth_segments": 0,
        "predicted_segments": 0,
        "samples_with_fragmentation": 0,
        "missed_intervals": 0,
        "false_positive_intervals": 0,
        "missed_duration_seconds": 0.0,
        "false_positive_duration_seconds": 0.0,
        "valid_ground_truth_duration_seconds": 0.0,
        "correct_duration_seconds": 0.0,
        "all_views_unavailable_duration_seconds": 0.0,
        "cause_breakdown": {},
        "examples": [],
        "start_delays": [],
        "end_delays": [],
        "occluded_missed_duration_seconds": 0.0,
        "disturbance_missed_duration_seconds": 0.0,
    }


def _record_causes(accumulator: dict[str, Any], intervals: list[ErrorInterval]) -> None:
    for interval in intervals:
        cause = accumulator["cause_breakdown"].setdefault(
            interval.cause,
            {"intervals": 0, "duration_seconds": 0.0},
        )
        cause["intervals"] += 1
        cause["duration_seconds"] += interval.duration_seconds
        accumulator["examples"].append(interval)


def _finalize_phase_accumulator(
    accumulator: dict[str, Any],
    *,
    samples: int,
) -> dict[str, Any]:
    ground_truth_segments = accumulator["ground_truth_segments"]
    valid_duration = accumulator["valid_ground_truth_duration_seconds"]
    missed_duration = accumulator["missed_duration_seconds"]

    cause_breakdown = {
        cause: {
            "intervals": values["intervals"],
            "duration_seconds": round(values["duration_seconds"], 6),
        }
        for cause, values in sorted(accumulator["cause_breakdown"].items())
    }
    examples = sorted(
        accumulator["examples"],
        key=lambda interval: interval.duration_seconds,
        reverse=True,
    )[:6]

    result = {
        "samples_evaluated": samples,
        "ground_truth_segments": ground_truth_segments,
        "predicted_segments": accumulator["predicted_segments"],
        "extra_fragments": max(
            0,
            accumulator["predicted_segments"] - ground_truth_segments,
        ),
        "samples_with_fragmentation": accumulator["samples_with_fragmentation"],
        "missed_intervals": accumulator["missed_intervals"],
        "false_positive_intervals": accumulator["false_positive_intervals"],
        "missed_duration_seconds": round(missed_duration, 6),
        "false_positive_duration_seconds": round(
            accumulator["false_positive_duration_seconds"],
            6,
        ),
        "coverage_ratio": round(
            accumulator["correct_duration_seconds"] / valid_duration,
            6,
        )
        if valid_duration > 0
        else 0.0,
        "all_views_unavailable_duration_seconds": round(
            accumulator["all_views_unavailable_duration_seconds"],
            6,
        ),
        "mean_start_delay_seconds": round(
            sum(accumulator["start_delays"]) / len(accumulator["start_delays"]),
            6,
        )
        if accumulator["start_delays"]
        else None,
        "mean_end_delay_seconds": round(
            sum(accumulator["end_delays"]) / len(accumulator["end_delays"]),
            6,
        )
        if accumulator["end_delays"]
        else None,
        "cause_breakdown": cause_breakdown,
        "examples": [interval.to_dict() for interval in examples],
    }

    if accumulator["occluded_missed_duration_seconds"] > 0 or missed_duration > 0:
        result["occluded_missed_duration_seconds"] = round(
            accumulator["occluded_missed_duration_seconds"],
            6,
        )
        result["occlusion_share_of_missed_duration"] = round(
            accumulator["occluded_missed_duration_seconds"] / missed_duration,
            6,
        ) if missed_duration > 0 else 0.0

    if accumulator["disturbance_missed_duration_seconds"] > 0 or missed_duration > 0:
        result["disturbance_missed_duration_seconds"] = round(
            accumulator["disturbance_missed_duration_seconds"],
            6,
        )
        result["disturbance_share_of_missed_duration"] = round(
            accumulator["disturbance_missed_duration_seconds"] / missed_duration,
            6,
        ) if missed_duration > 0 else 0.0

    return result


def analyze_error_version(
    *,
    sample_ids: Sequence[str],
    targets: torch.Tensor,
    predictions: torch.Tensor,
    time_mask: torch.Tensor,
    view_mask: torch.Tensor,
    occlusion_mask: torch.Tensor,
    patient_disturbance_mask: torch.Tensor,
    phase_names: Sequence[str],
    sampling_rate_hz: float,
    boundary_tolerance_seconds: float,
) -> dict[str, Any]:
    """Aggregate Patient Present and Operation failures across a batch."""
    if targets.ndim != 2 or predictions.shape != targets.shape:
        raise ValueError("targets and predictions must have shape [batch, time].")
    if time_mask.shape != targets.shape:
        raise ValueError("time_mask must match targets.")
    if view_mask.ndim != 3 or view_mask.shape[0] != targets.shape[0]:
        raise ValueError("view_mask must have shape [batch, views, time].")
    if occlusion_mask.shape != view_mask.shape:
        raise ValueError("occlusion_mask must match view_mask.")
    if patient_disturbance_mask.shape != targets.shape:
        raise ValueError("patient_disturbance_mask must match targets.")
    if len(sample_ids) != targets.shape[0]:
        raise ValueError("sample_ids must match the batch size.")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")

    names = tuple(phase_names)
    boundary_steps = round(boundary_tolerance_seconds * sampling_rate_hz)
    accumulators = {
        "patient_present": _empty_phase_accumulator(),
        "operation": _empty_phase_accumulator(),
    }

    for batch_index, sample_id in enumerate(sample_ids):
        sample_targets = targets[batch_index]
        sample_predictions = predictions[batch_index]
        sample_time_mask = time_mask[batch_index]
        sample_view_mask = view_mask[batch_index]
        sample_occlusion_mask = occlusion_mask[batch_index]
        sample_disturbance = patient_disturbance_mask[batch_index]

        product = product_quality_metrics(
            targets=sample_targets,
            predictions=sample_predictions,
            time_mask=sample_time_mask,
            phase_names=names,
            sampling_rate_hz=sampling_rate_hz,
            occlusion_mask=sample_occlusion_mask,
        )

        for phase_name in ("patient_present", "operation"):
            phase_id = _phase_id(names, phase_name)
            accumulator = accumulators[phase_name]
            target_segments = _phase_segment_count(
                sample_targets,
                torch.ones_like(sample_time_mask),
                phase_id,
            )
            predicted_segments = _phase_segment_count(
                sample_predictions,
                sample_time_mask,
                phase_id,
            )
            accumulator["ground_truth_segments"] += target_segments
            accumulator["predicted_segments"] += predicted_segments
            accumulator["samples_with_fragmentation"] += int(
                predicted_segments > target_segments
            )

            missed, false_positive = _phase_error_intervals(
                sample_id=str(sample_id),
                phase_name=phase_name,
                targets=sample_targets,
                predictions=sample_predictions,
                time_mask=sample_time_mask,
                view_mask=sample_view_mask,
                occlusion_mask=sample_occlusion_mask,
                patient_disturbance_mask=sample_disturbance,
                phase_names=names,
                sampling_rate_hz=sampling_rate_hz,
                boundary_tolerance_steps=boundary_steps,
            )
            accumulator["missed_intervals"] += len(missed)
            accumulator["false_positive_intervals"] += len(false_positive)
            accumulator["missed_duration_seconds"] += sum(
                interval.duration_seconds for interval in missed
            )
            accumulator["false_positive_duration_seconds"] += sum(
                interval.duration_seconds for interval in false_positive
            )
            _record_causes(accumulator, missed + false_positive)

            phase_ground_truth = sample_targets == phase_id
            valid_ground_truth = phase_ground_truth & sample_time_mask
            correct = valid_ground_truth & (sample_predictions == phase_id)
            accumulator["valid_ground_truth_duration_seconds"] += (
                valid_ground_truth.sum().item() / sampling_rate_hz
            )
            accumulator["correct_duration_seconds"] += (
                correct.sum().item() / sampling_rate_hz
            )
            accumulator["all_views_unavailable_duration_seconds"] += (
                (phase_ground_truth & ~sample_time_mask).sum().item()
                / sampling_rate_hz
            )

            product_phase = product[phase_name]
            start_key = (
                "mean_start_confirmation_delay_seconds"
                if phase_name == "patient_present"
                else "start_delay_seconds"
            )
            end_key = (
                "mean_end_delay_seconds"
                if phase_name == "patient_present"
                else "end_delay_seconds"
            )
            if product_phase[start_key] is not None:
                accumulator["start_delays"].append(product_phase[start_key])
            if product_phase[end_key] is not None:
                accumulator["end_delays"].append(product_phase[end_key])

            missed_valid = valid_ground_truth & (sample_predictions != phase_id)
            if phase_name == "operation":
                accumulator["occluded_missed_duration_seconds"] += (
                    (missed_valid & sample_occlusion_mask.any(dim=0)).sum().item()
                    / sampling_rate_hz
                )
            else:
                accumulator["disturbance_missed_duration_seconds"] += (
                    (missed_valid & sample_disturbance).sum().item()
                    / sampling_rate_hz
                )

    return {
        phase_name: _finalize_phase_accumulator(
            accumulator,
            samples=targets.shape[0],
        )
        for phase_name, accumulator in accumulators.items()
    }


def compare_error_versions(
    *,
    raw: dict[str, Any],
    cleaned: dict[str, Any],
) -> dict[str, Any]:
    """Summarize what post-processing fixed and what latency it introduced."""
    return {
        "patient_present_fragments_removed": (
            raw["patient_present"]["extra_fragments"]
            - cleaned["patient_present"]["extra_fragments"]
        ),
        "patient_present_missed_seconds_change": round(
            cleaned["patient_present"]["missed_duration_seconds"]
            - raw["patient_present"]["missed_duration_seconds"],
            6,
        ),
        "operation_fragments_removed": (
            raw["operation"]["extra_fragments"]
            - cleaned["operation"]["extra_fragments"]
        ),
        "operation_coverage_change": round(
            cleaned["operation"]["coverage_ratio"]
            - raw["operation"]["coverage_ratio"],
            6,
        ),
        "operation_start_delay_change_seconds": round(
            (cleaned["operation"]["mean_start_delay_seconds"] or 0.0)
            - (raw["operation"]["mean_start_delay_seconds"] or 0.0),
            6,
        ),
    }


def format_error_analysis_report(report: dict[str, Any]) -> str:
    """Render a short human-readable breakdown for the take-home deliverable."""
    raw = report["raw_predictions"]
    cleaned = report["cleaned_predictions"]
    tradeoff = report["postprocessing_effect"]

    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"

    raw_patient = raw["patient_present"]
    clean_patient = cleaned["patient_present"]
    raw_operation = raw["operation"]
    clean_operation = cleaned["operation"]

    patient_main_cause = max(
        raw_patient["cause_breakdown"].items(),
        key=lambda item: item[1]["duration_seconds"],
        default=("none", {"duration_seconds": 0.0}),
    )[0]
    operation_main_cause = max(
        raw_operation["cause_breakdown"].items(),
        key=lambda item: item[1]["duration_seconds"],
        default=("none", {"duration_seconds": 0.0}),
    )[0]

    return "\n".join(
        [
            "MOCK ERROR ANALYSIS",
            "=" * 72,
            f"Validation sequences: {report['samples_evaluated']}",
            "",
            "Patient Present",
            "-" * 72,
            (
                f"Raw: {raw_patient['predicted_segments']} predicted segments for "
                f"{raw_patient['ground_truth_segments']} true segments; "
                f"{raw_patient['extra_fragments']} extra fragments, "
                f"{raw_patient['missed_duration_seconds']:.1f}s missed, and "
                f"{raw_patient['false_positive_duration_seconds']:.1f}s false activation."
            ),
            (
                f"Main raw failure source: {patient_main_cause.replace('_', ' ')}. "
                f"Disturbance-attributed share of missed time: "
                f"{percent(raw_patient.get('disturbance_share_of_missed_duration'))}."
            ),
            (
                f"Cleaned: {clean_patient['extra_fragments']} extra fragments and "
                f"{clean_patient['mean_start_delay_seconds']:.1f}s mean confirmation delay."
            ),
            "",
            "Operation",
            "-" * 72,
            (
                f"Raw: coverage {percent(raw_operation['coverage_ratio'])}, "
                f"{raw_operation['predicted_segments']} predicted segments for "
                f"{raw_operation['ground_truth_segments']} true segments, and "
                f"{raw_operation['missed_duration_seconds']:.1f}s missed."
            ),
            (
                f"Main raw failure source: {operation_main_cause.replace('_', ' ')}. "
                f"Occlusion-attributed share of missed time: "
                f"{percent(raw_operation.get('occlusion_share_of_missed_duration'))}."
            ),
            (
                f"Cleaned: coverage {percent(clean_operation['coverage_ratio'])}, "
                f"{clean_operation['extra_fragments']} extra fragments, "
                f"start delay {clean_operation['mean_start_delay_seconds']:.1f}s, "
                f"end delay {clean_operation['mean_end_delay_seconds']:.1f}s."
            ),
            "",
            "Post-processing effect",
            "-" * 72,
            (
                f"Removed {tradeoff['patient_present_fragments_removed']} Patient Present "
                f"fragments and {tradeoff['operation_fragments_removed']} Operation fragments."
            ),
            (
                f"Operation coverage change: {tradeoff['operation_coverage_change']:+.3f}; "
                f"start-delay change: "
                f"{tradeoff['operation_start_delay_change_seconds']:+.1f}s."
            ),
            "Interpretation: smoothing improves timeline stability but can delay true boundaries.",
        ]
    )

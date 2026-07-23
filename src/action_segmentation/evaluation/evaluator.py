from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import torch

from action_segmentation.config import AppConfig
from action_segmentation.data.dataset import DatasetSplit, create_dataloader
from action_segmentation.data.ingestion import ingest_precomputed_feature_batch
from action_segmentation.evaluation.frame_metrics import frame_classification_metrics
from action_segmentation.evaluation.product_metrics import product_quality_metrics
from action_segmentation.evaluation.segment_metrics import segment_metric_stack
from action_segmentation.models.fusion import build_fusion_layer
from action_segmentation.postprocessing.timeline import (
    UNAVAILABLE_LABEL_ID,
    generate_batch_timelines,
    labels_to_timeline,
)
from action_segmentation.training.losses import masked_cross_entropy
from action_segmentation.training.trainer import load_model_checkpoint, resolve_device

PredictionVersion = Literal["raw", "cleaned"]


@dataclass(frozen=True)
class _SampleEvaluation:
    sample_id: str
    targets: torch.Tensor
    timestamps: torch.Tensor
    view_mask: torch.Tensor
    time_mask: torch.Tensor
    raw_predictions: torch.Tensor
    cleaned_predictions: torch.Tensor
    raw_probabilities: torch.Tensor
    smoothed_probabilities: torch.Tensor
    occlusion_mask: torch.Tensor
    available_view_count: torch.Tensor
    segment_results: dict[str, dict[str, Any]]
    product_results: dict[str, dict[str, Any]]


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _aggregate_segment_metrics(
    samples: Sequence[_SampleEvaluation],
    version: PredictionVersion,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    results = [sample.segment_results[version] for sample in samples]
    ground_truth_segments = sum(item["ground_truth_segments"] for item in results)
    predicted_segments = sum(item["predicted_segments"] for item in results)
    edit_scores = [item["edit_score"]["score"] for item in results]

    segmental: dict[str, Any] = {}
    for threshold in thresholds:
        key = f"f1_at_{int(round(threshold * 100))}"
        true_positive = sum(item["segmental_f1"][key]["true_positive"] for item in results)
        false_positive = sum(item["segmental_f1"][key]["false_positive"] for item in results)
        false_negative = sum(item["segmental_f1"][key]["false_negative"] for item in results)
        precision = _safe_ratio(true_positive, true_positive + false_positive) or 0.0
        recall = _safe_ratio(true_positive, true_positive + false_negative) or 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        segmental[key] = {
            "iou_threshold": threshold,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }

    matched = sum(item["boundaries"]["matched_segments"] for item in results)

    def weighted_boundary(name: str) -> float | None:
        values = [
            (item["boundaries"][name], item["boundaries"]["matched_segments"])
            for item in results
            if item["boundaries"][name] is not None
            and item["boundaries"]["matched_segments"] > 0
        ]
        denominator = sum(weight for _, weight in values)
        if denominator == 0:
            return None
        return round(sum(float(value) * weight for value, weight in values) / denominator, 6)

    return {
        "ground_truth_segments": ground_truth_segments,
        "predicted_segments": predicted_segments,
        "mean_edit_score": round(sum(edit_scores) / len(edit_scores), 6),
        "segmental_f1": segmental,
        "boundaries": {
            "matched_segments": matched,
            "mean_start_error_seconds": weighted_boundary("mean_start_error_seconds"),
            "mean_end_error_seconds": weighted_boundary("mean_end_error_seconds"),
            "mean_absolute_start_error_seconds": weighted_boundary(
                "mean_absolute_start_error_seconds"
            ),
            "mean_absolute_end_error_seconds": weighted_boundary(
                "mean_absolute_end_error_seconds"
            ),
            "start_within_tolerance_ratio": weighted_boundary(
                "start_within_tolerance_ratio"
            ),
            "end_within_tolerance_ratio": weighted_boundary(
                "end_within_tolerance_ratio"
            ),
        },
    }


def _aggregate_product_metrics(
    samples: Sequence[_SampleEvaluation],
    version: PredictionVersion,
    config: AppConfig,
) -> dict[str, Any]:
    patient_id = config.phases.names.index("patient_present")
    operation_id = config.phases.names.index("operation")
    sample_metrics = [sample.product_results[version] for sample in samples]

    patient_gt_seconds = sum(
        float(((sample.targets == patient_id) & sample.time_mask).sum().item())
        / config.data.sampling_rate_hz
        for sample in samples
    )
    operation_gt_seconds = sum(
        float(((sample.targets == operation_id) & sample.time_mask).sum().item())
        / config.data.sampling_rate_hz
        for sample in samples
    )

    patient = [item["patient_present"] for item in sample_metrics]
    operation = [item["operation"] for item in sample_metrics]

    def mean_present(items: Sequence[dict[str, Any]], key: str) -> float | None:
        values = [float(item[key]) for item in items if item[key] is not None]
        return round(sum(values) / len(values), 6) if values else None

    occluded_seconds = sum(float(item["occluded_operation_duration_seconds"]) for item in operation)
    occluded_correct_seconds = sum(
        float(item["occluded_operation_duration_seconds"])
        * float(item["coverage_during_occlusion_ratio"])
        for item in operation
        if item["coverage_during_occlusion_ratio"] is not None
    )
    recovery_windows = sum(int(item["occlusion_windows_evaluated"]) for item in operation)
    weighted_recovery = sum(
        float(item["mean_occlusion_recovery_seconds"])
        * int(item["occlusion_windows_evaluated"])
        for item in operation
        if item["mean_occlusion_recovery_seconds"] is not None
    )

    patient_switches = sum(int(item["state_switches_inside_phase"]) for item in patient)
    patient_hours = patient_gt_seconds / 3600.0
    operation_missed = sum(float(item["missed_duration_seconds"]) for item in operation)

    return {
        "patient_present": {
            "ground_truth_segments": sum(int(item["ground_truth_segments"]) for item in patient),
            "predicted_segments": sum(int(item["predicted_segments"]) for item in patient),
            "extra_fragments": sum(int(item["extra_fragments"]) for item in patient),
            "fragmentation_ratio": round(
                sum(int(item["predicted_segments"]) for item in patient)
                / max(1, sum(int(item["ground_truth_segments"]) for item in patient)),
                6,
            ),
            "false_positive_duration_seconds": round(
                sum(float(item["false_positive_duration_seconds"]) for item in patient), 6
            ),
            "missed_duration_seconds": round(
                sum(float(item["missed_duration_seconds"]) for item in patient), 6
            ),
            "state_switches_inside_phase": patient_switches,
            "state_switches_per_hour": round(
                patient_switches / patient_hours if patient_hours else 0.0, 6
            ),
            "mean_start_confirmation_delay_seconds": mean_present(
                patient, "mean_start_confirmation_delay_seconds"
            ),
            "mean_end_delay_seconds": mean_present(patient, "mean_end_delay_seconds"),
        },
        "operation": {
            "ground_truth_segments": sum(int(item["ground_truth_segments"]) for item in operation),
            "predicted_segments": sum(int(item["predicted_segments"]) for item in operation),
            "start_delay_seconds": mean_present(operation, "start_delay_seconds"),
            "end_delay_seconds": mean_present(operation, "end_delay_seconds"),
            "coverage_ratio": round(
                max(0.0, operation_gt_seconds - operation_missed) / operation_gt_seconds,
                6,
            )
            if operation_gt_seconds
            else 0.0,
            "missed_duration_seconds": round(operation_missed, 6),
            "false_positive_duration_seconds": round(
                sum(float(item["false_positive_duration_seconds"]) for item in operation), 6
            ),
            "occluded_operation_duration_seconds": round(occluded_seconds, 6),
            "coverage_during_occlusion_ratio": round(
                occluded_correct_seconds / occluded_seconds, 6
            )
            if occluded_seconds
            else None,
            "mean_occlusion_recovery_seconds": round(
                weighted_recovery / recovery_windows, 6
            )
            if recovery_windows
            else None,
            "occlusion_windows_evaluated": recovery_windows,
        },
    }


def _version_metrics(
    samples: Sequence[_SampleEvaluation],
    version: PredictionVersion,
    config: AppConfig,
) -> dict[str, Any]:
    targets = torch.cat([sample.targets for sample in samples])
    predictions = torch.cat(
        [
            sample.raw_predictions if version == "raw" else sample.cleaned_predictions
            for sample in samples
        ]
    )
    time_mask = torch.cat([sample.time_mask for sample in samples])
    return {
        "frame_metrics": frame_classification_metrics(
            targets, predictions, time_mask, config.phases.names
        ),
        "segment_metrics": _aggregate_segment_metrics(
            samples, version, config.evaluation.segment_iou_thresholds
        ),
        "product_metrics": _aggregate_product_metrics(samples, version, config),
    }



def _boolean_intervals(
    values: torch.Tensor,
    timestamps: torch.Tensor,
    sampling_rate_hz: float,
) -> list[dict[str, Any]]:
    """Convert one boolean sequence to exclusive-end availability intervals."""
    if values.ndim != 1 or timestamps.ndim != 1 or values.shape != timestamps.shape:
        raise ValueError("Availability values and timestamps must share shape [time].")
    if values.dtype != torch.bool:
        raise TypeError("Availability values must use torch.bool.")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")
    if values.numel() == 0:
        return []

    intervals: list[dict[str, Any]] = []
    start = 0
    current = bool(values[0].item())
    final_step = 1.0 / sampling_rate_hz
    for index in range(1, values.numel() + 1):
        changed = index == values.numel() or bool(values[index].item()) != current
        if not changed:
            continue
        start_seconds = float(timestamps[start].item())
        end_seconds = (
            float(timestamps[index].item())
            if index < timestamps.numel()
            else float(timestamps[-1].item()) + final_step
        )
        intervals.append(
            {
                "start_index": start,
                "end_index": index,
                "start_seconds": round(start_seconds, 6),
                "end_seconds": round(end_seconds, 6),
                "duration_seconds": round(end_seconds - start_seconds, 6),
                "available": current,
            }
        )
        if index < values.numel():
            start = index
            current = bool(values[index].item())
    return intervals


def _count_intervals(
    values: torch.Tensor,
    timestamps: torch.Tensor,
    sampling_rate_hz: float,
) -> list[dict[str, Any]]:
    """Convert available-view counts into exclusive-end constant intervals."""
    if values.ndim != 1 or timestamps.ndim != 1 or values.shape != timestamps.shape:
        raise ValueError("View counts and timestamps must share shape [time].")
    if values.numel() == 0:
        return []
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")

    intervals: list[dict[str, Any]] = []
    start = 0
    current = int(values[0].item())
    final_step = 1.0 / sampling_rate_hz
    for index in range(1, values.numel() + 1):
        changed = index == values.numel() or int(values[index].item()) != current
        if not changed:
            continue
        start_seconds = float(timestamps[start].item())
        end_seconds = (
            float(timestamps[index].item())
            if index < timestamps.numel()
            else float(timestamps[-1].item()) + final_step
        )
        intervals.append(
            {
                "start_seconds": round(start_seconds, 6),
                "end_seconds": round(end_seconds, 6),
                "available_views": current,
            }
        )
        if index < values.numel():
            start = index
            current = int(values[index].item())
    return intervals


def _sample_visualization_payload(
    sample: _SampleEvaluation,
    config: AppConfig,
) -> dict[str, Any]:
    """Serialize aligned model, ground-truth, camera, and confidence tracks."""
    target_labels = sample.targets.clone()
    target_labels[~sample.time_mask] = UNAVAILABLE_LABEL_ID
    ground_truth_probabilities = torch.nn.functional.one_hot(
        sample.targets.clamp_min(0),
        num_classes=len(config.phases.names),
    ).to(torch.float32)
    ground_truth_probabilities[~sample.time_mask] = 0.0

    ground_truth_segments = labels_to_timeline(
        target_labels,
        ground_truth_probabilities,
        sample.timestamps,
        config.phases.names,
        config.data.sampling_rate_hz,
    )
    raw_segments = labels_to_timeline(
        sample.raw_predictions,
        sample.raw_probabilities,
        sample.timestamps,
        config.phases.names,
        config.data.sampling_rate_hz,
    )
    cleaned_segments = labels_to_timeline(
        sample.cleaned_predictions,
        sample.smoothed_probabilities,
        sample.timestamps,
        config.phases.names,
        config.data.sampling_rate_hz,
    )

    final_step = 1.0 / config.data.sampling_rate_hz
    duration = (
        float(sample.timestamps[-1].item()) + final_step
        if sample.timestamps.numel()
        else 0.0
    )
    camera_availability = [
        {
            "camera_index": camera_index,
            "intervals": _boolean_intervals(
                sample.view_mask[camera_index],
                sample.timestamps,
                config.data.sampling_rate_hz,
            ),
        }
        for camera_index in range(sample.view_mask.shape[0])
    ]

    return {
        "sample_id": sample.sample_id,
        "duration_seconds": round(duration, 6),
        "phase_names": list(config.phases.names),
        "segments": [segment.to_dict() for segment in cleaned_segments],
        "tracks": {
            "ground_truth": [segment.to_dict() for segment in ground_truth_segments],
            "raw_prediction": [segment.to_dict() for segment in raw_segments],
            "cleaned_prediction": [segment.to_dict() for segment in cleaned_segments],
        },
        "camera_availability": camera_availability,
        "available_view_count": _count_intervals(
            sample.available_view_count,
            sample.timestamps,
            config.data.sampling_rate_hz,
        ),
        "confidence": {
            "timestamps_seconds": [
                round(float(value), 6) for value in sample.timestamps.tolist()
            ],
            "raw_max_probability": [
                round(float(value), 6)
                for value in sample.raw_probabilities.max(dim=-1).values.tolist()
            ],
            "smoothed_max_probability": [
                round(float(value), 6)
                for value in sample.smoothed_probabilities.max(dim=-1).values.tolist()
            ],
        },
    }

def evaluate_checkpoint(
    config: AppConfig,
    *,
    checkpoint_path: str | Path,
    split: DatasetSplit = "test",
) -> dict[str, Any]:
    """Evaluate a restored model on a complete synthetic dataset split."""
    device = resolve_device(config.training.device)
    model, checkpoint = load_model_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
    )
    fusion_layer = build_fusion_layer(config).to(device)
    loader = create_dataloader(config, split=split, shuffle=False)

    samples: list[_SampleEvaluation] = []
    total_loss = 0.0
    total_valid_timestamps = 0

    with torch.no_grad():
        for raw_batch in loader:
            ingested = ingest_precomputed_feature_batch(raw_batch, config)
            features = ingested.features.to(device)
            labels_device = ingested.labels.to(device)
            view_mask = ingested.view_mask.to(device)
            fused = fusion_layer(features, view_mask)
            output = model(fused.fused_features, fused.time_mask)
            loss = masked_cross_entropy(output.logits, labels_device, output.time_mask)

            valid_count = int(output.time_mask.sum().item())
            total_loss += float(loss.item()) * valid_count
            total_valid_timestamps += valid_count

            logits_cpu = output.logits.cpu()
            timelines = generate_batch_timelines(
                logits=logits_cpu,
                timestamps=ingested.timestamps,
                time_mask=ingested.time_mask,
                phase_names=config.phases.names,
                sampling_rate_hz=config.data.sampling_rate_hz,
                smoothing_window_seconds=config.postprocessing.smoothing_window_seconds,
                min_segment_duration_seconds=config.postprocessing.min_segment_duration_seconds,
            )

            for index, timeline in enumerate(timelines):
                targets = ingested.labels[index].cpu()
                time_mask_cpu = ingested.time_mask[index].cpu()
                occlusion = ingested.metadata["occlusion_mask"][index].cpu()
                available_count = fused.available_view_count[index].cpu()
                versions = {
                    "raw": timeline.raw_predictions.cpu(),
                    "cleaned": timeline.cleaned_predictions.cpu(),
                }
                segment_results = {
                    name: segment_metric_stack(
                        targets=targets,
                        predictions=predictions,
                        time_mask=time_mask_cpu,
                        iou_thresholds=config.evaluation.segment_iou_thresholds,
                        sampling_rate_hz=config.data.sampling_rate_hz,
                        boundary_tolerance_seconds=(
                            config.evaluation.boundary_tolerance_seconds
                        ),
                    )
                    for name, predictions in versions.items()
                }
                product_results = {
                    name: product_quality_metrics(
                        targets=targets,
                        predictions=predictions,
                        time_mask=time_mask_cpu,
                        phase_names=config.phases.names,
                        sampling_rate_hz=config.data.sampling_rate_hz,
                        occlusion_mask=occlusion,
                    )
                    for name, predictions in versions.items()
                }
                samples.append(
                    _SampleEvaluation(
                        sample_id=ingested.sample_ids[index],
                        targets=targets,
                        timestamps=ingested.timestamps[index].cpu(),
                        view_mask=ingested.view_mask[index].cpu(),
                        time_mask=time_mask_cpu,
                        raw_predictions=versions["raw"],
                        cleaned_predictions=versions["cleaned"],
                        raw_probabilities=timeline.probabilities.cpu(),
                        smoothed_probabilities=timeline.smoothed_probabilities.cpu(),
                        occlusion_mask=occlusion,
                        available_view_count=available_count,
                        segment_results=segment_results,
                        product_results=product_results,
                    )
                )

    if total_valid_timestamps == 0:
        raise RuntimeError("Evaluation split contains no valid timestamps.")

    raw_metrics = _version_metrics(samples, "raw", config)
    cleaned_metrics = _version_metrics(samples, "cleaned", config)

    return {
        "status": "trained_model_evaluation_complete",
        "split": split,
        "device": str(device),
        "checkpoint": {
            "path": str(Path(checkpoint_path).expanduser().resolve()),
            "best_epoch": int(checkpoint.get("epoch", 0)),
            "validation_loss": round(float(checkpoint.get("validation_loss", 0.0)), 6),
        },
        "samples_evaluated": len(samples),
        "valid_timestamps": total_valid_timestamps,
        "mean_cross_entropy_loss": round(total_loss / total_valid_timestamps, 6),
        "raw_predictions": raw_metrics,
        "cleaned_predictions": cleaned_metrics,
        "postprocessing_effect": {
            "frame_accuracy_change": round(
                cleaned_metrics["frame_metrics"]["accuracy"]
                - raw_metrics["frame_metrics"]["accuracy"],
                6,
            ),
            "mean_edit_score_change": round(
                cleaned_metrics["segment_metrics"]["mean_edit_score"]
                - raw_metrics["segment_metrics"]["mean_edit_score"],
                6,
            ),
            "patient_present_fragments_removed": (
                raw_metrics["product_metrics"]["patient_present"]["extra_fragments"]
                - cleaned_metrics["product_metrics"]["patient_present"]["extra_fragments"]
            ),
            "operation_fragments_removed": (
                raw_metrics["product_metrics"]["operation"]["predicted_segments"]
                - cleaned_metrics["product_metrics"]["operation"]["predicted_segments"]
            ),
        },
        "timelines": [
            _sample_visualization_payload(sample, config) for sample in samples
        ],
    }

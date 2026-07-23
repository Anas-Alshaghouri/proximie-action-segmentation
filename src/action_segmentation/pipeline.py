from __future__ import annotations

from typing import Any

import torch

from action_segmentation.config import AppConfig
from action_segmentation.data.dataset import create_dataloader
from action_segmentation.data.ingestion import ingest_precomputed_feature_batch
from action_segmentation.evaluation.error_analysis import (
    analyze_error_version,
    build_mock_noisy_logits,
    compare_error_versions,
)
from action_segmentation.evaluation.frame_metrics import frame_classification_metrics
from action_segmentation.evaluation.product_metrics import product_quality_metrics
from action_segmentation.evaluation.segment_metrics import segment_metric_stack
from action_segmentation.models.fusion import build_fusion_layer
from action_segmentation.models.temporal_tcn import build_temporal_model
from action_segmentation.postprocessing.timeline import (
    UNAVAILABLE_LABEL_ID,
    generate_batch_timelines,
    generate_timeline,
)
from action_segmentation.training.losses import masked_cross_entropy


def repository_readiness_summary(config: AppConfig) -> dict[str, Any]:
    """Return a compact, machine-readable summary of the frozen contract."""
    return {
        "project": config.project.name,
        "seed": config.project.seed,
        "phases": list(config.phases.names),
        "num_classes": config.model.num_classes,
        "sampling_rate_hz": config.data.sampling_rate_hz,
        "sequence_length": config.data.sequence_length,
        "sequence_duration_seconds": (
            config.data.sequence_length / config.data.sampling_rate_hz
        ),
        "feature_dim": config.data.feature_dim,
        "supported_views": {
            "minimum": config.data.min_views,
            "maximum": config.data.max_views,
        },
        "fusion_type": config.fusion.type,
        "model_type": config.model.type,
        "causal": config.model.causal,
    }


def _load_validation_batch(config: AppConfig) -> dict[str, Any]:
    validation_loader = create_dataloader(
        config=config,
        split="validation",
        shuffle=False,
    )
    return next(iter(validation_loader))


def synthetic_data_readiness_summary(
    config: AppConfig,
    batch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize the deterministic synthetic multi-view data contract."""
    batch = _load_validation_batch(config) if batch is None else batch

    first_sample_durations = batch["phase_durations"][0].tolist()
    duration_seconds = {
        phase_name: round(duration / config.data.sampling_rate_hz, 3)
        for phase_name, duration in zip(
            config.phases.names,
            first_sample_durations,
            strict=True,
        )
    }

    view_mask = batch["view_mask"]
    time_mask = batch["time_mask"]
    occlusion_mask = batch["occlusion_mask"]
    disturbance_mask = batch["patient_present_disturbance_mask"]

    return {
        "status": "synthetic_data_ready",
        "dataset_sizes": {
            "train": config.data.train_samples,
            "validation": config.data.validation_samples,
            "test": config.data.test_samples,
        },
        "batch_shapes": {
            "features": list(batch["features"].shape),
            "labels": list(batch["labels"].shape),
            "timestamps": list(batch["timestamps"].shape),
            "view_mask": list(view_mask.shape),
            "time_mask": list(time_mask.shape),
        },
        "first_batch_sample_ids": list(batch["sample_id"]),
        "physical_views_per_sample": batch["num_views"].tolist(),
        "first_sample_phase_durations_seconds": duration_seconds,
        "valid_view_timestamp_ratio": round(view_mask.float().mean().item(), 4),
        "fully_missing_timestamps": int((~time_mask).sum().item()),
        "operation_occluded_view_timestamps": int(occlusion_mask.sum().item()),
        "patient_present_disturbance_timestamps": int(disturbance_mask.sum().item()),
        "invalid_features_are_zero": bool(
            torch.all(batch["features"][~view_mask] == 0).item()
        ),
    }


def fusion_readiness_summary(
    config: AppConfig,
    batch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate ingestion and fuse one multi-view validation batch."""
    batch = _load_validation_batch(config) if batch is None else batch
    ingested = ingest_precomputed_feature_batch(batch, config)
    fusion_layer = build_fusion_layer(config)
    output = fusion_layer(ingested.features, ingested.view_mask)

    expected_time_mask = ingested.view_mask.any(dim=1)
    fully_missing = ~output.time_mask
    single_view_timestamps = output.available_view_count == 1

    if single_view_timestamps.any():
        single_view_reference = (
            ingested.features
            * ingested.view_mask.unsqueeze(-1).to(ingested.features.dtype)
        ).sum(dim=1)
        single_view_identity = torch.allclose(
            output.fused_features[single_view_timestamps],
            single_view_reference[single_view_timestamps],
        )
    else:
        single_view_identity = True

    return {
        "status": "multi_view_fusion_ready",
        "ingestion_batch_size": ingested.batch_size,
        "input_shape": list(ingested.features.shape),
        "output_shape": list(output.fused_features.shape),
        "fusion_type": config.fusion.type,
        "available_view_count_range": [
            int(output.available_view_count.min().item()),
            int(output.available_view_count.max().item()),
        ],
        "single_view_timestamps": int(single_view_timestamps.sum().item()),
        "fully_missing_timestamps": int(fully_missing.sum().item()),
        "time_mask_preserved": bool(
            torch.equal(output.time_mask, expected_time_mask)
        ),
        "single_view_features_preserved": bool(single_view_identity),
        "fully_missing_features_are_zero": bool(
            torch.all(output.fused_features[fully_missing] == 0).item()
        ),
        "labels_preserved_by_ingestion": bool(
            torch.equal(ingested.labels, batch["labels"])
        ),
        "timestamps_preserved_by_ingestion": bool(
            torch.equal(ingested.timestamps, batch["timestamps"])
        ),
    }


def temporal_model_readiness_summary(
    config: AppConfig,
    batch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one fused validation batch through the causal temporal model."""
    batch = _load_validation_batch(config) if batch is None else batch
    ingested = ingest_precomputed_feature_batch(batch, config)
    fusion_layer = build_fusion_layer(config)
    fusion_output = fusion_layer(ingested.features, ingested.view_mask)

    model = build_temporal_model(config)
    model.eval()

    with torch.no_grad():
        output = model(
            fusion_output.fused_features,
            fusion_output.time_mask,
        )
        loss = masked_cross_entropy(
            output.logits,
            ingested.labels,
            output.time_mask,
        )

        cutoff = config.data.sequence_length // 2
        future_modified = fusion_output.fused_features.clone()
        future_modified[:, cutoff + 1 :] = (
            torch.randn_like(future_modified[:, cutoff + 1 :]) * 100.0
        )
        modified_output = model(future_modified, fusion_output.time_mask)
        past_difference = (
            output.logits[:, : cutoff + 1]
            - modified_output.logits[:, : cutoff + 1]
        ).abs()

    valid_probabilities = torch.softmax(output.logits[output.time_mask], dim=-1)
    probability_sum_error = (
        valid_probabilities.sum(dim=-1) - 1.0
    ).abs().max()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    invalid_timestamps = ~output.time_mask

    return {
        "status": "causal_temporal_model_ready",
        "model_type": config.model.type,
        "input_shape": list(fusion_output.fused_features.shape),
        "logit_shape": list(output.logits.shape),
        "prediction_shape": list(output.logits.argmax(dim=-1).shape),
        "parameter_count": parameter_count,
        "temporal_blocks": len(config.model.dilations),
        "dilations": list(config.model.dilations),
        "receptive_field_steps": model.receptive_field_steps,
        "receptive_field_seconds": round(
            model.receptive_field_steps / config.data.sampling_rate_hz,
            3,
        ),
        "masked_cross_entropy": round(float(loss.item()), 6),
        "valid_probability_sum_max_error": round(
            float(probability_sum_error.item()),
            8,
        ),
        "invalid_logits_are_zero": bool(
            invalid_timestamps.numel() == 0
            or torch.all(output.logits[invalid_timestamps] == 0).item()
        ),
        "causality_check": {
            "future_change_start_index": cutoff + 1,
            "past_output_max_abs_difference": round(
                float(past_difference.max().item()),
                10,
            ),
            "passed": bool(torch.allclose(
                output.logits[:, : cutoff + 1],
                modified_output.logits[:, : cutoff + 1],
                atol=1e-6,
                rtol=0.0,
            )),
        },
    }



def _count_available_segments(predictions: torch.Tensor) -> int:
    """Count contiguous phase segments while excluding unavailable intervals."""
    if predictions.numel() == 0:
        return 0
    count = 0
    previous: int | None = None
    for value in predictions.tolist():
        label = int(value)
        if label == UNAVAILABLE_LABEL_ID:
            previous = None
            continue
        if previous != label:
            count += 1
            previous = label
    return count


def _controlled_noisy_logits(
    labels: torch.Tensor,
    time_mask: torch.Tensor,
    num_classes: int,
    phase_names: tuple[str, ...],
) -> torch.Tensor:
    """Create understandable short false-positive islands for the controlled demo."""
    logits = torch.full(
        (labels.shape[0], num_classes),
        -3.0,
        dtype=torch.float32,
        device=labels.device,
    )
    logits.scatter_(1, labels.unsqueeze(-1), 4.0)

    corruption_plan = (
        ("patient_present", "preparation", 3, 0.45),
        ("operation", "closing", 4, 0.55),
    )
    for source_name, false_name, duration, relative_position in corruption_plan:
        if source_name not in phase_names or false_name not in phase_names:
            continue
        source_id = phase_names.index(source_name)
        false_id = phase_names.index(false_name)
        indices = torch.nonzero(labels == source_id, as_tuple=False).flatten()
        if indices.numel() <= duration:
            continue
        source_start = int(indices[0].item())
        source_end = int(indices[-1].item()) + 1
        start = source_start + int((source_end - source_start - duration) * relative_position)
        end = start + duration
        logits[start:end] = -3.0
        logits[start:end, false_id] = 7.0

    return logits.masked_fill(~time_mask.unsqueeze(-1), 0.0)


def timeline_readiness_summary(
    config: AppConfig,
    batch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate causal smoothing, debouncing, and timeline conversion."""
    batch = _load_validation_batch(config) if batch is None else batch
    ingested = ingest_precomputed_feature_batch(batch, config)
    fusion_layer = build_fusion_layer(config)
    fusion_output = fusion_layer(ingested.features, ingested.view_mask)

    model = build_temporal_model(config)
    model.eval()
    with torch.no_grad():
        model_output = model(
            fusion_output.fused_features,
            fusion_output.time_mask,
        )
        model_timelines = generate_batch_timelines(
            logits=model_output.logits,
            timestamps=ingested.timestamps,
            time_mask=model_output.time_mask,
            phase_names=config.phases.names,
            sampling_rate_hz=config.data.sampling_rate_hz,
            smoothing_window_seconds=(
                config.postprocessing.smoothing_window_seconds
            ),
            min_segment_duration_seconds=(
                config.postprocessing.min_segment_duration_seconds
            ),
        )

    first_labels = ingested.labels[0]
    first_timestamps = ingested.timestamps[0]
    first_time_mask = ingested.time_mask[0]
    demo_logits = _controlled_noisy_logits(
        labels=first_labels,
        time_mask=first_time_mask,
        num_classes=config.model.num_classes,
        phase_names=config.phases.names,
    )
    demo = generate_timeline(
        logits=demo_logits,
        timestamps=first_timestamps,
        time_mask=first_time_mask,
        phase_names=config.phases.names,
        sampling_rate_hz=config.data.sampling_rate_hz,
        smoothing_window_seconds=config.postprocessing.smoothing_window_seconds,
        min_segment_duration_seconds=(
            config.postprocessing.min_segment_duration_seconds
        ),
    )

    cutoff = config.data.sequence_length // 2
    future_modified = demo_logits.clone()
    future_modified[cutoff + 1 :] = torch.randn_like(
        future_modified[cutoff + 1 :]
    ) * 100.0
    modified = generate_timeline(
        logits=future_modified,
        timestamps=first_timestamps,
        time_mask=first_time_mask,
        phase_names=config.phases.names,
        sampling_rate_hz=config.data.sampling_rate_hz,
        smoothing_window_seconds=config.postprocessing.smoothing_window_seconds,
        min_segment_duration_seconds=(
            config.postprocessing.min_segment_duration_seconds
        ),
    )

    timeline = [segment.to_dict() for segment in demo.segments]
    unavailable_segments = sum(
        not segment.is_available for segment in demo.segments
    )

    return {
        "status": "timeline_postprocessing_ready",
        "processing_mode": "strictly_causal_with_confirmation_delay",
        "smoothing_window_seconds": (
            config.postprocessing.smoothing_window_seconds
        ),
        "minimum_segment_duration_seconds": (
            config.postprocessing.min_segment_duration_seconds
        ),
        "model_output_accepted": len(model_timelines) == ingested.batch_size,
        "model_batch_timelines_generated": len(model_timelines),
        "controlled_demo": {
            "sample_id": ingested.sample_ids[0],
            "raw_available_segments": _count_available_segments(
                demo.raw_predictions
            ),
            "smoothed_available_segments": _count_available_segments(
                demo.smoothed_predictions
            ),
            "cleaned_available_segments": _count_available_segments(
                demo.cleaned_predictions
            ),
            "unavailable_segments": unavailable_segments,
            "timeline": timeline,
        },
        "invalid_timestamps_preserved_as_unavailable": bool(
            torch.all(
                demo.cleaned_predictions[~first_time_mask]
                == UNAVAILABLE_LABEL_ID
            ).item()
        ),
        "causality_check": {
            "future_change_start_index": cutoff + 1,
            "past_cleaned_predictions_unchanged": bool(
                torch.equal(
                    demo.cleaned_predictions[: cutoff + 1],
                    modified.cleaned_predictions[: cutoff + 1],
                )
            ),
            "past_smoothed_probability_max_abs_difference": round(
                float(
                    (
                        demo.smoothed_probabilities[: cutoff + 1]
                        - modified.smoothed_probabilities[: cutoff + 1]
                    ).abs().max().item()
                ),
                10,
            ),
        },
    }



def _evaluate_prediction_version(
    *,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    time_mask: torch.Tensor,
    occlusion_mask: torch.Tensor,
    config: AppConfig,
) -> dict[str, Any]:
    """Run the complete model- and product-quality metric stack."""
    return {
        "frame_metrics": frame_classification_metrics(
            targets=targets,
            predictions=predictions,
            time_mask=time_mask,
            phase_names=config.phases.names,
        ),
        "segment_metrics": segment_metric_stack(
            targets=targets,
            predictions=predictions,
            time_mask=time_mask,
            iou_thresholds=config.evaluation.segment_iou_thresholds,
            sampling_rate_hz=config.data.sampling_rate_hz,
            boundary_tolerance_seconds=(
                config.evaluation.boundary_tolerance_seconds
            ),
        ),
        "product_metrics": product_quality_metrics(
            targets=targets,
            predictions=predictions,
            time_mask=time_mask,
            phase_names=config.phases.names,
            sampling_rate_hz=config.data.sampling_rate_hz,
            occlusion_mask=occlusion_mask,
        ),
    }


def metric_stack_readiness_summary(
    config: AppConfig,
    batch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate frame, segment, boundary, and product-quality metrics."""
    batch = _load_validation_batch(config) if batch is None else batch
    ingested = ingest_precomputed_feature_batch(batch, config)
    fusion_output = build_fusion_layer(config)(
        ingested.features,
        ingested.view_mask,
    )

    model = build_temporal_model(config)
    model.eval()
    with torch.no_grad():
        model_output = model(
            fusion_output.fused_features,
            fusion_output.time_mask,
        )
        model_timelines = generate_batch_timelines(
            logits=model_output.logits,
            timestamps=ingested.timestamps,
            time_mask=model_output.time_mask,
            phase_names=config.phases.names,
            sampling_rate_hz=config.data.sampling_rate_hz,
            smoothing_window_seconds=(
                config.postprocessing.smoothing_window_seconds
            ),
            min_segment_duration_seconds=(
                config.postprocessing.min_segment_duration_seconds
            ),
        )

    first_labels = ingested.labels[0]
    first_time_mask = ingested.time_mask[0]
    controlled_logits = _controlled_noisy_logits(
        labels=first_labels,
        time_mask=first_time_mask,
        num_classes=config.model.num_classes,
        phase_names=config.phases.names,
    )
    controlled_timeline = generate_timeline(
        logits=controlled_logits,
        timestamps=ingested.timestamps[0],
        time_mask=first_time_mask,
        phase_names=config.phases.names,
        sampling_rate_hz=config.data.sampling_rate_hz,
        smoothing_window_seconds=(
            config.postprocessing.smoothing_window_seconds
        ),
        min_segment_duration_seconds=(
            config.postprocessing.min_segment_duration_seconds
        ),
    )

    raw_metrics = _evaluate_prediction_version(
        targets=first_labels,
        predictions=controlled_timeline.raw_predictions,
        time_mask=first_time_mask,
        occlusion_mask=batch["occlusion_mask"][0],
        config=config,
    )
    cleaned_metrics = _evaluate_prediction_version(
        targets=first_labels,
        predictions=controlled_timeline.cleaned_predictions,
        time_mask=first_time_mask,
        occlusion_mask=batch["occlusion_mask"][0],
        config=config,
    )

    raw_patient = raw_metrics["product_metrics"]["patient_present"]
    cleaned_patient = cleaned_metrics["product_metrics"]["patient_present"]
    raw_operation = raw_metrics["product_metrics"]["operation"]
    cleaned_operation = cleaned_metrics["product_metrics"]["operation"]

    return {
        "status": "dual_metric_stack_ready",
        "evaluation_levels": ["frame", "segment", "product"],
        "segment_iou_thresholds": list(
            config.evaluation.segment_iou_thresholds
        ),
        "boundary_tolerance_seconds": (
            config.evaluation.boundary_tolerance_seconds
        ),
        "untrained_model_outputs_accepted": (
            len(model_timelines) == ingested.batch_size
        ),
        "controlled_demo_sample_id": ingested.sample_ids[0],
        "raw_predictions": raw_metrics,
        "cleaned_predictions": cleaned_metrics,
        "postprocessing_tradeoff": {
            "frame_accuracy_change": round(
                cleaned_metrics["frame_metrics"]["accuracy"]
                - raw_metrics["frame_metrics"]["accuracy"],
                6,
            ),
            "edit_score_change": round(
                cleaned_metrics["segment_metrics"]["edit_score"]["score"]
                - raw_metrics["segment_metrics"]["edit_score"]["score"],
                6,
            ),
            "patient_present_extra_fragments": {
                "raw": raw_patient["extra_fragments"],
                "cleaned": cleaned_patient["extra_fragments"],
            },
            "operation_predicted_segments": {
                "raw": raw_operation["predicted_segments"],
                "cleaned": cleaned_operation["predicted_segments"],
            },
            "operation_start_delay_seconds": {
                "raw": raw_operation["start_delay_seconds"],
                "cleaned": cleaned_operation["start_delay_seconds"],
            },
        },
    }


def error_analysis_readiness_summary(config: AppConfig) -> dict[str, Any]:
    """Run a deterministic noisy validation evaluation and explain failures."""
    sample_ids: list[str] = []
    targets: list[torch.Tensor] = []
    raw_predictions: list[torch.Tensor] = []
    cleaned_predictions: list[torch.Tensor] = []
    time_masks: list[torch.Tensor] = []
    view_masks: list[torch.Tensor] = []
    occlusion_masks: list[torch.Tensor] = []
    disturbance_masks: list[torch.Tensor] = []

    validation_loader = create_dataloader(
        config=config,
        split="validation",
        shuffle=False,
    )

    for batch in validation_loader:
        ingested = ingest_precomputed_feature_batch(batch, config)
        for index, sample_id in enumerate(ingested.sample_ids):
            logits = build_mock_noisy_logits(
                labels=ingested.labels[index],
                time_mask=ingested.time_mask[index],
                view_mask=ingested.view_mask[index],
                occlusion_mask=batch["occlusion_mask"][index],
                patient_disturbance_mask=(
                    batch["patient_present_disturbance_mask"][index]
                ),
                phase_names=config.phases.names,
                boundary_noise_steps=round(
                    config.data.boundary_noise_seconds
                    * config.data.sampling_rate_hz
                ),
            )
            timeline = generate_timeline(
                logits=logits,
                timestamps=ingested.timestamps[index],
                time_mask=ingested.time_mask[index],
                phase_names=config.phases.names,
                sampling_rate_hz=config.data.sampling_rate_hz,
                smoothing_window_seconds=(
                    config.postprocessing.smoothing_window_seconds
                ),
                min_segment_duration_seconds=(
                    config.postprocessing.min_segment_duration_seconds
                ),
            )

            sample_ids.append(str(sample_id))
            targets.append(ingested.labels[index])
            raw_predictions.append(timeline.raw_predictions)
            cleaned_predictions.append(timeline.cleaned_predictions)
            time_masks.append(ingested.time_mask[index])
            view_masks.append(ingested.view_mask[index])
            occlusion_masks.append(batch["occlusion_mask"][index])
            disturbance_masks.append(
                batch["patient_present_disturbance_mask"][index]
            )

    stacked_targets = torch.stack(targets)
    stacked_raw = torch.stack(raw_predictions)
    stacked_cleaned = torch.stack(cleaned_predictions)
    stacked_time_mask = torch.stack(time_masks)
    stacked_view_mask = torch.stack(view_masks)
    stacked_occlusion = torch.stack(occlusion_masks)
    stacked_disturbance = torch.stack(disturbance_masks)

    common = {
        "sample_ids": sample_ids,
        "targets": stacked_targets,
        "time_mask": stacked_time_mask,
        "view_mask": stacked_view_mask,
        "occlusion_mask": stacked_occlusion,
        "patient_disturbance_mask": stacked_disturbance,
        "phase_names": config.phases.names,
        "sampling_rate_hz": config.data.sampling_rate_hz,
        "boundary_tolerance_seconds": (
            config.evaluation.boundary_tolerance_seconds
        ),
    }
    raw_report = analyze_error_version(
        predictions=stacked_raw,
        **common,
    )
    cleaned_report = analyze_error_version(
        predictions=stacked_cleaned,
        **common,
    )

    return {
        "status": "mock_error_analysis_ready",
        "samples_evaluated": len(sample_ids),
        "analysis_scope": ["patient_present", "operation"],
        "known_mock_error_sources": [
            "premature_false_activation",
            "background_disturbance",
            "boundary_transition",
            "view_occlusion",
            "severe_view_occlusion",
            "all_views_unavailable",
        ],
        "raw_predictions": raw_report,
        "cleaned_predictions": cleaned_report,
        "postprocessing_effect": compare_error_versions(
            raw=raw_report,
            cleaned=cleaned_report,
        ),
    }


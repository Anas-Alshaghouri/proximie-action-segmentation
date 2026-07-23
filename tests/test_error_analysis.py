from pathlib import Path

import torch

from action_segmentation.config import load_config
from action_segmentation.data.dataset import SyntheticTemporalDataset
from action_segmentation.evaluation.error_analysis import (
    analyze_error_version,
    build_mock_noisy_logits,
    format_error_analysis_report,
)
from action_segmentation.pipeline import error_analysis_readiness_summary
from action_segmentation.postprocessing.timeline import generate_timeline


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "default.yaml"


def test_mock_logits_follow_known_failure_masks() -> None:
    config = load_config(CONFIG_PATH)
    sample = SyntheticTemporalDataset(config, "validation")[0]
    logits = build_mock_noisy_logits(
        labels=sample["labels"],
        time_mask=sample["time_mask"],
        view_mask=sample["view_mask"],
        occlusion_mask=sample["occlusion_mask"],
        patient_disturbance_mask=sample["patient_present_disturbance_mask"],
        phase_names=config.phases.names,
        boundary_noise_steps=3,
    )

    predictions = logits.argmax(dim=-1)
    patient_id = config.phases.names.index("patient_present")
    preparation_id = config.phases.names.index("preparation")
    disturbance = sample["patient_present_disturbance_mask"]

    assert torch.all(predictions[disturbance] == preparation_id)
    assert torch.all(logits[~sample["time_mask"]] == 0)
    assert torch.any(predictions[sample["labels"] != patient_id] == patient_id)


def test_error_analysis_attributes_patient_disturbance() -> None:
    config = load_config(CONFIG_PATH)
    sample = SyntheticTemporalDataset(config, "validation")[0]
    logits = build_mock_noisy_logits(
        labels=sample["labels"],
        time_mask=sample["time_mask"],
        view_mask=sample["view_mask"],
        occlusion_mask=sample["occlusion_mask"],
        patient_disturbance_mask=sample["patient_present_disturbance_mask"],
        phase_names=config.phases.names,
        boundary_noise_steps=3,
    )
    timeline = generate_timeline(
        logits=logits,
        timestamps=sample["timestamps"],
        time_mask=sample["time_mask"],
        phase_names=config.phases.names,
        sampling_rate_hz=config.data.sampling_rate_hz,
        smoothing_window_seconds=config.postprocessing.smoothing_window_seconds,
        min_segment_duration_seconds=config.postprocessing.min_segment_duration_seconds,
    )

    report = analyze_error_version(
        sample_ids=[sample["sample_id"]],
        targets=sample["labels"].unsqueeze(0),
        predictions=timeline.raw_predictions.unsqueeze(0),
        time_mask=sample["time_mask"].unsqueeze(0),
        view_mask=sample["view_mask"].unsqueeze(0),
        occlusion_mask=sample["occlusion_mask"].unsqueeze(0),
        patient_disturbance_mask=(
            sample["patient_present_disturbance_mask"].unsqueeze(0)
        ),
        phase_names=config.phases.names,
        sampling_rate_hz=config.data.sampling_rate_hz,
        boundary_tolerance_seconds=config.evaluation.boundary_tolerance_seconds,
    )

    patient = report["patient_present"]
    assert patient["ground_truth_segments"] == 1
    assert patient["cause_breakdown"]["background_disturbance"]["intervals"] >= 1
    assert patient["false_positive_duration_seconds"] > 0


def test_pipeline_error_report_shows_cleanup_tradeoff() -> None:
    config = load_config(CONFIG_PATH)
    report = error_analysis_readiness_summary(config)

    assert report["status"] == "mock_error_analysis_ready"
    assert report["samples_evaluated"] == config.data.validation_samples
    assert (
        report["raw_predictions"]["patient_present"]["extra_fragments"]
        > report["cleaned_predictions"]["patient_present"]["extra_fragments"]
    )
    assert report["postprocessing_effect"]["patient_present_fragments_removed"] > 0
    assert report["cleaned_predictions"]["operation"]["mean_start_delay_seconds"] > 0


def test_human_report_contains_required_classes() -> None:
    config = load_config(CONFIG_PATH)
    text = format_error_analysis_report(error_analysis_readiness_summary(config))

    assert "Patient Present" in text
    assert "Operation" in text
    assert "Post-processing effect" in text
    assert "Validation sequences: 16" in text

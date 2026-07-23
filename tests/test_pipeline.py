from pathlib import Path

from action_segmentation.config import load_config
from action_segmentation.pipeline import (
    error_analysis_readiness_summary,
    fusion_readiness_summary,
    metric_stack_readiness_summary,
    repository_readiness_summary,
    synthetic_data_readiness_summary,
    temporal_model_readiness_summary,
    timeline_readiness_summary,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "default.yaml"


def test_readiness_summary_matches_contract() -> None:
    config = load_config(CONFIG_PATH)
    summary = repository_readiness_summary(config)

    assert summary["supported_views"] == {"minimum": 1, "maximum": 3}
    assert summary["sequence_duration_seconds"] == 600.0
    assert summary["fusion_type"] == "masked_mean"


def test_synthetic_data_summary_matches_contract() -> None:
    config = load_config(CONFIG_PATH)
    summary = synthetic_data_readiness_summary(config)

    assert summary["status"] == "synthetic_data_ready"
    assert summary["batch_shapes"]["features"] == [8, 3, 600, 64]
    assert summary["invalid_features_are_zero"] is True


def test_fusion_summary_matches_contract() -> None:
    config = load_config(CONFIG_PATH)
    summary = fusion_readiness_summary(config)

    assert summary["status"] == "multi_view_fusion_ready"
    assert summary["input_shape"] == [8, 3, 600, 64]
    assert summary["output_shape"] == [8, 600, 64]
    assert summary["time_mask_preserved"] is True
    assert summary["single_view_features_preserved"] is True
    assert summary["fully_missing_features_are_zero"] is True


def test_temporal_model_summary_matches_contract() -> None:
    config = load_config(CONFIG_PATH)
    summary = temporal_model_readiness_summary(config)

    assert summary["status"] == "causal_temporal_model_ready"
    assert summary["input_shape"] == [8, 600, 64]
    assert summary["logit_shape"] == [8, 600, 5]
    assert summary["receptive_field_steps"] == 61
    assert summary["invalid_logits_are_zero"] is True
    assert summary["causality_check"]["passed"] is True



def test_timeline_summary_matches_contract() -> None:
    config = load_config(CONFIG_PATH)
    summary = timeline_readiness_summary(config)

    assert summary["status"] == "timeline_postprocessing_ready"
    assert summary["processing_mode"] == "strictly_causal_with_confirmation_delay"
    assert summary["model_output_accepted"] is True
    assert summary["controlled_demo"]["cleaned_available_segments"] < summary["controlled_demo"]["raw_available_segments"]
    assert summary["causality_check"]["past_cleaned_predictions_unchanged"] is True
    assert summary["causality_check"]["past_smoothed_probability_max_abs_difference"] == 0.0


def test_metric_stack_summary_shows_stability_latency_tradeoff() -> None:
    config = load_config(CONFIG_PATH)
    summary = metric_stack_readiness_summary(config)

    assert summary["status"] == "dual_metric_stack_ready"
    assert summary["untrained_model_outputs_accepted"] is True
    assert summary["cleaned_predictions"]["segment_metrics"]["edit_score"]["score"] == 100.0
    assert summary["postprocessing_tradeoff"]["patient_present_extra_fragments"] == {"raw": 1, "cleaned": 0}
    assert summary["postprocessing_tradeoff"]["operation_predicted_segments"] == {"raw": 2, "cleaned": 1}
    assert summary["postprocessing_tradeoff"]["operation_start_delay_seconds"] == {"raw": 0.0, "cleaned": 9.0}


def test_error_analysis_summary_matches_contract() -> None:
    config = load_config(CONFIG_PATH)
    summary = error_analysis_readiness_summary(config)

    assert summary["status"] == "mock_error_analysis_ready"
    assert summary["samples_evaluated"] == 16
    assert summary["analysis_scope"] == ["patient_present", "operation"]
    assert summary["postprocessing_effect"]["patient_present_fragments_removed"] > 0

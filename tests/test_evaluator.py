from dataclasses import replace
import json
from pathlib import Path

from action_segmentation.config import load_config
from action_segmentation.evaluation.evaluator import evaluate_checkpoint
from action_segmentation.training.trainer import train_model

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "default.yaml"


def _tiny_config(tmp_path: Path):
    config = load_config(CONFIG_PATH)
    return replace(
        config,
        data=replace(
            config.data,
            sequence_length=120,
            feature_dim=16,
            train_samples=8,
            validation_samples=4,
            test_samples=4,
            minimum_phase_duration_seconds=10,
        ),
        model=replace(
            config.model,
            input_dim=16,
            hidden_dim=16,
            dilations=(1, 2),
        ),
        training=replace(
            config.training,
            batch_size=4,
            epochs=1,
            device="cpu",
        ),
        postprocessing=replace(
            config.postprocessing,
            smoothing_window_seconds=3,
            min_segment_duration_seconds=3,
        ),
        output=replace(
            config.output,
            artifacts_directory=tmp_path,
            checkpoint_filename="eval_model.pt",
            metrics_filename="eval_metrics.json",
        ),
    )


def test_checkpoint_evaluation_returns_serializable_metrics(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    training = train_model(config)
    result = evaluate_checkpoint(
        config,
        checkpoint_path=training.checkpoint_path,
        split="test",
    )

    assert result["status"] == "trained_model_evaluation_complete"
    assert result["samples_evaluated"] == 4
    assert result["valid_timestamps"] > 0
    assert 0.0 <= result["raw_predictions"]["frame_metrics"]["accuracy"] <= 1.0
    assert len(result["timelines"]) == 4
    first_timeline = result["timelines"][0]
    assert set(first_timeline["tracks"]) == {
        "ground_truth",
        "raw_prediction",
        "cleaned_prediction",
    }
    assert len(first_timeline["camera_availability"]) == config.data.max_views
    assert len(first_timeline["confidence"]["timestamps_seconds"]) == config.data.sequence_length
    json.dumps(result)

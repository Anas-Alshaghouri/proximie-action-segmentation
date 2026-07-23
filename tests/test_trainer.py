from dataclasses import replace
from pathlib import Path

import torch

from action_segmentation.config import load_config
from action_segmentation.training.trainer import (
    load_model_checkpoint,
    resolve_device,
    train_model,
)

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
            epochs=2,
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
            checkpoint_filename="tiny_model.pt",
            metrics_filename="tiny_metrics.json",
        ),
    )


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_training_saves_and_restores_checkpoint(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    result = train_model(config)

    assert result.checkpoint_path.is_file()
    assert result.epochs_completed == 2
    assert 1 <= result.best_epoch <= 2
    assert len(result.history) == 2
    assert all(item.train_loss > 0 for item in result.history)

    model, checkpoint = load_model_checkpoint(
        checkpoint_path=result.checkpoint_path,
        config=config,
        device=torch.device("cpu"),
    )
    assert model.num_classes == 5
    assert checkpoint["epoch"] == result.best_epoch

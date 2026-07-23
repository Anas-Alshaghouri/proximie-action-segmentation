from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when the project configuration is missing or inconsistent."""


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int


@dataclass(frozen=True)
class PhaseConfig:
    names: tuple[str, ...]


@dataclass(frozen=True)
class DataConfig:
    sampling_rate_hz: float
    sequence_length: int
    feature_dim: int
    phase_duration_weights: tuple[float, ...]
    minimum_phase_duration_seconds: float
    min_views: int
    max_views: int
    train_samples: int
    validation_samples: int
    test_samples: int
    feature_noise_std: float
    view_bias_std: float
    occlusion_probability: float
    camera_dropout_probability: float
    boundary_noise_seconds: int


@dataclass(frozen=True)
class FusionConfig:
    type: str


@dataclass(frozen=True)
class ModelConfig:
    type: str
    input_dim: int
    hidden_dim: int
    num_classes: int
    kernel_size: int
    dilations: tuple[int, ...]
    dropout: float
    causal: bool


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    device: str


@dataclass(frozen=True)
class PostprocessingConfig:
    smoothing_window_seconds: int
    min_segment_duration_seconds: int


@dataclass(frozen=True)
class EvaluationConfig:
    segment_iou_thresholds: tuple[float, ...]
    boundary_tolerance_seconds: int


@dataclass(frozen=True)
class OutputConfig:
    artifacts_directory: Path
    checkpoint_filename: str
    metrics_filename: str


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    phases: PhaseConfig
    data: DataConfig
    fusion: FusionConfig
    model: ModelConfig
    training: TrainingConfig
    postprocessing: PostprocessingConfig
    evaluation: EvaluationConfig
    output: OutputConfig


def _require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(
            f"Missing required key '{section}.{key}' in configuration."
        )
    return mapping[key]


def _probability(value: Any, name: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ConfigurationError(f"'{name}' must be between 0 and 1.")
    return result


def load_config(config_path: str | Path) -> AppConfig:
    """Load and validate the YAML project configuration."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)

    if not isinstance(raw, dict):
        raise ConfigurationError("The configuration root must be a mapping.")

    project_raw = _require(raw, "project", "root")
    phases_raw = _require(raw, "phases", "root")
    data_raw = _require(raw, "data", "root")
    fusion_raw = _require(raw, "fusion", "root")
    model_raw = _require(raw, "model", "root")
    training_raw = _require(raw, "training", "root")
    post_raw = _require(raw, "postprocessing", "root")
    evaluation_raw = _require(raw, "evaluation", "root")
    output_raw = _require(raw, "output", "root")

    phases = PhaseConfig(
        names=tuple(str(name) for name in _require(phases_raw, "names", "phases"))
    )
    if not phases.names:
        raise ConfigurationError("'phases.names' must contain at least one phase.")
    if len(set(phases.names)) != len(phases.names):
        raise ConfigurationError("'phases.names' must not contain duplicates.")

    data = DataConfig(
        sampling_rate_hz=float(_require(data_raw, "sampling_rate_hz", "data")),
        sequence_length=int(_require(data_raw, "sequence_length", "data")),
        feature_dim=int(_require(data_raw, "feature_dim", "data")),
        phase_duration_weights=tuple(
            float(value)
            for value in _require(
                data_raw, "phase_duration_weights", "data"
            )
        ),
        minimum_phase_duration_seconds=float(
            _require(data_raw, "minimum_phase_duration_seconds", "data")
        ),
        min_views=int(_require(data_raw, "min_views", "data")),
        max_views=int(_require(data_raw, "max_views", "data")),
        train_samples=int(_require(data_raw, "train_samples", "data")),
        validation_samples=int(_require(data_raw, "validation_samples", "data")),
        test_samples=int(_require(data_raw, "test_samples", "data")),
        feature_noise_std=float(_require(data_raw, "feature_noise_std", "data")),
        view_bias_std=float(_require(data_raw, "view_bias_std", "data")),
        occlusion_probability=_probability(
            _require(data_raw, "occlusion_probability", "data"),
            "data.occlusion_probability",
        ),
        camera_dropout_probability=_probability(
            _require(data_raw, "camera_dropout_probability", "data"),
            "data.camera_dropout_probability",
        ),
        boundary_noise_seconds=int(
            _require(data_raw, "boundary_noise_seconds", "data")
        ),
    )

    fusion = FusionConfig(
        type=str(_require(fusion_raw, "type", "fusion")),
    )

    model = ModelConfig(
        type=str(_require(model_raw, "type", "model")),
        input_dim=int(_require(model_raw, "input_dim", "model")),
        hidden_dim=int(_require(model_raw, "hidden_dim", "model")),
        num_classes=int(_require(model_raw, "num_classes", "model")),
        kernel_size=int(_require(model_raw, "kernel_size", "model")),
        dilations=tuple(
            int(value) for value in _require(model_raw, "dilations", "model")
        ),
        dropout=float(_require(model_raw, "dropout", "model")),
        causal=bool(_require(model_raw, "causal", "model")),
    )

    training = TrainingConfig(
        batch_size=int(_require(training_raw, "batch_size", "training")),
        epochs=int(_require(training_raw, "epochs", "training")),
        learning_rate=float(_require(training_raw, "learning_rate", "training")),
        weight_decay=float(_require(training_raw, "weight_decay", "training")),
        device=str(_require(training_raw, "device", "training")),
    )

    postprocessing = PostprocessingConfig(
        smoothing_window_seconds=int(
            _require(post_raw, "smoothing_window_seconds", "postprocessing")
        ),
        min_segment_duration_seconds=int(
            _require(post_raw, "min_segment_duration_seconds", "postprocessing")
        ),
    )

    evaluation = EvaluationConfig(
        segment_iou_thresholds=tuple(
            float(value)
            for value in _require(
                evaluation_raw, "segment_iou_thresholds", "evaluation"
            )
        ),
        boundary_tolerance_seconds=int(
            _require(
                evaluation_raw, "boundary_tolerance_seconds", "evaluation"
            )
        ),
    )

    output = OutputConfig(
        artifacts_directory=Path(
            _require(output_raw, "artifacts_directory", "output")
        ),
        checkpoint_filename=str(
            _require(output_raw, "checkpoint_filename", "output")
        ),
        metrics_filename=str(_require(output_raw, "metrics_filename", "output")),
    )

    config = AppConfig(
        project=ProjectConfig(
            name=str(_require(project_raw, "name", "project")),
            seed=int(_require(project_raw, "seed", "project")),
        ),
        phases=phases,
        data=data,
        fusion=fusion,
        model=model,
        training=training,
        postprocessing=postprocessing,
        evaluation=evaluation,
        output=output,
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    """Validate cross-section constraints that YAML alone cannot enforce."""
    if config.data.sampling_rate_hz <= 0:
        raise ConfigurationError("'data.sampling_rate_hz' must be positive.")
    if config.data.sequence_length <= 0:
        raise ConfigurationError("'data.sequence_length' must be positive.")
    if config.data.feature_dim <= 0:
        raise ConfigurationError("'data.feature_dim' must be positive.")
    if len(config.data.phase_duration_weights) != len(config.phases.names):
        raise ConfigurationError(
            "'data.phase_duration_weights' must contain one value per phase."
        )
    if any(weight <= 0 for weight in config.data.phase_duration_weights):
        raise ConfigurationError(
            "'data.phase_duration_weights' must contain positive values."
        )
    if config.data.minimum_phase_duration_seconds <= 0:
        raise ConfigurationError(
            "'data.minimum_phase_duration_seconds' must be positive."
        )
    minimum_phase_steps = math.ceil(
        config.data.minimum_phase_duration_seconds
        * config.data.sampling_rate_hz
    )
    if minimum_phase_steps * len(config.phases.names) > config.data.sequence_length:
        raise ConfigurationError(
            "Minimum phase durations exceed 'data.sequence_length'."
        )
    if config.data.min_views < 1:
        raise ConfigurationError("'data.min_views' must be at least 1.")
    if config.data.max_views > 3:
        raise ConfigurationError(
            "'data.max_views' cannot exceed the challenge limit of 3."
        )
    if config.data.min_views > config.data.max_views:
        raise ConfigurationError("'data.min_views' cannot exceed 'data.max_views'.")

    sample_counts = (
        config.data.train_samples,
        config.data.validation_samples,
        config.data.test_samples,
    )
    if any(count <= 0 for count in sample_counts):
        raise ConfigurationError("All dataset sample counts must be positive.")
    if config.data.feature_noise_std < 0 or config.data.view_bias_std < 0:
        raise ConfigurationError(
            "Synthetic noise standard deviations cannot be negative."
        )
    if config.data.boundary_noise_seconds < 0:
        raise ConfigurationError(
            "'data.boundary_noise_seconds' cannot be negative."
        )

    if config.fusion.type != "masked_mean":
        raise ConfigurationError(
            "The prototype currently supports only fusion.type=masked_mean."
        )

    if config.model.type != "causal_tcn":
        raise ConfigurationError(
            "The prototype currently supports only model.type=causal_tcn."
        )
    if config.model.input_dim != config.data.feature_dim:
        raise ConfigurationError("'model.input_dim' must equal 'data.feature_dim'.")
    if config.model.num_classes != len(config.phases.names):
        raise ConfigurationError(
            "'model.num_classes' must equal the number of phase names."
        )
    if config.model.hidden_dim <= 0:
        raise ConfigurationError("'model.hidden_dim' must be positive.")
    if config.model.kernel_size < 2:
        raise ConfigurationError("'model.kernel_size' must be at least 2.")
    if not config.model.dilations or any(
        dilation <= 0 for dilation in config.model.dilations
    ):
        raise ConfigurationError("'model.dilations' must contain positive integers.")
    if not 0.0 <= config.model.dropout < 1.0:
        raise ConfigurationError("'model.dropout' must be in [0, 1).")
    if not config.model.causal:
        raise ConfigurationError(
            "The prototype is defined as online; 'model.causal' must be true."
        )

    if config.training.batch_size <= 0 or config.training.epochs <= 0:
        raise ConfigurationError(
            "Training batch size and epoch count must be positive."
        )
    if config.training.learning_rate <= 0:
        raise ConfigurationError("'training.learning_rate' must be positive.")

    if config.postprocessing.smoothing_window_seconds <= 0:
        raise ConfigurationError(
            "'postprocessing.smoothing_window_seconds' must be positive."
        )
    if config.postprocessing.min_segment_duration_seconds <= 0:
        raise ConfigurationError(
            "'postprocessing.min_segment_duration_seconds' must be positive."
        )

    thresholds = config.evaluation.segment_iou_thresholds
    if not thresholds or any(not 0.0 < value <= 1.0 for value in thresholds):
        raise ConfigurationError(
            "Segment IoU thresholds must be in the interval (0, 1]."
        )
    if config.evaluation.boundary_tolerance_seconds < 0:
        raise ConfigurationError(
            "'evaluation.boundary_tolerance_seconds' cannot be negative."
        )

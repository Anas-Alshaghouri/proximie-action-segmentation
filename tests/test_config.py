from pathlib import Path

from action_segmentation.config import load_config


def test_default_configuration_loads() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "default.yaml"
    config = load_config(config_path)

    assert config.model.causal is True
    assert config.fusion.type == "masked_mean"
    assert config.model.num_classes == len(config.phases.names)
    assert config.model.input_dim == config.data.feature_dim
    assert config.data.min_views == 1
    assert config.data.max_views == 3
    assert len(config.data.phase_duration_weights) == len(config.phases.names)
    assert config.data.minimum_phase_duration_seconds == 20

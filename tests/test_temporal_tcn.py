from pathlib import Path

import torch

from action_segmentation.config import load_config
from action_segmentation.models.temporal_tcn import (
    CausalTemporalConvNet,
    build_temporal_model,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "default.yaml"


def _small_model() -> CausalTemporalConvNet:
    torch.manual_seed(7)
    model = CausalTemporalConvNet(
        input_dim=4,
        hidden_dim=8,
        num_classes=5,
        kernel_size=3,
        dilations=(1, 2, 4),
        dropout=0.0,
    )
    model.eval()
    return model


def test_model_output_shape_and_invalid_logits() -> None:
    model = _small_model()
    features = torch.randn(2, 20, 4)
    time_mask = torch.ones(2, 20, dtype=torch.bool)
    time_mask[0, 5] = False

    output = model(features, time_mask)

    assert output.logits.shape == (2, 20, 5)
    assert torch.equal(output.time_mask, time_mask)
    assert torch.all(output.logits[0, 5] == 0)


def test_model_is_causal() -> None:
    model = _small_model()
    features = torch.randn(1, 30, 4)
    time_mask = torch.ones(1, 30, dtype=torch.bool)

    original = model(features, time_mask).logits
    modified = features.clone()
    modified[:, 16:] = torch.randn_like(modified[:, 16:]) * 100.0
    changed = model(modified, time_mask).logits

    torch.testing.assert_close(original[:, :16], changed[:, :16])
    assert not torch.allclose(original[:, 16:], changed[:, 16:])


def test_receptive_field_uses_two_convolutions_per_block() -> None:
    model = _small_model()
    assert model.receptive_field_steps == 29


def test_default_model_factory_matches_configuration() -> None:
    config = load_config(CONFIG_PATH)
    model = build_temporal_model(config)

    assert model.input_dim == 64
    assert model.num_classes == 5
    assert model.receptive_field_steps == 61


def test_model_learns_and_checkpoint_round_trip(tmp_path: Path) -> None:
    model = _small_model()
    model.train()
    features = torch.randn(2, 12, 4)
    labels = torch.full((2, 12), 2, dtype=torch.int64)
    time_mask = torch.ones(2, 12, dtype=torch.bool)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    with torch.no_grad():
        initial_loss = torch.nn.functional.cross_entropy(
            model(features, time_mask).logits[time_mask],
            labels[time_mask],
        )

    for _ in range(15):
        optimizer.zero_grad()
        logits = model(features, time_mask).logits
        loss = torch.nn.functional.cross_entropy(
            logits[time_mask],
            labels[time_mask],
        )
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        trained_logits = model(features, time_mask).logits
        final_loss = torch.nn.functional.cross_entropy(
            trained_logits[time_mask],
            labels[time_mask],
        )

    assert final_loss < initial_loss

    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    restored = _small_model()
    restored.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    restored.eval()
    with torch.no_grad():
        restored_logits = restored(features, time_mask).logits

    torch.testing.assert_close(trained_logits, restored_logits)

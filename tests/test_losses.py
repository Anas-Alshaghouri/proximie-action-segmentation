import torch

from action_segmentation.training.losses import masked_cross_entropy


def test_masked_loss_ignores_invalid_timestamps() -> None:
    logits = torch.tensor(
        [[[5.0, 0.0], [0.0, 5.0], [4.0, 1.0]]],
        dtype=torch.float32,
    )
    labels_a = torch.tensor([[0, 1, 0]], dtype=torch.int64)
    labels_b = torch.tensor([[0, 1, 1]], dtype=torch.int64)
    time_mask = torch.tensor([[True, True, False]])

    loss_a = masked_cross_entropy(logits, labels_a, time_mask)
    loss_b = masked_cross_entropy(logits, labels_b, time_mask)

    torch.testing.assert_close(loss_a, loss_b)


def test_masked_loss_rejects_an_empty_valid_set() -> None:
    logits = torch.randn(1, 3, 2)
    labels = torch.zeros(1, 3, dtype=torch.int64)
    time_mask = torch.zeros(1, 3, dtype=torch.bool)

    try:
        masked_cross_entropy(logits, labels, time_mask)
    except ValueError as exc:
        assert "at least one valid timestamp" in str(exc)
    else:
        raise AssertionError("Expected ValueError for an empty valid set.")

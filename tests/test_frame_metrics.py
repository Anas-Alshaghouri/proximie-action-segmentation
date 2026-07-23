import torch

from action_segmentation.evaluation.frame_metrics import (
    confusion_matrix,
    frame_classification_metrics,
)


def test_confusion_matrix_uses_true_rows_and_predicted_columns() -> None:
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    predictions = torch.tensor([0, 1, 1, 1], dtype=torch.int64)
    mask = torch.ones(4, dtype=torch.bool)

    matrix = confusion_matrix(targets, predictions, mask, num_classes=2)

    assert matrix.tolist() == [[1, 1], [0, 2]]


def test_frame_metrics_ignore_unavailable_timestamps() -> None:
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.int64)
    predictions = torch.tensor([0, -1, 0, 0], dtype=torch.int64)
    mask = torch.tensor([True, False, True, True])

    metrics = frame_classification_metrics(
        targets,
        predictions,
        mask,
        phase_names=("empty", "operation"),
    )

    assert metrics["valid_timestamps"] == 3
    assert metrics["correct_timestamps"] == 2
    assert metrics["accuracy"] == 0.666667
    assert metrics["confusion_matrix"] == [[2, 0], [1, 0]]

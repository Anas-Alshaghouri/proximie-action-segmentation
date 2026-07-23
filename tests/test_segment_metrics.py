import torch

from action_segmentation.evaluation.segment_metrics import (
    TemporalSegment,
    boundary_metrics,
    extract_segments,
    segment_edit_score,
    segmental_f1,
    temporal_iou,
)


def test_extract_segments_splits_at_unavailable_gap() -> None:
    labels = torch.tensor([0, 0, -1, 0, 1, 1], dtype=torch.int64)
    mask = torch.tensor([True, True, False, True, True, True])

    segments = extract_segments(labels, mask)

    assert segments == (
        TemporalSegment(0, 0, 2),
        TemporalSegment(0, 3, 4),
        TemporalSegment(1, 4, 6),
    )


def test_temporal_iou_uses_overlap_over_union() -> None:
    first = TemporalSegment(0, 0, 10)
    second = TemporalSegment(0, 5, 15)

    assert temporal_iou(first, second) == 5 / 15


def test_segmental_f1_penalizes_extra_fragment() -> None:
    targets = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.int64)
    predictions = torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.int64)
    mask = torch.ones(6, dtype=torch.bool)

    metrics = segmental_f1(targets, predictions, mask, iou_threshold=0.25)

    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 2
    assert metrics["false_negative"] == 0
    assert metrics["f1"] == 0.666667


def test_edit_score_detects_wrong_segment_order() -> None:
    targets = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
    predictions = torch.tensor([0, 0, 2, 2, 1, 1], dtype=torch.int64)
    mask = torch.ones(6, dtype=torch.bool)

    result = segment_edit_score(targets, predictions, mask)

    assert result["ground_truth_sequence"] == [0, 1, 2]
    assert result["predicted_sequence"] == [0, 2, 1]
    assert result["distance"] == 2
    assert result["score"] == 33.333333


def test_boundary_metrics_report_late_start_and_end() -> None:
    targets = torch.tensor([0, 0, 1, 1, 1, 2, 2], dtype=torch.int64)
    predictions = torch.tensor([0, 0, 0, 1, 1, 1, 2], dtype=torch.int64)
    mask = torch.ones(7, dtype=torch.bool)

    result = boundary_metrics(
        targets,
        predictions,
        mask,
        sampling_rate_hz=1.0,
        tolerance_seconds=1.0,
    )

    assert result["matched_segments"] == 3
    assert result["mean_start_error_seconds"] == 0.666667
    assert result["mean_end_error_seconds"] == 0.666667

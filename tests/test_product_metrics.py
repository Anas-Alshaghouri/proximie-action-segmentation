import torch

from action_segmentation.evaluation.product_metrics import product_quality_metrics


PHASES = ("empty", "patient_present", "preparation", "operation", "closing")


def test_product_metrics_capture_fragmentation_and_operation_boundaries() -> None:
    targets = torch.tensor(
        [0, 1, 1, 1, 2, 3, 3, 3, 3, 4],
        dtype=torch.int64,
    )
    predictions = torch.tensor(
        [0, 1, 2, 1, 2, 2, 3, 3, 4, 4],
        dtype=torch.int64,
    )
    mask = torch.ones(10, dtype=torch.bool)
    occlusion = torch.zeros((2, 10), dtype=torch.bool)
    occlusion[0, 6:8] = True

    metrics = product_quality_metrics(
        targets,
        predictions,
        mask,
        PHASES,
        sampling_rate_hz=1.0,
        occlusion_mask=occlusion,
    )

    patient = metrics["patient_present"]
    operation = metrics["operation"]
    assert patient["predicted_segments"] == 2
    assert patient["extra_fragments"] == 1
    assert patient["missed_duration_seconds"] == 1.0
    assert patient["mean_start_confirmation_delay_seconds"] == 0.0
    assert patient["mean_end_delay_seconds"] == 0.0
    assert operation["start_delay_seconds"] == 1.0
    assert operation["end_delay_seconds"] == -1.0
    assert operation["coverage_ratio"] == 0.5
    assert operation["coverage_during_occlusion_ratio"] == 1.0


def test_product_metrics_report_zero_recovery_when_operation_is_maintained() -> None:
    targets = torch.tensor([3, 3, 3, 3, 3], dtype=torch.int64)
    predictions = targets.clone()
    mask = torch.ones(5, dtype=torch.bool)
    occlusion = torch.tensor([False, True, True, False, False])

    metrics = product_quality_metrics(
        targets,
        predictions,
        mask,
        PHASES,
        sampling_rate_hz=1.0,
        occlusion_mask=occlusion,
    )

    assert metrics["operation"]["coverage_during_occlusion_ratio"] == 1.0
    assert metrics["operation"]["mean_occlusion_recovery_seconds"] == 0.0

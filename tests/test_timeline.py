import torch

from action_segmentation.postprocessing.timeline import (
    UNAVAILABLE_LABEL_ID,
    causal_debounce_predictions,
    causal_smooth_probabilities,
    generate_batch_timelines,
    generate_timeline,
    labels_to_timeline,
    logits_to_probabilities,
)


PHASES = ("empty", "patient_present", "operation")


def test_logits_to_probabilities_masks_missing_timestamps() -> None:
    logits = torch.tensor([[2.0, 0.0], [1.0, 3.0], [0.0, 1.0]])
    mask = torch.tensor([True, False, True])

    probabilities = logits_to_probabilities(logits, mask)

    torch.testing.assert_close(probabilities[0].sum(), torch.tensor(1.0))
    torch.testing.assert_close(probabilities[2].sum(), torch.tensor(1.0))
    assert torch.equal(probabilities[1], torch.zeros(2))


def test_causal_smoothing_does_not_read_future_probabilities() -> None:
    probabilities = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )
    mask = torch.ones(6, dtype=torch.bool)
    modified = probabilities.clone()
    modified[4:] = torch.tensor([[0.99, 0.01], [0.99, 0.01]])

    first = causal_smooth_probabilities(probabilities, mask, window_steps=3)
    second = causal_smooth_probabilities(modified, mask, window_steps=3)

    torch.testing.assert_close(first[:4], second[:4])


def test_causal_smoothing_ignores_missing_values_inside_window() -> None:
    probabilities = torch.tensor(
        [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]
    )
    mask = torch.tensor([True, False, True])

    smoothed = causal_smooth_probabilities(probabilities, mask, window_steps=3)

    assert torch.equal(smoothed[1], torch.zeros(2))
    torch.testing.assert_close(smoothed[2], torch.tensor([0.5, 0.5]))


def test_debounce_removes_short_false_positive_island() -> None:
    predictions = torch.tensor([0, 0, 1, 1, 1, 0], dtype=torch.int64)
    mask = torch.ones(6, dtype=torch.bool)

    cleaned = causal_debounce_predictions(
        predictions,
        mask,
        min_duration_steps=4,
    )

    assert cleaned.tolist() == [0, 0, 0, 0, 0, 0]


def test_debounce_confirms_sustained_transition_after_minimum_duration() -> None:
    predictions = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int64)
    mask = torch.ones(5, dtype=torch.bool)

    cleaned = causal_debounce_predictions(
        predictions,
        mask,
        min_duration_steps=3,
    )

    assert cleaned.tolist() == [0, 0, 0, 0, 1]


def test_debounce_preserves_unavailable_timestamp() -> None:
    predictions = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64)
    mask = torch.tensor([True, True, False, True, True])

    cleaned = causal_debounce_predictions(
        predictions,
        mask,
        min_duration_steps=2,
    )

    assert cleaned.tolist() == [0, 0, UNAVAILABLE_LABEL_ID, 0, 1]


def test_labels_to_timeline_uses_exclusive_end_times() -> None:
    labels = torch.tensor([0, 0, 1, 1, -1, 2], dtype=torch.int64)
    timestamps = torch.arange(6, dtype=torch.float32)
    probabilities = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.1, 0.8, 0.1],
            [0.1, 0.9, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.9],
        ]
    )

    segments = labels_to_timeline(
        labels,
        probabilities,
        timestamps,
        PHASES,
        sampling_rate_hz=1.0,
    )

    assert [segment.label_name for segment in segments] == [
        "empty",
        "patient_present",
        "unavailable",
        "operation",
    ]
    assert [(segment.start_seconds, segment.end_seconds) for segment in segments] == [
        (0.0, 2.0),
        (2.0, 4.0),
        (4.0, 5.0),
        (5.0, 6.0),
    ]
    assert segments[2].is_available is False


def test_generate_timeline_returns_all_intermediate_outputs() -> None:
    logits = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 5.0, 0.0],
        ]
    )
    timestamps = torch.arange(6, dtype=torch.float32)
    mask = torch.tensor([True, True, True, True, True, False])

    result = generate_timeline(
        logits=logits,
        timestamps=timestamps,
        time_mask=mask,
        phase_names=PHASES,
        sampling_rate_hz=1.0,
        smoothing_window_seconds=1,
        min_segment_duration_seconds=2,
    )

    assert result.probabilities.shape == (6, 3)
    assert result.smoothed_probabilities.shape == (6, 3)
    assert result.cleaned_predictions.shape == (6,)
    assert result.cleaned_predictions[-1].item() == UNAVAILABLE_LABEL_ID
    assert result.segments[-1].label_name == "unavailable"


def test_generate_batch_timelines_returns_one_result_per_sample() -> None:
    logits = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 3.0], [0.0, 3.0]],
            [[0.0, 3.0], [0.0, 3.0], [3.0, 0.0]],
        ]
    )
    timestamps = torch.arange(3, dtype=torch.float32).repeat(2, 1)
    mask = torch.ones((2, 3), dtype=torch.bool)

    results = generate_batch_timelines(
        logits=logits,
        timestamps=timestamps,
        time_mask=mask,
        phase_names=("empty", "operation"),
        sampling_rate_hz=1.0,
        smoothing_window_seconds=1,
        min_segment_duration_seconds=1,
    )

    assert len(results) == 2
    assert results[0].cleaned_predictions.tolist() == [0, 1, 1]
    assert results[1].cleaned_predictions.tolist() == [1, 1, 0]

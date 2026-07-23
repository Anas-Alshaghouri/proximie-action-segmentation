from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch
from torch.nn import functional as F

UNAVAILABLE_LABEL_ID = -1
UNAVAILABLE_LABEL_NAME = "unavailable"


@dataclass(frozen=True)
class TimelineSegment:
    """One contiguous phase interval using an exclusive end boundary."""

    label_id: int
    label_name: str
    start_index: int
    end_index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    mean_confidence: float
    is_available: bool

    def to_dict(self) -> dict[str, int | float | str | bool]:
        return asdict(self)


@dataclass(frozen=True)
class TimelineResult:
    """Intermediate and final outputs of causal timeline post-processing."""

    probabilities: torch.Tensor
    smoothed_probabilities: torch.Tensor
    raw_predictions: torch.Tensor
    smoothed_predictions: torch.Tensor
    cleaned_predictions: torch.Tensor
    segments: tuple[TimelineSegment, ...]


def _validate_sequence_inputs(
    values: torch.Tensor,
    timestamps: torch.Tensor,
    time_mask: torch.Tensor,
) -> None:
    if values.ndim != 2:
        raise ValueError("Sequence values must have shape [time, classes].")
    if timestamps.ndim != 1 or time_mask.ndim != 1:
        raise ValueError("'timestamps' and 'time_mask' must have shape [time].")
    if values.shape[0] != timestamps.shape[0] or values.shape[0] != time_mask.shape[0]:
        raise ValueError("Values, timestamps, and time_mask must share time length.")
    if values.shape[1] < 2:
        raise ValueError("At least two phase classes are required.")
    if not values.is_floating_point() or not timestamps.is_floating_point():
        raise TypeError("Values and timestamps must be floating point tensors.")
    if time_mask.dtype != torch.bool:
        raise TypeError("'time_mask' must use torch.bool.")
    if values.device != timestamps.device or values.device != time_mask.device:
        raise ValueError("All timeline tensors must use the same device.")
    if timestamps.numel() > 1 and not torch.all(timestamps[1:] > timestamps[:-1]):
        raise ValueError("Timestamps must be strictly increasing.")


def logits_to_probabilities(
    logits: torch.Tensor,
    time_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert raw class logits into probabilities only at valid timestamps."""
    if logits.ndim != 2:
        raise ValueError("'logits' must have shape [time, classes].")
    if time_mask.ndim != 1 or logits.shape[0] != time_mask.shape[0]:
        raise ValueError("Logits and time_mask must share the time dimension.")
    if not logits.is_floating_point():
        raise TypeError("'logits' must be floating point.")
    if time_mask.dtype != torch.bool:
        raise TypeError("'time_mask' must use torch.bool.")
    if logits.device != time_mask.device:
        raise ValueError("'logits' and 'time_mask' must use the same device.")

    probabilities = torch.softmax(logits, dim=-1)
    return probabilities.masked_fill(~time_mask.unsqueeze(-1), 0.0)


def causal_smooth_probabilities(
    probabilities: torch.Tensor,
    time_mask: torch.Tensor,
    window_steps: int,
) -> torch.Tensor:
    """Average current-and-past probabilities using a mask-aware window.

    The output at time t never depends on t+1 or any later value. Missing current
    timestamps remain zero and therefore cannot become artificial phase evidence.
    """
    if probabilities.ndim != 2:
        raise ValueError("'probabilities' must have shape [time, classes].")
    if time_mask.ndim != 1 or probabilities.shape[0] != time_mask.shape[0]:
        raise ValueError("Probabilities and time_mask must share time length.")
    if window_steps <= 0:
        raise ValueError("'window_steps' must be positive.")
    if not probabilities.is_floating_point():
        raise TypeError("'probabilities' must be floating point.")
    if time_mask.dtype != torch.bool:
        raise TypeError("'time_mask' must use torch.bool.")
    if probabilities.device != time_mask.device:
        raise ValueError("Probabilities and time_mask must use the same device.")

    masked = probabilities * time_mask.unsqueeze(-1).to(probabilities.dtype)
    if window_steps == 1:
        return masked

    time_length, num_classes = probabilities.shape
    class_kernel = torch.ones(
        num_classes,
        1,
        window_steps,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    values = masked.transpose(0, 1).unsqueeze(0)
    sums = F.conv1d(
        F.pad(values, (window_steps - 1, 0)),
        class_kernel,
        groups=num_classes,
    )

    count_kernel = torch.ones(
        1,
        1,
        window_steps,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    counts = F.conv1d(
        F.pad(
            time_mask.to(probabilities.dtype).view(1, 1, time_length),
            (window_steps - 1, 0),
        ),
        count_kernel,
    )

    smoothed = (sums / counts.clamp_min(1.0)).squeeze(0).transpose(0, 1)
    return smoothed.masked_fill(~time_mask.unsqueeze(-1), 0.0)


def predictions_from_probabilities(
    probabilities: torch.Tensor,
    time_mask: torch.Tensor,
) -> torch.Tensor:
    """Return phase IDs and reserve -1 for completely unavailable timestamps."""
    if probabilities.ndim != 2:
        raise ValueError("'probabilities' must have shape [time, classes].")
    if time_mask.ndim != 1 or probabilities.shape[0] != time_mask.shape[0]:
        raise ValueError("Probabilities and time_mask must share time length.")

    predictions = probabilities.argmax(dim=-1).to(torch.int64)
    return predictions.masked_fill(~time_mask, UNAVAILABLE_LABEL_ID)


def causal_debounce_predictions(
    predictions: torch.Tensor,
    time_mask: torch.Tensor,
    min_duration_steps: int,
) -> torch.Tensor:
    """Suppress short label changes using only evidence seen so far.

    A new phase must persist for min_duration_steps before becoming confirmed.
    This removes brief islands but introduces a bounded confirmation delay. Fully
    missing timestamps are emitted as -1 and never treated as Empty.
    """
    if predictions.ndim != 1 or time_mask.ndim != 1:
        raise ValueError("Predictions and time_mask must have shape [time].")
    if predictions.shape != time_mask.shape:
        raise ValueError("Predictions and time_mask must share shape.")
    if predictions.dtype != torch.int64:
        raise TypeError("'predictions' must use torch.int64.")
    if time_mask.dtype != torch.bool:
        raise TypeError("'time_mask' must use torch.bool.")
    if min_duration_steps <= 0:
        raise ValueError("'min_duration_steps' must be positive.")

    cleaned = torch.full_like(predictions, UNAVAILABLE_LABEL_ID)
    confirmed_label: int | None = None
    candidate_label: int | None = None
    candidate_count = 0

    for index in range(predictions.numel()):
        if not bool(time_mask[index]):
            candidate_label = None
            candidate_count = 0
            continue

        current_label = int(predictions[index].item())
        if confirmed_label is None:
            confirmed_label = current_label
            cleaned[index] = confirmed_label
            continue

        if current_label == confirmed_label:
            candidate_label = None
            candidate_count = 0
            cleaned[index] = confirmed_label
            continue

        if candidate_label != current_label:
            candidate_label = current_label
            candidate_count = 1
        else:
            candidate_count += 1

        if candidate_count >= min_duration_steps:
            confirmed_label = candidate_label
            candidate_label = None
            candidate_count = 0

        cleaned[index] = confirmed_label

    return cleaned


def _segment_ranges(labels: torch.Tensor) -> list[tuple[int, int, int]]:
    if labels.ndim != 1:
        raise ValueError("'labels' must have shape [time].")
    if labels.numel() == 0:
        return []

    ranges: list[tuple[int, int, int]] = []
    start = 0
    current = int(labels[0].item())
    for index in range(1, labels.numel()):
        label = int(labels[index].item())
        if label != current:
            ranges.append((current, start, index))
            start = index
            current = label
    ranges.append((current, start, labels.numel()))
    return ranges


def labels_to_timeline(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    timestamps: torch.Tensor,
    phase_names: Sequence[str],
    sampling_rate_hz: float,
) -> tuple[TimelineSegment, ...]:
    """Convert per-timestamp labels into contiguous human-readable segments."""
    if labels.ndim != 1:
        raise ValueError("'labels' must have shape [time].")
    if probabilities.ndim != 2:
        raise ValueError("'probabilities' must have shape [time, classes].")
    if timestamps.ndim != 1:
        raise ValueError("'timestamps' must have shape [time].")
    if labels.shape[0] != probabilities.shape[0] or labels.shape[0] != timestamps.shape[0]:
        raise ValueError("Labels, probabilities, and timestamps must share time length.")
    if probabilities.shape[1] != len(phase_names):
        raise ValueError("Probability class count must match phase_names.")
    if sampling_rate_hz <= 0:
        raise ValueError("'sampling_rate_hz' must be positive.")
    if timestamps.numel() > 1 and not torch.all(timestamps[1:] > timestamps[:-1]):
        raise ValueError("Timestamps must be strictly increasing.")

    segments: list[TimelineSegment] = []
    final_step_seconds = 1.0 / sampling_rate_hz

    for label_id, start, end in _segment_ranges(labels):
        start_seconds = float(timestamps[start].item())
        if end < timestamps.numel():
            end_seconds = float(timestamps[end].item())
        else:
            end_seconds = float(timestamps[-1].item()) + final_step_seconds

        is_available = label_id != UNAVAILABLE_LABEL_ID
        if is_available:
            if not 0 <= label_id < len(phase_names):
                raise ValueError(f"Invalid phase label ID: {label_id}")
            label_name = str(phase_names[label_id])
            mean_confidence = float(
                probabilities[start:end, label_id].mean().item()
            )
        else:
            label_name = UNAVAILABLE_LABEL_NAME
            mean_confidence = 0.0

        segments.append(
            TimelineSegment(
                label_id=label_id,
                label_name=label_name,
                start_index=start,
                end_index=end,
                start_seconds=round(start_seconds, 6),
                end_seconds=round(end_seconds, 6),
                duration_seconds=round(end_seconds - start_seconds, 6),
                mean_confidence=round(mean_confidence, 6),
                is_available=is_available,
            )
        )

    return tuple(segments)


def generate_timeline(
    logits: torch.Tensor,
    timestamps: torch.Tensor,
    time_mask: torch.Tensor,
    phase_names: Sequence[str],
    sampling_rate_hz: float,
    smoothing_window_seconds: float,
    min_segment_duration_seconds: float,
) -> TimelineResult:
    """Create a clean causal timeline from one sequence of frame-level logits."""
    _validate_sequence_inputs(logits, timestamps, time_mask)
    if logits.shape[1] != len(phase_names):
        raise ValueError("Logit class count must match phase_names.")
    if sampling_rate_hz <= 0:
        raise ValueError("'sampling_rate_hz' must be positive.")
    if smoothing_window_seconds <= 0 or min_segment_duration_seconds <= 0:
        raise ValueError("Post-processing durations must be positive.")

    smoothing_steps = max(1, math.ceil(smoothing_window_seconds * sampling_rate_hz))
    minimum_steps = max(
        1,
        math.ceil(min_segment_duration_seconds * sampling_rate_hz),
    )

    probabilities = logits_to_probabilities(logits, time_mask)
    smoothed_probabilities = causal_smooth_probabilities(
        probabilities,
        time_mask,
        smoothing_steps,
    )
    raw_predictions = predictions_from_probabilities(probabilities, time_mask)
    smoothed_predictions = predictions_from_probabilities(
        smoothed_probabilities,
        time_mask,
    )
    cleaned_predictions = causal_debounce_predictions(
        smoothed_predictions,
        time_mask,
        minimum_steps,
    )
    segments = labels_to_timeline(
        cleaned_predictions,
        smoothed_probabilities,
        timestamps,
        phase_names,
        sampling_rate_hz,
    )

    return TimelineResult(
        probabilities=probabilities,
        smoothed_probabilities=smoothed_probabilities,
        raw_predictions=raw_predictions,
        smoothed_predictions=smoothed_predictions,
        cleaned_predictions=cleaned_predictions,
        segments=segments,
    )


def generate_batch_timelines(
    logits: torch.Tensor,
    timestamps: torch.Tensor,
    time_mask: torch.Tensor,
    phase_names: Sequence[str],
    sampling_rate_hz: float,
    smoothing_window_seconds: float,
    min_segment_duration_seconds: float,
) -> tuple[TimelineResult, ...]:
    """Apply timeline generation independently to every batch item."""
    if logits.ndim != 3:
        raise ValueError("'logits' must have shape [batch, time, classes].")
    if timestamps.ndim != 2 or time_mask.ndim != 2:
        raise ValueError("'timestamps' and 'time_mask' must have [batch, time].")
    if logits.shape[:2] != timestamps.shape or logits.shape[:2] != time_mask.shape:
        raise ValueError("Batch and time dimensions must match.")

    return tuple(
        generate_timeline(
            logits=logits[index],
            timestamps=timestamps[index],
            time_mask=time_mask[index],
            phase_names=phase_names,
            sampling_rate_hz=sampling_rate_hz,
            smoothing_window_seconds=smoothing_window_seconds,
            min_segment_duration_seconds=min_segment_duration_seconds,
        )
        for index in range(logits.shape[0])
    )

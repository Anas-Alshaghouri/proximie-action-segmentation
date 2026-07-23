from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from action_segmentation.config import AppConfig


def generate_phase_prototypes(config: AppConfig) -> torch.Tensor:
    """Create deterministic class prototypes shared by every dataset split.

    The prototype vectors stand in for embeddings produced by a visual backbone.
    Keeping them shared across train, validation, and test makes the mock learning
    problem meaningful while sample-level noise still changes between sequences.
    """
    rng = np.random.default_rng(config.project.seed)
    prototypes = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(len(config.phases.names), config.data.feature_dim),
    ).astype(np.float32)

    prototypes -= prototypes.mean(axis=1, keepdims=True)
    prototypes /= prototypes.std(axis=1, keepdims=True) + 1e-6
    return torch.from_numpy(prototypes)


def _allocate_phase_durations(
    config: AppConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Allocate one contiguous duration to every workflow phase."""
    num_phases = len(config.phases.names)
    minimum_steps = math.ceil(
        config.data.minimum_phase_duration_seconds
        * config.data.sampling_rate_hz
    )
    durations = np.full(num_phases, minimum_steps, dtype=np.int64)
    remaining = config.data.sequence_length - int(durations.sum())

    weights = np.asarray(config.data.phase_duration_weights, dtype=np.float64)
    concentration = weights / weights.sum() * 40.0
    proportions = rng.dirichlet(concentration)
    raw_allocation = proportions * remaining
    extra_steps = np.floor(raw_allocation).astype(np.int64)
    durations += extra_steps

    unassigned = remaining - int(extra_steps.sum())
    if unassigned:
        fractional_order = np.argsort(raw_allocation - extra_steps)[::-1]
        durations[fractional_order[:unassigned]] += 1

    return durations


def _build_labels(durations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.concatenate(
        [
            np.full(duration, phase_id, dtype=np.int64)
            for phase_id, duration in enumerate(durations)
        ]
    )
    boundaries = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(durations)]
    )
    return labels, boundaries


def _build_base_features(
    labels: np.ndarray,
    boundaries: np.ndarray,
    prototypes: np.ndarray,
    config: AppConfig,
) -> np.ndarray:
    """Create phase-conditioned features with ambiguous transition regions."""
    base_features = prototypes[labels].copy()
    boundary_radius = round(
        config.data.boundary_noise_seconds * config.data.sampling_rate_hz
    )
    if boundary_radius <= 0:
        return base_features

    for next_phase_id, boundary in enumerate(boundaries[1:-1], start=1):
        start = max(0, int(boundary) - boundary_radius)
        end = min(config.data.sequence_length, int(boundary) + boundary_radius)
        if end - start <= 1:
            continue

        previous_prototype = prototypes[next_phase_id - 1]
        next_prototype = prototypes[next_phase_id]
        alpha = np.linspace(0.0, 1.0, end - start, dtype=np.float32)[:, None]
        base_features[start:end] = (
            (1.0 - alpha) * previous_prototype + alpha * next_prototype
        )

    return base_features


def _correlated_noise(
    rng: np.random.Generator,
    sequence_length: int,
    feature_dim: int,
    standard_deviation: float,
    correlation: float = 0.85,
) -> np.ndarray:
    """Generate smooth temporal noise instead of independent frame noise."""
    if standard_deviation == 0:
        return np.zeros((sequence_length, feature_dim), dtype=np.float32)

    white_noise = rng.normal(
        loc=0.0,
        scale=standard_deviation,
        size=(sequence_length, feature_dim),
    ).astype(np.float32)
    result = np.empty_like(white_noise)
    result[0] = white_noise[0]
    innovation_scale = math.sqrt(1.0 - correlation**2)

    for timestamp in range(1, sequence_length):
        result[timestamp] = (
            correlation * result[timestamp - 1]
            + innovation_scale * white_noise[timestamp]
        )
    return result


def _inject_patient_present_disturbances(
    features: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    phase_names: tuple[str, ...],
    rng: np.random.Generator,
    sampling_rate_hz: float,
) -> np.ndarray:
    """Add short background-noise events inside Patient Present.

    These events move the embedding toward Empty or Preparation without changing
    the ground-truth label. They create the type of local ambiguity that may lead
    to false positives and fragmented Patient Present predictions.
    """
    disturbance_mask = np.zeros(labels.shape[0], dtype=np.bool_)
    try:
        patient_id = phase_names.index("patient_present")
    except ValueError:
        return disturbance_mask

    patient_indices = np.flatnonzero(labels == patient_id)
    if patient_indices.size == 0:
        return disturbance_mask

    alternative_ids = [
        phase_names.index(name)
        for name in ("empty", "preparation")
        if name in phase_names
    ]
    if not alternative_ids:
        return disturbance_mask

    duration_seconds = patient_indices.size / sampling_rate_hz
    number_of_windows = max(1, min(3, round(duration_seconds / 90.0)))
    minimum_window = max(2, round(2 * sampling_rate_hz))
    maximum_window = max(minimum_window, round(7 * sampling_rate_hz))
    patient_start = int(patient_indices[0])
    patient_end = int(patient_indices[-1]) + 1

    for _ in range(number_of_windows):
        window_length = int(
            rng.integers(minimum_window, maximum_window + 1)
        )
        latest_start = max(patient_start, patient_end - window_length)
        start = int(rng.integers(patient_start, latest_start + 1))
        end = min(patient_end, start + window_length)
        target_id = int(rng.choice(alternative_ids))
        mixture = float(rng.uniform(0.45, 0.70))

        features[:, start:end] = (
            (1.0 - mixture) * features[:, start:end]
            + mixture * prototypes[target_id][None, None, :]
        )
        disturbance_mask[start:end] = True

    return disturbance_mask


def _apply_camera_dropout(
    view_mask: np.ndarray,
    physical_views: int,
    probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate complete stream loss while preserving at least one camera."""
    dropout_mask = np.zeros(view_mask.shape[0], dtype=np.bool_)
    if physical_views <= 1 or probability <= 0:
        return dropout_mask

    candidates = rng.random(physical_views) < probability
    if candidates.all():
        candidates[int(rng.integers(0, physical_views))] = False

    dropout_mask[:physical_views] = candidates
    view_mask[dropout_mask] = False
    return dropout_mask


def _apply_operation_occlusions(
    view_mask: np.ndarray,
    labels: np.ndarray,
    phase_names: tuple[str, ...],
    physical_views: int,
    probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create contiguous missing-view windows primarily during Operation."""
    occlusion_mask = np.zeros_like(view_mask, dtype=np.bool_)
    if probability <= 0 or "operation" not in phase_names:
        return occlusion_mask

    operation_id = phase_names.index("operation")
    operation_indices = np.flatnonzero(labels == operation_id)
    if operation_indices.size == 0:
        return occlusion_mask

    operation_start = int(operation_indices[0])
    operation_end = int(operation_indices[-1]) + 1
    target_steps = max(1, round(operation_indices.size * probability))

    for view_id in range(physical_views):
        if not view_mask[view_id].any():
            continue

        remaining = target_steps
        number_of_windows = 1 if target_steps < 8 else 2
        for window_index in range(number_of_windows):
            windows_left = number_of_windows - window_index
            minimum_length = max(1, remaining // windows_left)
            variation = max(1, minimum_length // 3)
            window_length = int(
                np.clip(
                    rng.integers(
                        max(1, minimum_length - variation),
                        minimum_length + variation + 1,
                    ),
                    1,
                    remaining,
                )
            )
            latest_start = max(operation_start, operation_end - window_length)
            start = int(rng.integers(operation_start, latest_start + 1))
            end = min(operation_end, start + window_length)

            view_mask[view_id, start:end] = False
            occlusion_mask[view_id, start:end] = True
            remaining -= end - start
            if remaining <= 0:
                break

    return occlusion_mask


def generate_synthetic_sample(
    config: AppConfig,
    prototypes: torch.Tensor,
    seed: int,
    sample_id: str,
) -> dict[str, Any]:
    """Generate one deterministic multi-view temporal sample."""
    rng = np.random.default_rng(seed)
    prototype_array = prototypes.numpy()

    phase_durations = _allocate_phase_durations(config, rng)
    labels, boundaries = _build_labels(phase_durations)
    base_features = _build_base_features(
        labels,
        boundaries,
        prototype_array,
        config,
    )

    physical_views = int(
        rng.integers(config.data.min_views, config.data.max_views + 1)
    )
    features = np.zeros(
        (
            config.data.max_views,
            config.data.sequence_length,
            config.data.feature_dim,
        ),
        dtype=np.float32,
    )
    view_mask = np.zeros(
        (config.data.max_views, config.data.sequence_length),
        dtype=np.bool_,
    )
    view_mask[:physical_views] = True

    for view_id in range(physical_views):
        view_bias = rng.normal(
            loc=0.0,
            scale=config.data.view_bias_std,
            size=(1, config.data.feature_dim),
        ).astype(np.float32)
        temporal_noise = _correlated_noise(
            rng=rng,
            sequence_length=config.data.sequence_length,
            feature_dim=config.data.feature_dim,
            standard_deviation=config.data.feature_noise_std,
        )
        features[view_id] = base_features + view_bias + temporal_noise

    patient_present_disturbance_mask = _inject_patient_present_disturbances(
        features=features,
        labels=labels,
        prototypes=prototype_array,
        phase_names=config.phases.names,
        rng=rng,
        sampling_rate_hz=config.data.sampling_rate_hz,
    )
    camera_dropout_mask = _apply_camera_dropout(
        view_mask=view_mask,
        physical_views=physical_views,
        probability=config.data.camera_dropout_probability,
        rng=rng,
    )
    occlusion_mask = _apply_operation_occlusions(
        view_mask=view_mask,
        labels=labels,
        phase_names=config.phases.names,
        physical_views=physical_views,
        probability=config.data.occlusion_probability,
        rng=rng,
    )

    features[~view_mask] = 0.0
    time_mask = view_mask.any(axis=0)
    timestamps = (
        np.arange(config.data.sequence_length, dtype=np.float32)
        / config.data.sampling_rate_hz
    )

    return {
        "sample_id": sample_id,
        "features": torch.from_numpy(features),
        "labels": torch.from_numpy(labels),
        "timestamps": torch.from_numpy(timestamps),
        "view_mask": torch.from_numpy(view_mask),
        "time_mask": torch.from_numpy(time_mask),
        "occlusion_mask": torch.from_numpy(occlusion_mask),
        "camera_dropout_mask": torch.from_numpy(camera_dropout_mask),
        "patient_present_disturbance_mask": torch.from_numpy(
            patient_present_disturbance_mask
        ),
        "phase_durations": torch.from_numpy(phase_durations),
        "phase_boundaries": torch.from_numpy(boundaries),
        "num_views": torch.tensor(physical_views, dtype=torch.int64),
    }

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


class TimelineVisualizationError(ValueError):
    """Raised when an evaluation report cannot be visualized."""


def load_evaluation_report(path: str | Path) -> dict[str, Any]:
    """Load and validate an evaluation JSON report."""
    report_path = Path(path).expanduser().resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"Evaluation report not found: {report_path}")

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TimelineVisualizationError(
            f"Evaluation report is not valid JSON: {report_path}"
        ) from exc

    timelines = payload.get("timelines")
    if not isinstance(timelines, list) or not timelines:
        raise TimelineVisualizationError(
            "Evaluation report must contain a non-empty 'timelines' list."
        )
    return payload


def select_timeline(
    report: Mapping[str, Any],
    *,
    sample_id: str | None = None,
    sample_index: int = 0,
) -> dict[str, Any]:
    """Select one timeline by ID or zero-based list index."""
    timelines = report.get("timelines")
    if not isinstance(timelines, list) or not timelines:
        raise TimelineVisualizationError("No timelines are available in the report.")

    if sample_id is not None:
        for item in timelines:
            if isinstance(item, dict) and item.get("sample_id") == sample_id:
                return item
        available = [
            str(item.get("sample_id"))
            for item in timelines
            if isinstance(item, dict)
        ]
        raise TimelineVisualizationError(
            f"Unknown sample ID '{sample_id}'. Available IDs: {', '.join(available)}"
        )

    if sample_index < 0 or sample_index >= len(timelines):
        raise TimelineVisualizationError(
            f"Sample index {sample_index} is outside [0, {len(timelines) - 1}]."
        )
    selected = timelines[sample_index]
    if not isinstance(selected, dict):
        raise TimelineVisualizationError("Selected timeline entry must be an object.")
    return selected


def _normalized_tracks(sample: Mapping[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    tracks = sample.get("tracks")
    if isinstance(tracks, dict):
        result: list[tuple[str, list[dict[str, Any]]]] = []
        for key, title in (
            ("ground_truth", "Ground truth"),
            ("raw_prediction", "Raw prediction"),
            ("cleaned_prediction", "Cleaned prediction"),
        ):
            value = tracks.get(key)
            if isinstance(value, list):
                result.append((title, value))
        if result:
            return result

    legacy = sample.get("segments")
    if isinstance(legacy, list):
        return [("Cleaned prediction", legacy)]

    raise TimelineVisualizationError(
        "Timeline entry does not contain visualization tracks or legacy segments."
    )


def _duration_seconds(sample: Mapping[str, Any], tracks: Sequence[tuple[str, list[dict[str, Any]]]]) -> float:
    explicit = sample.get("duration_seconds")
    if isinstance(explicit, (int, float)) and float(explicit) > 0:
        return float(explicit)

    end_values = [
        float(segment.get("end_seconds", 0.0))
        for _, segments in tracks
        for segment in segments
        if isinstance(segment, dict)
    ]
    if not end_values or max(end_values) <= 0:
        raise TimelineVisualizationError("Could not determine timeline duration.")
    return max(end_values)


def _phase_names(sample: Mapping[str, Any], tracks: Sequence[tuple[str, list[dict[str, Any]]]]) -> list[str]:
    configured = sample.get("phase_names")
    if isinstance(configured, list) and all(isinstance(value, str) for value in configured):
        names = list(configured)
    else:
        names = []
        for _, segments in tracks:
            for segment in segments:
                name = str(segment.get("label_name", "unknown"))
                if name != "unavailable" and name not in names:
                    names.append(name)
    return names


def _phase_color_map(phase_names: Sequence[str]) -> dict[str, Any]:
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_cycle:
        default_cycle = [f"C{index}" for index in range(max(1, len(phase_names)))]
    return {
        name: default_cycle[index % len(default_cycle)]
        for index, name in enumerate(phase_names)
    }


def _draw_phase_track(
    axis: Any,
    segments: Sequence[Mapping[str, Any]],
    y_position: float,
    color_map: Mapping[str, Any],
) -> None:
    for segment in segments:
        start = float(segment.get("start_seconds", 0.0))
        end = float(segment.get("end_seconds", start))
        duration = max(0.0, end - start)
        if duration <= 0:
            continue
        label_name = str(segment.get("label_name", "unknown"))
        if label_name == "unavailable" or not bool(segment.get("is_available", True)):
            axis.broken_barh(
                [(start, duration)],
                (y_position - 0.32, 0.64),
                facecolors="none",
                edgecolors=plt.rcParams["text.color"],
                hatch="////",
                linewidth=0.8,
            )
        else:
            axis.broken_barh(
                [(start, duration)],
                (y_position - 0.32, 0.64),
                facecolors=color_map.get(label_name),
                edgecolors="none",
            )


def _draw_camera_track(axis: Any, camera: Mapping[str, Any], y_position: float) -> None:
    intervals = camera.get("intervals")
    if not isinstance(intervals, list):
        return
    for interval in intervals:
        if not isinstance(interval, dict):
            continue
        start = float(interval.get("start_seconds", 0.0))
        end = float(interval.get("end_seconds", start))
        duration = max(0.0, end - start)
        if duration <= 0:
            continue
        available = bool(interval.get("available", False))
        axis.broken_barh(
            [(start, duration)],
            (y_position - 0.24, 0.48),
            facecolors="C5" if available else "none",
            edgecolors="none" if available else plt.rcParams["text.color"],
            hatch=None if available else "////",
            alpha=0.8,
            linewidth=0.7,
        )


def render_timeline_figure(
    sample: Mapping[str, Any],
    output_path: str | Path,
    *,
    show_confidence: bool = False,
    dpi: int = 160,
) -> Path:
    """Render aligned phase, camera, and optional confidence tracks to PNG."""
    if dpi <= 0:
        raise TimelineVisualizationError("DPI must be positive.")

    tracks = _normalized_tracks(sample)
    duration = _duration_seconds(sample, tracks)
    phase_names = _phase_names(sample, tracks)
    color_map = _phase_color_map(phase_names)
    cameras = sample.get("camera_availability")
    camera_rows = cameras if isinstance(cameras, list) else []
    confidence = sample.get("confidence") if show_confidence else None
    has_confidence = isinstance(confidence, dict)

    row_count = len(tracks) + len(camera_rows)
    if row_count == 0:
        raise TimelineVisualizationError("No timeline rows are available to plot.")

    height = max(4.0, 1.0 + 0.65 * row_count + (2.0 if has_confidence else 0.0))
    if has_confidence:
        figure, (timeline_axis, confidence_axis) = plt.subplots(
            2,
            1,
            figsize=(12, height),
            sharex=True,
            gridspec_kw={"height_ratios": [max(2, row_count), 2]},
            constrained_layout=True,
        )
    else:
        figure, timeline_axis = plt.subplots(
            1,
            1,
            figsize=(12, height),
            constrained_layout=True,
        )
        confidence_axis = None

    labels: list[str] = []
    y_positions: list[float] = []
    current_y = float(row_count)

    for title, segments in tracks:
        _draw_phase_track(timeline_axis, segments, current_y, color_map)
        labels.append(title)
        y_positions.append(current_y)
        current_y -= 1.0

    for camera in camera_rows:
        if not isinstance(camera, dict):
            continue
        _draw_camera_track(timeline_axis, camera, current_y)
        camera_index = camera.get("camera_index", len(labels))
        labels.append(f"Camera {camera_index} availability")
        y_positions.append(current_y)
        current_y -= 1.0

    timeline_axis.set_xlim(0.0, duration)
    timeline_axis.set_ylim(0.2, row_count + 0.8)
    timeline_axis.set_yticks(y_positions, labels)
    timeline_axis.set_xlabel("Time (seconds)")
    timeline_axis.grid(axis="x", alpha=0.2)
    timeline_axis.set_title(
        f"Workflow timeline: {sample.get('sample_id', 'selected sample')}",
        loc="left",
    )

    legend_items = [
        Patch(facecolor=color_map[name], label=name.replace("_", " ").title())
        for name in phase_names
    ]
    legend_items.extend(
        [
            Patch(facecolor="C5", alpha=0.8, label="Camera available"),
            Patch(facecolor="none", edgecolor=plt.rcParams["text.color"], hatch="////", label="Unavailable"),
        ]
    )
    timeline_axis.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(4, max(1, len(legend_items))),
        frameon=False,
    )

    if confidence_axis is not None and isinstance(confidence, dict):
        timestamps = confidence.get("timestamps_seconds")
        raw_max = confidence.get("raw_max_probability")
        cleaned_max = confidence.get("smoothed_max_probability")
        if isinstance(timestamps, list) and isinstance(raw_max, list):
            confidence_axis.plot(timestamps, raw_max, label="Raw max confidence")
        if isinstance(timestamps, list) and isinstance(cleaned_max, list):
            confidence_axis.plot(timestamps, cleaned_max, label="Smoothed max confidence")
        confidence_axis.set_ylim(0.0, 1.05)
        confidence_axis.set_ylabel("Confidence")
        confidence_axis.set_xlabel("Time (seconds)")
        confidence_axis.grid(alpha=0.2)
        confidence_axis.legend(frameon=False, loc="lower right")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output

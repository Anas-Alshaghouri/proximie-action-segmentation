import json
from pathlib import Path

import pytest

from action_segmentation.visualization.timeline_plot import (
    TimelineVisualizationError,
    load_evaluation_report,
    render_timeline_figure,
    select_timeline,
)


def _segment(label_id: int, label_name: str, start: float, end: float):
    return {
        "label_id": label_id,
        "label_name": label_name,
        "start_index": int(start),
        "end_index": int(end),
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": end - start,
        "mean_confidence": 0.9,
        "is_available": label_id >= 0,
    }


def _report():
    return {
        "timelines": [
            {
                "sample_id": "test_0000",
                "duration_seconds": 10.0,
                "phase_names": ["empty", "operation"],
                "tracks": {
                    "ground_truth": [
                        _segment(0, "empty", 0.0, 4.0),
                        _segment(1, "operation", 4.0, 10.0),
                    ],
                    "raw_prediction": [
                        _segment(0, "empty", 0.0, 5.0),
                        _segment(1, "operation", 5.0, 10.0),
                    ],
                    "cleaned_prediction": [
                        _segment(0, "empty", 0.0, 6.0),
                        _segment(1, "operation", 6.0, 10.0),
                    ],
                },
                "camera_availability": [
                    {
                        "camera_index": 0,
                        "intervals": [
                            {
                                "start_seconds": 0.0,
                                "end_seconds": 8.0,
                                "available": True,
                            },
                            {
                                "start_seconds": 8.0,
                                "end_seconds": 10.0,
                                "available": False,
                            },
                        ],
                    }
                ],
                "confidence": {
                    "timestamps_seconds": list(range(10)),
                    "raw_max_probability": [0.9] * 10,
                    "smoothed_max_probability": [0.85] * 10,
                },
            },
            {
                "sample_id": "test_0001",
                "segments": [_segment(0, "empty", 0.0, 10.0)],
            },
        ]
    }


def test_load_and_select_timeline_by_id(tmp_path: Path) -> None:
    report_path = tmp_path / "metrics.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")

    loaded = load_evaluation_report(report_path)
    selected = select_timeline(loaded, sample_id="test_0001")

    assert selected["sample_id"] == "test_0001"


def test_select_timeline_rejects_unknown_sample() -> None:
    with pytest.raises(TimelineVisualizationError, match="Unknown sample ID"):
        select_timeline(_report(), sample_id="missing")


def test_render_full_timeline_with_confidence(tmp_path: Path) -> None:
    output = render_timeline_figure(
        _report()["timelines"][0],
        tmp_path / "timeline.png",
        show_confidence=True,
        dpi=80,
    )

    assert output.is_file()
    assert output.stat().st_size > 1_000


def test_render_legacy_cleaned_timeline(tmp_path: Path) -> None:
    output = render_timeline_figure(
        _report()["timelines"][1],
        tmp_path / "legacy.png",
        dpi=80,
    )

    assert output.is_file()
    assert output.stat().st_size > 1_000

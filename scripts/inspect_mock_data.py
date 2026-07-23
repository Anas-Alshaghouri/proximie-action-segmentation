from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from action_segmentation.config import load_config
from action_segmentation.data import SyntheticTemporalDataset
from action_segmentation.seed import set_global_seed


def _contiguous_ranges(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Return inclusive-exclusive ranges where a 1D boolean mask is true."""
    indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    if not indices:
        return []

    ranges: list[tuple[int, int]] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            ranges.append((start, previous + 1))
            start = index
        previous = index
    ranges.append((start, previous + 1))
    return ranges


def _phase_segments(
    labels: torch.Tensor,
    phase_names: tuple[str, ...],
    sampling_rate_hz: float,
) -> list[dict[str, Any]]:
    """Convert frame-level labels into contiguous phase segments."""
    if labels.numel() == 0:
        return []

    segments: list[dict[str, Any]] = []
    start = 0
    current_label = int(labels[0].item())

    for index in range(1, labels.numel()):
        label = int(labels[index].item())
        if label == current_label:
            continue

        segments.append(
            {
                "phase": phase_names[current_label],
                "start_index": start,
                "end_index": index,
                "start_seconds": start / sampling_rate_hz,
                "end_seconds": index / sampling_rate_hz,
                "duration_seconds": (index - start) / sampling_rate_hz,
            }
        )
        start = index
        current_label = label

    end = labels.numel()
    segments.append(
        {
            "phase": phase_names[current_label],
            "start_index": start,
            "end_index": end,
            "start_seconds": start / sampling_rate_hz,
            "end_seconds": end / sampling_rate_hz,
            "duration_seconds": (end - start) / sampling_rate_hz,
        }
    )
    return segments


def _print_tensor_contract(sample: dict[str, Any]) -> None:
    print("\nTensor contract")
    print("-" * 72)
    for key in ("features", "labels", "timestamps", "view_mask", "time_mask"):
        tensor = sample[key]
        print(f"{key:12s} shape={tuple(tensor.shape)!s:18s} dtype={tensor.dtype}")


def _print_split_summary(dataset: SyntheticTemporalDataset) -> None:
    view_distribution: Counter[int] = Counter()
    valid_view_timestamps = 0
    total_view_timestamps = 0
    fully_missing_timestamps = 0
    total_timestamps = 0
    phase_counts = torch.zeros(
        len(dataset.config.phases.names), dtype=torch.int64
    )

    for sample in dataset:
        view_distribution[int(sample["num_views"].item())] += 1
        valid_view_timestamps += int(sample["view_mask"].sum().item())
        total_view_timestamps += sample["view_mask"].numel()
        fully_missing_timestamps += int((~sample["time_mask"]).sum().item())
        total_timestamps += sample["time_mask"].numel()
        phase_counts += torch.bincount(
            sample["labels"], minlength=len(dataset.config.phases.names)
        )

    print("Split summary")
    print("-" * 72)
    print(f"split:                       {dataset.split}")
    print(f"samples:                     {len(dataset)}")
    print(f"physical view distribution:  {dict(sorted(view_distribution.items()))}")
    print(
        "valid view timestamp ratio: "
        f"{valid_view_timestamps / max(total_view_timestamps, 1):.4f}"
    )
    print(
        "fully missing time ratio:    "
        f"{fully_missing_timestamps / max(total_timestamps, 1):.4f}"
    )
    print("phase timestamp counts:")
    for phase_name, count in zip(dataset.config.phases.names, phase_counts.tolist()):
        print(f"  {phase_name:18s} {count}")


def _print_selected_sample(
    sample: dict[str, Any],
    phase_names: tuple[str, ...],
    sampling_rate_hz: float,
    start: int,
    end: int,
    feature_values: int,
) -> None:
    features = sample["features"]
    labels = sample["labels"]
    timestamps = sample["timestamps"]
    view_mask = sample["view_mask"]
    time_mask = sample["time_mask"]

    end = min(end, labels.numel())
    start = max(0, min(start, end))
    feature_values = max(1, min(feature_values, features.shape[-1]))

    print("\nSelected sample")
    print("-" * 72)
    print(f"sample_id:       {sample['sample_id']}")
    print(f"physical views:  {int(sample['num_views'].item())}")
    print(f"time range:      [{start}, {end})")
    _print_tensor_contract(sample)

    print("\nGround-truth timeline")
    print("-" * 72)
    for segment in _phase_segments(labels, phase_names, sampling_rate_hz):
        print(
            f"{segment['phase']:18s} "
            f"indices [{segment['start_index']:3d}, {segment['end_index']:3d})  "
            f"seconds [{segment['start_seconds']:6.1f}, "
            f"{segment['end_seconds']:6.1f})  "
            f"duration={segment['duration_seconds']:6.1f}s"
        )

    print("\nFeature statistics by view, valid timestamps only")
    print("-" * 72)
    for view_index in range(features.shape[0]):
        valid = view_mask[view_index]
        valid_features = features[view_index][valid]
        if valid_features.numel() == 0:
            print(f"view {view_index}: no valid features")
            continue
        print(
            f"view {view_index}: valid={int(valid.sum().item()):3d}/"
            f"{valid.numel()}  mean={valid_features.mean().item(): .4f}  "
            f"std={valid_features.std().item(): .4f}  "
            f"min={valid_features.min().item(): .4f}  "
            f"max={valid_features.max().item(): .4f}"
        )

    print("\nMissing intervals")
    print("-" * 72)
    for view_index in range(view_mask.shape[0]):
        ranges = _contiguous_ranges(~view_mask[view_index])
        print(f"view {view_index}: {ranges if ranges else 'none'}")
    all_missing_ranges = _contiguous_ranges(~time_mask)
    print(f"all views missing: {all_missing_ranges if all_missing_ranges else 'none'}")

    print("\nSelected time slice")
    print("-" * 72)
    print("timestamps:")
    print(timestamps[start:end])
    print("labels (IDs):")
    print(labels[start:end])
    print("labels (names):")
    print([phase_names[int(label)] for label in labels[start:end].tolist()])
    print("view_mask [views, selected_time]:")
    print(view_mask[:, start:end])
    print("time_mask [selected_time]:")
    print(time_mask[start:end])
    print(
        f"features [views, selected_time, first {feature_values} dimensions]:"
    )
    print(features[:, start:end, :feature_values])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect the generated synthetic temporal dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=20)
    parser.add_argument(
        "--feature-values",
        type=int,
        default=8,
        help="Number of feature dimensions printed per timestamp.",
    )
    parser.add_argument(
        "--skip-split-summary",
        action="store_true",
        help="Skip iterating over the complete split.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional .pt path for saving the selected sample.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    set_global_seed(config.project.seed)
    dataset = SyntheticTemporalDataset(config=config, split=args.split)

    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(
            f"sample-index must be between 0 and {len(dataset) - 1}."
        )

    if not args.skip_split_summary:
        _print_split_summary(dataset)

    sample = dataset[args.sample_index]
    _print_selected_sample(
        sample=sample,
        phase_names=config.phases.names,
        sampling_rate_hz=config.data.sampling_rate_hz,
        start=args.start,
        end=args.end,
        feature_values=args.feature_values,
    )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sample, args.save)
        print(f"\nSaved selected sample to: {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from action_segmentation.config import ConfigurationError, load_config
from action_segmentation.evaluation.evaluator import evaluate_checkpoint
from action_segmentation.logging_utils import configure_logging
from action_segmentation.seed import set_global_seed

LOGGER = logging.getLogger(__name__)


def build_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained causal TCN checkpoint and export timelines and metrics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path; defaults to artifacts_directory/checkpoint_filename.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Metrics JSON path; defaults to artifacts_directory/metrics_filename.",
    )
    return parser


def evaluate_main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = build_evaluate_parser().parse_args(argv)

    try:
        config = load_config(args.config)
        set_global_seed(config.project.seed)
        checkpoint = args.checkpoint or (
            config.output.artifacts_directory / config.output.checkpoint_filename
        )
        output_path = args.output or (
            config.output.artifacts_directory / config.output.metrics_filename
        )
        result = evaluate_checkpoint(
            config,
            checkpoint_path=checkpoint,
            split=args.split,
        )
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except (
        ConfigurationError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        LOGGER.error("Evaluation failed: %s", exc)
        return 1

    LOGGER.info("Trained checkpoint evaluated successfully on the %s split.", args.split)
    LOGGER.info("Metrics and timelines saved to: %s", output_path)
    print(json.dumps({key: value for key, value in result.items() if key != "timelines"}, indent=2))
    LOGGER.info("End-to-end evaluation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(evaluate_main())

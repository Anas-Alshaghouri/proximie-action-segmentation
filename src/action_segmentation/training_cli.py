from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from action_segmentation.config import ConfigurationError, load_config
from action_segmentation.logging_utils import configure_logging
from action_segmentation.seed import set_global_seed
from action_segmentation.training.trainer import train_model

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the lightweight causal TCN on synthetic multi-view features."
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
        help="Optional checkpoint output path; defaults to the configured artifact path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        set_global_seed(config.project.seed)
        result = train_model(config, checkpoint_path=args.checkpoint)
    except (
        ConfigurationError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        LOGGER.error("Training failed: %s", exc)
        return 1

    print(json.dumps(result.to_dict(), indent=2))
    LOGGER.info("Training completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

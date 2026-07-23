from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from action_segmentation.config import ConfigurationError, load_config
from action_segmentation.data.ingestion import FeatureIngestionError
from action_segmentation.logging_utils import configure_logging
from action_segmentation.pipeline import (
    error_analysis_readiness_summary,
    fusion_readiness_summary,
    metric_stack_readiness_summary,
    repository_readiness_summary,
    synthetic_data_readiness_summary,
    temporal_model_readiness_summary,
    timeline_readiness_summary,
)
from action_segmentation.seed import set_global_seed

LOGGER = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Validate the complete prototype architecture without training.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        set_global_seed(config.project.seed)
        summary = {
            "contract": repository_readiness_summary(config),
            "synthetic_data": synthetic_data_readiness_summary(config),
            "multi_view_fusion": fusion_readiness_summary(config),
            "temporal_model": temporal_model_readiness_summary(config),
            "timeline_postprocessing": timeline_readiness_summary(config),
            "evaluation_metrics": metric_stack_readiness_summary(config),
            "error_analysis": error_analysis_readiness_summary(config),
        }
    except (
        ConfigurationError,
        FeatureIngestionError,
        FileNotFoundError,
        OSError,
        TypeError,
        RuntimeError,
        ValueError,
    ) as exc:
        LOGGER.error("Pipeline validation failed: %s", exc)
        return 1
    print(json.dumps(summary, indent=2))
    LOGGER.info("Pipeline validation completed successfully.")
    return 0

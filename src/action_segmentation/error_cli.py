from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from action_segmentation.config import load_config
from action_segmentation.evaluation.error_analysis import format_error_analysis_report
from action_segmentation.pipeline import error_analysis_readiness_summary
from action_segmentation.seed import set_global_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run mock temporal error analysis across the validation split."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the full JSON report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    set_global_seed(config.project.seed)
    report = error_analysis_readiness_summary(config)
    print(format_error_analysis_report(report))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nFull JSON report saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

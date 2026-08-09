#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace

from _bootstrap import ROOT  # noqa: F401
from dit_research.config import load_config
from dit_research.training import Trainer, with_max_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a class-conditional DiT experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="strict dotted config override; repeat for multiple values",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", default=None, help="checkpoint path or 'auto'")
    parser.add_argument("--device", default=None, help="override runtime.device")
    parser.add_argument("--output-root", default=None, help="override experiment.output_root")
    parser.add_argument(
        "--allow-code-change",
        action="store_true",
        help="resume despite a changed training-code hash after manual review",
    )
    parser.add_argument(
        "--allow-environment-change",
        action="store_true",
        help="resume despite changed actual precision/device after manual review",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    config = with_max_steps(config, args.max_steps)
    if args.device is not None:
        config = replace(config, runtime=replace(config.runtime, device=args.device))
    if args.output_root is not None:
        config = replace(
            config,
            experiment=replace(config.experiment, output_root=args.output_root),
        )
    trainer = Trainer(
        config,
        resume=args.resume,
        allow_code_change=args.allow_code_change,
        allow_environment_change=args.allow_environment_change,
    )
    trainer.run()


if __name__ == "__main__":
    main()

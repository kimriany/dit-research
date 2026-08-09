#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from _bootstrap import ROOT  # noqa: F401
from dit_research.config import load_config
from dit_research.evaluation.benchmark import benchmark_model
from dit_research.evaluation.complexity import analytic_complexity, assert_exact_match
from dit_research.evaluation.quality import clean_fid, torch_fidelity_metrics
from dit_research.model import build_model
from dit_research.utils import atomic_json_dump, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DiT complexity, speed, or sample quality")
    subparsers = parser.add_subparsers(dest="command", required=True)

    complexity = subparsers.add_parser("complexity")
    complexity.add_argument("--config", action="append", required=True)
    complexity.add_argument("--assert-matched", action="store_true")
    complexity.add_argument("--output", default=None)

    throughput = subparsers.add_parser("throughput")
    throughput.add_argument("--config", required=True)
    throughput.add_argument("--mode", choices=("forward", "train"), default="train")
    throughput.add_argument("--batch-size", type=int, default=16)
    throughput.add_argument("--warmup", type=int, default=50)
    throughput.add_argument("--iterations", type=int, default=200)
    throughput.add_argument("--repeats", type=int, default=1)
    throughput.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="override accumulation for batch-fit benchmarking",
    )
    throughput.add_argument("--device", default="auto")
    throughput.add_argument("--output", default=None)

    fid = subparsers.add_parser("fid")
    fid.add_argument("--samples", required=True)
    fid.add_argument("--split", choices=("train", "test"), default="train")
    fid.add_argument("--mode", default="clean")
    fid.add_argument("--expected-count", type=int, default=None)
    fid.add_argument("--with-torch-fidelity", action="store_true")
    fid.add_argument(
        "--allow-unmanifested",
        action="store_true",
        help="allow an external sample folder without the generation manifest",
    )
    fid.add_argument("--cpu", action="store_true")
    fid.add_argument("--output", default=None)
    return parser.parse_args()


def write_or_print(payload: Any, output: str | None, *, merge: bool = False) -> None:
    if output is not None:
        path = Path(output)
        if merge and path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            existing.update(payload)
            payload = existing
        atomic_json_dump(payload, path)
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.command == "complexity":
        results = []
        for path in args.config:
            config = load_config(path)
            model = build_model(config.model, config.diffusion)
            results.append(
                {
                    "config": path,
                    "experiment": config.experiment.name,
                    **analytic_complexity(model, config.model),
                }
            )
        if args.assert_matched:
            assert_exact_match(results)
        write_or_print({"matched": bool(args.assert_matched), "models": results}, args.output)
        return

    if args.command == "throughput":
        config = load_config(args.config)
        if args.grad_accum_steps is not None:
            if args.grad_accum_steps <= 0:
                raise ValueError("grad-accum-steps must be positive")
            config = replace(
                config,
                train=replace(config.train, grad_accum_steps=args.grad_accum_steps),
            )
        model = build_model(config.model, config.diffusion)
        result = benchmark_model(
            model,
            config,
            resolve_device(args.device),
            mode=args.mode,
            batch_size=args.batch_size,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        write_or_print(result, args.output)
        return

    require_manifest = not args.allow_unmanifested
    result = clean_fid(
        args.samples,
        split=args.split,
        mode=args.mode,
        require_manifest=require_manifest,
        expected_count=args.expected_count,
    )
    if args.with_torch_fidelity:
        result.update(
            torch_fidelity_metrics(
                args.samples,
                split=args.split,
                cuda=not args.cpu,
                require_manifest=require_manifest,
                expected_count=args.expected_count,
            )
        )
    write_or_print(result, args.output, merge=True)


if __name__ == "__main__":
    main()

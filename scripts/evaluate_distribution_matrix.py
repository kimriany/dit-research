#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from _bootstrap import ROOT
from dit_research.config import load_config
from run_matrix import load_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print or execute KID and precision/recall for every run in a matrix"
    )
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--sample-subdir", default="fid_samples_50k")
    parser.add_argument("--expected-count", type=int, default=50_000)
    parser.add_argument("--expected-reference-count", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pr-sample-count", type=int, default=10_000)
    parser.add_argument("--distance-chunk-size", type=int, default=1000)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def metrics_are_complete(path: Path, expected_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("sample_count") == expected_count
        and all(isinstance(payload.get(key), (int, float)) for key in ("kid", "precision", "recall"))
    )


def main() -> None:
    args = parse_args()
    if args.expected_count <= 0 or args.expected_reference_count <= 0:
        raise ValueError("expected counts must be positive")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    matrix = load_matrix(args.matrix)
    selected_ids = set(args.only)
    known_ids = {str(run.get("id")) for run in matrix["runs"]}
    unknown_ids = selected_ids - known_ids
    if unknown_ids:
        raise ValueError(f"unknown matrix run IDs: {sorted(unknown_ids)}")
    reference = Path(args.reference)
    if args.execute and not reference.is_dir():
        raise FileNotFoundError(reference)

    commands: list[list[str]] = []
    skipped: list[Path] = []
    for run in matrix["runs"]:
        run_id = str(run["id"])
        if selected_ids and run_id not in selected_ids:
            continue
        config_path = Path(run["config"])
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        config = load_config(config_path)
        output_root = Path(config.experiment.output_root)
        if not output_root.is_absolute():
            output_root = ROOT / output_root
        for seed in run["seeds"]:
            run_name = f"{matrix['phase']}_{run_id}_seed{seed}"
            run_directory = output_root / run_name
            metrics_path = run_directory / "final_metrics.json"
            if args.skip_complete and metrics_are_complete(metrics_path, args.expected_count):
                skipped.append(metrics_path)
                continue
            command = [
                sys.executable,
                str(ROOT / "scripts" / "evaluate.py"),
                "distribution",
                "--samples",
                str(run_directory / args.sample_subdir),
                "--reference",
                str(reference),
                "--expected-count",
                str(args.expected_count),
                "--expected-reference-count",
                str(args.expected_reference_count),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--pr-sample-count",
                str(args.pr_sample_count),
                "--distance-chunk-size",
                str(args.distance_chunk_size),
                "--output",
                str(metrics_path),
            ]
            if args.cpu:
                command.append("--cpu")
            commands.append(command)
    for path in skipped:
        print(f"skip complete: {path}", flush=True)
    if not commands and not skipped:
        raise ValueError("matrix selection produced no evaluation commands")
    for command in commands:
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

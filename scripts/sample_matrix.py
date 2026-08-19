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
from dit_research.evaluation.quality import _validate_sample_directory
from run_matrix import load_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print or execute balanced sampling for every run in an experiment matrix"
    )
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument(
        "--expected-checkpoint-step",
        type=int,
        default=None,
        help="only treat an existing sample directory as complete at this checkpoint step",
    )
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument(
        "--restart-incomplete",
        action="store_true",
        help="quarantine a partial sample directory before regenerating it",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def samples_are_complete(
    path: Path, expected_count: int, expected_checkpoint_step: int | None = None
) -> bool:
    try:
        _validate_sample_directory(path, expected_count=expected_count)
    except (FileNotFoundError, ValueError, TypeError):
        return False
    if expected_checkpoint_step is not None:
        manifest_path = path / "sample_manifest.json"
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return False
        if manifest.get("checkpoint_step") != expected_checkpoint_step:
            return False
    return True


def quarantine_directory(path: Path) -> Path:
    candidate = path.with_name(f"{path.name}.incomplete")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.incomplete-{suffix}")
        suffix += 1
    path.rename(candidate)
    return candidate


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("num-samples and batch-size must be positive")
    if args.expected_checkpoint_step is not None and args.expected_checkpoint_step <= 0:
        raise ValueError("expected-checkpoint-step must be positive")
    if args.restart_incomplete and not args.skip_complete:
        raise ValueError("restart-incomplete requires skip-complete")
    matrix = load_matrix(args.matrix)
    selected_ids = set(args.only)
    known_ids = {str(run.get("id")) for run in matrix["runs"]}
    unknown_ids = selected_ids - known_ids
    if unknown_ids:
        raise ValueError(f"unknown matrix run IDs: {sorted(unknown_ids)}")
    count_label = (
        f"{args.num_samples // 1000}k"
        if args.num_samples % 1000 == 0
        else str(args.num_samples)
    )
    output_subdir = args.output_subdir or f"fid_samples_{count_label}"

    commands: list[list[str]] = []
    checkpoints: list[Path] = []
    skipped: list[Path] = []
    incomplete: list[Path] = []
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
            sample_directory = run_directory / output_subdir
            if args.skip_complete and samples_are_complete(
                sample_directory, args.num_samples, args.expected_checkpoint_step
            ):
                skipped.append(sample_directory)
                continue
            if (
                args.restart_incomplete
                and sample_directory.is_dir()
                and any(sample_directory.glob("*.png"))
            ):
                incomplete.append(sample_directory)
            checkpoint = run_directory / "checkpoints" / "latest.pt"
            checkpoints.append(checkpoint)
            commands.append(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "sample.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--num-samples",
                    str(args.num_samples),
                    "--batch-size",
                    str(args.batch_size),
                    "--output",
                    str(sample_directory),
                ]
            )
    if args.execute:
        missing = [path for path in checkpoints if not path.is_file()]
        if missing:
            formatted = "\n".join(str(path) for path in missing)
            raise FileNotFoundError(
                "every selected checkpoint must exist before matrix sampling starts:\n"
                f"{formatted}"
            )
        for path in incomplete:
            quarantined = quarantine_directory(path)
            print(f"quarantine incomplete: {path} -> {quarantined}", flush=True)
    else:
        for path in incomplete:
            print(f"would quarantine incomplete: {path}", flush=True)
    for path in skipped:
        print(f"skip complete: {path}", flush=True)
    if not commands and not skipped:
        raise ValueError("matrix selection produced no sampling commands")
    for command in commands:
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

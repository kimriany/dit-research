#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from _bootstrap import ROOT
from dit_research.config import load_config
from run_matrix import load_matrix


EXPECTED_GROUPS = {"e1_uniform_b", "e3_front_b", "a1_reverse_b"}
EXPECTED_SEEDS = {42, 123, 777, 2026, 9001}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked tapered-FFN confirmation from existing shakedown "
            "checkpoints through training, sampling, evaluation, and summaries"
        )
    )
    parser.add_argument(
        "--matrix",
        default=str(ROOT / "configs" / "matrices" / "ffn_b_confirmation_template.yaml"),
    )
    parser.add_argument(
        "--reference", default=str(ROOT / "datasets" / "cifar10_train_png")
    )
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--sample-batch-size", type=int, default=256)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def run_records(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run in matrix["runs"]:
        config_path = Path(run["config"])
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        config = load_config(config_path)
        output_root = Path(config.experiment.output_root)
        if not output_root.is_absolute():
            output_root = ROOT / output_root
        for seed in run["seeds"]:
            run_name = f"{matrix['phase']}_{run['id']}_seed{seed}"
            directory = output_root / run_name
            records.append(
                {
                    "group": run["id"],
                    "seed": seed,
                    "directory": directory,
                    "checkpoint": directory / "checkpoints" / "latest.pt",
                    "metrics": directory / "final_metrics.json",
                }
            )
    return records


def validate_locked_matrix(matrix: dict[str, Any]) -> None:
    if matrix["template"]:
        raise ValueError("confirmation matrix is still locked as a template")
    if matrix["phase"] != "ffn_confirmation":
        raise ValueError(f"unexpected phase: {matrix['phase']}")
    if matrix["max_steps"] != 200_000:
        raise ValueError(f"confirmation budget must be 200000, got {matrix['max_steps']}")
    groups = {str(run["id"]) for run in matrix["runs"]}
    if groups != EXPECTED_GROUPS:
        raise ValueError(f"confirmation groups={sorted(groups)}; expected={sorted(EXPECTED_GROUPS)}")
    for run in matrix["runs"]:
        seeds = {int(seed) for seed in run["seeds"]}
        if seeds != EXPECTED_SEEDS:
            raise ValueError(
                f"{run['id']} seeds={sorted(seeds)}; expected={sorted(EXPECTED_SEEDS)}"
            )


def print_or_run(stage: str, command: list[str], execute: bool) -> None:
    print(f"\n[pipeline] stage={stage}", flush=True)
    print(shlex.join(command), flush=True)
    if execute:
        subprocess.run(command, cwd=ROOT, check=True)


def load_metrics(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid metrics file: {path}") from error


def validate_training(records: list[dict[str, Any]], expected_step: int) -> None:
    problems: list[str] = []
    for record in records:
        checkpoint = record["checkpoint"]
        metrics_path = record["metrics"]
        if not checkpoint.is_file():
            problems.append(f"missing checkpoint: {checkpoint}")
            continue
        metrics = load_metrics(metrics_path)
        if metrics.get("step") != expected_step:
            problems.append(f"step={metrics.get('step')}: {metrics_path}")
        for key in ("validation_failures", "preview_failures", "skipped_updates"):
            if metrics.get(key) != 0:
                problems.append(f"{key}={metrics.get(key)}: {metrics_path}")
        for key in ("train_loss", "validation_loss"):
            value = metrics.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                problems.append(f"non-finite {key}: {metrics_path}")
    if problems:
        raise RuntimeError("training validation failed:\n" + "\n".join(problems))


def validate_distribution(records: list[dict[str, Any]], expected_count: int) -> None:
    problems: list[str] = []
    for record in records:
        metrics_path = record["metrics"]
        metrics = load_metrics(metrics_path)
        if metrics.get("step") != 200_000:
            problems.append(f"step={metrics.get('step')}: {metrics_path}")
        if metrics.get("sample_count") != expected_count:
            problems.append(f"sample_count={metrics.get('sample_count')}: {metrics_path}")
        for key in ("fid", "kid", "precision", "recall"):
            value = metrics.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                problems.append(f"missing/non-finite {key}: {metrics_path}")
    if problems:
        raise RuntimeError("distribution validation failed:\n" + "\n".join(problems))


def main() -> None:
    args = parse_args()
    positive = (
        args.train_batch_size,
        args.grad_accum_steps,
        args.num_samples,
        args.sample_batch_size,
        args.evaluation_batch_size,
    )
    if min(positive) <= 0 or args.num_workers < 0:
        raise ValueError("batch sizes, accumulation, and sample count must be positive")

    matrix_path = Path(args.matrix)
    matrix = load_matrix(matrix_path)
    validate_locked_matrix(matrix)
    records = run_records(matrix)
    if len(records) != 15:
        raise ValueError(f"confirmation must contain 15 runs, got {len(records)}")

    execute_flag = ["--execute"] if args.execute else []
    print_or_run(
        "train_200k",
        [
            sys.executable,
            str(ROOT / "scripts" / "run_matrix.py"),
            "--matrix",
            str(matrix_path),
            "--batch-size",
            str(args.train_batch_size),
            "--grad-accum-steps",
            str(args.grad_accum_steps),
            "--resume-existing",
            "--skip-complete",
            *execute_flag,
        ],
        args.execute,
    )
    if args.execute:
        validate_training(records, matrix["max_steps"])

    print_or_run(
        "sample_50k",
        [
            sys.executable,
            str(ROOT / "scripts" / "sample_matrix.py"),
            "--matrix",
            str(matrix_path),
            "--num-samples",
            str(args.num_samples),
            "--batch-size",
            str(args.sample_batch_size),
            "--output-subdir",
            "fid_samples_50k",
            "--expected-checkpoint-step",
            str(matrix["max_steps"]),
            "--skip-complete",
            "--restart-incomplete",
            *execute_flag,
        ],
        args.execute,
    )

    print_or_run(
        "evaluate_distribution",
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_distribution_matrix.py"),
            "--matrix",
            str(matrix_path),
            "--reference",
            str(Path(args.reference)),
            "--sample-subdir",
            "fid_samples_50k",
            "--expected-count",
            str(args.num_samples),
            "--expected-reference-count",
            "50000",
            "--batch-size",
            str(args.evaluation_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--skip-complete",
            *execute_flag,
        ],
        args.execute,
    )
    if args.execute:
        validate_distribution(records, args.num_samples)

    inputs = [str(record["metrics"]) for record in records]
    expected_seeds = ",".join(str(seed) for seed in sorted(EXPECTED_SEEDS))
    for control, suffix in (("e1_uniform_b", "e1"), ("a1_reverse_b", "a1")):
        print_or_run(
            f"summarize_vs_{suffix}",
            [
                sys.executable,
                str(ROOT / "scripts" / "summarize_results.py"),
                *inputs,
                "--control-group",
                control,
                "--phase",
                "ffn_confirmation",
                "--step",
                str(matrix["max_steps"]),
                "--expected-sample-count",
                str(args.num_samples),
                "--expected-seeds",
                expected_seeds,
                "--output",
                str(ROOT / "results" / f"ffn_b_5seed_vs_{suffix}.csv"),
                "--raw-output",
                str(ROOT / "results" / f"ffn_b_5seed_runs_vs_{suffix}.csv"),
            ],
            args.execute,
        )

    if args.execute:
        print("\n[pipeline] complete", flush=True)


if __name__ == "__main__":
    main()

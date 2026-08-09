#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from _bootstrap import ROOT
from dit_research.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print or execute a sequential experiment matrix")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--only", action="append", default=[], help="only run this matrix ID")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="override matrix max_steps for staged 2k/10k runs",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="append --resume auto; every selected run must already have a checkpoint",
    )
    parser.add_argument(
        "--skip-backend-check",
        action="store_true",
        help="skip the automatic FLA parity preflight for fused-memory matrices",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    return parser.parse_args()


def load_matrix(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError("matrix must be a mapping")
    expected = {"name", "template", "phase", "max_steps", "runs"}
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if unknown or missing:
        raise ValueError(f"matrix keys mismatch; unknown={sorted(unknown)}, missing={sorted(missing)}")
    if not isinstance(raw["runs"], list) or not raw["runs"]:
        raise ValueError("matrix.runs must be a non-empty list")
    if type(raw["name"]) is not str or type(raw["phase"]) is not str:
        raise TypeError("matrix name and phase must be strings")
    if type(raw["template"]) is not bool:
        raise TypeError("matrix template must be boolean")
    if type(raw["max_steps"]) is not int or raw["max_steps"] <= 0:
        raise TypeError("matrix max_steps must be a positive integer")
    return raw


def main() -> None:
    args = parse_args()
    matrix = load_matrix(args.matrix)
    if args.execute and matrix["template"]:
        raise RuntimeError("refusing to execute a template matrix; select the model and set template: false")
    max_steps = args.max_steps if args.max_steps is not None else matrix["max_steps"]
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.grad_accum_steps is not None and args.grad_accum_steps <= 0:
        raise ValueError("grad-accum-steps must be positive")
    selected_ids = set(args.only)
    commands: list[list[str]] = []
    resume_checkpoints: list[Path] = []
    requires_fla = False
    matrix_run_ids: set[str] = set()
    for run in matrix["runs"]:
        if set(run) != {"id", "config", "seeds"}:
            raise ValueError(f"run must contain exactly id/config/seeds: {run}")
        if type(run["id"]) is not str or type(run["config"]) is not str:
            raise TypeError("run id and config must be strings")
        if not isinstance(run["seeds"], list) or not run["seeds"]:
            raise TypeError("run seeds must be a non-empty list")
        if any(type(seed) is not int or seed < 0 for seed in run["seeds"]):
            raise TypeError("run seeds must be non-negative integers")
        if run["id"] in matrix_run_ids:
            raise ValueError(f"duplicate matrix run id: {run['id']}")
        matrix_run_ids.add(run["id"])
        if len(set(run["seeds"])) != len(run["seeds"]):
            raise ValueError(f"run {run['id']} contains duplicate seeds")
        if selected_ids and run["id"] not in selected_ids:
            continue
        config_path = ROOT / run["config"]
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        experiment_config = load_config(config_path)
        requires_fla = requires_fla or experiment_config.model.memory.backend == "fla"
        output_root = Path(experiment_config.experiment.output_root)
        if not output_root.is_absolute():
            output_root = ROOT / output_root
        for seed in run["seeds"]:
            run_name = f"{matrix['phase']}_{run['id']}_seed{seed}"
            commands.append(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "train.py"),
                    "--config",
                    str(config_path),
                    "--set",
                    f"experiment.name={run_name}",
                    "--set",
                    f"experiment.phase={matrix['phase']}",
                    "--set",
                    f"seeds.base={seed}",
                    "--max-steps",
                    str(max_steps),
                ]
            )
            if args.resume_existing:
                commands[-1].extend(("--resume", "auto"))
                resume_checkpoints.append(
                    output_root / run_name / "checkpoints" / "latest.pt"
                )
            if args.batch_size is not None:
                commands[-1].extend(("--set", f"train.batch_size={args.batch_size}"))
            if args.grad_accum_steps is not None:
                commands[-1].extend(
                    ("--set", f"train.grad_accum_steps={args.grad_accum_steps}")
                )
    if not commands:
        raise ValueError("matrix selection produced no commands")
    if args.resume_existing:
        missing_checkpoints = [
            checkpoint for checkpoint in resume_checkpoints if not checkpoint.is_file()
        ]
        if missing_checkpoints:
            formatted = "\n".join(str(path) for path in missing_checkpoints)
            raise FileNotFoundError(
                "resume-existing requires every selected checkpoint before any run starts:\n"
                f"{formatted}"
            )
    if requires_fla and not args.skip_backend_check:
        preflight = [
            sys.executable,
            str(ROOT / "scripts" / "check_memory_backend.py"),
            "--require-sm120",
        ]
        print(shlex.join(preflight), flush=True)
        if args.execute:
            subprocess.run(preflight, cwd=ROOT, check=True)
    for command in commands:
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

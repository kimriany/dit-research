#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from _bootstrap import ROOT


METRICS = (
    "train_loss",
    "validation_loss",
    "validation_loss_tq1",
    "validation_loss_tq2",
    "validation_loss_tq3",
    "validation_loss_tq4",
    "fid",
    "kid",
    "precision",
    "recall",
    "kernel_inception_distance_mean",
    "inception_score_mean",
    "parameters_trainable",
    "gmacs_per_image",
    "wall_seconds",
    "gpu_hours",
    "peak_allocated_mb",
    "peak_reserved_mb",
)
PAIRED_METRICS = (
    "train_loss",
    "validation_loss",
    "validation_loss_tq1",
    "validation_loss_tq2",
    "validation_loss_tq3",
    "validation_loss_tq4",
    "fid",
    "kid",
    "precision",
    "recall",
    "kernel_inception_distance_mean",
    "inception_score_mean",
    "wall_seconds",
    "gpu_hours",
)

# Two-sided 95% Student-t critical values for paired differences across
# independent training seeds.
# Current studies use n=5 or n=10; the wider table keeps the summary utility
# well-defined for later fixed cohorts without adding a SciPy dependency.
T_CRITICAL_95 = {
    1: 12.706204736,
    2: 4.30265273,
    3: 3.182446305,
    4: 2.776445105,
    5: 2.570581836,
    6: 2.446911851,
    7: 2.364624252,
    8: 2.306004135,
    9: 2.262157163,
    10: 2.228138852,
    11: 2.20098516,
    12: 2.17881283,
    13: 2.160368656,
    14: 2.144786688,
    15: 2.131449546,
    16: 2.119905299,
    17: 2.109815578,
    18: 2.10092204,
    19: 2.093024054,
    20: 2.085963447,
    21: 2.079613845,
    22: 2.073873068,
    23: 2.06865761,
    24: 2.063898562,
    25: 2.059538553,
    26: 2.055529439,
    27: 2.051830516,
    28: 2.048407142,
    29: 2.045229642,
    30: 2.042272456,
}


def mean_t95_interval(values: list[float]) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    degrees_of_freedom = len(values) - 1
    critical = T_CRITICAL_95.get(degrees_of_freedom, 1.959963985)
    mean = statistics.mean(values)
    margin = critical * statistics.stdev(values) / math.sqrt(len(values))
    return mean - margin, mean + margin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-run final_metrics JSON files")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", default=str(ROOT / "results" / "summary.csv"))
    parser.add_argument("--raw-output", default=str(ROOT / "results" / "runs.csv"))
    parser.add_argument("--control-group", default="e0_original")
    parser.add_argument("--phase", default=None, help="include only this experiment phase")
    parser.add_argument(
        "--pool-phases-as",
        default=None,
        help=(
            "replace the phase of all selected rows with this label after filtering; "
            "use only for an explicitly reported pooled estimate"
        ),
    )
    parser.add_argument("--step", type=int, default=None, help="include only this update budget")
    parser.add_argument(
        "--expected-sample-count",
        type=int,
        default=None,
        help="require this sample_count for every row containing FID",
    )
    parser.add_argument(
        "--expected-seeds",
        default=None,
        help="comma-separated exact seed set required independently for every group",
    )
    parser.add_argument(
        "--allow-validation-failures",
        action="store_true",
        help="include runs whose validation_failures is nonzero",
    )
    parser.add_argument(
        "--allow-missing-control",
        action="store_true",
        help="allow rows without a same-phase, same-step, same-seed control",
    )
    return parser.parse_args()


def load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for encoded in paths:
        path = Path(encoded)
        with path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        row["source"] = str(path)
        rows.append(row)
    return rows


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.inputs)
    if args.phase is not None:
        rows = [row for row in rows if row.get("phase") == args.phase]
    if args.step is not None:
        rows = [row for row in rows if row.get("step") == args.step]
    if not rows:
        raise ValueError("result filters selected no runs")
    if args.pool_phases_as is not None:
        pooled_phase = args.pool_phases_as.strip()
        if not pooled_phase:
            raise ValueError("pool-phases-as must be a non-empty label")
        for row in rows:
            row["source_phase"] = row.get("phase")
            row["phase"] = pooled_phase
    if not args.allow_validation_failures:
        failed = [
            row.get("source", row.get("experiment"))
            for row in rows
            if int(row.get("validation_failures", 0)) != 0
        ]
        if failed:
            raise ValueError(f"runs contain validation failures: {failed}")
    if args.expected_sample_count is not None:
        bad_samples = [
            row.get("source", row.get("experiment"))
            for row in rows
            if not isinstance(row.get("fid"), (int, float))
            or row.get("sample_count") != args.expected_sample_count
        ]
        if bad_samples:
            raise ValueError(
                f"FID rows do not have sample_count={args.expected_sample_count}: {bad_samples}"
            )

    unique_runs: dict[tuple[object, object, str, int], str] = {}
    for row in rows:
        seed = row.get("seed")
        if not isinstance(seed, int):
            continue
        group = str(row.get("group", row.get("experiment", "unknown")))
        key = (row.get("phase"), row.get("step"), group, seed)
        source = str(row.get("source", row.get("experiment", "unknown")))
        if key in unique_runs:
            raise ValueError(
                "duplicate run for phase/step/group/seed "
                f"{key}: {unique_runs[key]} and {source}"
            )
        unique_runs[key] = source

    controls: dict[tuple[object, object, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("group") != args.control_group or not isinstance(row.get("seed"), int):
            continue
        key = (row.get("phase"), row.get("step"), int(row["seed"]))
        controls[key] = row
    missing_controls = []
    for row in rows:
        seed = row.get("seed")
        control = (
            controls.get((row.get("phase"), row.get("step"), int(seed)))
            if isinstance(seed, int)
            else None
        )
        if row.get("group") == args.control_group:
            continue
        if control is None:
            missing_controls.append(row.get("source", row.get("experiment")))
            continue
        row["paired_control_group"] = args.control_group
        for metric in PAIRED_METRICS:
            if isinstance(row.get(metric), (int, float)) and isinstance(
                control.get(metric), (int, float)
            ):
                row[f"delta_{metric}_vs_control"] = float(row[metric]) - float(control[metric])
    if missing_controls and not args.allow_missing_control:
        raise ValueError(
            "runs are missing a same-phase, same-step, same-seed control: "
            f"{missing_controls}"
        )
    write_csv(args.raw_output, rows)
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_key = (
            str(row.get("phase", "unknown")),
            int(row.get("step", -1)),
            str(row.get("group", row.get("experiment", "unknown"))),
        )
        groups[group_key].append(row)
    expected_seeds = None
    if args.expected_seeds is not None:
        expected_seeds = {
            int(value.strip())
            for value in args.expected_seeds.split(",")
            if value.strip()
        }
        if not expected_seeds:
            raise ValueError("expected-seeds must contain at least one integer")
    summary = []
    for (phase, step, group), members in sorted(groups.items()):
        if expected_seeds is not None:
            actual_seeds = {
                int(row["seed"]) for row in members if isinstance(row.get("seed"), int)
            }
            if actual_seeds != expected_seeds:
                raise ValueError(
                    f"{phase}/{step}/{group} seeds={sorted(actual_seeds)}; "
                    f"expected={sorted(expected_seeds)}"
                )
        item: dict[str, Any] = {
            "phase": phase,
            "step": step,
            "group": group,
            "run_count": len(members),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in members if isinstance(row.get(metric), (int, float))]
            if values:
                item[f"{metric}_mean"] = statistics.mean(values)
                item[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        for metric in PAIRED_METRICS:
            key = f"delta_{metric}_vs_control"
            values = [float(row[key]) for row in members if isinstance(row.get(key), (int, float))]
            if values:
                item[f"{key}_mean"] = statistics.mean(values)
                item[f"{key}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
                interval = mean_t95_interval(values)
                if interval is not None:
                    item[f"{key}_ci95_low"] = interval[0]
                    item[f"{key}_ci95_high"] = interval[1]
        summary.append(item)
    write_csv(args.output, summary)
    print(f"wrote {len(rows)} runs to {args.raw_output}")
    print(f"wrote {len(summary)} groups to {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from _path import ROOT


class ResultSummaryTests(unittest.TestCase):
    def test_duplicate_group_seed_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                {"phase": "pilot", "step": 25000, "group": "e0_original", "seed": 11},
                {"phase": "pilot", "step": 25000, "group": "m2_adaptive", "seed": 11},
                {"phase": "pilot", "step": 25000, "group": "m2_adaptive", "seed": 11},
            ]
            paths = []
            for index, row in enumerate(rows):
                path = root / f"run_{index}.json"
                path.write_text(json.dumps(row), encoding="utf-8")
                paths.append(str(path))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "summarize_results.py"),
                    *paths,
                    "--output",
                    str(root / "summary.csv"),
                    "--raw-output",
                    str(root / "runs.csv"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("duplicate run for phase/step/group/seed", completed.stderr)

    def test_explicit_phase_pooling_preserves_source_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                {
                    "phase": phase,
                    "step": 50000,
                    "group": group,
                    "seed": seed,
                    "fid": fid,
                    "sample_count": 50000,
                }
                for phase, seed, control_fid, treatment_fid in (
                    ("confirmation", 42, 70.0, 68.0),
                    ("replication", 1001, 69.0, 66.0),
                )
                for group, fid in (
                    ("m0_coupled", control_fid),
                    ("m1_separated", treatment_fid),
                )
            ]
            paths = []
            for index, row in enumerate(rows):
                path = root / f"run_{index}.json"
                path.write_text(json.dumps(row), encoding="utf-8")
                paths.append(str(path))
            summary_path = root / "summary.csv"
            raw_path = root / "runs.csv"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "summarize_results.py"),
                    *paths,
                    "--step",
                    "50000",
                    "--expected-seeds",
                    "42,1001",
                    "--expected-sample-count",
                    "50000",
                    "--control-group",
                    "m0_coupled",
                    "--pool-phases-as",
                    "pooled",
                    "--output",
                    str(summary_path),
                    "--raw-output",
                    str(raw_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with summary_path.open(newline="", encoding="utf-8") as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual({row["phase"] for row in summary}, {"pooled"})
            self.assertEqual({row["run_count"] for row in summary}, {"2"})
            with raw_path.open(newline="", encoding="utf-8") as handle:
                raw = list(csv.DictReader(handle))
            self.assertEqual(
                {row["source_phase"] for row in raw}, {"confirmation", "replication"}
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()

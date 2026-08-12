from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from _path import ROOT


class RunMatrixTests(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        with (ROOT / "configs" / "smoke" / "dit_tiny_memory.yaml").open(
            "r", encoding="utf-8"
        ) as handle:
            config = yaml.safe_load(handle)
        config["experiment"]["output_root"] = str(root / "outputs")
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    def _run(self, matrix: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_matrix.py"),
                "--matrix",
                str(matrix),
                *arguments,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_resume_existing_checks_the_whole_cohort_before_running(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._write_config(root)
            matrix = root / "matrix.yaml"
            matrix.write_text(
                yaml.safe_dump(
                    {
                        "name": "resume_test",
                        "template": False,
                        "phase": "pilot",
                        "max_steps": 10,
                        "runs": [{"id": "m2", "config": str(config), "seeds": [11]}],
                    }
                ),
                encoding="utf-8",
            )
            completed = self._run(matrix, "--resume-existing")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "resume-existing requires every selected checkpoint", completed.stderr
            )

    def test_duplicate_run_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._write_config(root)
            matrix = root / "matrix.yaml"
            matrix.write_text(
                yaml.safe_dump(
                    {
                        "name": "duplicate_test",
                        "template": False,
                        "phase": "pilot",
                        "max_steps": 10,
                        "runs": [
                            {"id": "m2", "config": str(config), "seeds": [11]},
                            {"id": "m2", "config": str(config), "seeds": [12]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            completed = self._run(matrix)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("duplicate matrix run id", completed.stderr)

    def test_followup_matrix_expands_to_thirteen_train_sample_and_eval_commands(self) -> None:
        matrix = ROOT / "configs" / "matrices" / "memory_followup_50k.yaml"
        training = self._run(matrix, "--batch-size", "64", "--grad-accum-steps", "2")
        self.assertEqual(training.returncode, 0, training.stderr)
        self.assertEqual(training.stdout.count("scripts/train.py"), 13)
        self.assertIn("train.batch_size=64", training.stdout)

        sampling = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "sample_matrix.py"),
                "--matrix",
                str(matrix),
                "--num-samples",
                "50000",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(sampling.returncode, 0, sampling.stderr)
        self.assertEqual(sampling.stdout.count("scripts/sample.py"), 13)

        evaluation = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_distribution_matrix.py"),
                "--matrix",
                str(matrix),
                "--reference",
                "datasets/cifar10_train_png",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(evaluation.returncode, 0, evaluation.stderr)
        self.assertEqual(evaluation.stdout.count("scripts/evaluate.py"), 13)


if __name__ == "__main__":
    unittest.main()

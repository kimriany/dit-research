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

    def test_skip_complete_avoids_reopening_a_finished_training_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._write_config(root)
            matrix = root / "matrix.yaml"
            matrix.write_text(
                yaml.safe_dump(
                    {
                        "name": "skip_test",
                        "template": False,
                        "phase": "pilot",
                        "max_steps": 10,
                        "runs": [{"id": "m2", "config": str(config), "seeds": [11]}],
                    }
                ),
                encoding="utf-8",
            )
            run_directory = root / "outputs" / "pilot_m2_seed11"
            checkpoint = run_directory / "checkpoints" / "latest.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            (run_directory / "final_metrics.json").write_text(
                '{"step": 10}', encoding="utf-8"
            )

            completed = self._run(
                matrix, "--resume-existing", "--skip-complete"
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("skip complete training", completed.stdout)
            self.assertNotIn("scripts/train.py", completed.stdout)

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

    def test_separation_replication_expands_to_ten_train_sample_and_eval_commands(self) -> None:
        matrix = ROOT / "configs" / "matrices" / "memory_separation_replication_50k.yaml"
        training = self._run(matrix, "--batch-size", "64", "--grad-accum-steps", "2")
        self.assertEqual(training.returncode, 0, training.stderr)
        self.assertEqual(training.stdout.count("scripts/train.py"), 10)
        self.assertEqual(training.stdout.count("dit_s2_coupled.yaml"), 5)
        self.assertEqual(training.stdout.count("dit_s2_separated.yaml"), 5)

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
        self.assertEqual(sampling.stdout.count("scripts/sample.py"), 10)

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
        self.assertEqual(evaluation.stdout.count("scripts/evaluate.py"), 10)

    def test_ffn_b_matrices_expand_to_calibration_and_fifteen_confirmation_runs(self) -> None:
        calibration = ROOT / "configs" / "matrices" / "ffn_b_calibration_100k.yaml"
        training = self._run(
            calibration,
            "--max-steps",
            "500",
            "--batch-size",
            "64",
            "--grad-accum-steps",
            "2",
        )
        self.assertEqual(training.returncode, 0, training.stderr)
        self.assertEqual(training.stdout.count("scripts/train.py"), 1)
        self.assertIn("ffn_calibration_e1_uniform_b_seed11", training.stdout)
        self.assertIn(
            "hidden_size: 768",
            (ROOT / "configs" / "ffn" / "dit_b_uniform_r5.yaml").read_text(),
        )

        confirmation = (
            ROOT / "configs" / "matrices" / "ffn_b_confirmation_template.yaml"
        )
        training = self._run(confirmation)
        self.assertEqual(training.returncode, 0, training.stderr)
        self.assertEqual(training.stdout.count("scripts/train.py"), 15)
        self.assertEqual(training.stdout.count("dit_b_uniform_r5.yaml"), 5)
        self.assertEqual(training.stdout.count("dit_b_front_b.yaml"), 5)
        self.assertEqual(training.stdout.count("dit_b_reverse_b.yaml"), 5)

        sampling = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "sample_matrix.py"),
                "--matrix",
                str(confirmation),
                "--num-samples",
                "50000",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(sampling.returncode, 0, sampling.stderr)
        self.assertEqual(sampling.stdout.count("scripts/sample.py"), 15)

        evaluation = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_distribution_matrix.py"),
                "--matrix",
                str(confirmation),
                "--reference",
                "datasets/cifar10_train_png",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(evaluation.returncode, 0, evaluation.stderr)
        self.assertEqual(evaluation.stdout.count("scripts/evaluate.py"), 15)

        with confirmation.open("r", encoding="utf-8") as handle:
            resolved_confirmation = yaml.safe_load(handle)
        self.assertFalse(resolved_confirmation["template"])
        self.assertEqual(resolved_confirmation["max_steps"], 200000)

        with tempfile.TemporaryDirectory() as temporary:
            template_matrix = Path(temporary) / "template.yaml"
            resolved_confirmation["template"] = True
            template_matrix.write_text(
                yaml.safe_dump(resolved_confirmation), encoding="utf-8"
            )
            blocked = self._run(template_matrix, "--execute")
        self.assertNotEqual(blocked.returncode, 0)
        self.assertIn("refusing to execute a template matrix", blocked.stderr)


if __name__ == "__main__":
    unittest.main()

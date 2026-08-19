from __future__ import annotations

import subprocess
import sys
import unittest

from _path import ROOT


class FfnPipelineTests(unittest.TestCase):
    def test_locked_confirmation_pipeline_dry_run_lists_all_stages(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_ffn_confirmation_pipeline.py")],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("stage=train_200k", completed.stdout)
        self.assertIn("--resume-existing --skip-complete", completed.stdout)
        self.assertIn("stage=sample_50k", completed.stdout)
        self.assertIn("--expected-checkpoint-step 200000", completed.stdout)
        self.assertIn("--restart-incomplete", completed.stdout)
        self.assertIn("stage=evaluate_distribution", completed.stdout)
        self.assertIn("stage=summarize_vs_e1", completed.stdout)
        self.assertIn("stage=summarize_vs_a1", completed.stdout)
        self.assertNotIn(" --execute", completed.stdout)


if __name__ == "__main__":
    unittest.main()

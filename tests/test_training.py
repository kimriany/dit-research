from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from _path import ROOT  # noqa: F401
from dit_research.config import (
    AllocationConfig,
    DataConfig,
    DiffusionConfig,
    EvaluationConfig,
    ExperimentConfig,
    ExperimentMeta,
    ModelConfig,
    RuntimeConfig,
    SamplingConfig,
    SeedsConfig,
    TrainConfig,
)
from dit_research.training import Trainer


def tiny_training_config(root: str, max_steps: int, name: str = "resume_test") -> ExperimentConfig:
    return ExperimentConfig(
        experiment=ExperimentMeta(name, "test", "smoke", root),
        data=DataConfig("fake", "./datasets", 0.2, 7, 20, 0, False),
        model=ModelConfig(8, 3, 2, 32, 3, 4, 10, 0.1, 2.0, AllocationConfig()),
        diffusion=DiffusionConfig(10, "linear", 0.0001, 0.02),
        train=TrainConfig(2, 1, max_steps, 0.0001, 0.0, 0.9, "fp32", 1.0, 1, 1, 1, 1, 2),
        sampling=SamplingConfig("ddim", 2, 0.0, 1.0, 2, True),
        evaluation=EvaluationConfig(),
        runtime=RuntimeConfig("cpu", False, False, False),
        seeds=SeedsConfig(3),
    )


class TrainingTests(unittest.TestCase):
    def assert_nested_equal(self, expected: object, actual: object) -> None:
        if isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor)
            self.assertTrue(expected.equal(actual))
        elif isinstance(expected, dict):
            self.assertIsInstance(actual, dict)
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self.assert_nested_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertIsInstance(actual, type(expected))
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(expected, actual)

    def test_train_checkpoint_and_exact_single_gpu_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            uninterrupted_config = tiny_training_config(temporary, 6, "uninterrupted")
            uninterrupted = Trainer(uninterrupted_config)
            uninterrupted.run()

            first_config = tiny_training_config(temporary, 3, "resume_test")
            first_config.validate()
            first = Trainer(first_config)
            first.run()
            checkpoint = Path(temporary) / "resume_test" / "checkpoints" / "latest.pt"
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(first.step, 3)

            second_config = replace(first_config, train=replace(first_config.train, max_steps=6))
            second = Trainer(second_config, resume="auto")
            self.assertEqual(second.step, 3)
            second.run()
            self.assertEqual(second.step, 6)
            self.assertEqual(second.ema.num_updates, 6)
            self.assertEqual(second.microbatches_consumed, 6)
            self.assertEqual(second.latest_metrics["steps_remaining"], 0)
            self.assertEqual(second.latest_metrics["progress_percent"], 100.0)
            self.assertEqual(second.latest_metrics["epochs_completed"], 1)
            self.assertEqual(second.latest_metrics["epoch"], 2)
            self.assertEqual(second.latest_metrics["epoch_progress_percent"], 20.0)
            self.assertIn("eta_human", second.latest_metrics)
            with (Path(temporary) / "resume_test" / "metrics.jsonl").open() as handle:
                events = [json.loads(line) for line in handle]
            epoch_events = [event for event in events if event["event"] == "epoch_complete"]
            self.assertEqual([event["epoch"] for event in epoch_events], [1])

            for name, expected in uninterrupted.model.state_dict().items():
                with self.subTest(model_key=name):
                    self.assertTrue(expected.equal(second.model.state_dict()[name]))
            for name, expected in uninterrupted.ema.model.state_dict().items():
                with self.subTest(ema_key=name):
                    self.assertTrue(expected.equal(second.ema.model.state_dict()[name]))
            self.assert_nested_equal(
                uninterrupted.optimizer.state_dict(), second.optimizer.state_dict()
            )
            self.assert_nested_equal(uninterrupted.scaler.state_dict(), second.scaler.state_dict())


if __name__ == "__main__":
    unittest.main()

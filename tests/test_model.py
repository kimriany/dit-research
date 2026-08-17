from __future__ import annotations

import unittest

import torch

from _path import ROOT  # noqa: F401
from dit_research.config import AllocationConfig, ModelConfig
from dit_research.evaluation.complexity import analytic_complexity
from dit_research.model import build_model, resolve_ffn_widths


def small_model_config(allocation: str = "uniform", strength: float = 0.0) -> ModelConfig:
    return ModelConfig(
        image_size=8,
        in_channels=3,
        patch_size=2,
        hidden_size=48,
        depth=6,
        num_heads=4,
        num_classes=10,
        class_dropout_prob=0.1,
        mlp_ratio=5.0,
        allocation=AllocationConfig(allocation, strength, 8),
    )


class ModelTests(unittest.TestCase):
    def test_documented_s2_stage_schedules(self) -> None:
        expected = {
            0.5: (2112,) * 4 + (1920,) * 4 + (1728,) * 4,
            1.0: (2304,) * 4 + (1920,) * 4 + (1536,) * 4,
            2.0: (2688,) * 4 + (1920,) * 4 + (1152,) * 4,
        }
        for strength, widths in expected.items():
            self.assertEqual(
                resolve_ffn_widths(384, 12, 5.0, "frontloaded", strength, 8),
                widths,
            )
            self.assertEqual(sum(widths), 12 * 1920)

    def test_documented_b_stage_schedules(self) -> None:
        uniform = resolve_ffn_widths(768, 12, 5.0, "uniform", 0.0, 8)
        front = resolve_ffn_widths(768, 12, 5.0, "frontloaded", 1.0, 8)
        reverse = resolve_ffn_widths(768, 12, 5.0, "backloaded", 1.0, 8)
        self.assertEqual(uniform, (3840,) * 12)
        self.assertEqual(front, (4608,) * 4 + (3840,) * 4 + (3072,) * 4)
        self.assertEqual(reverse, tuple(reversed(front)))
        self.assertEqual({sum(uniform), sum(front), sum(reverse)}, {46080})

    def test_forward_shape_zero_init_and_backward_are_finite(self) -> None:
        config = small_model_config()
        model = build_model(config)
        images = torch.randn(2, 3, 8, 8)
        timesteps = torch.tensor([0, 9])
        labels = torch.tensor([0, 10])
        output = model(images, timesteps, labels)
        self.assertEqual(output.shape, images.shape)
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        loss = (output - torch.randn_like(output)).square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_invalid_label_is_rejected(self) -> None:
        model = build_model(small_model_config())
        with self.assertRaises(ValueError):
            model(
                torch.randn(1, 3, 8, 8),
                torch.tensor([1]),
                torch.tensor([11]),
            )

    def test_parameter_and_mac_budgets_match_exactly(self) -> None:
        configs = [
            small_model_config("uniform", 0.0),
            small_model_config("frontloaded", 0.5),
            small_model_config("frontloaded", 1.0),
            small_model_config("frontloaded", 2.0),
            small_model_config("backloaded", 1.0),
        ]
        stats = [analytic_complexity(build_model(config), config) for config in configs]
        self.assertEqual(len({item["parameters_trainable"] for item in stats}), 1)
        self.assertEqual(len({item["macs_per_image"] for item in stats}), 1)
        self.assertEqual(len({item["mlp_hidden_sum"] for item in stats}), 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from _path import ROOT  # noqa: F401
from dit_research.config import AllocationConfig, DiffusionConfig, MemoryConfig, ModelConfig
from dit_research.diffusion import GaussianDiffusion
from dit_research.evaluation.complexity import analytic_complexity, assert_exact_match
from dit_research.memory import (
    NoiseAdaptiveGatedDeltaMixer,
    make_scan_order,
    recurrent_gdn2_reference,
)
from dit_research.model import build_model


def tiny_memory_config(gate_mode: str = "adaptive") -> ModelConfig:
    return ModelConfig(
        image_size=8,
        in_channels=3,
        patch_size=2,
        hidden_size=48,
        depth=6,
        num_heads=4,
        num_classes=10,
        class_dropout_prob=0.1,
        mlp_ratio=2.0,
        allocation=AllocationConfig(),
        memory=MemoryConfig(
            kind="hybrid_gdn2",
            gate_mode=gate_mode,
            block_indices=(2, 5),
            gate_rank=16,
            lambda_hidden_size=8,
            backend="reference",
        ),
    )


def tiny_mixer(gate_mode: str) -> NoiseAdaptiveGatedDeltaMixer:
    return NoiseAdaptiveGatedDeltaMixer(
        hidden_size=16,
        num_heads=2,
        grid_size=2,
        direction="lr",
        gate_mode=gate_mode,
        gate_rank=8,
        lambda_hidden_size=4,
        backend="reference",
    )


class MemoryTests(unittest.TestCase):
    def test_all_scan_orders_have_exact_inverse(self) -> None:
        tokens = torch.arange(16)
        orders = []
        for direction in ("lr", "rl", "tb", "bt"):
            order, inverse = make_scan_order(4, direction)
            self.assertTrue(torch.equal(tokens[order][inverse], tokens))
            orders.append(tuple(order.tolist()))
        self.assertEqual(len(set(orders)), 4)
        self.assertEqual(
            orders,
            [
                (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12),
                (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15),
                (12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3),
            ],
        )

    def test_gate_mode_endpoints_and_parameter_budget(self) -> None:
        mixers = [
            tiny_mixer(mode) for mode in ("coupled", "separated", "static", "adaptive")
        ]
        parameters = [sum(parameter.numel() for parameter in mixer.parameters()) for mixer in mixers]
        self.assertEqual(len(set(parameters)), 1)
        noise = torch.tensor([-1.0, 1.0])
        coupled, separated, static, adaptive = [
            mixer.decoupling_strength(noise).reshape(-1) for mixer in mixers
        ]
        self.assertTrue(torch.equal(coupled, torch.zeros_like(coupled)))
        self.assertTrue(torch.equal(separated, torch.ones_like(separated)))
        self.assertTrue(torch.equal(static, torch.full_like(static, 0.5)))
        self.assertEqual(float(static.std(unbiased=False)), 0.0)
        self.assertTrue(torch.equal(adaptive, torch.full_like(adaptive, 0.5)))

        with torch.no_grad():
            for mixer in (mixers[2], mixers[3]):
                mixer.lambda_mlp[0].weight.fill_(1.0)
                mixer.lambda_mlp[0].bias.zero_()
                mixer.lambda_mlp[-1].weight.fill_(1.0)
                mixer.lambda_mlp[-1].bias.zero_()
        static_changed = mixers[2].decoupling_strength(noise).reshape(-1)
        adaptive_changed = mixers[3].decoupling_strength(noise).reshape(-1)
        self.assertEqual(float(static_changed.std(unbiased=False)), 0.0)
        self.assertGreater(float(adaptive_changed.std(unbiased=False)), 0.0)
        mixers[3].set_lambda_override(0.25)
        self.assertTrue(
            torch.equal(
                mixers[3].decoupling_strength(noise).reshape(-1),
                torch.full_like(noise, 0.25),
            )
        )
        mixers[3].set_lambda_override(None)
        self.assertTrue(
            torch.equal(
                mixers[3].decoupling_strength(noise),
                adaptive_changed.reshape(-1, 1, 1, 1),
            )
        )

    def test_reference_one_token_value_and_zero_norm_are_stable(self) -> None:
        q = torch.ones(1, 1, 1, 1)
        k = torch.ones_like(q)
        v = torch.full_like(q, 2.0)
        g = torch.zeros_like(q)
        b = torch.ones_like(q)
        w = torch.full_like(q, 0.5)
        output, state = recurrent_gdn2_reference(q, k, v, g, b, w)
        self.assertTrue(torch.allclose(output, torch.tensor([[[[1 / 1.000001]]]])))
        self.assertTrue(torch.isfinite(state).all())
        zero_output, zero_state = recurrent_gdn2_reference(
            torch.zeros_like(q), torch.zeros_like(k), v, g, b, w
        )
        self.assertTrue(torch.equal(zero_output, torch.zeros_like(zero_output)))
        self.assertTrue(torch.equal(zero_state, torch.zeros_like(zero_state)))

    def test_reference_forward_backward_is_finite_and_state_is_local(self) -> None:
        mixer = tiny_mixer("adaptive")
        mixer.set_diagnostics(True)
        x = torch.randn(2, 4, 16, requires_grad=True)
        noise = torch.tensor([-1.0, 1.0])
        first = mixer(x, noise)
        second = mixer(x, noise)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.shape, x.shape)
        first.square().mean().backward()
        gradients = [
            parameter.grad for parameter in mixer.parameters() if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        diagnostics = mixer.latest_diagnostics()
        self.assertIn("state_rms", diagnostics)
        self.assertTrue(
            all(bool(torch.isfinite(value).all()) for value in diagnostics.values())
        )

    def test_coupled_mode_has_identical_erase_and_write(self) -> None:
        mixer = tiny_mixer("coupled")
        mixer.set_diagnostics(True)
        mixer(torch.randn(2, 4, 16), torch.zeros(2))
        gap = mixer.latest_diagnostics()["erase_write_abs_gap"]
        self.assertTrue(torch.equal(gap, torch.zeros_like(gap)))

    def test_hybrid_model_assignment_log_snr_and_zero_init(self) -> None:
        diffusion = DiffusionConfig(10, "linear", 0.0001, 0.02)
        model = build_model(tiny_memory_config(), diffusion)
        self.assertEqual(model.memory_block_indices, (2, 5))
        self.assertEqual(model.scan_directions[2], "lr")
        self.assertEqual(model.scan_directions[5], "rl")
        self.assertIsNotNone(model.log_snr_table)
        self.assertTrue(torch.isfinite(model.log_snr_table).all())
        images = torch.randn(2, 3, 8, 8)
        output = model(images, torch.tensor([0, 9]), torch.tensor([0, 10]))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_memory_controls_match_parameters_and_analytic_macs(self) -> None:
        diffusion = DiffusionConfig(10, "linear", 0.0001, 0.02)
        base = tiny_memory_config("adaptive")
        configs = [
            replace(base, memory=replace(base.memory, gate_mode=mode))
            for mode in ("coupled", "separated", "static", "adaptive")
        ]
        stats = [
            analytic_complexity(build_model(config, diffusion), config) for config in configs
        ]
        assert_exact_match(stats)
        self.assertEqual(stats[0]["block_types"].count("gdn2_memory"), 2)

    def test_blockwise_mean_lambda_intervention_is_constant_per_block(self) -> None:
        diffusion = DiffusionConfig(10, "linear", 0.0001, 0.02)
        model = build_model(tiny_memory_config("adaptive"), diffusion)
        with torch.no_grad():
            for block_index in model.memory_block_indices:
                mixer = model.blocks[block_index].attention
                self.assertIsInstance(mixer, NoiseAdaptiveGatedDeltaMixer)
                mixer.lambda_mlp[0].weight.fill_(1.0)
                mixer.lambda_mlp[-1].weight.fill_(1.0)
        resolved = model.set_memory_intervention(blockwise_mean_lambda=True)
        self.assertEqual(set(resolved), {"b03", "b06"})
        noise = torch.tensor([-2.0, 0.0, 2.0])
        for block_index in model.memory_block_indices:
            mixer = model.blocks[block_index].attention
            separation = mixer.decoupling_strength(noise).reshape(-1)
            self.assertEqual(float(separation.std(unbiased=False)), 0.0)
            self.assertAlmostEqual(
                float(separation[0]), resolved[f"b{block_index + 1:02d}"]
            )
        model.set_memory_intervention()
        restored = model.blocks[model.memory_block_indices[0]].attention
        self.assertGreater(
            float(restored.decoupling_strength(noise).std(unbiased=False)), 0.0
        )

    def test_cfg_and_ddim_use_memory_model_without_cross_sample_state(self) -> None:
        diffusion_config = DiffusionConfig(10, "linear", 0.0001, 0.02)
        diffusion = GaussianDiffusion(diffusion_config)
        model = build_model(tiny_memory_config(), diffusion_config).eval()
        with torch.no_grad():
            model.final.projection.weight.normal_(std=0.02)
            for index in model.memory_block_indices:
                bias = model.blocks[index].ada_ln[-1].bias.reshape(6, model.hidden_size)
                bias[2].fill_(1.0)
        noisy = torch.randn(2, 3, 8, 8)
        timesteps = torch.tensor([3, 7])
        labels = torch.tensor([1, 2])
        conditional = model(noisy, timesteps, labels)
        null_labels = torch.full_like(labels, model.null_class_id)
        unconditional = model(noisy, timesteps, null_labels)
        expected = unconditional + 1.5 * (conditional - unconditional)
        actual = diffusion.predict_eps_cfg(model, noisy, timesteps, labels, 1.5)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        sampled = diffusion.ddim_sample_loop(
            model,
            noisy.shape,
            labels,
            steps=2,
            cfg_scale=1.5,
            initial_noise=noisy,
        )
        self.assertTrue(torch.isfinite(sampled).all())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import torch
from torch import nn

from _path import ROOT  # noqa: F401
from dit_research.config import DiffusionConfig
from dit_research.diffusion import GaussianDiffusion


class ZeroModel(nn.Module):
    null_class_id = 10

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        del timesteps, labels
        return torch.zeros_like(x)


class LabelModel(nn.Module):
    null_class_id = 10

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        del timesteps
        return labels.float().reshape(-1, 1, 1, 1).expand_as(x)


class DiffusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.diffusion = GaussianDiffusion(
            DiffusionConfig(timesteps=10, schedule="linear", beta_start=0.0001, beta_end=0.02)
        )

    def test_q_sample_matches_closed_form(self) -> None:
        clean = torch.ones(2, 1, 2, 2)
        noise = torch.full_like(clean, 0.25)
        timesteps = torch.tensor([0, 9])
        actual = self.diffusion.q_sample(clean, timesteps, noise)
        expected = (
            self.diffusion.sqrt_alpha_bars[timesteps].reshape(2, 1, 1, 1) * clean
            + self.diffusion.sqrt_one_minus_alpha_bars[timesteps].reshape(2, 1, 1, 1)
            * noise
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_cfg_scale_zero_and_one(self) -> None:
        model = LabelModel()
        x = torch.zeros(2, 1, 2, 2)
        timesteps = torch.zeros(2, dtype=torch.long)
        labels = torch.tensor([2, 4])
        scale_zero = self.diffusion.predict_eps_cfg(model, x, timesteps, labels, 0.0)
        scale_one = self.diffusion.predict_eps_cfg(model, x, timesteps, labels, 1.0)
        self.assertTrue(torch.equal(scale_zero, torch.full_like(x, 10.0)))
        self.assertTrue(torch.equal(scale_one[:, 0, 0, 0], labels.float()))

    def test_ddim_is_reproducible_with_eta_zero(self) -> None:
        labels = torch.tensor([0, 1])
        initial = torch.randn(2, 1, 4, 4)
        first = self.diffusion.ddim_sample_loop(
            ZeroModel(),
            tuple(initial.shape),
            labels,
            steps=5,
            eta=0.0,
            initial_noise=initial,
        )
        second = self.diffusion.ddim_sample_loop(
            ZeroModel(),
            tuple(initial.shape),
            labels,
            steps=5,
            eta=0.0,
            initial_noise=initial,
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())


if __name__ == "__main__":
    unittest.main()

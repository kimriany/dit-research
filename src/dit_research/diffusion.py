from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

from .config import DiffusionConfig


def make_beta_schedule(config: DiffusionConfig) -> Tensor:
    if config.schedule == "linear":
        betas = torch.linspace(
            config.beta_start,
            config.beta_end,
            config.timesteps,
            dtype=torch.float64,
        )
    elif config.schedule == "cosine":
        steps = config.timesteps + 1
        x = torch.linspace(0, config.timesteps, steps, dtype=torch.float64)
        cumulative = torch.cos(((x / config.timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
        cumulative = cumulative / cumulative[0]
        betas = 1 - cumulative[1:] / cumulative[:-1]
        betas = betas.clamp(min=1e-8, max=0.999)
    else:
        raise ValueError(f"unsupported beta schedule: {config.schedule}")
    if torch.any(betas <= 0) or torch.any(betas >= 1):
        raise ValueError("beta schedule must stay inside (0, 1)")
    return betas.float()


def _extract(values: Tensor, timesteps: Tensor, target_shape: torch.Size) -> Tensor:
    extracted = values.gather(0, timesteps)
    return extracted.reshape(timesteps.shape[0], *((1,) * (len(target_shape) - 1)))


class GaussianDiffusion:
    def __init__(self, config: DiffusionConfig, device: torch.device | str = "cpu") -> None:
        self.config = config
        self.timesteps = config.timesteps
        betas = make_beta_schedule(config).to(device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = torch.cat(
            (torch.ones(1, device=betas.device), alpha_bars[:-1]), dim=0
        )

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.alpha_bars_previous = alpha_bars_previous
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
        self.sqrt_recip_alpha_bars = torch.sqrt(1.0 / alpha_bars)
        self.sqrt_recipm1_alpha_bars = torch.sqrt(1.0 / alpha_bars - 1)

    @property
    def device(self) -> torch.device:
        return self.betas.device

    def to(self, device: torch.device | str) -> "GaussianDiffusion":
        for name, value in vars(self).items():
            if isinstance(value, Tensor):
                setattr(self, name, value.to(device))
        return self

    def q_sample(self, clean: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            noise = torch.randn_like(clean)
        if noise.shape != clean.shape:
            raise ValueError("noise and clean tensors must have identical shapes")
        return (
            _extract(self.sqrt_alpha_bars, timesteps, clean.shape) * clean
            + _extract(self.sqrt_one_minus_alpha_bars, timesteps, clean.shape) * noise
        )

    def predict_clean_from_eps(self, noisy: Tensor, timesteps: Tensor, epsilon: Tensor) -> Tensor:
        return (
            _extract(self.sqrt_recip_alpha_bars, timesteps, noisy.shape) * noisy
            - _extract(self.sqrt_recipm1_alpha_bars, timesteps, noisy.shape) * epsilon
        )

    def training_losses(
        self,
        model: nn.Module,
        clean: Tensor,
        timesteps: Tensor,
        labels: Tensor,
        *,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        if noise is None:
            noise = torch.randn(
                clean.shape,
                device=clean.device,
                dtype=clean.dtype,
                generator=generator,
            )
        noisy = self.q_sample(clean, timesteps, noise)
        predicted = model(noisy, timesteps, labels)
        per_sample = (predicted.float() - noise.float()).square().flatten(1).mean(1)
        return {"loss": per_sample.mean(), "per_sample": per_sample, "prediction": predicted}

    @torch.no_grad()
    def predict_eps_cfg(
        self,
        model: nn.Module,
        noisy: Tensor,
        timesteps: Tensor,
        labels: Tensor,
        cfg_scale: float,
    ) -> Tensor:
        if cfg_scale == 1.0:
            return model(noisy, timesteps, labels)
        original = getattr(model, "_orig_mod", model)
        null_class_id = getattr(original, "null_class_id", None)
        if null_class_id is None:
            raise AttributeError("model must expose null_class_id for classifier-free guidance")
        null_labels = torch.full_like(labels, null_class_id)
        combined_noisy = torch.cat((noisy, noisy), dim=0)
        combined_timesteps = torch.cat((timesteps, timesteps), dim=0)
        combined_labels = torch.cat((labels, null_labels), dim=0)
        conditional, unconditional = model(
            combined_noisy, combined_timesteps, combined_labels
        ).chunk(2, dim=0)
        return unconditional + cfg_scale * (conditional - unconditional)

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        labels: Tensor,
        *,
        steps: int,
        eta: float = 0.0,
        cfg_scale: float = 1.0,
        clip_denoised: bool = True,
        generator: torch.Generator | None = None,
        initial_noise: Tensor | None = None,
        callback: Callable[[int, Tensor], None] | None = None,
    ) -> Tensor:
        if not 1 <= steps <= self.timesteps:
            raise ValueError("DDIM steps must be between one and diffusion timesteps")
        if labels.shape != (shape[0],):
            raise ValueError("labels must have one entry per generated image")
        device = labels.device
        if initial_noise is None:
            sample = torch.randn(shape, device=device, generator=generator)
        else:
            if initial_noise.shape != shape:
                raise ValueError("initial_noise has the wrong shape")
            sample = initial_noise.to(device)

        sequence = torch.linspace(self.timesteps - 1, 0, steps, dtype=torch.float64)
        sequence = sequence.round().long()
        if torch.unique(sequence).numel() != steps:
            raise ValueError("rounded DDIM schedule contains duplicate timesteps")

        for index, timestep_value in enumerate(sequence.tolist()):
            timesteps = torch.full(
                (shape[0],), timestep_value, device=device, dtype=torch.long
            )
            epsilon = self.predict_eps_cfg(model, sample, timesteps, labels, cfg_scale)
            alpha_bar = self.alpha_bars[timestep_value]
            if index + 1 < len(sequence):
                previous_value = int(sequence[index + 1].item())
                alpha_bar_previous = self.alpha_bars[previous_value]
            else:
                alpha_bar_previous = torch.ones((), device=device)

            clean = (sample - torch.sqrt(1 - alpha_bar) * epsilon) / torch.sqrt(alpha_bar)
            if clip_denoised:
                clean = clean.clamp(-1, 1)
                epsilon = (sample - torch.sqrt(alpha_bar) * clean) / torch.sqrt(
                    1 - alpha_bar
                )
            variance = (1 - alpha_bar_previous) / (1 - alpha_bar) * (
                1 - alpha_bar / alpha_bar_previous
            )
            sigma = eta * torch.sqrt(variance.clamp_min(0))
            direction_scale = torch.sqrt((1 - alpha_bar_previous - sigma.square()).clamp_min(0))
            if eta > 0 and index + 1 < len(sequence):
                random_noise = torch.randn(
                    sample.shape,
                    device=device,
                    dtype=sample.dtype,
                    generator=generator,
                )
            else:
                random_noise = torch.zeros_like(sample)
            sample = (
                torch.sqrt(alpha_bar_previous) * clean
                + direction_scale * epsilon
                + sigma * random_noise
            )
            if callback is not None:
                callback(timestep_value, sample)
        return sample

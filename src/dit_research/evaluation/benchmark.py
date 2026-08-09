from __future__ import annotations

import statistics
import time
from typing import Any

import torch
from torch import nn

from ..config import ExperimentConfig
from ..diffusion import GaussianDiffusion
from ..utils import EMA, autocast_context, resolve_precision


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_model(
    model: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
    *,
    mode: str,
    batch_size: int,
    warmup: int,
    iterations: int,
    repeats: int = 1,
) -> dict[str, Any]:
    if mode not in {"forward", "train"}:
        raise ValueError("mode must be forward or train")
    if min(batch_size, iterations, repeats) <= 0 or warmup < 0:
        raise ValueError("invalid benchmark sizes")
    actual_precision, warning = resolve_precision(config.train.precision, device)
    model = model.to(device)
    model.train(mode == "train")
    ema = EMA(model, config.train.ema_decay) if mode == "train" else None
    forward_model = model
    if config.runtime.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("runtime.compile requires torch.compile support")
        forward_model = torch.compile(model)
    diffusion = GaussianDiffusion(config.diffusion, device)
    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        if mode == "train"
        else None
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and actual_precision == "fp16"
    )
    images = torch.randn(
        batch_size,
        config.model.in_channels,
        config.model.image_size,
        config.model.image_size,
        device=device,
    )
    labels = torch.arange(batch_size, device=device) % config.model.num_classes
    timesteps = torch.arange(batch_size, device=device) % config.diffusion.timesteps

    def one_iteration() -> None:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if mode == "forward":
            with torch.no_grad(), autocast_context(device, actual_precision):
                forward_model(images, timesteps, labels)
        else:
            for _ in range(config.train.grad_accum_steps):
                with autocast_context(device, actual_precision):
                    loss = diffusion.training_losses(
                        forward_model, images, timesteps, labels
                    )["loss"]
                scaler.scale(loss / config.train.grad_accum_steps).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

    for _ in range(warmup):
        one_iteration()
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rates: list[float] = []
    latencies: list[float] = []
    images_per_iteration = batch_size * (
        config.train.grad_accum_steps if mode == "train" else 1
    )
    for _ in range(repeats):
        _synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            one_iteration()
        _synchronize(device)
        elapsed = time.perf_counter() - started
        rates.append(images_per_iteration * iterations / elapsed)
        latencies.append(1000 * elapsed / iterations)

    result: dict[str, Any] = {
        "mode": mode,
        "device": str(device),
        "precision_requested": config.train.precision,
        "precision_actual": actual_precision,
        "precision_warning": warning,
        "batch_size": batch_size,
        "effective_batch_size_single_gpu": images_per_iteration,
        "gradient_accumulation_steps": config.train.grad_accum_steps if mode == "train" else 1,
        "ema_included": mode == "train",
        "compiled": config.runtime.compile,
        "warmup": warmup,
        "iterations": iterations,
        "repeats": repeats,
        "images_per_second_values": rates,
        "images_per_second_median": statistics.median(rates),
        "latency_ms_median": statistics.median(latencies),
    }
    if len(rates) >= 4:
        quartiles = statistics.quantiles(rates, n=4, method="inclusive")
        result["images_per_second_iqr"] = quartiles[2] - quartiles[0]
    if device.type == "cuda":
        result["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
        result["peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 2**20
    return result

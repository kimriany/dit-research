#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

import torch

from _bootstrap import ROOT  # noqa: F401
from dit_research.config import ExperimentConfig
from dit_research.diffusion import GaussianDiffusion
from dit_research.model import build_model
from dit_research.utils import (
    atomic_json_dump,
    autocast_context,
    make_generator,
    resolve_device,
    resolve_precision,
    save_tensor_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate balanced class-conditional PNG samples")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--raw", action="store_true", help="sample raw rather than EMA weights")
    lambda_group = parser.add_mutually_exclusive_group()
    lambda_group.add_argument(
        "--lambda-override",
        type=float,
        default=None,
        help="force every memory block's separation lambda to a value in [0,1]",
    )
    lambda_group.add_argument(
        "--blockwise-mean-lambda",
        action="store_true",
        help="replace each block's adaptive lambda with its schedule-wide mean",
    )
    parser.add_argument(
        "--log-snr-mode",
        choices=("normal", "reversed", "shuffled", "zero"),
        default="normal",
        help="paired mechanism intervention for the adaptive controller",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("num-samples and batch-size must be positive")
    if args.lambda_override is not None and not 0 <= args.lambda_override <= 1:
        raise ValueError("lambda-override must be in [0, 1]")
    if (
        args.lambda_override is not None or args.blockwise_mean_lambda
    ) and args.log_snr_mode != "normal":
        raise ValueError(
            "lambda interventions cannot be combined with a log-SNR intervention"
        )
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    # Project checkpoints include the dataclass-compatible experiment config,
    # not just tensor weights.  Full pickle loading is therefore required and
    # must only be used for checkpoints from a trusted training run.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = ExperimentConfig.from_dict(checkpoint["config"])
    sampling = replace(
        config.sampling,
        steps=args.steps if args.steps is not None else config.sampling.steps,
        cfg_scale=args.cfg_scale if args.cfg_scale is not None else config.sampling.cfg_scale,
        eta=args.eta if args.eta is not None else config.sampling.eta,
    )
    config = replace(config, sampling=sampling)
    config.validate()
    device = resolve_device(args.device)
    precision, warning = resolve_precision(config.train.precision, device)
    model = build_model(config.model, config.diffusion)
    if args.raw:
        model.load_state_dict(checkpoint["model"])
        weight_source = "raw"
    else:
        model.load_state_dict(checkpoint["ema"]["model"])
        weight_source = "ema"
    resolved_lambda_overrides = model.set_memory_intervention(
        lambda_override=args.lambda_override,
        blockwise_mean_lambda=args.blockwise_mean_lambda,
        log_snr_mode=args.log_snr_mode,
    )
    model.to(device).eval()
    diffusion = GaussianDiffusion(config.diffusion, device)
    seed = args.seed if args.seed is not None else config.seeds.resolved()["sampling"]
    generator = make_generator(device, seed)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    existing = list(output.glob("*.png"))
    if existing:
        raise FileExistsError(f"sample directory already contains {len(existing)} PNG files: {output}")

    generated = 0
    with torch.no_grad(), autocast_context(device, precision):
        while generated < args.num_samples:
            current = min(args.batch_size, args.num_samples - generated)
            labels = torch.arange(generated, generated + current, device=device)
            labels = labels.remainder(config.model.num_classes).long()
            samples = diffusion.ddim_sample_loop(
                model,
                (
                    current,
                    config.model.in_channels,
                    config.model.image_size,
                    config.model.image_size,
                ),
                labels,
                steps=config.sampling.steps,
                eta=config.sampling.eta,
                cfg_scale=config.sampling.cfg_scale,
                clip_denoised=config.sampling.clip_denoised,
                generator=generator,
            )
            for local_index, image in enumerate(samples):
                save_tensor_image(image, output / f"{generated + local_index:06d}.png")
            generated += current
            print(f"generated {generated}/{args.num_samples}", flush=True)

    atomic_json_dump(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "checkpoint_step": int(checkpoint["step"]),
            "weight_source": weight_source,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "class_balance_rule": "global_index_mod_num_classes",
            "seed": seed,
            "sampler": "ddim",
            "steps": config.sampling.steps,
            "eta": config.sampling.eta,
            "cfg_scale": config.sampling.cfg_scale,
            "precision": precision,
            "precision_warning": warning,
            "config_hash": checkpoint.get("config_hash"),
            "memory_intervention": {
                "lambda_override": args.lambda_override,
                "blockwise_mean_lambda": args.blockwise_mean_lambda,
                "resolved_lambda_overrides": resolved_lambda_overrides,
                "log_snr_mode": args.log_snr_mode,
            },
        },
        output / "sample_manifest.json",
    )


if __name__ == "__main__":
    main()

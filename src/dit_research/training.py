from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .config import ExperimentConfig, dump_config
from .data import DatasetBundle, build_datasets, build_loaders
from .diffusion import GaussianDiffusion
from .evaluation.complexity import analytic_complexity
from .model import DiT, build_model
from .utils import (
    EMA,
    append_jsonl,
    apply_class_dropout,
    atomic_json_dump,
    atomic_torch_save,
    autocast_context,
    capture_rng_state,
    environment_manifest,
    make_generator,
    make_grid,
    resolve_device,
    resolve_precision,
    restore_rng_state,
    save_tensor_image,
    seed_process,
)


def _resume_fingerprint(config: ExperimentConfig) -> str:
    payload = config.to_dict()
    for key in (
        "max_steps",
        "log_every",
        "validation_every",
        "validation_batches",
        "checkpoint_every",
        "sample_every",
    ):
        payload["train"].pop(key, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        resume: str | None = None,
        *,
        allow_code_change: bool = False,
        allow_environment_change: bool = False,
    ) -> None:
        self.config = config
        self.device = resolve_device(config.runtime.device)
        self.precision, precision_warning = resolve_precision(config.train.precision, self.device)
        self.seeds = config.seeds.resolved()
        self.allow_code_change = allow_code_change
        self.allow_environment_change = allow_environment_change
        self.environment = environment_manifest()
        cuda_info = self.environment["cuda"]
        self.runtime_signature = {
            "device_type": self.device.type,
            "precision_actual": self.precision,
            "torch": self.environment["torch"],
            "torch_cuda": cuda_info.get("torch_cuda"),
            "python": ".".join(self.environment["python"].split()[0].split(".")[:3]),
            "packages": self.environment["packages"],
            "gpu_name": cuda_info.get("device_name") if self.device.type == "cuda" else None,
            "gpu_capability": cuda_info.get("capability") if self.device.type == "cuda" else None,
        }
        seed_process(self.seeds["init"], config.runtime.deterministic)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
            torch.backends.cudnn.allow_tf32 = config.runtime.allow_tf32
        torch.set_float32_matmul_precision("high" if config.runtime.allow_tf32 else "highest")

        self.output_dir = config.output_dir
        checkpoint_path = self._resolve_resume(resume)
        if checkpoint_path is None and self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(
                f"output directory is not empty: {self.output_dir}; choose a new name or use --resume"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "samples").mkdir(exist_ok=True)

        self.model: DiT = build_model(config.model, config.diffusion).to(self.device)
        self.ema = EMA(self.model, config.train.ema_decay)
        self.forward_model = self.model
        if config.runtime.compile:
            if not hasattr(torch, "compile"):
                raise RuntimeError("runtime.compile requires a PyTorch build with torch.compile")
            self.forward_model = torch.compile(self.model)
        self.diffusion = GaussianDiffusion(config.diffusion, self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        scaler_enabled = self.device.type == "cuda" and self.precision == "fp16"
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        else:
            # PyTorch 2.2 compatibility; newer releases expose torch.amp.GradScaler.
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        self.diffusion_generator = make_generator(self.device, self.seeds["diffusion"])
        self.dropout_generator = make_generator(self.device, self.seeds["dropout"])

        self.datasets: DatasetBundle = build_datasets(config.data, config.model, self.seeds["data"])
        self.microbatches_consumed = 0
        self._reset_loaders()
        self.step = 0
        self.samples_seen = 0
        self.skipped_updates = 0
        self.preview_failures = 0
        self.validation_failures = 0
        self.wall_seconds_before_resume = 0.0
        self.started_at = time.perf_counter()
        self.last_log_samples = 0
        self.train_seconds_since_log = 0.0
        self.latest_metrics: dict[str, Any] = {}

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
            self.last_log_samples = self.samples_seen
        else:
            dump_config(config, self.output_dir / "resolved_config.yaml")
            self._write_manifest(precision_warning)

    def _reset_loaders(self) -> None:
        self.train_loader, self.validation_loader = build_loaders(
            self.datasets,
            self.config.data,
            self.config.train.batch_size,
            self.seeds["data"],
            pin_memory=self.device.type == "cuda",
            consumed_train_batches=self.microbatches_consumed,
        )
        self.train_batches = iter(self.train_loader)

    def _resolve_resume(self, resume: str | None) -> Path | None:
        if resume is None:
            return None
        if resume == "auto":
            path = self.output_dir / "checkpoints" / "latest.pt"
        else:
            path = Path(resume)
        if not path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {path}")
        return path

    def _write_manifest(self, precision_warning: str | None) -> None:
        stats = analytic_complexity(self.model, self.config.model)
        manifest = dict(self.environment)
        manifest.update(
            {
                "experiment": self.config.experiment.name,
                "group": self.config.experiment.group,
                "phase": self.config.experiment.phase,
                "config_hash": self.config.fingerprint(),
                "resume_compatibility_hash": _resume_fingerprint(self.config),
                "dataset": self.datasets.metadata,
                "seeds": self.seeds,
                "precision_requested": self.config.train.precision,
                "precision_actual": self.precision,
                "precision_warning": precision_warning,
                "runtime_signature": self.runtime_signature,
                "effective_batch_size_single_gpu": self.config.effective_batch_size,
                "model_complexity": stats,
            }
        )
        atomic_json_dump(manifest, self.output_dir / "manifest.json")

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "version": 1,
            "config": self.config.to_dict(),
            "config_hash": self.config.fingerprint(),
            "resume_compatibility_hash": _resume_fingerprint(self.config),
            "dataset_split_hash": self.datasets.metadata["split_hash"],
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "samples_seen": self.samples_seen,
            "wall_seconds": self.wall_seconds,
            "microbatches_consumed": self.microbatches_consumed,
            "skipped_updates": self.skipped_updates,
            "preview_failures": self.preview_failures,
            "validation_failures": self.validation_failures,
            "rng": capture_rng_state(),
            "training_code_hash": self.environment["training_code_hash"],
            "runtime_signature": self.runtime_signature,
            "diffusion_generator": (
                self.diffusion_generator.get_state() if self.diffusion_generator is not None else None
            ),
            "dropout_generator": (
                self.dropout_generator.get_state() if self.dropout_generator is not None else None
            ),
        }

    def save_checkpoint(self) -> Path:
        path = self.output_dir / "checkpoints" / "latest.pt"
        atomic_torch_save(self._checkpoint_payload(), path)
        return path

    def _load_checkpoint(self, path: Path) -> None:
        # Training checkpoints contain optimizer/RNG/config objects in addition
        # to tensors, so they are intentionally not weights-only archives.
        # Only resume checkpoints produced by this project or another trusted
        # source; pickle-based full loading can execute code from a hostile file.
        checkpoint = torch.load(
            path,
            # Keep CPU and CUDA RNG byte states on CPU.  Model and optimizer
            # load_state_dict calls below move their tensors to the parameter
            # devices, while torch.set_rng_state requires a CPU ByteTensor.
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_code_hash = checkpoint.get("training_code_hash")
        current_code_hash = self.environment["training_code_hash"]
        if checkpoint_code_hash != current_code_hash and not self.allow_code_change:
            raise ValueError(
                "training code hash differs from the checkpoint; use --allow-code-change only after "
                "reviewing the semantic change"
            )
        checkpoint_runtime = checkpoint.get("runtime_signature")
        if checkpoint_runtime != self.runtime_signature and not self.allow_environment_change:
            raise ValueError(
                "resolved precision/device runtime differs from the checkpoint; use "
                "--allow-environment-change only when a non-bitwise resume is intentional"
            )
        expected = _resume_fingerprint(self.config)
        if checkpoint.get("resume_compatibility_hash") != expected:
            raise ValueError(
                "checkpoint and config differ in a training-affecting field; only max_steps and "
                "log/validation/checkpoint/sample intervals may change"
            )
        if checkpoint.get("dataset_split_hash") != self.datasets.metadata["split_hash"]:
            raise ValueError("dataset split hash differs from the checkpoint")
        self.model.load_state_dict(checkpoint["model"])
        self.ema.load_state_dict(checkpoint["ema"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.step = int(checkpoint["step"])
        self.samples_seen = int(checkpoint["samples_seen"])
        self.microbatches_consumed = int(
            checkpoint.get("microbatches_consumed", self.step * self.config.train.grad_accum_steps)
        )
        self.skipped_updates = int(checkpoint.get("skipped_updates", 0))
        self.preview_failures = int(checkpoint.get("preview_failures", 0))
        self.validation_failures = int(checkpoint.get("validation_failures", 0))
        self.wall_seconds_before_resume = float(checkpoint.get("wall_seconds", 0.0))
        restore_rng_state(checkpoint["rng"])
        if self.diffusion_generator is not None and checkpoint.get("diffusion_generator") is not None:
            self.diffusion_generator.set_state(checkpoint["diffusion_generator"])
        if self.dropout_generator is not None and checkpoint.get("dropout_generator") is not None:
            self.dropout_generator.set_state(checkpoint["dropout_generator"])
        self._reset_loaders()
        if self.step >= self.config.train.max_steps:
            raise ValueError(
                f"checkpoint step {self.step} is not below requested max_steps {self.config.train.max_steps}"
            )
        dump_config(self.config, self.output_dir / "resolved_config.yaml")
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["config_hash"] = self.config.fingerprint()
            manifest["last_resume"] = {
                "checkpoint": str(path),
                "resumed_from_step": self.step,
                "requested_max_steps": self.config.train.max_steps,
                "code_hash_changed": checkpoint_code_hash != current_code_hash,
                "environment_changed": checkpoint_runtime != self.runtime_signature,
                "runtime_signature": self.runtime_signature,
                "training_code_hash": current_code_hash,
                "source_tree_hash": self.environment["source_tree_hash"],
                "unix": time.time(),
            }
            atomic_json_dump(manifest, manifest_path)

    @property
    def wall_seconds(self) -> float:
        return self.wall_seconds_before_resume + (time.perf_counter() - self.started_at)

    def _train_step(self) -> tuple[float, float, bool]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        starting_microbatches = self.microbatches_consumed
        starting_samples_seen = self.samples_seen
        diffusion_generator_state = (
            self.diffusion_generator.get_state() if self.diffusion_generator is not None else None
        )
        dropout_generator_state = (
            self.dropout_generator.get_state() if self.dropout_generator is not None else None
        )
        accumulated_loss = 0.0
        for _ in range(self.config.train.grad_accum_steps):
            images, labels = next(self.train_batches)
            self.microbatches_consumed += 1
            self.samples_seen += images.shape[0]
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
            labels = apply_class_dropout(
                labels,
                self.config.model.class_dropout_prob,
                self.model.null_class_id,
                self.dropout_generator,
            )
            timesteps = torch.randint(
                0,
                self.diffusion.timesteps,
                (images.shape[0],),
                device=self.device,
                generator=self.diffusion_generator,
            )
            with autocast_context(self.device, self.precision):
                loss = self.diffusion.training_losses(
                    self.forward_model,
                    images,
                    timesteps,
                    labels,
                    generator=self.diffusion_generator,
                )["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at step {self.step + 1}: {loss}")
            accumulated_loss += float(loss.detach())
            self.scaler.scale(loss / self.config.train.grad_accum_steps).backward()

        self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.train.gradient_clip
        )
        if not torch.isfinite(gradient_norm):
            if not self.scaler.is_enabled():
                raise FloatingPointError(f"non-finite gradient norm at step {self.step + 1}")
            old_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() >= old_scale:
                raise FloatingPointError("non-finite fp16 gradients did not reduce the loss scale")
            self.microbatches_consumed = starting_microbatches
            self.samples_seen = starting_samples_seen
            if self.diffusion_generator is not None and diffusion_generator_state is not None:
                self.diffusion_generator.set_state(diffusion_generator_state)
            if self.dropout_generator is not None and dropout_generator_state is not None:
                self.dropout_generator.set_state(dropout_generator_state)
            self._reset_loaders()
            self.skipped_updates += 1
            append_jsonl(
                {
                    "event": "fp16_overflow",
                    "step": self.step,
                    "samples_seen": self.samples_seen,
                    "old_scale": old_scale,
                    "new_scale": self.scaler.get_scale(),
                },
                self.output_dir / "metrics.jsonl",
            )
            return accumulated_loss / self.config.train.grad_accum_steps, float(gradient_norm), False
        old_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scaler.is_enabled() and self.scaler.get_scale() < old_scale:
            raise FloatingPointError("fp16 GradScaler skipped an optimizer step")
        self.ema.update(self.model)
        self.step += 1
        return accumulated_loss / self.config.train.grad_accum_steps, float(gradient_norm), True

    @torch.no_grad()
    def validate(self) -> dict[str, float | None]:
        model = self.ema.model.eval()
        model.set_memory_diagnostics(True)
        generator = make_generator(self.device, self.seeds["evaluation"])
        loss_sum = 0.0
        count = 0
        bucket_sum = [0.0, 0.0, 0.0, 0.0]
        bucket_count = [0, 0, 0, 0]
        diagnostic_sums: dict[str, Tensor] = {}
        diagnostic_square_sums: dict[str, Tensor] = {}
        diagnostic_counts: dict[str, int] = {}
        diagnostic_bucket_sums: dict[str, list[Tensor]] = {}
        diagnostic_bucket_counts: dict[str, list[int]] = {}
        try:
            for batch_index, (images, labels) in enumerate(self.validation_loader):
                if batch_index >= self.config.train.validation_batches:
                    break
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
                timesteps = torch.randint(
                    0,
                    self.diffusion.timesteps,
                    (images.shape[0],),
                    device=self.device,
                    generator=generator,
                )
                with autocast_context(self.device, self.precision):
                    losses = self.diffusion.training_losses(
                        model,
                        images,
                        timesteps,
                        labels,
                        generator=generator,
                    )["per_sample"]
                loss_sum += float(losses.sum())
                count += losses.numel()
                buckets = torch.clamp(timesteps * 4 // self.diffusion.timesteps, max=3)
                for name, value in model.memory_diagnostics().items():
                    values = value.detach().float().reshape(-1)
                    if values.numel() != losses.numel():
                        raise RuntimeError(
                            f"memory diagnostic {name} must have one value per sample"
                        )
                    diagnostic_sums[name] = diagnostic_sums.get(
                        name, torch.zeros((), device=values.device)
                    ) + values.sum()
                    diagnostic_square_sums[name] = diagnostic_square_sums.get(
                        name, torch.zeros((), device=values.device)
                    ) + values.square().sum()
                    diagnostic_counts[name] = diagnostic_counts.get(name, 0) + values.numel()
                    if name not in diagnostic_bucket_sums:
                        diagnostic_bucket_sums[name] = [
                            torch.zeros((), device=values.device) for _ in range(4)
                        ]
                        diagnostic_bucket_counts[name] = [0, 0, 0, 0]
                    for bucket in range(4):
                        selected_values = values[buckets == bucket]
                        if selected_values.numel():
                            diagnostic_bucket_sums[name][bucket] += selected_values.sum()
                            diagnostic_bucket_counts[name][bucket] += selected_values.numel()
                for bucket in range(4):
                    selected = losses[buckets == bucket]
                    if selected.numel():
                        bucket_sum[bucket] += float(selected.sum())
                        bucket_count[bucket] += selected.numel()
        finally:
            model.set_memory_diagnostics(False)
        if count == 0:
            raise RuntimeError("validation loader produced no samples")
        result: dict[str, float | None] = {"validation_loss": loss_sum / count}
        for bucket in range(4):
            result[f"validation_loss_tq{bucket + 1}"] = (
                bucket_sum[bucket] / bucket_count[bucket] if bucket_count[bucket] else None
            )
        for name, value in diagnostic_sums.items():
            diagnostic_mean = value / diagnostic_counts[name]
            diagnostic_variance = (
                diagnostic_square_sums[name] / diagnostic_counts[name]
                - diagnostic_mean.square()
            ).clamp_min(0)
            result[f"{name}_mean"] = float(diagnostic_mean)
            result[f"{name}_std"] = float(diagnostic_variance.sqrt())
            for bucket in range(4):
                bucket_diagnostic_count = diagnostic_bucket_counts[name][bucket]
                result[f"{name}_tq{bucket + 1}"] = (
                    float(
                        diagnostic_bucket_sums[name][bucket]
                        / bucket_diagnostic_count
                    )
                    if bucket_diagnostic_count
                    else None
                )
        return result

    @torch.no_grad()
    def sample_preview(self) -> Path:
        count = self.config.sampling.preview_count
        generator = make_generator(self.device, self.seeds["sampling"])
        sample_batches = []
        generated = 0
        while generated < count:
            current = min(self.config.train.batch_size, count - generated)
            labels = torch.arange(generated, generated + current, device=self.device)
            labels = labels.remainder(self.config.model.num_classes).long()
            with autocast_context(self.device, self.precision):
                batch = self.diffusion.ddim_sample_loop(
                    self.ema.model.eval(),
                    (
                        current,
                        self.config.model.in_channels,
                        self.config.model.image_size,
                        self.config.model.image_size,
                    ),
                    labels,
                    steps=self.config.sampling.steps,
                    eta=self.config.sampling.eta,
                    cfg_scale=self.config.sampling.cfg_scale,
                    clip_denoised=self.config.sampling.clip_denoised,
                    generator=generator,
                )
            sample_batches.append(batch.float().cpu())
            generated += current
        samples = torch.cat(sample_batches, dim=0)
        grid = make_grid(samples)
        path = self.output_dir / "samples" / f"step_{self.step:08d}.png"
        save_tensor_image(grid, path)
        return path

    def safe_sample_preview(self) -> Path | None:
        try:
            return self.sample_preview()
        except Exception as exc:
            self.preview_failures += 1
            event = {
                "event": "preview_error",
                "step": self.step,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            append_jsonl(event, self.output_dir / "metrics.jsonl")
            print(json.dumps(event, sort_keys=True), flush=True)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return None

    def _log(self, train_loss: float, gradient_norm: float) -> dict[str, Any]:
        interval_samples = self.samples_seen - self.last_log_samples
        metrics: dict[str, Any] = {
            "event": "train",
            "step": self.step,
            "samples_seen": self.samples_seen,
            "microbatches_consumed": self.microbatches_consumed,
            "skipped_updates": self.skipped_updates,
            "preview_failures": self.preview_failures,
            "validation_failures": self.validation_failures,
            "train_loss": train_loss,
            "gradient_norm": gradient_norm,
            "wall_seconds": self.wall_seconds,
            "interval_images_per_second": interval_samples
            / max(self.train_seconds_since_log, 1e-9),
        }
        if self.step % self.config.train.validation_every == 0 or self.step == self.config.train.max_steps:
            try:
                metrics.update(self.validate())
            except Exception as exc:
                self.validation_failures += 1
                metrics.update(
                    {
                        "validation_error_type": type(exc).__name__,
                        "validation_error": str(exc),
                        "validation_failures": self.validation_failures,
                    }
                )
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        if self.device.type == "cuda":
            metrics["peak_allocated_mb"] = torch.cuda.max_memory_allocated(self.device) / 2**20
            metrics["peak_reserved_mb"] = torch.cuda.max_memory_reserved(self.device) / 2**20
        append_jsonl(metrics, self.output_dir / "metrics.jsonl")
        self.latest_metrics = metrics
        self.last_log_samples = self.samples_seen
        self.train_seconds_since_log = 0.0
        print(json.dumps(metrics, sort_keys=True), flush=True)
        return metrics

    def _write_final_metrics(self) -> None:
        stats = analytic_complexity(self.model, self.config.model)
        payload = {
            "experiment": self.config.experiment.name,
            "group": self.config.experiment.group,
            "phase": self.config.experiment.phase,
            "seed": self.config.seeds.base,
            "step": self.step,
            "samples_seen": self.samples_seen,
            "microbatches_consumed": self.microbatches_consumed,
            "skipped_updates": self.skipped_updates,
            "preview_failures": self.preview_failures,
            "validation_failures": self.validation_failures,
            "precision": self.precision,
            "effective_batch_size_single_gpu": self.config.effective_batch_size,
            **stats,
            **self.latest_metrics,
        }
        payload["preview_failures"] = self.preview_failures
        payload["validation_failures"] = self.validation_failures
        payload["skipped_updates"] = self.skipped_updates
        payload["wall_seconds"] = self.wall_seconds
        payload["gpu_hours"] = self.wall_seconds / 3600 if self.device.type == "cuda" else None
        atomic_json_dump(payload, self.output_dir / "final_metrics.json")

    def run(self) -> dict[str, Any]:
        last_loss = float("nan")
        last_gradient_norm = float("nan")
        while self.step < self.config.train.max_steps:
            train_started = time.perf_counter()
            last_loss, last_gradient_norm, updated = self._train_step()
            self.train_seconds_since_log += time.perf_counter() - train_started
            if not updated:
                continue
            if (
                self.step % self.config.train.checkpoint_every == 0
                or self.step == self.config.train.max_steps
            ):
                self.save_checkpoint()
            should_log = self.step % self.config.train.log_every == 0
            if should_log or self.step == self.config.train.max_steps:
                self._log(last_loss, last_gradient_norm)
            if self.step == self.config.train.max_steps:
                self._write_final_metrics()
            if self.step % self.config.train.sample_every == 0:
                self.safe_sample_preview()

        self.save_checkpoint()
        if self.step % self.config.train.sample_every:
            self.safe_sample_preview()
        if not self.latest_metrics or self.latest_metrics.get("step") != self.step:
            self._log(last_loss, last_gradient_norm)
        self._write_final_metrics()
        return self.latest_metrics


def with_max_steps(config: ExperimentConfig, max_steps: int | None) -> ExperimentConfig:
    if max_steps is None:
        return config
    if max_steps <= 0:
        raise ValueError("max_steps override must be positive")
    return replace(config, train=replace(config.train, max_steps=max_steps))

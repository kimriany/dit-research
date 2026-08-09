from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeVar, get_type_hints

import yaml


@dataclass(frozen=True)
class ExperimentMeta:
    name: str
    group: str
    phase: str
    output_root: str = "outputs"


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "cifar10"
    root: str = "./datasets"
    validation_fraction: float = 0.1
    split_seed: int = 20260712
    fake_size: int = 0
    num_workers: int = 4
    download: bool = False


@dataclass(frozen=True)
class AllocationConfig:
    kind: str = "uniform"
    strength: float = 0.0
    multiple_of: int = 8


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for interleaved GDN2-style spatial memory mixers.

    ``block_indices`` are zero-based DiT block indices.  The default keeps the
    pre-existing all-softmax model and checkpoint layout unchanged.
    """

    kind: str = "softmax"
    gate_mode: str = "adaptive"
    block_indices: tuple[int, ...] = ()
    scan_pattern: str = "lr,rl,tb,bt"
    gate_rank: int = 64
    lambda_hidden_size: int = 16
    log_snr_clip: float = 20.0
    backend: str = "reference"


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 32
    in_channels: int = 3
    patch_size: int = 2
    hidden_size: int = 384
    depth: int = 12
    num_heads: int = 6
    num_classes: int = 10
    class_dropout_prob: float = 0.1
    mlp_ratio: float = 4.0
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int = 1000
    schedule: str = "linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    grad_accum_steps: int = 1
    max_steps: int = 50000
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    ema_decay: float = 0.9999
    precision: str = "bf16"
    gradient_clip: float = 1.0
    log_every: int = 100
    validation_every: int = 1000
    validation_batches: int = 80
    checkpoint_every: int = 5000
    sample_every: int = 5000


@dataclass(frozen=True)
class SamplingConfig:
    method: str = "ddim"
    steps: int = 50
    eta: float = 0.0
    cfg_scale: float = 1.5
    preview_count: int = 40
    clip_denoised: bool = True


@dataclass(frozen=True)
class EvaluationConfig:
    fid_samples_pilot: int = 5000
    fid_samples_final: int = 50000
    fid_reference_split: str = "train"
    fid_mode: str = "clean"


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    compile: bool = False
    deterministic: bool = False
    allow_tf32: bool = True


@dataclass(frozen=True)
class SeedsConfig:
    base: int = 42

    def resolved(self) -> dict[str, int]:
        return {
            "init": self.base,
            "data": self.base + 10_000,
            "diffusion": self.base + 20_000,
            "dropout": self.base + 30_000,
            "evaluation": self.base + 40_000,
            "sampling": self.base + 50_000,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: ExperimentMeta
    data: DataConfig
    model: ModelConfig
    diffusion: DiffusionConfig
    train: TrainConfig
    sampling: SamplingConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig
    seeds: SeedsConfig

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        expected = {
            "experiment",
            "data",
            "model",
            "diffusion",
            "train",
            "sampling",
            "evaluation",
            "runtime",
            "seeds",
        }
        unknown = set(raw) - expected
        missing = expected - set(raw)
        if unknown:
            raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
        if missing:
            raise ValueError(f"missing top-level config keys: {sorted(missing)}")

        model_raw = _mapping(raw["model"], "model")
        allocation_raw = model_raw.pop("allocation", {})
        memory_raw = _mapping(model_raw.pop("memory", {}), "model.memory")
        if "block_indices" in memory_raw:
            block_indices = memory_raw["block_indices"]
            if not isinstance(block_indices, (list, tuple)):
                raise TypeError("model.memory.block_indices must be a list of integers")
            if any(type(index) is not int for index in block_indices):
                raise TypeError("model.memory.block_indices must contain only integers")
            memory_raw["block_indices"] = tuple(block_indices)
        config = cls(
            experiment=_strict_dataclass(ExperimentMeta, raw["experiment"], "experiment"),
            data=_strict_dataclass(DataConfig, raw["data"], "data"),
            model=_strict_dataclass(
                ModelConfig,
                {
                    **model_raw,
                    "allocation": _strict_dataclass(
                        AllocationConfig, allocation_raw, "model.allocation"
                    ),
                    "memory": _strict_dataclass(
                        MemoryConfig, memory_raw, "model.memory"
                    ),
                },
                "model",
            ),
            diffusion=_strict_dataclass(DiffusionConfig, raw["diffusion"], "diffusion"),
            train=_strict_dataclass(TrainConfig, raw["train"], "train"),
            sampling=_strict_dataclass(SamplingConfig, raw["sampling"], "sampling"),
            evaluation=_strict_dataclass(EvaluationConfig, raw["evaluation"], "evaluation"),
            runtime=_strict_dataclass(RuntimeConfig, raw["runtime"], "runtime"),
            seeds=_strict_dataclass(SeedsConfig, raw["seeds"], "seeds"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        m = self.model
        if m.image_size <= 0 or m.patch_size <= 0:
            raise ValueError("model.image_size and model.patch_size must be positive")
        if m.image_size % m.patch_size:
            raise ValueError("model.image_size must be positive and divisible by model.patch_size")
        if m.hidden_size <= 0 or m.num_heads <= 0:
            raise ValueError("model.hidden_size and model.num_heads must be positive")
        if m.hidden_size % m.num_heads:
            raise ValueError("model.hidden_size must be positive and divisible by model.num_heads")
        if m.hidden_size % 4:
            raise ValueError("model.hidden_size must be divisible by 4 for 2D sin-cos embeddings")
        if m.depth <= 0 or m.num_classes <= 0 or m.in_channels <= 0:
            raise ValueError("model depth/heads/classes/channels must be positive")
        if not 0.0 <= m.class_dropout_prob < 1.0:
            raise ValueError("model.class_dropout_prob must be in [0, 1)")
        if m.mlp_ratio <= 0:
            raise ValueError("model.mlp_ratio must be positive")
        if m.allocation.kind not in {"uniform", "frontloaded", "backloaded", "middle_heavy"}:
            raise ValueError(f"unsupported allocation kind: {m.allocation.kind}")
        if m.allocation.strength < 0 or m.allocation.multiple_of <= 0:
            raise ValueError("allocation strength must be non-negative and multiple_of positive")
        if m.allocation.kind != "uniform" and m.depth % 3:
            raise ValueError("stage-wise non-uniform allocations require model.depth divisible by 3")
        if m.allocation.kind in {"frontloaded", "backloaded"} and m.allocation.strength >= m.mlp_ratio:
            raise ValueError("allocation strength must be smaller than mean mlp_ratio")
        if m.allocation.kind == "middle_heavy" and m.allocation.strength * 0.5 >= m.mlp_ratio:
            raise ValueError("middle-heavy allocation would create a non-positive FFN width")

        memory = m.memory
        if memory.kind not in {"softmax", "hybrid_gdn2"}:
            raise ValueError("model.memory.kind must be softmax or hybrid_gdn2")
        if memory.gate_mode not in {"coupled", "separated", "static", "adaptive"}:
            raise ValueError(
                "model.memory.gate_mode must be coupled, separated, static, or adaptive"
            )
        if memory.backend not in {"reference", "fla"}:
            raise ValueError("model.memory.backend must be reference or fla")
        if memory.gate_rank <= 0 or memory.lambda_hidden_size <= 0:
            raise ValueError("memory gate_rank and lambda_hidden_size must be positive")
        if memory.log_snr_clip <= 0:
            raise ValueError("memory log_snr_clip must be positive")
        directions = tuple(
            direction.strip() for direction in memory.scan_pattern.split(",")
        )
        if not directions or any(
            direction not in {"lr", "rl", "tb", "bt"} for direction in directions
        ):
            raise ValueError(
                "model.memory.scan_pattern must be a comma-separated sequence of lr/rl/tb/bt"
            )
        if memory.kind == "softmax" and memory.block_indices:
            raise ValueError("softmax memory kind requires empty block_indices")
        if memory.kind == "hybrid_gdn2" and not memory.block_indices:
            raise ValueError("hybrid_gdn2 requires at least one memory block index")
        if len(set(memory.block_indices)) != len(memory.block_indices):
            raise ValueError("memory block_indices must not contain duplicates")
        if tuple(sorted(memory.block_indices)) != memory.block_indices:
            raise ValueError("memory block_indices must be in ascending order")
        if any(index < 0 or index >= m.depth for index in memory.block_indices):
            raise ValueError("memory block index is outside model depth")
        if memory.backend == "fla" and m.hidden_size // m.num_heads > 256:
            raise ValueError("FLA GDN2 requires head dimension <= 256")

        d = self.data
        if d.dataset not in {"fake", "cifar10"}:
            raise ValueError(f"unsupported data.dataset: {d.dataset}")
        if not 0.0 < d.validation_fraction < 1.0:
            raise ValueError("data.validation_fraction must be in (0, 1)")
        if d.num_workers < 0 or (d.dataset == "fake" and d.fake_size <= 0):
            raise ValueError("invalid data worker/fake_size settings")

        diffusion = self.diffusion
        if diffusion.timesteps <= 1 or diffusion.schedule not in {"linear", "cosine"}:
            raise ValueError("diffusion timesteps must exceed one and schedule must be linear/cosine")
        if not 0 < diffusion.beta_start < diffusion.beta_end < 1:
            raise ValueError("diffusion beta range must satisfy 0 < start < end < 1")

        train = self.train
        positive_ints = {
            "batch_size": train.batch_size,
            "grad_accum_steps": train.grad_accum_steps,
            "max_steps": train.max_steps,
            "log_every": train.log_every,
            "validation_every": train.validation_every,
            "validation_batches": train.validation_batches,
            "checkpoint_every": train.checkpoint_every,
            "sample_every": train.sample_every,
        }
        if any(value <= 0 for value in positive_ints.values()):
            raise ValueError(f"train integer settings must be positive: {positive_ints}")
        if train.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("train.precision must be fp32/fp16/bf16")
        if (
            train.learning_rate <= 0
            or train.weight_decay < 0
            or train.gradient_clip <= 0
            or not 0 <= train.ema_decay < 1
        ):
            raise ValueError("invalid optimizer/EMA settings")

        sampling = self.sampling
        if sampling.method != "ddim":
            raise ValueError("only DDIM sampling is implemented")
        if not 1 <= sampling.steps <= diffusion.timesteps:
            raise ValueError("sampling.steps must be between 1 and diffusion.timesteps")
        if sampling.eta < 0 or sampling.cfg_scale < 0 or sampling.preview_count <= 0:
            raise ValueError("invalid sampling settings")
        if self.evaluation.fid_reference_split not in {"train", "test"}:
            raise ValueError("evaluation.fid_reference_split must be train/test")
        if self.evaluation.fid_mode not in {"clean", "legacy_tensorflow", "legacy_pytorch"}:
            raise ValueError("evaluation.fid_mode is not a supported Clean-FID mode")
        if self.evaluation.fid_samples_pilot <= 0 or self.evaluation.fid_samples_final <= 0:
            raise ValueError("evaluation FID sample counts must be positive")
        if self.seeds.base < 0:
            raise ValueError("seeds.base must be non-negative")

    @property
    def effective_batch_size(self) -> int:
        return self.train.batch_size * self.train.grad_accum_steps

    @property
    def output_dir(self) -> Path:
        return Path(self.experiment.output_root) / self.experiment.name

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


T = TypeVar("T")


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return dict(value)


def _strict_dataclass(cls: type[T], value: Any, path: str) -> T:
    raw = _mapping(value, path)
    known = {item.name for item in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys at {path}: {sorted(unknown)}")
    hints = get_type_hints(cls)
    for key, item in raw.items():
        expected = hints[key]
        if expected is bool and type(item) is not bool:
            raise TypeError(f"{path}.{key} must be bool, received {type(item).__name__}")
        if expected is int and type(item) is not int:
            raise TypeError(f"{path}.{key} must be int, received {type(item).__name__}")
        if expected is float:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(f"{path}.{key} must be float, received {type(item).__name__}")
            raw[key] = float(item)
        if expected is str and type(item) is not str:
            raise TypeError(f"{path}.{key} must be str, received {type(item).__name__}")
        if dataclasses.is_dataclass(expected) and not isinstance(item, expected):
            raise TypeError(f"{path}.{key} must be {expected.__name__}")
    try:
        return cls(**raw)
    except TypeError as exc:
        raise TypeError(f"invalid config at {path}: {exc}") from exc


_BRACED_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}")


def _expand_env_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        fallback = match.group(3)
        current = os.environ.get(name)
        if current:
            return current
        if fallback is not None:
            return fallback
        return match.group(0)

    return os.path.expanduser(os.path.expandvars(_BRACED_ENV.sub(replace, value)))


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env_string(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if any(not part for part in parts):
        raise ValueError(f"invalid override path: {dotted_key}")
    cursor: dict[str, Any] = mapping
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            raise KeyError(f"override path does not exist: {dotted_key}")
        cursor = child
    if parts[-1] not in cursor:
        raise KeyError(f"override key does not exist: {dotted_key}")
    cursor[parts[-1]] = value


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"config root must be a mapping: {config_path}")
    raw = _expand_env(raw)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must have KEY=VALUE form: {override}")
        key, encoded = override.split("=", 1)
        _set_nested(raw, key, yaml.safe_load(encoded))
    return ExperimentConfig.from_dict(raw)


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False, allow_unicode=True)

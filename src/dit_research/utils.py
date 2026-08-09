from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch
from PIL import Image
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def seed_process(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(requested: str, device: torch.device) -> tuple[str, str | None]:
    if requested == "fp32":
        return "fp32", None
    if device.type == "cuda":
        if requested == "bf16" and not torch.cuda.is_bf16_supported():
            return "fp16", "bf16 is unavailable on this CUDA device; using fp16"
        return requested, None
    if device.type == "cpu":
        if requested == "bf16":
            return "bf16", None
        return "fp32", f"{requested} autocast is not used on CPU; using fp32"
    if device.type == "mps":
        if requested == "bf16":
            return "fp16", "bf16 MPS support is not assumed; using fp16"
        return requested, None
    return "fp32", f"unsupported autocast device {device.type}; using fp32"


@contextlib.contextmanager
def autocast_context(device: torch.device, precision: str) -> Iterator[None]:
    if precision == "fp32":
        with contextlib.nullcontext():
            yield
        return
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.autocast(device_type=device.type, dtype=dtype):
        yield


def make_generator(device: torch.device, seed: int) -> torch.Generator | None:
    try:
        generator = torch.Generator(device=device)
    except RuntimeError:
        torch.manual_seed(seed)
        return None
    generator.manual_seed(seed)
    return generator


def apply_class_dropout(
    labels: Tensor,
    probability: float,
    null_class_id: int,
    generator: torch.Generator | None,
) -> Tensor:
    if probability <= 0:
        return labels
    mask = torch.rand(labels.shape, device=labels.device, generator=generator) < probability
    return torch.where(mask, torch.full_like(labels, null_class_id), labels)


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, source: nn.Module) -> None:
        source = getattr(source, "_orig_mod", source)
        source_parameters = dict(source.named_parameters())
        for name, target in self.model.named_parameters():
            value = source_parameters[name].detach().to(dtype=target.dtype)
            target.mul_(self.decay).add_(value, alpha=1 - self.decay)
        source_buffers = dict(source.named_buffers())
        for name, target in self.model.named_buffers():
            value = source_buffers[name].detach()
            if target.is_floating_point():
                target.mul_(self.decay).add_(value.to(dtype=target.dtype), alpha=1 - self.decay)
            else:
                target.copy_(value)
        self.num_updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "decay": self.decay,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.decay = float(state["decay"])
        self.num_updates = int(state["num_updates"])


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, destination)


def append_jsonl(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch_state = state["torch"]
    if not isinstance(torch_state, Tensor):
        raise TypeError("saved torch RNG state must be a tensor")
    torch.set_rng_state(
        torch_state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    )
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = state["cuda"]
        if not isinstance(cuda_states, (list, tuple)) or not all(
            isinstance(item, Tensor) for item in cuda_states
        ):
            raise TypeError("saved CUDA RNG state must be a sequence of tensors")
        torch.cuda.set_rng_state_all(
            [
                item.detach().to(device="cpu", dtype=torch.uint8).contiguous()
                for item in cuda_states
            ]
        )


def make_grid(images: Tensor, nrow: int | None = None, padding: int = 2) -> Tensor:
    if images.ndim != 4 or images.shape[1] not in {1, 3}:
        raise ValueError("images must have shape [B,1|3,H,W]")
    batch, channels, height, width = images.shape
    if nrow is None:
        nrow = max(1, int(math_sqrt_ceil(batch)))
    columns = min(nrow, batch)
    rows = (batch + columns - 1) // columns
    grid = torch.full(
        (
            channels,
            rows * height + padding * max(rows - 1, 0),
            columns * width + padding * max(columns - 1, 0),
        ),
        -1.0,
        dtype=images.dtype,
        device=images.device,
    )
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        y = row * (height + padding)
        x = column * (width + padding)
        grid[:, y : y + height, x : x + width] = image
    return grid


def math_sqrt_ceil(value: int) -> int:
    candidate = int(value**0.5)
    return candidate if candidate * candidate == value else candidate + 1


def save_tensor_image(image: Tensor, path: str | Path) -> None:
    if image.ndim != 3 or image.shape[0] not in {1, 3}:
        raise ValueError("image must have shape [1|3,H,W]")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        image.detach()
        .float()
        .clamp(-1, 1)
        .add(1)
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .contiguous()
    )
    height, width, channels = encoded.shape
    data = bytes(encoded.reshape(-1).tolist())
    if channels == 1:
        pil_image = Image.frombytes("L", (width, height), data)
    else:
        pil_image = Image.frombytes("RGB", (width, height), data)
    pil_image.save(destination)


def _run_command(args: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=cwd,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _run_git(args: list[str]) -> str | None:
    return _run_command(["git", *args], cwd=PROJECT_ROOT)


def _paths_hash(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in files if path.is_file()):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _source_tree_hash() -> str:
    roots = ["README.md", "pyproject.toml", "requirements-lock.txt", "env.sh"]
    directories = ["configs", "docs", "scripts", "src", "tests"]
    files = [PROJECT_ROOT / name for name in roots]
    for directory in directories:
        files.extend((PROJECT_ROOT / directory).rglob("*"))
    return _paths_hash(files)


def _training_code_hash() -> str:
    files = [PROJECT_ROOT / "pyproject.toml", PROJECT_ROOT / "scripts" / "train.py"]
    files.extend((PROJECT_ROOT / "src" / "dit_research").rglob("*.py"))
    return _paths_hash(files)


def environment_manifest() -> dict[str, Any]:
    commit = _run_git(["rev-parse", "HEAD"])
    status = _run_git(["status", "--porcelain"])
    diff = _run_git(["diff", "--binary", "HEAD"])
    cuda: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        device_uuid = getattr(properties, "uuid", None)
        cuda.update(
            {
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(device_index),
                "capability": torch.cuda.get_device_capability(device_index),
                "arch_list": torch.cuda.get_arch_list(),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "total_memory_bytes": properties.total_memory,
                "device_uuid": str(device_uuid) if device_uuid is not None else None,
            }
        )
        nvidia_smi = _run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        if nvidia_smi:
            cuda["nvidia_smi_gpus"] = [
                {
                    "name": fields[0],
                    "uuid": fields[1],
                    "driver_version": fields[2],
                    "memory_total_mib": int(fields[3]),
                }
                for line in nvidia_smi.splitlines()
                if len(fields := [field.strip() for field in line.split(",")]) == 4
            ]
    package_versions = {}
    for distribution in (
        "torchvision",
        "triton",
        "fla-core",
        "numpy",
        "PyYAML",
        "Pillow",
    ):
        try:
            package_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_versions[distribution] = None
    return {
        "created_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "packages": package_versions,
        "cuda": cuda,
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_diff_hash": hashlib.sha256((diff or "").encode()).hexdigest(),
        "source_tree_hash": _source_tree_hash(),
        "training_code_hash": _training_code_hash(),
    }

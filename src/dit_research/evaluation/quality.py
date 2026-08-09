from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

from PIL import Image


def _validate_sample_directory(
    sample_dir: str | Path,
    *,
    require_manifest: bool = True,
    expected_count: int | None = None,
) -> tuple[Path, int, dict[str, Any] | None]:
    directory = Path(sample_dir)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    png_paths = sorted(path for path in directory.iterdir() if path.suffix.lower() == ".png")
    png_count = len(png_paths)
    if png_count == 0:
        raise ValueError(f"no PNG samples found in {directory}")
    manifest_path = directory / "sample_manifest.json"
    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise TypeError("sample_manifest.json must contain an object")
        manifest = loaded
        declared_count = manifest.get("num_samples")
        if type(declared_count) is not int or declared_count <= 0:
            raise ValueError("sample manifest num_samples must be a positive integer")
        if declared_count != png_count:
            raise ValueError(
                f"sample manifest declares {declared_count} images but found {png_count} PNG files"
            )
        expected_names = {f"{index:06d}.png" for index in range(declared_count)}
        actual_names = {path.name for path in png_paths}
        if actual_names != expected_names:
            raise ValueError("sample PNG filenames are incomplete or do not follow the generation manifest")
    elif require_manifest:
        raise FileNotFoundError(
            f"missing {manifest_path}; refuse metrics on an unverified or interrupted sample set"
        )
    if png_count < 2:
        raise ValueError("at least two samples are required for distribution metrics")
    if expected_count is not None:
        if expected_count <= 0:
            raise ValueError("expected_count must be positive")
        if png_count != expected_count:
            raise ValueError(f"expected {expected_count} PNG samples but found {png_count}")
    for path in png_paths:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (32, 32):
                raise ValueError(
                    f"expected 32x32 RGB PNG samples, found {image.format}/{image.mode}/{image.size} at {path}"
                )
    return directory, png_count, manifest


def clean_fid(
    sample_dir: str | Path,
    *,
    split: str = "train",
    mode: str = "clean",
    require_manifest: bool = True,
    expected_count: int | None = None,
) -> dict[str, Any]:
    directory, png_count, _ = _validate_sample_directory(
        sample_dir,
        require_manifest=require_manifest,
        expected_count=expected_count,
    )
    if split not in {"train", "test"}:
        raise ValueError("CIFAR-10 Clean-FID split must be train/test")
    if mode not in {"clean", "legacy_tensorflow", "legacy_pytorch"}:
        raise ValueError(f"unsupported Clean-FID mode: {mode}")
    try:
        from cleanfid import fid
    except ImportError as exc:
        raise RuntimeError('Clean-FID is optional; install with pip install -e ".[eval]"') from exc
    score = fid.compute_fid(
        str(directory),
        dataset_name="cifar10",
        dataset_res=32,
        dataset_split=split,
        mode=mode,
    )
    return {
        "fid": float(score),
        "fid_implementation": "clean-fid",
        "fid_implementation_version": importlib.metadata.version("clean-fid"),
        "fid_mode": mode,
        "fid_reference": f"cifar10-{split}-32",
        "sample_count": png_count,
    }


def torch_fidelity_metrics(
    sample_dir: str | Path,
    *,
    split: str = "train",
    cuda: bool = True,
    require_manifest: bool = True,
    expected_count: int | None = None,
) -> dict[str, Any]:
    directory, png_count, _ = _validate_sample_directory(
        sample_dir,
        require_manifest=require_manifest,
        expected_count=expected_count,
    )
    try:
        import torch_fidelity
    except ImportError as exc:
        raise RuntimeError('torch-fidelity is optional; install with pip install -e ".[eval]"') from exc
    reference = "cifar10-train" if split == "train" else "cifar10-val"
    values = torch_fidelity.calculate_metrics(
        input1=str(directory),
        input2=reference,
        cuda=cuda,
        isc=True,
        fid=True,
        kid=True,
        verbose=False,
    )
    return {
        **{key: float(value) for key, value in values.items()},
        "metric_implementation": "torch-fidelity",
        "metric_implementation_version": importlib.metadata.version("torch-fidelity"),
        "reference": reference,
        "sample_count": png_count,
    }

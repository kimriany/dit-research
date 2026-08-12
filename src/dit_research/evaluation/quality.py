from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


_CLEAN_FID_INPUT_SUFFIXES = {
    ".bmp",
    ".jpg",
    ".jpeg",
    ".pgm",
    ".png",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
    ".npy",
}


def _reject_extra_clean_fid_inputs(directory: Path) -> None:
    unexpected = [
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in _CLEAN_FID_INPUT_SUFFIXES
        and (path.parent != directory or path.suffix.lower() != ".png")
    ]
    if unexpected:
        raise ValueError(
            f"directory contains an extra file Clean-FID would treat as input: {unexpected[0]}"
        )


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
    _reject_extra_clean_fid_inputs(directory)
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


def _validate_reference_directory(
    reference_dir: str | Path,
    *,
    require_manifest: bool = True,
    expected_count: int | None = None,
) -> tuple[Path, int, dict[str, Any] | None]:
    directory = Path(reference_dir)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    png_paths = sorted(path for path in directory.iterdir() if path.suffix.lower() == ".png")
    png_count = len(png_paths)
    if png_count == 0:
        raise ValueError(f"no PNG reference images found in {directory}")
    _reject_extra_clean_fid_inputs(directory)
    manifest_path = directory / "reference_manifest.json"
    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise TypeError("reference_manifest.json must contain an object")
        manifest = loaded
        declared_count = manifest.get("num_images")
        if type(declared_count) is not int or declared_count <= 0:
            raise ValueError("reference manifest num_images must be a positive integer")
        if declared_count != png_count:
            raise ValueError(
                f"reference manifest declares {declared_count} images but found {png_count} PNG files"
            )
    elif require_manifest:
        raise FileNotFoundError(
            f"missing {manifest_path}; create the reference with export_cifar10_reference.py"
        )
    if png_count < 2:
        raise ValueError("at least two reference images are required")
    if expected_count is not None:
        if expected_count <= 0:
            raise ValueError("expected_count must be positive")
        if png_count != expected_count:
            raise ValueError(f"expected {expected_count} reference PNGs but found {png_count}")
    expected_names = {f"{index:06d}.png" for index in range(png_count)}
    actual_names = {path.name for path in png_paths}
    if actual_names != expected_names:
        raise ValueError("reference PNG filenames must be a complete zero-based sequence")
    for path in png_paths:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (32, 32):
                raise ValueError(
                    f"expected 32x32 RGB PNG references, found "
                    f"{image.format}/{image.mode}/{image.size} at {path}"
                )
    return directory, png_count, manifest


def _directory_fingerprint(directory: Path, manifest_name: str) -> str:
    digest = hashlib.sha256()
    manifest_path = directory / manifest_name
    if manifest_path.is_file():
        digest.update(manifest_path.read_bytes())
    for path in sorted(directory.glob("*.png")):
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _feature_cache_paths(directory: Path, mode: str) -> tuple[Path, Path]:
    stem = f".cleanfid_{mode}_features"
    # Clean-FID recursively treats .npy as an image input, so the cache uses a
    # neutral extension even though its contents follow NumPy's .npy format.
    return directory / f"{stem}.bin", directory / f"{stem}.json"


def _load_feature_cache(
    directory: Path,
    *,
    mode: str,
    image_count: int,
    fingerprint: str,
    clean_fid_version: str,
) -> np.ndarray | None:
    feature_path, metadata_path = _feature_cache_paths(directory, mode)
    if not feature_path.is_file() or not metadata_path.is_file():
        return None
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        expected = {
            "schema_version": 1,
            "mode": mode,
            "image_count": image_count,
            "directory_fingerprint": fingerprint,
            "clean_fid_version": clean_fid_version,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            return None
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if features.ndim != 2 or features.shape[0] != image_count or features.shape[1] <= 0:
            return None
        return features
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _save_feature_cache(
    directory: Path,
    features: np.ndarray,
    *,
    mode: str,
    image_count: int,
    fingerprint: str,
    clean_fid_version: str,
) -> None:
    feature_path, metadata_path = _feature_cache_paths(directory, mode)
    temporary_feature = feature_path.with_suffix(feature_path.suffix + ".tmp")
    with temporary_feature.open("wb") as handle:
        np.save(handle, features, allow_pickle=False)
    os.replace(temporary_feature, feature_path)
    metadata = {
        "schema_version": 1,
        "mode": mode,
        "image_count": image_count,
        "feature_dimension": int(features.shape[1]),
        "feature_dtype": str(features.dtype),
        "directory_fingerprint": fingerprint,
        "clean_fid_version": clean_fid_version,
    }
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with temporary_metadata.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_metadata, metadata_path)


def _kernel_inception_distance(
    real_features: np.ndarray,
    sample_features: np.ndarray,
    *,
    num_subsets: int = 100,
    max_subset_size: int = 1000,
    seed: int = 0,
) -> tuple[float, float, int]:
    real = np.asarray(real_features)
    sample = np.asarray(sample_features)
    if real.ndim != 2 or sample.ndim != 2 or real.shape[1] != sample.shape[1]:
        raise ValueError("real and sample features must be 2D with equal feature dimensions")
    if num_subsets <= 0 or max_subset_size < 2:
        raise ValueError("KID requires positive num_subsets and max_subset_size >= 2")
    subset_size = min(real.shape[0], sample.shape[0], max_subset_size)
    if subset_size < 2:
        raise ValueError("KID requires at least two real and sample features")
    rng = np.random.default_rng(seed)
    feature_dimension = real.shape[1]
    estimates: list[float] = []
    for _ in range(num_subsets):
        x = np.asarray(
            sample[rng.choice(sample.shape[0], subset_size, replace=False)],
            dtype=np.float64,
        )
        y = np.asarray(
            real[rng.choice(real.shape[0], subset_size, replace=False)],
            dtype=np.float64,
        )
        kernel_xx = (x @ x.T / feature_dimension + 1.0) ** 3
        kernel_yy = (y @ y.T / feature_dimension + 1.0) ** 3
        kernel_xy = (x @ y.T / feature_dimension + 1.0) ** 3
        estimate = (
            (kernel_xx.sum() - np.trace(kernel_xx))
            / (subset_size * (subset_size - 1))
            + (kernel_yy.sum() - np.trace(kernel_yy))
            / (subset_size * (subset_size - 1))
            - 2.0 * kernel_xy.mean()
        )
        estimates.append(float(estimate))
    mean = float(np.mean(estimates))
    subset_std = float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0
    return mean, subset_std, subset_size


def _improved_precision_recall(
    real_features: np.ndarray,
    sample_features: np.ndarray,
    *,
    sample_count: int = 10_000,
    nearest_k: int = 3,
    seed: int = 0,
    chunk_size: int = 1000,
    device: str = "cuda",
) -> tuple[float, float, int]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for precision/recall distances") from exc
    real = np.asarray(real_features)
    sample = np.asarray(sample_features)
    if real.ndim != 2 or sample.ndim != 2 or real.shape[1] != sample.shape[1]:
        raise ValueError("real and sample features must be 2D with equal feature dimensions")
    if sample_count <= nearest_k or nearest_k <= 0:
        raise ValueError("precision/recall sample_count must be greater than nearest_k > 0")
    if chunk_size <= 0:
        raise ValueError("precision/recall chunk_size must be positive")
    resolved_count = min(sample_count, real.shape[0], sample.shape[0])
    if resolved_count <= nearest_k:
        raise ValueError("not enough features for the requested precision/recall neighborhood")
    rng = np.random.default_rng(seed)
    real_indices = rng.choice(real.shape[0], resolved_count, replace=False)
    sample_indices = rng.choice(sample.shape[0], resolved_count, replace=False)
    torch_device = torch.device(device)
    real_tensor = torch.from_numpy(
        np.array(real[real_indices], dtype=np.float32, copy=True)
    ).to(torch_device)
    sample_tensor = torch.from_numpy(
        np.array(sample[sample_indices], dtype=np.float32, copy=True)
    ).to(torch_device)

    def manifold_radii(features: Any) -> Any:
        radii = []
        for start in range(0, resolved_count, chunk_size):
            distances = torch.cdist(features[start : start + chunk_size], features)
            radii.append(
                torch.topk(distances, nearest_k + 1, largest=False, dim=1).values[:, -1]
            )
        return torch.cat(radii)

    def coverage(manifold: Any, radii: Any, probes: Any) -> float:
        covered_total = 0
        for probe_start in range(0, resolved_count, chunk_size):
            probe_batch = probes[probe_start : probe_start + chunk_size]
            covered = torch.zeros(
                probe_batch.shape[0], dtype=torch.bool, device=torch_device
            )
            for manifold_start in range(0, resolved_count, chunk_size):
                manifold_batch = manifold[manifold_start : manifold_start + chunk_size]
                radius_batch = radii[manifold_start : manifold_start + chunk_size]
                distances = torch.cdist(probe_batch, manifold_batch)
                covered |= (distances <= radius_batch.unsqueeze(0) + 1e-5).any(dim=1)
                if bool(covered.all()):
                    break
            covered_total += int(covered.sum().item())
        return covered_total / resolved_count

    with torch.no_grad():
        real_radii = manifold_radii(real_tensor)
        sample_radii = manifold_radii(sample_tensor)
        precision = coverage(real_tensor, real_radii, sample_tensor)
        recall = coverage(sample_tensor, sample_radii, real_tensor)
    return float(precision), float(recall), resolved_count


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


def distribution_metrics(
    sample_dir: str | Path,
    reference_dir: str | Path,
    *,
    mode: str = "clean",
    require_sample_manifest: bool = True,
    require_reference_manifest: bool = True,
    expected_sample_count: int | None = None,
    expected_reference_count: int | None = None,
    batch_size: int = 256,
    num_workers: int = 8,
    kid_num_subsets: int = 100,
    kid_subset_size: int = 1000,
    kid_seed: int = 0,
    pr_sample_count: int = 10_000,
    pr_nearest_k: int = 3,
    pr_seed: int = 0,
    distance_chunk_size: int = 1000,
    device: str = "cuda",
    use_feature_cache: bool = True,
) -> dict[str, Any]:
    sample_directory, sample_count, _ = _validate_sample_directory(
        sample_dir,
        require_manifest=require_sample_manifest,
        expected_count=expected_sample_count,
    )
    reference_directory, reference_count, reference_manifest = _validate_reference_directory(
        reference_dir,
        require_manifest=require_reference_manifest,
        expected_count=expected_reference_count,
    )
    if mode not in {"clean", "legacy_tensorflow", "legacy_pytorch"}:
        raise ValueError(f"unsupported Clean-FID mode: {mode}")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers must be non-negative")
    try:
        import torch
        from cleanfid import fid
        from cleanfid.features import build_feature_extractor
    except ImportError as exc:
        raise RuntimeError('Clean-FID is optional; install with pip install -e ".[eval]"') from exc

    clean_fid_version = importlib.metadata.version("clean-fid")
    sample_fingerprint = _directory_fingerprint(sample_directory, "sample_manifest.json")
    reference_fingerprint = _directory_fingerprint(
        reference_directory, "reference_manifest.json"
    )
    sample_features = None
    reference_features = None
    if use_feature_cache:
        sample_features = _load_feature_cache(
            sample_directory,
            mode=mode,
            image_count=sample_count,
            fingerprint=sample_fingerprint,
            clean_fid_version=clean_fid_version,
        )
        reference_features = _load_feature_cache(
            reference_directory,
            mode=mode,
            image_count=reference_count,
            fingerprint=reference_fingerprint,
            clean_fid_version=clean_fid_version,
        )
    sample_cache_hit = sample_features is not None
    reference_cache_hit = reference_features is not None
    torch_device = torch.device(device)
    if sample_features is None or reference_features is None:
        feature_model = build_feature_extractor(
            mode, torch_device, use_dataparallel=False
        )

        def extract(directory: Path, description: str) -> np.ndarray:
            features = fid.get_folder_features(
                str(directory),
                feature_model,
                num_workers=num_workers,
                batch_size=batch_size,
                device=torch_device,
                mode=mode,
                description=description,
                verbose=True,
            )
            resolved = np.asarray(features, dtype=np.float32)
            if resolved.ndim != 2 or resolved.shape[1] <= 0:
                raise ValueError(f"Clean-FID returned invalid feature shape {resolved.shape}")
            if not np.isfinite(resolved).all():
                raise ValueError("Clean-FID returned non-finite features")
            return resolved

        if reference_features is None:
            reference_features = extract(reference_directory, "reference features")
            if reference_features.shape[0] != reference_count:
                raise ValueError(
                    f"expected {reference_count} reference features, got {reference_features.shape[0]}"
                )
            if use_feature_cache:
                _save_feature_cache(
                    reference_directory,
                    reference_features,
                    mode=mode,
                    image_count=reference_count,
                    fingerprint=reference_fingerprint,
                    clean_fid_version=clean_fid_version,
                )
        if sample_features is None:
            sample_features = extract(sample_directory, "sample features")
            if sample_features.shape[0] != sample_count:
                raise ValueError(
                    f"expected {sample_count} sample features, got {sample_features.shape[0]}"
                )
            if use_feature_cache:
                _save_feature_cache(
                    sample_directory,
                    sample_features,
                    mode=mode,
                    image_count=sample_count,
                    fingerprint=sample_fingerprint,
                    clean_fid_version=clean_fid_version,
                )

    assert sample_features is not None and reference_features is not None
    if sample_features.shape[1] != reference_features.shape[1]:
        raise ValueError("sample and reference feature dimensions differ")
    fid_payload: dict[str, Any] = {}
    if reference_manifest is not None:
        reference_dataset = reference_manifest.get("dataset")
        reference_split = reference_manifest.get("split")
        if reference_dataset != "cifar10" or reference_split not in {"train", "test"}:
            raise ValueError(
                "distribution metrics currently require a CIFAR-10 train/test reference manifest"
            )
        reference_mu, reference_sigma = fid.get_reference_statistics(
            "cifar10",
            32,
            mode=mode,
            model_name="inception_v3",
            seed=0,
            split=reference_split,
        )
        sample_mu = np.mean(sample_features, axis=0)
        sample_sigma = np.cov(sample_features, rowvar=False)
        fid_payload = {
            "fid": float(
                fid.frechet_distance(
                    sample_mu,
                    sample_sigma,
                    reference_mu,
                    reference_sigma,
                )
            ),
            "fid_implementation": "clean-fid",
            "fid_implementation_version": clean_fid_version,
            "fid_mode": mode,
            "fid_reference": f"cifar10-{reference_split}-32",
        }
    kid, kid_subset_std, resolved_kid_subset_size = _kernel_inception_distance(
        reference_features,
        sample_features,
        num_subsets=kid_num_subsets,
        max_subset_size=kid_subset_size,
        seed=kid_seed,
    )
    precision, recall, resolved_pr_sample_count = _improved_precision_recall(
        reference_features,
        sample_features,
        sample_count=pr_sample_count,
        nearest_k=pr_nearest_k,
        seed=pr_seed,
        chunk_size=distance_chunk_size,
        device=str(torch_device),
    )
    reference_name = (
        f"{reference_manifest.get('dataset', 'folder')}-"
        f"{reference_manifest.get('split', 'unknown')}"
        if reference_manifest is not None
        else str(reference_directory)
    )
    return {
        **fid_payload,
        "kid": kid,
        "kid_subset_std": kid_subset_std,
        "kid_num_subsets": kid_num_subsets,
        "kid_subset_size": resolved_kid_subset_size,
        "kid_seed": kid_seed,
        "precision": precision,
        "recall": recall,
        "precision_recall_sample_count": resolved_pr_sample_count,
        "precision_recall_nearest_k": pr_nearest_k,
        "precision_recall_seed": pr_seed,
        "precision_recall_distance_chunk_size": distance_chunk_size,
        "distribution_metric_implementation": "dit-research-clean-fid-features",
        "precision_recall_definition": (
            "Kynkaanniemi-style-kNN-manifold-on-clean-fid-inception-features"
        ),
        "feature_implementation": "clean-fid",
        "feature_implementation_version": clean_fid_version,
        "feature_mode": mode,
        "feature_dimension": int(sample_features.shape[1]),
        "feature_cache_enabled": use_feature_cache,
        "sample_feature_cache_hit": sample_cache_hit,
        "reference_feature_cache_hit": reference_cache_hit,
        "reference": reference_name,
        "reference_count": reference_count,
        "sample_count": sample_count,
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

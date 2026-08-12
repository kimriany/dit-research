#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path

from PIL import Image

from _bootstrap import ROOT  # noqa: F401
from dit_research.evaluation.quality import _validate_reference_directory
from dit_research.utils import atomic_json_dump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export torchvision CIFAR-10 as a validated PNG reference folder"
    )
    parser.add_argument("--data-root", default=str(ROOT / "datasets"))
    parser.add_argument("--output", default=str(ROOT / "datasets" / "cifar10_train_png"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--download",
        action="store_true",
        help="allow torchvision to download CIFAR-10 when it is not already present",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from torchvision.datasets import CIFAR10
    except ImportError as exc:
        raise RuntimeError("torchvision is required to export the CIFAR-10 reference") from exc

    output = Path(args.output)
    manifest_path = output / "reference_manifest.json"
    expected_count = 50_000 if args.split == "train" else 10_000
    if manifest_path.is_file():
        _, count, manifest = _validate_reference_directory(
            output, expected_count=expected_count
        )
        assert manifest is not None
        if manifest.get("dataset") != "cifar10" or manifest.get("split") != args.split:
            raise ValueError("existing reference manifest does not match CIFAR-10 split")
        print(f"reference already complete: {output} ({count} images)")
        return

    dataset = CIFAR10(
        root=args.data_root,
        train=args.split == "train",
        download=args.download,
    )
    if len(dataset) != expected_count:
        raise ValueError(f"expected {expected_count} CIFAR-10 images, found {len(dataset)}")
    output.mkdir(parents=True, exist_ok=True)
    existing = sorted(path for path in output.iterdir() if path.suffix.lower() == ".png")
    expected_existing_names = [f"{index:06d}.png" for index in range(len(existing))]
    if [path.name for path in existing] != expected_existing_names:
        raise ValueError(
            "partial reference PNGs are not a contiguous zero-based prefix; "
            "move the directory aside and export again"
        )
    for path in existing:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (32, 32):
                raise ValueError(f"invalid partial reference image: {path}")
    start = len(existing)
    if start:
        print(f"resuming reference export at {start}/{expected_count}")
    for index in range(start, expected_count):
        image, _ = dataset[index]
        if image.mode != "RGB" or image.size != (32, 32):
            raise ValueError(f"unexpected CIFAR-10 image format at index {index}")
        destination = output / f"{index:06d}.png"
        temporary = output / f".{index:06d}.png.tmp"
        image.save(temporary, format="PNG")
        os.replace(temporary, destination)
        if (index + 1) % 1000 == 0 or index + 1 == expected_count:
            print(f"exported {index + 1}/{expected_count}", flush=True)

    atomic_json_dump(
        {
            "schema_version": 1,
            "dataset": "cifar10",
            "split": args.split,
            "num_images": expected_count,
            "image_mode": "RGB",
            "image_size": [32, 32],
            "filename_rule": "zero_based_six_digit_png",
            "torchvision_version": importlib.metadata.version("torchvision"),
        },
        manifest_path,
    )
    _validate_reference_directory(output, expected_count=expected_count)
    print(f"reference ready: {output} ({expected_count} images)")


if __name__ == "__main__":
    main()

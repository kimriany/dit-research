from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from _path import ROOT  # noqa: F401
from dit_research.evaluation.quality import (
    _improved_precision_recall,
    _kernel_inception_distance,
    _validate_reference_directory,
    _validate_sample_directory,
    distribution_metrics,
)


try:
    import torch

    torch.from_numpy(np.zeros((1,), dtype=np.float32))
    TORCH_NUMPY_AVAILABLE = True
except (ImportError, RuntimeError):
    TORCH_NUMPY_AVAILABLE = False


class QualityValidationTests(unittest.TestCase):
    def test_manifest_count_and_image_shape_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(2):
                Image.new("RGB", (32, 32)).save(root / f"{index:06d}.png")
            (root / "sample_manifest.json").write_text(
                json.dumps({"num_samples": 3}), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                _validate_sample_directory(root)

            (root / "sample_manifest.json").write_text(
                json.dumps({"num_samples": 2}), encoding="utf-8"
            )
            _, count, manifest = _validate_sample_directory(root)
            self.assertEqual(count, 2)
            self.assertEqual(manifest["num_samples"], 2)

    def test_preview_grid_without_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (100, 100)).save(root / "step_00000001.png")
            with self.assertRaises(FileNotFoundError):
                _validate_sample_directory(root)

    def test_extra_clean_fid_input_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(2):
                Image.new("RGB", (32, 32)).save(root / f"{index:06d}.png")
            (root / "sample_manifest.json").write_text(
                json.dumps({"num_samples": 2}), encoding="utf-8"
            )
            Image.new("RGB", (32, 32)).save(root / "preview.jpg")
            with self.assertRaisesRegex(ValueError, "Clean-FID would treat as input"):
                _validate_sample_directory(root)

    def test_reference_manifest_and_sequence_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(3):
                Image.new("RGB", (32, 32)).save(root / f"{index:06d}.png")
            (root / "reference_manifest.json").write_text(
                json.dumps({"dataset": "cifar10", "split": "train", "num_images": 3}),
                encoding="utf-8",
            )
            _, count, manifest = _validate_reference_directory(root, expected_count=3)
            self.assertEqual(count, 3)
            self.assertEqual(manifest["dataset"], "cifar10")
            (root / "000002.png").rename(root / "000003.png")
            with self.assertRaises(ValueError):
                _validate_reference_directory(root, expected_count=3)

    def test_kid_subsets_are_seeded_and_report_the_resolved_size(self) -> None:
        real = np.arange(48, dtype=np.float32).reshape(12, 4) / 10
        sample = real + 0.25
        first = _kernel_inception_distance(
            real, sample, num_subsets=5, max_subset_size=7, seed=123
        )
        second = _kernel_inception_distance(
            real, sample, num_subsets=5, max_subset_size=7, seed=123
        )
        self.assertEqual(first, second)
        self.assertEqual(first[2], 7)
        self.assertTrue(np.isfinite(first[0]))
        self.assertGreaterEqual(first[1], 0.0)

    @unittest.skipUnless(TORCH_NUMPY_AVAILABLE, "requires PyTorch with NumPy interop")
    def test_improved_precision_recall_handles_identical_and_disjoint_sets(self) -> None:
        real = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        identical = _improved_precision_recall(
            real,
            real.copy(),
            sample_count=4,
            nearest_k=1,
            seed=7,
            chunk_size=2,
            device="cpu",
        )
        self.assertEqual(identical, (1.0, 1.0, 4))
        far = _improved_precision_recall(
            real,
            real + 100.0,
            sample_count=4,
            nearest_k=1,
            seed=7,
            chunk_size=2,
            device="cpu",
        )
        self.assertEqual(far, (0.0, 0.0, 4))

    @unittest.skipUnless(TORCH_NUMPY_AVAILABLE, "requires PyTorch with NumPy interop")
    def test_distribution_metrics_reuses_one_feature_definition_for_all_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = root / "samples"
            reference = root / "reference"
            samples.mkdir()
            reference.mkdir()
            for directory in (samples, reference):
                for index in range(4):
                    Image.new("RGB", (32, 32), color=(index, index, index)).save(
                        directory / f"{index:06d}.png"
                    )
            (samples / "sample_manifest.json").write_text(
                json.dumps({"num_samples": 4}), encoding="utf-8"
            )
            (reference / "reference_manifest.json").write_text(
                json.dumps({"dataset": "cifar10", "split": "train", "num_images": 4}),
                encoding="utf-8",
            )

            fake_fid = types.ModuleType("cleanfid.fid")
            fake_fid.get_folder_features = lambda directory, *_args, **_kwargs: np.asarray(
                [[0.0], [1.0], [2.0], [3.0]], dtype=np.float32
            )
            fake_fid.get_reference_statistics = lambda *_args, **_kwargs: (
                np.asarray([0.0]),
                np.asarray([[1.0]]),
            )
            fake_fid.frechet_distance = lambda mu1, _sigma1, mu2, _sigma2: float(
                np.square(mu1 - mu2).sum()
            )
            fake_features = types.ModuleType("cleanfid.features")
            fake_features.build_feature_extractor = lambda *_args, **_kwargs: object()
            fake_cleanfid = types.ModuleType("cleanfid")
            fake_cleanfid.fid = fake_fid
            with mock.patch.dict(
                sys.modules,
                {
                    "cleanfid": fake_cleanfid,
                    "cleanfid.fid": fake_fid,
                    "cleanfid.features": fake_features,
                },
            ), mock.patch("importlib.metadata.version", return_value="0.test"):
                result = distribution_metrics(
                    samples,
                    reference,
                    expected_sample_count=4,
                    expected_reference_count=4,
                    kid_num_subsets=2,
                    kid_subset_size=4,
                    pr_sample_count=4,
                    pr_nearest_k=1,
                    distance_chunk_size=2,
                    device="cpu",
                    use_feature_cache=False,
                )
            self.assertEqual(result["sample_count"], 4)
            self.assertEqual(result["reference_count"], 4)
            self.assertEqual(result["fid"], 2.25)
            self.assertEqual(result["precision"], 1.0)
            self.assertEqual(result["recall"], 1.0)
            self.assertIn("kid", result)


if __name__ == "__main__":
    unittest.main()

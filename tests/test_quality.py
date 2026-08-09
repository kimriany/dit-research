from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from _path import ROOT  # noqa: F401
from dit_research.evaluation.quality import _validate_sample_directory


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


if __name__ == "__main__":
    unittest.main()

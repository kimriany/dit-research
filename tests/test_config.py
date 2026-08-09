from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from _path import ROOT  # noqa: F401
from dit_research.config import load_config


class ConfigTests(unittest.TestCase):
    def test_all_experiment_yaml_files_are_strict_and_valid(self) -> None:
        paths = sorted((ROOT / "configs").glob("**/*.yaml"))
        paths = [path for path in paths if "matrices" not in path.parts]
        self.assertGreaterEqual(len(paths), 7)
        for path in paths:
            with self.subTest(path=path):
                load_config(path)

    def test_environment_fallback_is_expanded(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            config = load_config(ROOT / "configs" / "smoke" / "dit_tiny.yaml")
        self.assertEqual(config.data.root, "./datasets")

    def test_unknown_override_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            load_config(
                ROOT / "configs" / "smoke" / "dit_tiny.yaml",
                ["model.typo=123"],
            )

    def test_invalid_hidden_head_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_config(
                ROOT / "configs" / "smoke" / "dit_tiny.yaml",
                ["model.hidden_size=190"],
            )

    def test_zero_divisors_are_cleanly_rejected(self) -> None:
        for override in ("model.patch_size=0", "model.num_heads=0"):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_config(
                    ROOT / "configs" / "smoke" / "dit_tiny.yaml",
                    [override],
                )

    def test_quoted_boolean_and_boolean_integer_are_rejected(self) -> None:
        for override in ('runtime.compile="false"', "train.batch_size=true"):
            with self.subTest(override=override), self.assertRaises(TypeError):
                load_config(
                    ROOT / "configs" / "smoke" / "dit_tiny.yaml",
                    [override],
                )

    def test_invalid_evaluation_settings_are_rejected(self) -> None:
        for override in ("evaluation.fid_samples_final=0", "evaluation.fid_mode=made_up"):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_config(
                    ROOT / "configs" / "smoke" / "dit_tiny.yaml",
                    [override],
                )

    def test_memory_config_list_is_normalized_and_validated(self) -> None:
        config = load_config(ROOT / "configs" / "smoke" / "dit_tiny_memory.yaml")
        self.assertEqual(config.model.memory.block_indices, (2, 5))
        self.assertEqual(config.model.memory.kind, "hybrid_gdn2")
        for override in (
            "model.memory.block_indices=[2,2]",
            "model.memory.block_indices=[2,7]",
            "model.memory.gate_mode=unknown",
            "model.memory.scan_pattern=diagonal",
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_config(
                    ROOT / "configs" / "smoke" / "dit_tiny_memory.yaml",
                    [override],
                )


if __name__ == "__main__":
    unittest.main()

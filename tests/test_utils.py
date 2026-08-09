from __future__ import annotations

import unittest

import torch

from _path import ROOT  # noqa: F401
from dit_research.utils import capture_rng_state, restore_rng_state


class RNGStateTests(unittest.TestCase):
    def test_restore_normalizes_cpu_rng_state_to_byte_tensor(self) -> None:
        state = capture_rng_state()
        expected = torch.rand(8)
        state["torch"] = state["torch"].to(torch.int16)

        restore_rng_state(state)
        actual = torch.rand(8)

        self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()

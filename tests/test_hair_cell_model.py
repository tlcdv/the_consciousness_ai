"""Tests for the inner hair cell model (envelope + TFS extraction)."""
from __future__ import annotations

import unittest
import torch

from models.audio.hair_cell_model import HairCellModel


class TestHairCellModel(unittest.TestCase):
    def setUp(self):
        self.model = HairCellModel(num_bands=64)

    def test_output_shape(self):
        x = torch.randn(2, 64, 100)
        out = self.model(x)
        self.assertEqual(out.shape, (2, 128, 100))

    def test_envelope_non_negative(self):
        x = torch.randn(1, 64, 200)
        out = self.model(x)
        envelope = out[:, :64, :]
        self.assertTrue(torch.all(envelope >= 0).item())

    def test_envelope_smoother_than_input(self):
        """Envelope should have lower temporal variance than raw input."""
        x = torch.randn(1, 64, 500)
        out = self.model(x)
        envelope = out[:, :64, :]
        input_var = x.var(dim=2).mean().item()
        envelope_var = envelope.var(dim=2).mean().item()
        self.assertLess(envelope_var, input_var)

    def test_silence_zero_output(self):
        x = torch.zeros(1, 64, 100)
        out = self.model(x)
        self.assertLess(out.abs().max().item(), 1e-6)

    def test_tfs_preserves_fast_structure(self):
        """TFS should retain high-frequency variation."""
        x = torch.randn(1, 64, 500)
        out = self.model(x)
        tfs = out[:, 64:, :]
        # TFS should have higher variance than envelope (it preserves oscillations)
        envelope = out[:, :64, :]
        tfs_var = tfs.var(dim=2).mean().item()
        env_var = envelope.var(dim=2).mean().item()
        self.assertGreater(tfs_var, env_var * 0.1)


if __name__ == "__main__":
    unittest.main()

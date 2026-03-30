"""Tests for the gammatone filterbank (cochlear decomposition)."""
from __future__ import annotations

import unittest
import torch
import numpy as np

from models.audio.gammatone_filterbank import GammatoneFilterbank, erb_space


class TestERBSpace(unittest.TestCase):
    def test_frequency_count(self):
        freqs = erb_space(20.0, 16000.0, 64)
        self.assertEqual(len(freqs), 64)

    def test_ascending_order(self):
        freqs = erb_space(20.0, 16000.0, 64)
        for i in range(len(freqs) - 1):
            self.assertLess(freqs[i], freqs[i + 1])

    def test_frequency_range(self):
        freqs = erb_space(20.0, 16000.0, 64)
        self.assertGreaterEqual(freqs[0], 15.0)
        self.assertLessEqual(freqs[-1], 16100.0)


class TestGammatoneFilterbank(unittest.TestCase):
    def setUp(self):
        self.fb = GammatoneFilterbank(num_bands=64, sample_rate=16000)

    def test_output_shape(self):
        waveform = torch.randn(1, 1, 16000)  # 1 sec mono
        out = self.fb(waveform)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 64)
        self.assertGreater(out.shape[2], 0)

    def test_frequency_selectivity(self):
        """Pure 440 Hz tone should activate band near 440 Hz most."""
        sr = 16000
        t = torch.arange(sr, dtype=torch.float32) / sr
        tone_440 = (0.5 * torch.sin(2 * 3.14159 * 440 * t)).unsqueeze(0).unsqueeze(0)
        out = self.fb(tone_440)
        band_energy = out.pow(2).sum(dim=2).squeeze()
        peak_band = band_energy.argmax().item()
        # 440 Hz should be in roughly the lower-middle bands (not at edges)
        cfs = self.fb.get_center_frequencies().numpy()
        peak_cf = cfs[peak_band]
        self.assertGreater(peak_cf, 200)
        self.assertLess(peak_cf, 800)

    def test_silence_near_zero(self):
        waveform = torch.zeros(1, 1, 4000)
        out = self.fb(waveform)
        self.assertLess(out.abs().max().item(), 1e-5)

    def test_batch_consistency(self):
        waveform = torch.randn(1, 1, 4000)
        batch = waveform.expand(3, -1, -1)
        out = self.fb(batch)
        self.assertEqual(out.shape[0], 3)
        torch.testing.assert_close(out[0], out[1])

    def test_no_gradients(self):
        for p in self.fb.parameters():
            self.assertFalse(p.requires_grad)

    def test_center_frequencies_accessible(self):
        cfs = self.fb.get_center_frequencies()
        self.assertEqual(cfs.shape[0], 64)
        self.assertTrue(torch.all(cfs > 0))

    def test_short_audio(self):
        """66ms at 16kHz = 1056 samples (typical env step)."""
        waveform = torch.randn(1, 1, 1056)
        out = self.fb(waveform)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 64)
        self.assertGreater(out.shape[2], 0)


if __name__ == "__main__":
    unittest.main()

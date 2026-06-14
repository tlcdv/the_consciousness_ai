"""
Tests for the working-memory mechanisms.

WorkingMemoryLatch captures a state while the frame is blank and freezes it when a
stimulus is present. ObsMapSampleMemory is the corrected, working DMTS mechanism: it
captures the obs_map at the sample onset (stimulus after a SHORT fixation blank) and
holds it through the delay and choice, skipping the choices (stimulus after a LONG
delay blank). These tests use synthetic blank/stimulus frames and verify the gating
logic; the leakage-free behavioral validation lives in
scripts/analysis/probe_wm_leakage_free.py.
"""
import unittest

import torch

from models.self_model.working_memory_latch import (
    WorkingMemoryLatch, ObsMapSampleMemory,
)


def _blank():
    return torch.zeros(1, 3, 8, 8)


def _stim():
    f = torch.zeros(1, 3, 8, 8)
    f[0, 0, :4, :] = 1.0  # high-energy frame (std > 0)
    return f


class TestWorkingMemoryLatch(unittest.TestCase):
    def test_captures_blank_freezes_stimulus(self):
        latch = WorkingMemoryLatch()
        h1 = torch.full((1, 8), 1.0)
        h2 = torch.full((1, 8), 2.0)
        latch.update(h1, _blank())          # blank -> capture h1
        self.assertTrue(torch.equal(latch.latched, h1))
        out = latch.update(h2, _stim())     # stimulus -> freeze, hold h1
        self.assertTrue(torch.equal(out, h1))

    def test_reset_clears(self):
        latch = WorkingMemoryLatch()
        latch.update(torch.ones(1, 8), _blank())
        self.assertIsNotNone(latch.latched)
        latch.reset()
        self.assertIsNone(latch.latched)


class TestObsMapSampleMemory(unittest.TestCase):
    def test_captures_sample_skips_choice(self):
        """Capture the sample (stimulus after the short fixation blank); do NOT
        overwrite with the choices (stimulus after the long delay blank)."""
        mem = ObsMapSampleMemory(short_blank_max=12)
        obs_sample = torch.full((1, 16), 1.0)
        obs_choice = torch.full((1, 16), 2.0)
        for _ in range(10):                 # fixation: 10 blanks
            mem.update(torch.zeros(1, 16), _blank())
        mem.update(obs_sample, _stim())     # sample onset -> capture
        for _ in range(5):                  # sample continues
            mem.update(obs_sample, _stim())
        self.assertTrue(torch.equal(mem.slot, obs_sample))
        for _ in range(20):                 # delay: 20 blanks
            mem.update(torch.zeros(1, 16), _blank())
        slot = mem.update(obs_choice, _stim())  # choice onset -> do NOT capture
        self.assertTrue(torch.equal(slot, obs_sample))
        self.assertFalse(torch.equal(slot, obs_choice))

    def test_returns_current_before_any_capture(self):
        mem = ObsMapSampleMemory()
        cur = torch.full((1, 16), 7.0)
        out = mem.update(cur, _blank())     # nothing captured yet
        self.assertTrue(torch.equal(out, cur))

    def test_reset_clears(self):
        mem = ObsMapSampleMemory(short_blank_max=12)
        for _ in range(5):
            mem.update(torch.zeros(1, 16), _blank())
        mem.update(torch.ones(1, 16), _stim())
        self.assertIsNotNone(mem.slot)
        mem.reset()
        self.assertIsNone(mem.slot)
        self.assertEqual(mem._blank_run, 0)


if __name__ == "__main__":
    unittest.main()

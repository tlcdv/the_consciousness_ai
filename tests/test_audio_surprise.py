"""The audio bid from auditory surprise.

Before 2026-09-15 the audio bid came from an untrained MLP that no optimizer updated,
while the class docstring claimed "acoustic novelty and loudness change". With
--audio-salience surprise, a small predictor learns the next cochlear band profile from
the current one; the bid is the running z-score of its error (the same reduction as the
vision bid). A constant sound habituates; a new sound raises the bid.
"""

import unittest

import numpy as np
import torch

from models.audio.auditory_specialist import AuditorySpecialist
from scripts.training.session_recorder import describe_modules

SAMPLE_RATE = 16000
FRAME = 1056


def _tone(freq, amplitude=0.3):
    t = np.arange(FRAME) / SAMPLE_RATE
    return torch.tensor(amplitude * np.sin(2 * np.pi * freq * t), dtype=torch.float32).view(1, 1, -1)


def _specialist(mode=None):
    config = {"workspace_dim": 256, "tectum_feature_dim": 64, "tectum_grid_size": 16}
    if mode is not None:
        config["audio_salience"] = mode
    torch.manual_seed(0)
    return AuditorySpecialist(config)


class TestDefaultUnchanged(unittest.TestCase):

    def test_default_has_no_surprise_module(self):
        self.assertIsNone(_specialist().surprise)
        self.assertEqual(_specialist().owned_optimizers(), [])

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            _specialist("loudness")


class TestSurpriseBid(unittest.TestCase):

    def test_first_sound_bids_the_midpoint(self):
        _, bid = _specialist("surprise")(_tone(440))
        self.assertEqual(bid, 0.5)

    def test_a_constant_sound_habituates(self):
        specialist = _specialist("surprise")
        bids = [specialist(_tone(440))[1] for _ in range(300)]
        errors_early = specialist.surprise.error_history[1:11]
        errors_late = specialist.surprise.error_history[-10:]
        self.assertLess(np.mean(errors_late), 0.1 * np.mean(errors_early))
        self.assertLessEqual(np.mean(bids[-20:]), 0.55)

    def test_a_new_sound_after_habituation_bids_high(self):
        specialist = _specialist("surprise")
        for _ in range(300):
            specialist(_tone(440))
        _, bid = specialist(_tone(2500, amplitude=0.6))
        self.assertGreater(bid, 0.9)

    def test_no_learning_without_gradients(self):
        specialist = _specialist("surprise")
        specialist(_tone(440))
        before = [p.detach().clone() for p in specialist.surprise.predictor.parameters()]
        with torch.no_grad():
            for _ in range(5):
                specialist(_tone(880))
        after = list(specialist.surprise.predictor.parameters())
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, after)))

    def test_no_learning_in_eval_mode(self):
        specialist = _specialist("surprise")
        specialist(_tone(440))
        specialist.eval()
        before = [p.detach().clone() for p in specialist.surprise.predictor.parameters()]
        for _ in range(5):
            specialist(_tone(880))
        after = list(specialist.surprise.predictor.parameters())
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, after)))

    def test_the_session_record_reports_the_module_as_trained(self):
        specialist = _specialist("surprise")
        facts = describe_modules(None, specialist, None, has_self_model=False,
                                 drive_off=False, optimizers=[])
        self.assertTrue(facts["audio"]["trained"])


if __name__ == "__main__":
    unittest.main()

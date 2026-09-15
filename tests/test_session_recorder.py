"""Tests for the session recorder.

Every in-scope training run records selected episodes so a session can be
replayed. The recorder must also show which modules received real input,
because a module with no input produces a constant bid that looks like a
mechanism failure (see docs/results/modality_starvation_2026_09.md).
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.training.session_recorder import (
    SessionRecorder,
    audio_input_source,
    body_input_source,
    memory_input_source,
    module_is_optimized,
    parse_record_episodes,
)

FRAME_SHAPE = (8, 8, 3)
BIDS = {"vision": 1.0, "audio": 0.3, "memory": 0.1, "body": 0.05, "semantic": 0.0}
SOURCES = {"vision": "frame", "audio": "sound", "memory": "default",
           "body": "constant", "semantic": "none"}
INTERO = {"energy": 0.9, "fatigue": 0.1, "damage": 0.0}


def _frame(value):
    return np.full(FRAME_SHAPE, value, dtype=np.uint8)


def _record_steps(recorder, count):
    for step in range(count):
        recorder.record_inputs(_frame(step), SOURCES, BIDS, INTERO)
        recorder.record_outcome(winner="vision", ignited=True, sync_r=0.25,
                                bound_bids=BIDS, reward=-0.01, action=[0.1, -0.2],
                                env_phase="")
        recorder.record_extra(scalars={"x": step}, vectors={"v": np.full(4, step)},
                              maps={"m": np.full((2, 2), step)},
                              full={"f": np.full(3, step)})


class TestEpisodeSelection(unittest.TestCase):

    def test_default_is_first_last_and_every_tenth(self):
        self.assertEqual(parse_record_episodes("default", 25), frozenset({0, 10, 20, 24}))

    def test_single_episode_run_records_it(self):
        self.assertEqual(parse_record_episodes("default", 1), frozenset({0}))

    def test_none_records_nothing(self):
        self.assertEqual(parse_record_episodes("none", 25), frozenset())

    def test_explicit_list_is_kept_inside_the_run(self):
        self.assertEqual(parse_record_episodes("0,3,99", 10), frozenset({0, 3}))

    def test_malformed_spec_raises(self):
        with self.assertRaises(ValueError):
            parse_record_episodes("first", 10)


class TestInputSources(unittest.TestCase):

    def test_audio_sources(self):
        self.assertEqual(audio_input_source(has_specialist=False, waveform=None), "none")
        self.assertEqual(audio_input_source(has_specialist=True, waveform=None), "none")
        silent = np.zeros(16, dtype=np.float32)
        self.assertEqual(audio_input_source(has_specialist=True, waveform=silent), "silent")
        tone = np.full(16, 0.2, dtype=np.float32)
        self.assertEqual(audio_input_source(has_specialist=True, waveform=tone), "sound")

    def test_memory_sources(self):
        self.assertEqual(memory_input_source(0.1), "default")
        self.assertEqual(memory_input_source(0.35), "retrieved")

    def test_body_sources(self):
        self.assertEqual(body_input_source(has_self_model=True, drive_off=False), "energy_counter")
        self.assertEqual(body_input_source(has_self_model=True, drive_off=True),
                         "constant_by_ethics_framework")
        self.assertEqual(body_input_source(has_self_model=False, drive_off=False), "constant")


class TestModuleIsOptimized(unittest.TestCase):
    """Whether a module learns is read from the real optimizers, never typed in."""

    def test_module_in_an_optimizer(self):
        trained = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(trained.parameters(), lr=1e-3)
        self.assertTrue(module_is_optimized(trained, [optimizer]))

    def test_module_in_no_optimizer(self):
        trained, untrained = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(trained.parameters(), lr=1e-3)
        self.assertFalse(module_is_optimized(untrained, [optimizer]))

    def test_absent_module_is_none(self):
        self.assertIsNone(module_is_optimized(None, []))


class TestRecording(unittest.TestCase):
    """Owner decision 2026-09-15: small data and frames for every episode; maps and
    weights for the recorded episodes; full world-model tensors for the first and
    last episode."""

    TOTAL = 5
    MAX_STEPS = 6

    def setUp(self):
        self.log_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.log_dir, ignore_errors=True)

    def _recorder(self, selected=frozenset({0, 4}), min_free_gb=0.0):
        return SessionRecorder(
            self.log_dir, total_episodes=self.TOTAL, selected=selected,
            run_facts={"env": "dark_room", "seed": 42},
            module_facts={"audio": {"trained": False}},
            config={"env": "dark_room", "device": "cpu"},
            max_steps=self.MAX_STEPS, min_free_gb=min_free_gb)

    def _episode(self, recorder, index, steps=3, modules=None):
        recorder.start_episode(index)
        _record_steps(recorder, steps)
        recorder.end_episode(modules or {})
        return Path(self.log_dir) / "episodes" / ("ep_%04d" % index)

    def _meta(self, folder):
        return json.loads((folder / "meta.json").read_text(encoding="utf-8"))

    def test_run_files_are_written_at_start(self):
        self._recorder()
        folder = Path(self.log_dir)
        session = json.loads((folder / "session.json").read_text(encoding="utf-8"))
        self.assertEqual(session["run"], {"env": "dark_room", "seed": 42})
        self.assertEqual(session["modules"], {"audio": {"trained": False}})
        self.assertEqual(session["recorded_episodes"], [0, 4])
        self.assertEqual(session["full_tensor_episodes"], [0, 4])
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config, {"env": "dark_room", "device": "cpu"})
        environment = json.loads((folder / "environment.json").read_text(encoding="utf-8"))
        self.assertEqual(environment["torch"], torch.__version__)
        self.assertIn("python", environment)
        self.assertTrue((folder / "git_diff.patch").exists())

    def test_every_episode_gets_steps_frames_and_vectors(self):
        folder = self._episode(self._recorder(), 1)
        steps = [json.loads(line) for line in
                 (folder / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(np.load(folder / "frames.npz")["frames"].shape, (3,) + FRAME_SHAPE)
        self.assertEqual(steps[1]["input_source"], SOURCES)
        self.assertEqual(steps[1]["raw_bids"], BIDS)
        self.assertEqual(steps[1]["winner"], "vision")
        self.assertEqual(steps[2]["internals"], {"x": 2})
        self.assertEqual(np.load(folder / "vectors.npz")["v"][2].tolist(), [2.0] * 4)
        meta = self._meta(folder)
        self.assertEqual(meta["tiers"], ["light"])
        self.assertFalse((folder / "maps.npz").exists())
        self.assertFalse((folder / "weights.pt").exists())

    def test_recorded_episode_gets_maps_full_tensors_and_weights(self):
        net = torch.nn.Linear(2, 2)
        folder = self._episode(self._recorder(), 0, modules={"net": net})
        self.assertEqual(np.load(folder / "maps.npz")["m"].shape, (3, 2, 2))
        full = np.load(folder / "full_f.npy")
        self.assertEqual(full.dtype, np.float16)
        self.assertEqual(full.shape, (self.MAX_STEPS, 3))
        self.assertEqual(full[2].tolist(), [2.0, 2.0, 2.0])
        weights = torch.load(folder / "weights.pt")
        self.assertTrue(torch.equal(weights["net"]["weight"], net.weight.detach().cpu()))
        meta = self._meta(folder)
        self.assertEqual(meta["tiers"], ["light", "maps", "full", "weights"])
        self.assertEqual(meta["full_rows"], 3)

    def test_last_episode_gets_full_tensors_even_when_not_selected(self):
        folder = self._episode(self._recorder(selected=frozenset({0})), 4)
        self.assertTrue((folder / "full_f.npy").exists())
        self.assertFalse((folder / "maps.npz").exists())

    def test_low_disk_skips_heavy_tiers_and_keeps_light(self):
        folder = self._episode(self._recorder(min_free_gb=1e12), 0,
                               modules={"net": torch.nn.Linear(2, 2)})
        meta = self._meta(folder)
        self.assertEqual(meta["tiers"], ["light"])
        self.assertEqual(meta["skipped_tiers"], ["maps", "full", "weights"])
        self.assertTrue((folder / "steps.jsonl").exists())
        self.assertFalse((folder / "full_f.npy").exists())

    def test_wants_reports_the_active_tiers(self):
        recorder = self._recorder(selected=frozenset({0}))
        self.assertFalse(recorder.wants("light"))
        recorder.start_episode(2)
        self.assertEqual([recorder.wants(t) for t in ("light", "maps", "full")],
                         [True, False, False])
        recorder.start_episode(0)
        self.assertEqual([recorder.wants(t) for t in ("light", "maps", "full")],
                         [True, True, True])

    def test_attached_modules_are_saved_without_passing_them(self):
        recorder = self._recorder()
        net = torch.nn.Linear(2, 2)
        recorder.attach_modules({"net": net})
        recorder.start_episode(0)
        _record_steps(recorder, 2)
        recorder.end_episode()
        weights = torch.load(Path(self.log_dir) / "episodes" / "ep_0000" / "weights.pt")
        self.assertEqual(sorted(weights), ["net"])

    def test_a_value_missing_on_some_steps_keeps_its_step_numbers(self):
        # Measured 2026-09-15: audio_waveform has no row on step 0, so a plain stack
        # of 29 rows for 30 steps would shift every sound one step early.
        recorder = self._recorder()
        recorder.start_episode(1)
        for step in range(3):
            recorder.record_inputs(_frame(step), SOURCES, BIDS, INTERO)
            recorder.record_outcome(winner="vision", ignited=True, sync_r=0.25,
                                    bound_bids=BIDS, reward=0.0, action=[0.0], env_phase="")
            vectors = {} if step == 0 else {"wave": np.full(2, step)}
            recorder.record_extra(scalars={}, vectors=vectors, maps={}, full={})
        recorder.end_episode({})
        saved = np.load(Path(self.log_dir) / "episodes" / "ep_0001" / "vectors.npz")
        self.assertEqual(saved["wave__steps"].tolist(), [1, 2])
        self.assertEqual(saved["wave"][:, 0].tolist(), [1.0, 2.0])

    def test_each_step_records_its_duration(self):
        folder = self._episode(self._recorder(), 1)
        first = json.loads((folder / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertGreaterEqual(first["seconds_inputs_to_end"], 0.0)

    def test_inputs_without_outcome_mark_the_episode_incomplete(self):
        recorder = self._recorder()
        recorder.start_episode(0)
        _record_steps(recorder, 2)
        recorder.record_inputs(_frame(9), SOURCES, BIDS, INTERO)
        recorder.end_episode({})
        meta = self._meta(Path(self.log_dir) / "episodes" / "ep_0000")
        self.assertEqual(meta["steps"], 2)
        self.assertEqual(meta["frames"], 3)
        self.assertFalse(meta["complete"])

    def test_recording_uses_no_random_state(self):
        np.random.seed(123)
        torch.manual_seed(123)
        numpy_state = np.random.get_state()[1].copy()
        torch_state = torch.get_rng_state().clone()
        self._episode(self._recorder(), 0, steps=4)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_state))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))


if __name__ == "__main__":
    unittest.main()

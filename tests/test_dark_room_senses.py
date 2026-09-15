"""The dark room's senses must carry information about the task.

Measured 2026-09-15 on three seeds (1800 steps): the agent heard only noise. The
proximity tone needed a distance below 10 in a room 224 wide, the collision sound
never played because the environment never reported a collision, and the sound was
mono, so no direction reached the agent. These tests pin the repaired senses and
pin that the default environment is unchanged.
"""

import pathlib
import unittest

import numpy as np
import torch

from models.audio.spatial_audio import SpatialAudioComputer
from scripts.training.train_rlhf import audio_waveform_tensor
from simulations.environments.simple_visual_env import WALL_COLOUR, SimpleVisualEnv

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000


def _env(**options):
    env = SimpleVisualEnv(width=224, height=224, **options)
    env.seed_audio(0)
    np.random.seed(0)
    env.reset()
    return env


def _place(env, agent, light):
    env.agent_pos = np.array(agent, dtype=np.float64)
    env.light_pos = np.array(light, dtype=np.float64)


def _still_step(env):
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    return info


class TestDefaultEnvironmentUnchanged(unittest.TestCase):

    def test_default_info_keys_and_mono_waveform(self):
        info = _still_step(_env())
        self.assertEqual(sorted(info), ["audio_waveform", "battery", "distance_to_light", "in_light"])
        self.assertEqual(info["audio_waveform"].ndim, 1)


class TestCollision(unittest.TestCase):

    def test_moving_into_a_wall_reports_collision(self):
        env = _env(report_collision=True)
        _place(env, agent=(2.0, 100.0), light=(200.0, 200.0))
        _, _, _, _, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
        self.assertTrue(info["collision"])

    def test_moving_inside_the_room_reports_no_collision(self):
        env = _env(report_collision=True)
        _place(env, agent=(100.0, 100.0), light=(200.0, 200.0))
        _, _, _, _, info = env.step(np.array([0.5, 0.5], dtype=np.float32))
        self.assertFalse(info["collision"])


class TestBinauralSound(unittest.TestCase):

    def _channels(self, agent, light, channels):
        env = _env(audio="binaural", audio_channels=channels)
        _place(env, agent=agent, light=light)
        return _still_step(env)["audio_waveform"]

    def test_stereo_shape(self):
        self.assertEqual(self._channels((100, 100), (180, 100), 2).shape, (2, 1056))

    def test_four_channel_shape(self):
        self.assertEqual(self._channels((100, 100), (180, 100), 4).shape, (4, 1056))

    def test_source_on_the_right_is_louder_in_the_right_ear(self):
        left, right = self._channels((60, 100), (200, 100), 2)
        self.assertGreater(np.sum(right ** 2), np.sum(left ** 2))

    def test_source_below_is_louder_in_the_lower_ear(self):
        _, _, upper, lower = self._channels((100, 40), (100, 200), 4)
        self.assertGreater(np.sum(lower ** 2), np.sum(upper ** 2))

    def test_sound_level_falls_with_distance_across_the_room(self):
        near = self._channels((100, 100), (120, 100), 2)
        middle = self._channels((20, 100), (120, 100), 2)
        far = self._channels((20, 20), (210, 210), 2)
        levels = [float(np.sqrt(np.mean(w ** 2))) for w in (near, middle, far)]
        self.assertGreater(levels[0], levels[1])
        self.assertGreater(levels[1], levels[2])

    def test_farthest_tone_is_above_the_noise_floor(self):
        far = self._channels((0, 0), (224, 224), 2)
        env = _env()
        noise = env._generate_audio({"distance_to_light": 1e6, "in_light": False})
        self.assertGreater(np.sqrt(np.mean(far ** 2)), 1.5 * np.sqrt(np.mean(noise ** 2)))

    def test_truth_offset_is_reported_for_analysis(self):
        env = _env(audio="binaural", audio_channels=2)
        _place(env, agent=(50.0, 60.0), light=(150.0, 20.0))
        info = _still_step(env)
        self.assertEqual(info["_truth_light_offset"], [100.0, -40.0])


class TestDirectionEstimate(unittest.TestCase):
    """SpatialAudioComputer: positive azimuth is right, positive elevation is down."""

    def _estimate(self, waveform):
        return SpatialAudioComputer(sample_rate=SAMPLE_RATE)(
            torch.tensor(waveform[None], dtype=torch.float32))[0].tolist()

    def _tone(self):
        t = np.arange(1056) / SAMPLE_RATE
        return 0.3 * np.sin(2 * np.pi * 500 * t)

    def test_time_and_level_differences_agree_for_a_right_source(self):
        tone = self._tone()
        itd_only = self._estimate(np.stack([np.roll(tone, 6), tone]))[0]
        ild_only = self._estimate(np.stack([0.5 * tone, tone]))[0]
        self.assertGreater(itd_only, 0.0)
        self.assertGreater(ild_only, 0.0)

    def test_left_source_is_negative(self):
        tone = self._tone()
        self.assertLess(self._estimate(np.stack([tone, 0.5 * np.roll(tone, 6)]))[0], 0.0)

    def test_environment_sound_gives_the_right_signs(self):
        env = _env(audio="binaural", audio_channels=4)
        _place(env, agent=(40, 40), light=(200, 200))
        azimuth, elevation = self._estimate(_still_step(env)["audio_waveform"])
        self.assertGreater(azimuth, 0.0)
        self.assertGreater(elevation, 0.0)


LIGHT_COLOUR = (255, 255, 200)
AGENT_COLOUR = (0, 100, 255)


def _has_colour(frame, colour):
    return bool(np.any(np.all(frame == np.array(colour, dtype=np.uint8), axis=-1)))


class TestAgentCenteredView(unittest.TestCase):
    """Feinberg & Mallatt, Merker, Stein & Meredith: tectal maps are egocentric."""

    def _frame(self, agent, light, **options):
        env = _env(view="agent_centered", **options)
        _place(env, agent=agent, light=light)
        observation, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        return observation, info

    def test_frame_shape_is_unchanged(self):
        observation, _ = self._frame((100, 100), (110, 100))
        self.assertEqual(observation.shape, (224, 224, 3))

    def test_agent_is_drawn_at_the_centre(self):
        observation, _ = self._frame((100, 100), (200, 200))
        self.assertEqual(tuple(observation[112, 112]), AGENT_COLOUR)

    def test_far_light_is_out_of_view(self):
        observation, info = self._frame((20, 20), (200, 200), view_radius=48)
        self.assertFalse(_has_colour(observation, LIGHT_COLOUR))
        self.assertFalse(info["_truth_light_in_view"])

    def test_near_light_is_in_view(self):
        observation, info = self._frame((100, 100), (150, 100), view_radius=48)
        self.assertTrue(_has_colour(observation, LIGHT_COLOUR))
        self.assertTrue(info["_truth_light_in_view"])

    def test_room_edge_is_visible_near_a_wall(self):
        observation, _ = self._frame((5, 100), (200, 200), view_radius=48)
        self.assertTrue(_has_colour(observation[:, :40], WALL_COLOUR))

    def test_full_view_shows_a_far_light(self):
        env = _env()
        _place(env, agent=(20, 20), light=(200, 200))
        observation, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
        self.assertTrue(_has_colour(observation, LIGHT_COLOUR))

    def test_unknown_view_is_rejected(self):
        with self.assertRaises(ValueError):
            SimpleVisualEnv(width=224, height=224, view="room_corner")


class TestTrainingLoopWaveform(unittest.TestCase):

    def test_mono_becomes_one_channel(self):
        self.assertEqual(tuple(audio_waveform_tensor(np.zeros(1056, np.float32), "cpu").shape),
                         (1, 1, 1056))

    def test_multichannel_keeps_its_channels(self):
        self.assertEqual(tuple(audio_waveform_tensor(np.zeros((4, 1056), np.float32), "cpu").shape),
                         (1, 4, 1056))


class TestTruthIsNeverReadByTheAgent(unittest.TestCase):

    def test_truth_key_appears_only_in_the_environment_and_tests(self):
        # Agent code only. Read-only analysis probes in scripts/analysis may read the
        # truth to score the senses; they never feed the agent.
        readers = [path for folder in ("models", "scripts/training")
                   for path in (REPO_ROOT / folder).rglob("*.py")
                   if "_truth_" in path.read_text(encoding="utf-8", errors="ignore")]
        self.assertEqual(readers, [])


if __name__ == "__main__":
    unittest.main()

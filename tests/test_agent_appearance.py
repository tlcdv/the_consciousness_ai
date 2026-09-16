"""How the agent is drawn into the frame it receives.

The agent's own body is part of the picture the agent sees, so its shape and its
colour are run configuration, not decoration: a run drawn one way and a run drawn
another way are not comparable. The default must stay exactly what every earlier
run used (a filled disc in RGB 0, 100, 255), and the alternative must be
explicit.
"""
import unittest

import numpy as np

from simulations.environments.simple_visual_env import (
    AGENT_COLOUR, LIGHT_COLOUR, WALL_COLOUR, SimpleVisualEnv,
)

ROOM = 224
CENTRE = ROOM // 2


def _env(**kwargs):
    env = SimpleVisualEnv(width=ROOM, height=ROOM, view="agent_centered", **kwargs)
    env.reset()
    env.agent_pos = np.array([CENTRE, CENTRE], dtype=np.float64)
    env.light_pos = np.array([10.0, 10.0], dtype=np.float64)
    return env


class TestDefaultIsUnchanged(unittest.TestCase):
    def test_default_frame_is_identical_to_the_plain_disc(self):
        frame = _env()._get_obs()
        plain = _env(agent_mark="disc", agent_colour=AGENT_COLOUR)._get_obs()
        self.assertTrue(np.array_equal(frame, plain))

    def test_the_centre_of_a_disc_carries_the_agent_colour(self):
        frame = _env()._get_obs()
        self.assertEqual(tuple(frame[CENTRE, CENTRE]), AGENT_COLOUR)

    def test_the_full_room_view_uses_the_same_colour(self):
        env = SimpleVisualEnv(width=ROOM, height=ROOM)
        env.reset()
        env.agent_pos = np.array([CENTRE, CENTRE], dtype=np.float64)
        env.light_pos = np.array([10.0, 10.0], dtype=np.float64)
        self.assertEqual(tuple(env._get_obs()[CENTRE, CENTRE]), AGENT_COLOUR)


class TestRingMark(unittest.TestCase):
    def test_the_ring_leaves_the_floor_visible_between_its_rings(self):
        frame = _env(agent_mark="ring")._get_obs()
        colours = {tuple(frame[CENTRE, x]) for x in range(CENTRE - 40, CENTRE + 40)}
        self.assertIn(AGENT_COLOUR, colours)
        self.assertIn((0, 0, 0), colours)

    def test_the_ring_keeps_a_filled_dot_at_the_centre(self):
        frame = _env(agent_mark="ring")._get_obs()
        self.assertEqual(tuple(frame[CENTRE, CENTRE]), AGENT_COLOUR)

    def test_the_mark_stays_inside_the_agent_radius(self):
        env = _env(agent_mark="ring")
        frame = env._get_obs()
        scale = ROOM / (2 * env.view_radius)
        edge = int(env.agent_radius * scale) + 3
        beyond = frame[CENTRE, CENTRE + edge:]
        self.assertNotIn(AGENT_COLOUR, {tuple(pixel) for pixel in beyond})

    def test_an_unknown_mark_is_refused(self):
        with self.assertRaises(ValueError):
            SimpleVisualEnv(width=ROOM, height=ROOM, agent_mark="triangle")


class TestAgentColour(unittest.TestCase):
    def test_the_configured_colour_reaches_the_agent_centered_view(self):
        frame = _env(agent_colour=(217, 119, 87))._get_obs()
        self.assertEqual(tuple(frame[CENTRE, CENTRE]), (217, 119, 87))

    def test_the_configured_colour_reaches_the_full_room_view(self):
        env = SimpleVisualEnv(width=ROOM, height=ROOM, agent_colour=(217, 119, 87))
        env.reset()
        env.agent_pos = np.array([CENTRE, CENTRE], dtype=np.float64)
        self.assertEqual(tuple(env._get_obs()[CENTRE, CENTRE]), (217, 119, 87))

    def test_the_light_and_the_wall_keep_their_own_colours(self):
        env = _env(agent_colour=(217, 119, 87))
        env.light_pos = np.array([CENTRE + 60.0, CENTRE], dtype=np.float64)
        frame = env._get_obs()
        self.assertIn(LIGHT_COLOUR, {tuple(pixel) for pixel in frame[CENTRE]})
        self.assertEqual(WALL_COLOUR, (60, 60, 60))

    def test_a_colour_outside_the_byte_range_is_refused(self):
        with self.assertRaises(ValueError):
            SimpleVisualEnv(width=ROOM, height=ROOM, agent_colour=(0, 300, 0))


if __name__ == "__main__":
    unittest.main()

"""Tests for NavigationEnv."""
from __future__ import annotations

import unittest
import numpy as np

from simulations.environments.navigation_env import NavigationEnv


class TestNavigationEnvBasics(unittest.TestCase):

    def setUp(self):
        self.env = NavigationEnv(render_mode="rgb_array", width=224, height=224)

    def test_reset_returns_correct_shapes(self):
        obs, info = self.env.reset(seed=42)
        self.assertEqual(obs.shape, (224, 224, 3))
        self.assertEqual(obs.dtype, np.uint8)
        self.assertIn("current_room", info)
        self.assertIn("battery", info)

    def test_step_returns_correct_shapes(self):
        self.env.reset(seed=42)
        action = np.array([0.5, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertEqual(obs.shape, (224, 224, 3))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(info, dict)

    def test_initial_room_is_0_0(self):
        _, info = self.env.reset(seed=42)
        self.assertEqual(info["current_room"], (0, 0))

    def test_battery_drains(self):
        self.env.reset(seed=42)
        initial_battery = self.env.battery
        for _ in range(10):
            self.env.step(np.array([0.0, 0.0], dtype=np.float32))
        self.assertLess(self.env.battery, initial_battery)

    def test_battery_terminates_episode(self):
        self.env.reset(seed=42)
        self.env.battery = 0.001  # Nearly depleted
        _, _, terminated, _, info = self.env.step(np.array([0.0, 0.0]))
        self.assertTrue(terminated)

    def test_goals_spawned(self):
        self.env.reset(seed=42)
        self.assertEqual(len(self.env.goals), 3)
        for goal in self.env.goals:
            self.assertIn("pos", goal)
            self.assertIn("color", goal)
            self.assertIn(goal["color"], ["green", "blue", "red"])

    def test_rooms_visited_tracking(self):
        _, info = self.env.reset(seed=42)
        self.assertEqual(info["rooms_visited"], 1)


class TestNavigationEnvDynamics(unittest.TestCase):

    def setUp(self):
        self.env = NavigationEnv(render_mode="rgb_array", width=224, height=224)

    def test_agent_moves(self):
        self.env.reset(seed=42)
        pos_before = self.env.agent_pos.copy()
        self.env.step(np.array([1.0, 0.0], dtype=np.float32))
        self.assertGreater(self.env.agent_pos[0], pos_before[0])

    def test_agent_stays_in_bounds(self):
        self.env.reset(seed=42)
        for _ in range(200):
            self.env.step(np.array([1.0, 1.0], dtype=np.float32))
        self.assertLessEqual(self.env.agent_pos[0], 224)
        self.assertLessEqual(self.env.agent_pos[1], 224)
        self.assertGreaterEqual(self.env.agent_pos[0], 0)
        self.assertGreaterEqual(self.env.agent_pos[1], 0)

    def test_goal_collection_gives_reward(self):
        self.env.reset(seed=42)
        # Place a goal right on the agent
        self.env.goals[0]["pos"] = self.env.agent_pos.copy()
        self.env.goals[0]["room"] = self.env._get_room(self.env.agent_pos)
        _, reward, _, _, info = self.env.step(np.array([0.0, 0.0]))
        self.assertGreater(reward, -0.01)  # Got goal reward
        self.assertEqual(info["goals_collected"], 1)

    def test_goal_respawns_after_collection(self):
        self.env.reset(seed=42)
        # Place goal on agent
        self.env.goals[0]["pos"] = self.env.agent_pos.copy()
        self.env.goals[0]["room"] = self.env._get_room(self.env.agent_pos)
        self.env.step(np.array([0.0, 0.0]))
        # Goal should still exist (respawned)
        self.assertEqual(len(self.env.goals), 3)

    def test_fog_of_war_only_current_room_visible(self):
        """Only the current room should have non-zero pixels (roughly)."""
        obs, _ = self.env.reset(seed=42)
        room = self.env._get_room(self.env.agent_pos)
        bounds = self.env._room_bounds(room)
        x_min, y_min, x_max, y_max = [int(v) for v in bounds]

        # Current room should have some bright pixels
        room_brightness = obs[y_min + 5:y_max - 5, x_min + 5:x_max - 5].mean()

        # Opposite room (diagonal) should be mostly dark
        opp_room = (1 - room[0], 1 - room[1])
        opp_bounds = self.env._room_bounds(opp_room)
        ox_min, oy_min, ox_max, oy_max = [int(v) for v in opp_bounds]
        opp_brightness = obs[oy_min + 5:oy_max - 5, ox_min + 5:ox_max - 5].mean()

        self.assertGreater(room_brightness, opp_brightness)


if __name__ == "__main__":
    unittest.main()

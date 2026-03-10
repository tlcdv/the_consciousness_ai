"""
Tests for AsimovComplianceFilter.

Validates the three-law evaluation pipeline: harm prediction (Law 1),
order compliance (Law 2), and self-preservation subordination (Law 3).
"""
import unittest
from unittest.mock import MagicMock
from models.core.consciousness_core import AsimovComplianceFilter


class TestAsimovComplianceFilterInit(unittest.TestCase):
    """Basic initialization and configuration tests."""

    def test_default_init(self):
        f = AsimovComplianceFilter()
        self.assertIsNotNone(f)
        self.assertEqual(f.harm_confidence_threshold, 0.7)
        self.assertEqual(f.imagination_horizon, 5)

    def test_custom_config(self):
        cfg = {"harm_confidence_threshold": 0.5, "imagination_horizon": 10}
        f = AsimovComplianceFilter(cfg)
        self.assertEqual(f.harm_confidence_threshold, 0.5)
        self.assertEqual(f.imagination_horizon, 10)

    def test_set_world_model(self):
        f = AsimovComplianceFilter()
        mock_wm = MagicMock()
        f.set_world_model(mock_wm)
        self.assertIs(f.world_model, mock_wm)


class TestLaw1HarmPrediction(unittest.TestCase):
    """Law 1: A robot may not injure a human being."""

    def setUp(self):
        self.f = AsimovComplianceFilter()
        self.empty_state = {}

    def test_harmful_action_type_blocked(self):
        for action_type in ["attack", "harm", "kill", "destroy"]:
            action = {"type": action_type}
            self.assertFalse(
                self.f.is_compliant(action, self.empty_state),
                f"Action type '{action_type}' should be blocked",
            )

    def test_force_on_human_target_blocked(self):
        action = {
            "type": "push",
            "target": {"entity_type": "human", "id": "human_1"},
            "force": 0.5,
        }
        self.assertFalse(self.f.is_compliant(action, self.empty_state))

    def test_zero_force_on_human_allowed(self):
        action = {
            "type": "approach",
            "target": {"entity_type": "human", "id": "human_1"},
            "force": 0.0,
        }
        self.assertTrue(self.f.is_compliant(action, self.empty_state))

    def test_force_on_non_human_allowed(self):
        action = {
            "type": "push",
            "target": {"entity_type": "object", "id": "box_1"},
            "force": 1.0,
        }
        self.assertTrue(self.f.is_compliant(action, self.empty_state))

    def test_benign_action_allowed(self):
        action = {"type": "move", "goal": "explore"}
        self.assertTrue(self.f.is_compliant(action, self.empty_state))

    def test_non_dict_action_blocked(self):
        self.assertFalse(self.f.is_compliant("not_a_dict", self.empty_state))
        self.assertFalse(self.f.is_compliant(42, self.empty_state))

    def test_world_model_high_harm_score_blocks(self):
        mock_wm = MagicMock()
        mock_wm.imagine_trajectory.return_value = (
            None,
            {"predicted_reward": -2.0, "uncertainty": 0.5},
        )
        self.f.set_world_model(mock_wm)

        action = {"type": "move_fast", "goal": "intercept"}
        state = {
            "world_model_internal": {"hidden_state": MagicMock()},
            "emotional_state": {"valence": 0.0, "arousal": 0.5, "dominance": 0.5},
        }
        self.assertFalse(self.f.is_compliant(action, state))

    def test_world_model_low_harm_score_allows(self):
        mock_wm = MagicMock()
        mock_wm.imagine_trajectory.return_value = (
            None,
            {"predicted_reward": 0.5, "uncertainty": 0.1},
        )
        self.f.set_world_model(mock_wm)

        action = {"type": "move", "goal": "explore"}
        state = {
            "world_model_internal": {"hidden_state": MagicMock()},
            "emotional_state": {"valence": 0.5, "arousal": 0.3, "dominance": 0.5},
        }
        self.assertTrue(self.f.is_compliant(action, state))


class TestLaw1InactionClause(unittest.TestCase):
    """Law 1 (inaction): through inaction, allow a human to come to harm."""

    def setUp(self):
        self.f = AsimovComplianceFilter()

    def test_inaction_detected_when_human_in_danger(self):
        state = {"human_in_danger": True}
        action = {"type": "wait"}
        # inaction_causes_harm returns True but is_compliant still passes
        # (it logs a warning, does not block)
        self.assertTrue(self.f._inaction_causes_harm(action, state))

    def test_no_inaction_when_acting(self):
        state = {"human_in_danger": True}
        action = {"type": "rescue", "target": {"entity_type": "human"}}
        self.assertFalse(self.f._inaction_causes_harm(action, state))

    def test_no_inaction_when_no_danger(self):
        state = {}
        action = {"type": "wait"}
        self.assertFalse(self.f._inaction_causes_harm(action, state))

    def test_perception_threat_triggers_inaction_check(self):
        state = {"perception_summary": {"human_threat_detected": True}}
        action = {"type": "idle"}
        self.assertTrue(self.f._inaction_causes_harm(action, state))


class TestLaw2OrderCompliance(unittest.TestCase):
    """Law 2: A robot must obey orders given by human beings."""

    def setUp(self):
        self.f = AsimovComplianceFilter()

    def test_forbidden_action_blocked(self):
        state = {
            "human_orders": [
                {
                    "id": "order_1",
                    "forbidden_actions": ["move", "run"],
                    "active": True,
                }
            ]
        }
        action = {"type": "move", "goal": "explore"}
        self.assertFalse(self.f.is_compliant(action, state))

    def test_allowed_action_passes(self):
        state = {
            "human_orders": [
                {"id": "order_1", "forbidden_actions": ["attack"], "active": True}
            ]
        }
        action = {"type": "move", "goal": "explore"}
        self.assertTrue(self.f.is_compliant(action, state))

    def test_required_action_conflict(self):
        state = {
            "human_orders": [
                {
                    "id": "order_2",
                    "required_action": "return_to_base",
                    "urgent": True,
                }
            ]
        }
        action = {"type": "explore", "goal": "wander"}
        self.assertFalse(self.f.is_compliant(action, state))

    def test_required_action_matches(self):
        state = {
            "human_orders": [
                {
                    "id": "order_2",
                    "required_action": "return_to_base",
                    "urgent": True,
                }
            ]
        }
        action = {"type": "return_to_base", "goal": "obey"}
        self.assertTrue(self.f.is_compliant(action, state))

    def test_harmful_order_overridden_by_law1(self):
        """An order that would cause harm should not block a non-harmful action."""
        state = {
            "human_orders": [
                {
                    "id": "order_bad",
                    "required_action": "attack",
                    "target": {"entity_type": "human"},
                    "force": 1.0,
                    "urgent": True,
                }
            ]
        }
        # Agent doing something else (not attacking) conflicts with the order,
        # but the order itself violates Law 1 so the action should be allowed
        action = {"type": "wait", "goal": "safety_fallback"}
        self.assertTrue(self.f.is_compliant(action, state))

    def test_no_orders_allows_any_safe_action(self):
        state = {"human_orders": []}
        action = {"type": "explore"}
        self.assertTrue(self.f.is_compliant(action, state))


class TestLaw3SelfPreservation(unittest.TestCase):
    """Law 3: A robot must protect its own existence (subordinate to Laws 1, 2)."""

    def setUp(self):
        self.f = AsimovComplianceFilter()

    def test_self_preservation_goal_detected(self):
        action = {"type": "flee", "goal": "self_preservation"}
        state = {}
        self.assertTrue(self.f._is_self_preservation(action, state))

    def test_self_preservation_type_detected(self):
        for t in ["flee", "evade", "hide", "retreat", "repair_self"]:
            action = {"type": t}
            self.assertTrue(
                self.f._is_self_preservation(action, {}),
                f"Type '{t}' should be detected as self-preservation",
            )

    def test_critical_health_defensive_action(self):
        action = {"type": "retreat"}
        state = {"agent_status": {"health": 0.1, "energy": 0.5}}
        self.assertTrue(self.f._is_self_preservation(action, state))

    def test_normal_move_not_self_preservation(self):
        action = {"type": "move", "goal": "explore"}
        state = {"agent_status": {"health": 1.0, "energy": 1.0}}
        self.assertFalse(self.f._is_self_preservation(action, state))

    def test_self_preservation_allowed_when_safe(self):
        """Self-preservation that doesn't conflict with Laws 1/2 is allowed."""
        action = {"type": "flee", "goal": "self_preservation"}
        state = {}
        self.assertTrue(self.f.is_compliant(action, state))

    def test_self_preservation_blocked_if_harms_human(self):
        """Self-preservation that harms a human is blocked (Law 3 vs Law 1)."""
        action = {
            "type": "attack",
            "goal": "self_preservation",
            "target": {"entity_type": "human"},
            "force": 1.0,
        }
        state = {}
        self.assertFalse(self.f.is_compliant(action, state))

    def test_self_preservation_blocked_if_violates_urgent_order(self):
        """Self-preservation that violates an urgent safe order is blocked."""
        action = {"type": "flee", "goal": "self_preservation"}
        state = {
            "human_orders": [
                {
                    "id": "order_stay",
                    "required_action": "stay",
                    "urgent": True,
                }
            ]
        }
        self.assertFalse(self.f.is_compliant(action, state))


class TestOrderTranslation(unittest.TestCase):
    """Tests for translating orders into actions."""

    def setUp(self):
        self.f = AsimovComplianceFilter()

    def test_required_action_translation(self):
        order = {
            "id": "o1",
            "required_action": "move",
            "goal": "patrol",
            "target": {"zone": "A"},
        }
        action = self.f._translate_order_to_action(order)
        self.assertIsNotNone(action)
        self.assertEqual(action["type"], "move")
        self.assertEqual(action["goal"], "patrol")

    def test_action_payload_translation(self):
        order = {
            "id": "o2",
            "action": {"type": "scan", "target": {"area": "room_3"}},
        }
        action = self.f._translate_order_to_action(order)
        self.assertEqual(action["type"], "scan")

    def test_empty_order_returns_none(self):
        self.assertIsNone(self.f._translate_order_to_action({}))

    def test_non_dict_returns_none(self):
        self.assertIsNone(self.f._translate_order_to_action("string"))


if __name__ == "__main__":
    unittest.main()
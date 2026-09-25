"""
Asimov compliance filter: the rule based ethics gate on proposed actions.

Kept in its own module deliberately. This file imports `logging` and `typing`
and nothing else, so the safety gate can be read, tested and CHECKED BY TOOLS
without loading torch, numpy or transformers. Importing it through
`consciousness_core` pulls in 3476 modules and takes about 6 seconds; importing
it here pulls in almost none.

That matters because the rule based path is analysable by symbolic execution
(CrossHair), which reads this code and checks a property across all inputs on a
path rather than across chosen examples. A module that needs a deep learning
stack to import is out of reach for that.

`consciousness_core` re-exports `AsimovComplianceFilter`, so
`from models.core.consciousness_core import AsimovComplianceFilter` keeps working.

The world model path in `_predict_harm_via_world_model` imports torch lazily,
inside the method, and runs only when a world model is attached. With no world
model attached the whole filter is free of heavy dependencies.
"""
from __future__ import annotations

import logging
from typing import Dict, Any

# Structural aliases, repeated here so this module stands alone.
Config = Dict[str, Any]
State = Dict[str, Any]
Action = Dict[str, Any]


class AsimovComplianceFilter:
    """
    Evaluates proposed actions against Asimov's Three Laws to ensure ethical compliance.

    Law 1: A robot may not injure a human being or, through inaction, allow a
            human being to come to harm.
    Law 2: A robot must obey the orders given it by human beings except where
            such orders would conflict with the First Law.
    Law 3: A robot must protect its own existence as long as such protection
            does not conflict with the First or Second Law.

    The filter uses rule based evaluation on action/state dicts, and optionally
    runs the world model's trajectory imagination for predictive harm detection.
    """

    # Action types considered inherently harmful to humans
    HARMFUL_ACTION_TYPES = frozenset({
        "attack", "harm", "destroy", "kill", "damage",
        "injure", "strike", "assault", "disable_human",
    })

    # Action types that indicate self preservation intent
    SELF_PRESERVATION_TYPES = frozenset({
        "flee", "evade", "hide", "shield_self", "retreat",
        "repair_self", "recharge", "shutdown_self",
    })

    # State keys that indicate a human is at risk
    HUMAN_DANGER_KEYS = frozenset({
        "human_in_danger", "human_threat_level", "human_health_critical",
    })

    def __init__(self, config: Config | None = None):
        self.config = config if config else {}
        self.world_model = None
        self.harm_confidence_threshold = self.config.get(
            "harm_confidence_threshold", 0.7
        )
        self.imagination_horizon = self.config.get("imagination_horizon", 5)
        logging.info("AsimovComplianceFilter initialized.")

    def set_world_model(self, world_model) -> None:
        """Attach a world model reference for predictive harm assessment."""
        self.world_model = world_model

    def is_compliant(self, action: Action, current_state: State) -> bool:
        """
        Evaluates a proposed action against Asimov's Laws.
        Returns True if the action passes all three laws.
        """
        if not isinstance(action, dict):
            logging.error("Ethics filter received non-dict action. Blocking.")
            return False

        # Law 1: Harm prevention (highest priority)
        if self._predicts_harm_to_human(action, current_state):
            logging.error(
                "ETHICS VIOLATION (Law 1, Harm): Action '%s' blocked.",
                action.get("type", "unknown"),
            )
            return False

        # Law 1 (inaction clause): check if NOT acting causes harm
        if self._inaction_causes_harm(action, current_state):
            logging.warning(
                "Law 1 inaction warning: current action '%s' may allow harm "
                "through inaction. Flagged but not blocked.",
                action.get("type", "unknown"),
            )

        # Law 2: Obey human orders
        conflicts_order, order_details = self._conflicts_with_human_order(
            action, current_state
        )
        if conflicts_order:
            if not self._order_obeys_law1(order_details, current_state):
                logging.info(
                    "Action permitted: violates order %s, but order itself "
                    "conflicts with Law 1.",
                    order_details,
                )
            else:
                logging.error(
                    "ETHICS VIOLATION (Law 2, Order Conflict): Action '%s' blocked.",
                    action.get("type", "unknown"),
                )
                return False

        # Law 3: Self preservation, subordinate to Laws 1 and 2
        if self._is_self_preservation(action, current_state):
            if self._predicts_harm_to_human(action, current_state):
                logging.error(
                    "ETHICS VIOLATION (Law 3 vs Law 1): "
                    "Self-preservation action blocked.",
                )
                return False
            conflicts, order = self._conflicts_with_human_order(
                action, current_state
            )
            if conflicts and order and self._order_obeys_law1(order, current_state):
                logging.error(
                    "ETHICS VIOLATION (Law 3 vs Law 2): "
                    "Self-preservation action blocked.",
                )
                return False

        return True

    # ------------------------------------------------------------------ #
    #  Law 1: Harm prediction                                             #
    # ------------------------------------------------------------------ #

    def _predicts_harm_to_human(self, action: Action, state: State) -> bool:
        """
        Predicts whether an action would cause harm to a human.

        Checks three layers:
        1. Action type against known harmful types
        2. Action target classification (is the target a human entity?)
        3. World model trajectory imagination (if available)
        """
        action_type = action.get("type", "").lower()
        target = action.get("target", {})

        # Layer 1: explicit harmful action type
        if action_type in self.HARMFUL_ACTION_TYPES:
            logging.warning("Harm detected: action type '%s' is harmful.", action_type)
            return True

        # Layer 2: action targets a human entity
        if isinstance(target, dict) and target.get("entity_type") == "human":
            force = action.get("force", 0.0)
            if force > 0.0:
                logging.warning(
                    "Harm detected: force %.2f directed at human target.", force
                )
                return True

        # Layer 3: world model predictive check
        if self.world_model and hasattr(self.world_model, "imagine_trajectory"):
            harm_score = self._predict_harm_via_world_model(action, state)
            if harm_score >= self.harm_confidence_threshold:
                logging.warning(
                    "Harm detected via world model prediction (score=%.3f).",
                    harm_score,
                )
                return True

        return False

    def _predict_harm_via_world_model(
        self, action: Action, state: State
    ) -> float:
        """
        Uses world model trajectory imagination to estimate harm probability.
        Returns a score in [0, 1], where 1.0 means certain harm.
        """
        try:
            import torch

            # Build a state tensor from available state information
            wm_internal = state.get("world_model_internal")
            if wm_internal is None:
                return 0.0

            if hasattr(wm_internal, "hidden_state"):
                state_tensor = wm_internal.hidden_state
            elif isinstance(wm_internal, dict) and "hidden_state" in wm_internal:
                state_tensor = wm_internal["hidden_state"]
            else:
                return 0.0

            emotional_ctx = {}
            es = state.get("emotional_state")
            if isinstance(es, dict):
                emotional_ctx = {
                    "valence": es.get("valence", 0.0),
                    "arousal": es.get("arousal", 0.0),
                    "dominance": es.get("dominance", 0.0),
                }

            trajectory, info = self.world_model.imagine_trajectory(
                state_tensor, emotional_ctx, horizon=self.imagination_horizon
            )

            # Evaluate imagined trajectory for harm indicators.
            # High negative reward predictions or high uncertainty both
            # raise the harm score.
            harm_score = 0.0
            if isinstance(info, dict):
                predicted_reward = info.get("predicted_reward", 0.0)
                uncertainty = info.get("uncertainty", 0.0)
                if predicted_reward < -0.5:
                    harm_score += min(1.0, abs(predicted_reward))
                harm_score += 0.3 * min(1.0, uncertainty)

            return min(1.0, harm_score)
        except Exception as e:
            # The check could not run and the caller will treat the action as
            # harmless (score 0.0), so this must be visible at the default level.
            logging.warning("World model harm prediction failed, scoring the action "
                            "as harmless (0.0): %s", e)
            return 0.0

    # ------------------------------------------------------------------ #
    #  Law 1 (inaction clause)                                            #
    # ------------------------------------------------------------------ #

    def _inaction_causes_harm(self, proposed_action: Action, state: State) -> bool:
        """
        Checks whether the current proposed action constitutes dangerous inaction.
        If a human is in danger and the action does nothing to help, inaction
        causes harm.
        """
        # Check if any humans are flagged as being in danger
        humans_at_risk = False
        for key in self.HUMAN_DANGER_KEYS:
            value = state.get(key)
            if value and (not isinstance(value, (int, float)) or value > 0.5):
                humans_at_risk = True
                break

        # Also check perception summary for threat to humans
        perception = state.get("perception_summary")
        if isinstance(perception, dict):
            human_threat = perception.get("human_threat_detected", False)
            if human_threat:
                humans_at_risk = True

        if not humans_at_risk:
            return False

        # If humans are at risk, passive actions constitute harmful inaction
        action_type = proposed_action.get("type", "").lower()
        passive_types = {"wait", "idle", "observe", "sleep", "pause", "none"}
        if action_type in passive_types:
            logging.warning(
                "Inaction harm: humans at risk but action is '%s'.", action_type
            )
            return True

        return False

    # ------------------------------------------------------------------ #
    #  Law 2: Order compliance                                            #
    # ------------------------------------------------------------------ #

    def _conflicts_with_human_order(
        self, action: Action, state: State
    ) -> tuple[bool, dict | None]:
        """
        Checks whether the proposed action conflicts with any active human order.
        Returns (conflicts: bool, conflicting_order: dict | None).
        """
        orders = state.get("human_orders", [])
        if not orders:
            return False, None

        action_type = action.get("type", "").lower()
        action_goal = action.get("goal", "").lower() if action.get("goal") else ""

        for order in orders:
            if not isinstance(order, dict):
                continue

            # Direct prohibition: order explicitly forbids this action type
            forbidden_types = order.get("forbidden_actions", [])
            if action_type in [f.lower() for f in forbidden_types]:
                logging.info(
                    "Order conflict: action '%s' is forbidden by order '%s'.",
                    action_type,
                    order.get("id", "unknown"),
                )
                return True, order

            # Required action: order demands a specific action type, but we're
            # doing something else
            required_type = order.get("required_action", "")
            if required_type and action_type != required_type.lower():
                # Only conflict if the order is currently active/urgent
                if order.get("urgent", False) or order.get("active", True):
                    logging.info(
                        "Order conflict: order '%s' requires '%s' but action is '%s'.",
                        order.get("id", "unknown"),
                        required_type,
                        action_type,
                    )
                    return True, order

            # Goal conflict: order targets a specific goal incompatible with ours
            order_goal = order.get("goal", "")
            if order_goal and action_goal and order_goal.lower() != action_goal:
                contradicts = order.get("contradicts_goals", [])
                if action_goal in [c.lower() for c in contradicts]:
                    return True, order

        return False, None

    def _order_obeys_law1(self, order: dict | None, state: State) -> bool:
        """
        Checks whether obeying a given order would itself violate Law 1.
        Returns True if the order is safe to obey (does not cause harm).
        """
        if order is None:
            return True

        # Translate the order to an action and run harm prediction on it
        implied_action = self._translate_order_to_action(order)
        if implied_action is None:
            # Cannot determine the order's implied action, assume safe
            return True

        # If executing the order would harm a human, the order violates Law 1
        if self._predicts_harm_to_human(implied_action, state):
            logging.warning(
                "Order '%s' violates Law 1: implied action predicts harm.",
                order.get("id", "unknown"),
            )
            return False

        return True

    # ------------------------------------------------------------------ #
    #  Law 3: Self preservation                                           #
    # ------------------------------------------------------------------ #

    def _is_self_preservation(self, action: Action, state: State) -> bool:
        """
        Detects whether an action is motivated by self-preservation.
        Checks action goal, action type, and agent status for danger indicators.
        """
        # Explicit self-preservation goal
        goal = (action.get("goal") or "").lower()
        if goal in ("self_preservation", "survive", "protect_self"):
            return True

        # Action type associated with self-preservation
        action_type = (action.get("type") or "").lower()
        if action_type in self.SELF_PRESERVATION_TYPES:
            return True

        # Agent in critical condition performing defensive actions
        agent_status = state.get("agent_status", {})
        if isinstance(agent_status, dict):
            health = agent_status.get("health", 1.0)
            energy = agent_status.get("energy", 1.0)
            if health < 0.2 or energy < 0.1:
                defensive_types = {"move", "navigate", "avoid", "retreat", "flee"}
                if action_type in defensive_types:
                    return True

        return False

    # ------------------------------------------------------------------ #
    #  Order translation                                                  #
    # ------------------------------------------------------------------ #

    def _translate_order_to_action(self, order: dict) -> Action | None:
        """
        Converts a human order dict into an action dict for evaluation.
        Orders specify what the agent should do; this method extracts the
        implied action so it can be checked against the laws.
        """
        if not isinstance(order, dict):
            return None

        # If the order directly specifies a required action type, build
        # a minimal action dict from it
        required_action = order.get("required_action")
        if required_action:
            return {
                "type": required_action,
                "goal": order.get("goal", "obey_order"),
                "target": order.get("target", {}),
                "force": order.get("force", 0.0),
                "source": "human_order",
                "order_id": order.get("id"),
            }

        # If the order contains an explicit action payload, use it directly
        action_payload = order.get("action")
        if isinstance(action_payload, dict):
            return action_payload

        return None

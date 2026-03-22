"""
Wisconsin Card Sort Test (WCST) Analog Environment.

Tests meta-cognition and cognitive flexibility. The sorting rule changes
without warning after N consecutive correct sorts.

Cards have three feature dimensions: shape, color, count.
Four reference cards are always visible. The agent sorts the current card
to the reference that matches on the active (hidden) rule dimension.

Observation: RGB image [height, width, 3] uint8.
Action: Discrete(4), sort to reference card 0-3.
"""
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from simulations.environments._stimulus_renderer import (
    COLORS, COLOR_NAMES, SHAPE_NAMES, BACKGROUND_GRAY,
    draw_card, draw_filled_rect,
    FEEDBACK_GREEN, FEEDBACK_RED,
)

# Feature dimensions and values for WCST
_WCST_SHAPES = SHAPE_NAMES[:4]   # triangle, square, pentagon, hexagon
_WCST_COLORS = COLOR_NAMES[:4]   # red, blue, green, yellow
_WCST_COUNTS = [1, 2, 3, 4]
_RULE_DIMS = ["shape", "color", "count"]


class WCSTEnv(gym.Env):
    """
    Wisconsin Card Sort analog.

    Consciousness components exercised:
      - Meta-cognition (self-model): detect own performance drop at rule change
      - Inhibition (basal ganglia No-Go): suppress the old learned rule
      - Hypothesis testing (reentrant workspace): explore dimensions, update
        based on prediction error
      - Affective modulation: consecutive errors shift arousal and threshold
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        width: int = 224,
        height: int = 224,
        num_trials: int = 60,
        correct_to_switch: int = 6,
        max_rule_changes: int = 6,
        feedback_duration: int = 5,
    ):
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.num_trials = num_trials
        self.correct_to_switch = correct_to_switch
        self.max_rule_changes = max_rule_changes
        self.feedback_duration = feedback_duration

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )

        self._rng = np.random.default_rng()
        self._init_state()

    # ── Gym interface ────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._init_state()
        self._generate_reference_cards()
        self._deal_card()
        return self._render_frame(), self._info()

    def step(self, action: int):
        reward = 0.0
        terminated = False

        if self._feedback_remaining > 0:
            # Still showing feedback, ignore actions
            self._feedback_remaining -= 1
            if self._feedback_remaining == 0:
                self._trial += 1
                if self._trial >= self.num_trials:
                    terminated = True
                else:
                    self._deal_card()
            return self._render_frame(), 0.0, terminated, False, self._info()

        # Evaluate the sort
        correct_ref = self._correct_reference()
        is_correct = (action == correct_ref)

        # Check perseverative error
        is_perseverative = False
        if not is_correct and self._prev_rule is not None:
            prev_correct = self._match_reference(self._prev_rule)
            if action == prev_correct:
                is_perseverative = True

        if is_correct:
            reward = 1.0
            self._consecutive_correct += 1
            self._trials_correct += 1
            self._last_feedback = "correct"

            # Check rule switch
            if (self._consecutive_correct >= self.correct_to_switch
                    and self._rule_changes < self.max_rule_changes):
                self._switch_rule()
        else:
            if is_perseverative:
                reward = -0.5
                self._perseverative_errors += 1
            else:
                reward = -0.3
            self._consecutive_correct = 0
            self._last_feedback = "incorrect"

        self._feedback_remaining = self.feedback_duration

        obs = self._render_frame()
        return obs, reward, terminated, False, self._info()

    # ── State management ─────────────────────────────────────────────────

    def _init_state(self):
        self._trial = 0
        self._trials_correct = 0
        self._consecutive_correct = 0
        self._rule_changes = 0
        self._categories_completed = 0
        self._perseverative_errors = 0
        self._last_feedback = None
        self._feedback_remaining = 0
        self._prev_rule = None

        # Pick initial rule
        self._active_rule = self._rng.choice(_RULE_DIMS)
        self._reference_cards = []
        self._current_card = None

    def _generate_reference_cards(self):
        """Generate 4 reference cards forming a Latin square.

        Each value in each dimension appears exactly once across the 4 cards.
        """
        # Shuffle each dimension independently
        shapes = list(_WCST_SHAPES)
        colors = list(_WCST_COLORS)
        counts = list(_WCST_COUNTS)
        self._rng.shuffle(shapes)
        self._rng.shuffle(colors)
        self._rng.shuffle(counts)

        self._reference_cards = [
            {"shape": shapes[i], "color": colors[i], "count": counts[i]}
            for i in range(4)
        ]

    def _deal_card(self):
        """Generate a new card to sort. Must match exactly one reference per rule dimension."""
        shape = self._rng.choice(_WCST_SHAPES)
        color = self._rng.choice(_WCST_COLORS)
        count = int(self._rng.choice(_WCST_COUNTS))
        self._current_card = {"shape": shape, "color": color, "count": count}

    def _correct_reference(self) -> int:
        """Return index of the reference card matching the current card on active rule."""
        return self._match_reference(self._active_rule)

    def _match_reference(self, rule: str) -> int:
        """Find best matching reference for current card on given rule dimension."""
        card_val = self._current_card[rule]
        for i, ref in enumerate(self._reference_cards):
            if ref[rule] == card_val:
                return i
        # Fallback: closest match (shouldn't happen with Latin square)
        return 0

    def _switch_rule(self):
        """Switch to a new rule dimension (never the same as current)."""
        self._prev_rule = self._active_rule
        available = [r for r in _RULE_DIMS if r != self._active_rule]
        self._active_rule = self._rng.choice(available)
        self._consecutive_correct = 0
        self._rule_changes += 1
        self._categories_completed += 1

    # ── Rendering ────────────────────────────────────────────────────────

    def _render_frame(self) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), BACKGROUND_GRAY, dtype=np.uint8)

        # Reference cards at top
        card_w = self.width // 5
        card_h = self.height // 4
        spacing = (self.width - 4 * card_w) // 5

        for i, ref in enumerate(self._reference_cards):
            x = spacing + i * (card_w + spacing)
            y = 8
            draw_card(canvas, x, y, card_w, card_h,
                      ref["shape"], ref["color"], ref["count"])

        # Current card at center
        if self._current_card is not None:
            cur_w = int(card_w * 1.3)
            cur_h = int(card_h * 1.3)
            cur_x = (self.width - cur_w) // 2
            cur_y = self.height // 2 - cur_h // 2
            draw_card(canvas, cur_x, cur_y, cur_w, cur_h,
                      self._current_card["shape"],
                      self._current_card["color"],
                      self._current_card["count"])

        # Feedback indicator at bottom
        if self._feedback_remaining > 0 and self._last_feedback is not None:
            fb_size = 20
            fb_x = (self.width - fb_size) // 2
            fb_y = self.height - fb_size - 10
            if self._last_feedback == "correct":
                draw_filled_rect(canvas, fb_x, fb_y, fb_size, fb_size, FEEDBACK_GREEN)
            else:
                draw_filled_rect(canvas, fb_x, fb_y, fb_size, fb_size, FEEDBACK_RED)

        return canvas

    # ── Info ─────────────────────────────────────────────────────────────

    def _info(self) -> dict:
        return {
            "active_rule": self._active_rule,
            "rule_changes": self._rule_changes,
            "consecutive_correct": self._consecutive_correct,
            "trial": self._trial,
            "trials_correct": self._trials_correct,
            "trials_total": self._trial,
            "perseverative_errors": self._perseverative_errors,
            "categories_completed": self._categories_completed,
            "last_feedback": self._last_feedback,
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def close(self):
        pass

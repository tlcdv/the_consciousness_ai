"""
Delayed Match-to-Sample (DMTS) Environment.

The gold standard task from animal consciousness research. A simple reactive
agent fails because the sample disappears during the delay period.

Trial structure (4 phases):
  1. Fixation (10 steps): gray screen with fixation cross, agent must wait
  2. Sample (20 steps): single colored shape appears at center
  3. Delay (15-40 steps): blank screen, must hold sample in working memory
  4. Choice (up to 30 steps): 2-4 stimuli appear, agent picks the match

Observation: RGB image [height, width, 3] uint8.
Action: Discrete(5), 0=wait, 1=left, 2=right, 3=up, 4=down.
"""
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from simulations.environments._stimulus_renderer import (
    COLORS, COLOR_NAMES, SHAPE_NAMES, BACKGROUND_GRAY,
    draw_shape, draw_cross, draw_filled_rect,
)


class DMTSEnv(gym.Env):
    """
    Delayed Match-to-Sample with configurable distractors.

    Consciousness components exercised:
      - Working memory (GNW reverberation): sample persists across blank delay
      - Feature binding (AKOrN): shape+color+size bound as one object
      - Selective attention (ignition): distractors filtered during choice
      - Capsule hierarchy: shape identity is compositional
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    SIZES = {"small": 20, "large": 35}

    def __init__(
        self,
        render_mode: str | None = None,
        width: int = 224,
        height: int = 224,
        num_trials: int = 10,
        min_delay: int = 15,
        max_delay: int = 40,
        num_choices: int = 2,
        distractor_overlap: int = 0,
        delay_distractors: bool = False,
        max_steps_per_trial: int = 100,
        fixation_steps: int = 10,
        sample_steps: int = 20,
        choice_timeout: int = 30,
    ):
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.num_trials = num_trials
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.num_choices = max(2, min(4, num_choices))
        self.distractor_overlap = max(0, min(3, distractor_overlap))
        self.delay_distractors = delay_distractors
        self.max_steps_per_trial = max_steps_per_trial
        self.fixation_steps = fixation_steps
        self.sample_steps = sample_steps
        self.choice_timeout = choice_timeout

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )

        self._rng = np.random.default_rng()
        self._reset_trial_state()

    # ── Gym interface ────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._trial = 0
        self._trials_correct = 0
        self._total_steps = 0
        self._start_new_trial()
        return self._render_frame(), self._info()

    def step(self, action: int):
        reward = 0.0
        terminated = False
        truncated = False

        self._phase_step += 1
        self._total_steps += 1

        if self._phase == "fixation":
            if action == 0:
                reward = 0.01  # small shaping reward for waiting
            else:
                reward = -0.2  # premature response
            if self._phase_step >= self.fixation_steps:
                self._transition_to("sample")

        elif self._phase == "sample":
            if action != 0:
                reward = -0.2  # premature response
            if self._phase_step >= self.sample_steps:
                self._transition_to("delay")

        elif self._phase == "delay":
            if action != 0:
                reward = -0.2  # premature response
            if self._phase_step >= self._current_delay:
                self._transition_to("choice")

        elif self._phase == "choice":
            if action != 0:
                # Agent made a selection
                if action == self._target_position:
                    reward = 1.0
                    self._correct = True
                    self._trials_correct += 1
                else:
                    reward = -0.5
                    self._correct = False
                self._trial += 1
                if self._trial >= self.num_trials:
                    terminated = True
                else:
                    self._start_new_trial()
            elif self._phase_step >= self.choice_timeout:
                reward = -0.3  # timeout
                self._correct = False
                self._trial += 1
                if self._trial >= self.num_trials:
                    terminated = True
                else:
                    self._start_new_trial()

        obs = self._render_frame()
        return obs, reward, terminated, truncated, self._info()

    # ── Trial management ─────────────────────────────────────────────────

    def _reset_trial_state(self):
        self._phase = "fixation"
        self._phase_step = 0
        self._trial = 0
        self._trials_correct = 0
        self._total_steps = 0
        self._sample_shape = "triangle"
        self._sample_color = "red"
        self._sample_size = "small"
        self._current_delay = 15
        self._target_position = 1
        self._choice_stimuli = []
        self._correct = None
        self._delay_distractor_active = False
        self._delay_distractor_shape = None
        self._delay_distractor_color = None

    def _start_new_trial(self):
        self._phase = "fixation"
        self._phase_step = 0
        self._correct = None

        # Generate sample stimulus
        self._sample_shape = self._rng.choice(SHAPE_NAMES)
        self._sample_color = self._rng.choice(COLOR_NAMES)
        self._sample_size = self._rng.choice(["small", "large"])
        self._current_delay = int(self._rng.integers(self.min_delay, self.max_delay + 1))

        # Generate choice array
        self._generate_choices()

        # Delay distractor setup
        self._delay_distractor_active = False
        if self.delay_distractors:
            self._delay_distractor_shape = self._rng.choice(
                [s for s in SHAPE_NAMES if s != self._sample_shape]
            )
            self._delay_distractor_color = self._rng.choice(
                [c for c in COLOR_NAMES if c != self._sample_color]
            )

    def _generate_choices(self):
        """Generate target + distractors for the choice phase."""
        # Positions: 1=left, 2=right, 3=up, 4=down
        positions = list(range(1, self.num_choices + 1))
        self._rng.shuffle(positions)
        self._target_position = positions[0]

        target = {
            "shape": self._sample_shape,
            "color": self._sample_color,
            "size": self._sample_size,
            "position": self._target_position,
        }
        stimuli = [target]

        for i in range(1, self.num_choices):
            distractor = self._make_distractor()
            distractor["position"] = positions[i]
            stimuli.append(distractor)

        self._choice_stimuli = stimuli

    def _make_distractor(self) -> dict:
        """Create a distractor with controlled feature overlap."""
        features = ["shape", "color", "size"]
        sample_features = {
            "shape": self._sample_shape,
            "color": self._sample_color,
            "size": self._sample_size,
        }

        # Determine which features to share
        shared_count = min(self.distractor_overlap, len(features) - 1)
        shared_indices = self._rng.choice(len(features), size=shared_count, replace=False)
        shared_set = set(shared_indices)

        result = {}
        for i, feat in enumerate(features):
            if i in shared_set:
                result[feat] = sample_features[feat]
            else:
                if feat == "shape":
                    others = [s for s in SHAPE_NAMES if s != self._sample_shape]
                    result["shape"] = self._rng.choice(others)
                elif feat == "color":
                    others = [c for c in COLOR_NAMES if c != self._sample_color]
                    result["color"] = self._rng.choice(others)
                elif feat == "size":
                    result["size"] = "large" if self._sample_size == "small" else "small"
        return result

    def _transition_to(self, phase: str):
        self._phase = phase
        self._phase_step = 0

    # ── Rendering ────────────────────────────────────────────────────────

    def _render_frame(self) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), BACKGROUND_GRAY, dtype=np.uint8)
        cx, cy = self.width // 2, self.height // 2

        if self._phase == "fixation":
            draw_cross(canvas, cx, cy, size=20, color=(255, 255, 255), thickness=2)

        elif self._phase == "sample":
            radius = self.SIZES[self._sample_size]
            rgb = COLORS[self._sample_color]
            draw_shape(canvas, self._sample_shape, rgb, cx, cy, radius)

        elif self._phase == "delay":
            # Blank screen by default
            if (self.delay_distractors and self._delay_distractor_shape
                    and 5 <= self._phase_step < 10):
                # Brief flash of non-matching stimulus
                rgb = COLORS[self._delay_distractor_color]
                draw_shape(canvas, self._delay_distractor_shape, rgb, cx, cy, 25)
                self._delay_distractor_active = True
            else:
                self._delay_distractor_active = False

        elif self._phase == "choice":
            self._render_choices(canvas)

        return canvas

    def _render_choices(self, canvas: np.ndarray):
        """Draw choice stimuli at cardinal positions."""
        cx, cy = self.width // 2, self.height // 2
        offset = min(self.width, self.height) // 3

        position_coords = {
            1: (cx - offset, cy),       # left
            2: (cx + offset, cy),       # right
            3: (cx, cy - offset),       # up
            4: (cx, cy + offset),       # down
        }

        for stim in self._choice_stimuli:
            pos = stim["position"]
            if pos not in position_coords:
                continue
            sx, sy = position_coords[pos]
            radius = self.SIZES[stim["size"]]
            rgb = COLORS[stim["color"]]
            draw_shape(canvas, stim["shape"], rgb, sx, sy, radius)

    # ── Info ─────────────────────────────────────────────────────────────

    def _info(self) -> dict:
        return {
            "phase": self._phase,
            "trial": self._trial,
            "sample_shape": self._sample_shape,
            "sample_color": self._sample_color,
            "sample_size": self._sample_size,
            "target_position": self._target_position,
            "distractor_overlap": self.distractor_overlap,
            "delay_length": self._current_delay,
            "correct": self._correct,
            "trials_correct": self._trials_correct,
            "trials_total": self._trial,
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def close(self):
        pass

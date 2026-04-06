"""
Emotional Processing Core (PAD Model)

Implements the Pleasure, Arousal, Dominance continuous emotion model.
Processes perception context and previous internal state to produce
updated emotional state vectors. Tracks temporal dynamics for consistency
metrics and supports PCI noise perturbation.

References:
- Mehrabian (1996) PAD temperament model
- Russell (2003) Core affect theory
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any
from collections import deque
from .emotion_processing_interface import EmotionProcessingInterface, UpdateContext, EmotionalState

# Mapping from discrete emotion labels to PAD coordinates
EMOTION_PAD_MAP: dict[str, dict[str, float]] = {
    "joy":       {"valence": 0.8,  "arousal": 0.5,  "dominance": 0.6},
    "sadness":   {"valence": -0.7, "arousal": -0.3, "dominance": -0.5},
    "anger":     {"valence": -0.6, "arousal": 0.8,  "dominance": 0.7},
    "fear":      {"valence": -0.7, "arousal": 0.7,  "dominance": -0.6},
    "surprise":  {"valence": 0.2,  "arousal": 0.8,  "dominance": 0.0},
    "disgust":   {"valence": -0.6, "arousal": 0.3,  "dominance": 0.4},
    "trust":     {"valence": 0.6,  "arousal": -0.2, "dominance": 0.3},
    "neutral":   {"valence": 0.0,  "arousal": 0.0,  "dominance": 0.0},
}


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class EmotionalProcessingCore(EmotionProcessingInterface):
    """
    PAD based emotion processor for the consciousness core.

    Receives perception summaries and previous internal state,
    extracts emotional signals, and updates a continuous PAD vector
    with exponential moving average smoothing.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # PAD state (all start at neutral)
        self._current_state: EmotionalState = {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0,
        }

        # Smoothing factor: higher means faster response to new input
        self._alpha = config.get("emotion_alpha", 0.3)

        # Temporal history for consistency metrics
        self._history: deque = deque(maxlen=config.get("history_length", 100))

        # Reward signal influence on valence
        self._reward_sensitivity = config.get("reward_sensitivity", 0.4)

        # Stress threshold for arousal spike detection
        self._stress_threshold = config.get("stress_threshold", 0.7)

        logging.info("EmotionalProcessingCore initialized (PAD model).")

    def update(self, context: UpdateContext) -> EmotionalState:
        """
        Update emotional state from perception and previous internal state.

        Expected context keys:
            perception: dict with optional keys: emotion_label, reward,
                        stress_level, social_valence, threat_level, novelty
            previous_state: dict with any prior consciousness state info
        """
        perception = context.get("perception") or {}
        previous = context.get("previous_state") or {}

        # Start from current state
        v = self._current_state["valence"]
        a = self._current_state["arousal"]
        d = self._current_state["dominance"]

        # 1. Discrete emotion label mapping (if perception detected one)
        label = perception.get("emotion_label", "").lower()
        if label in EMOTION_PAD_MAP:
            pad = EMOTION_PAD_MAP[label]
            v = self._ema(v, pad["valence"])
            a = self._ema(a, pad["arousal"])
            d = self._ema(d, pad["dominance"])

        # 2. Reward signal drives valence
        reward = perception.get("reward")
        if reward is not None:
            reward_signal = float(reward) * self._reward_sensitivity
            v = self._ema(v, _clamp(reward_signal))

        # 3. Stress / threat drives arousal up and dominance down
        stress = perception.get("stress_level", 0.0)
        threat = perception.get("threat_level", 0.0)
        combined_stress = max(float(stress), float(threat))
        if combined_stress > 0:
            a = self._ema(a, _clamp(combined_stress))
            d = self._ema(d, _clamp(-combined_stress * 0.5))

        # 4. Novelty increases arousal
        novelty = perception.get("novelty", 0.0)
        if float(novelty) > 0:
            a = self._ema(a, _clamp(float(novelty) * 0.3))

        # 5. Social valence (positive social interaction raises valence and dominance)
        social = perception.get("social_valence", 0.0)
        if float(social) != 0:
            v = self._ema(v, _clamp(v + float(social) * 0.3))
            d = self._ema(d, _clamp(d + float(social) * 0.2))

        # 6. Decay toward neutral when no strong signals
        signal_strength = abs(v) + abs(a) + abs(d)
        if signal_strength > 0:
            decay = self.config.get("decay_rate", 0.02)
            v *= (1.0 - decay)
            a *= (1.0 - decay)
            d *= (1.0 - decay)

        # Clamp final values
        self._current_state = {
            "valence": _clamp(v),
            "arousal": _clamp(a),
            "dominance": _clamp(d),
        }

        # Record history
        self._history.append(dict(self._current_state))

        logging.debug(f"Emotion updated: V={v:.3f} A={a:.3f} D={d:.3f}")
        return dict(self._current_state)

    def get_state(self) -> EmotionalState:
        return dict(self._current_state)

    def add_noise(self, magnitude: float):
        """Add random perturbation to PAD state (used by PCI measurement)."""
        for dim in ("valence", "arousal", "dominance"):
            noise = random.gauss(0, magnitude)
            self._current_state[dim] = _clamp(self._current_state[dim] + noise)

    def get_temporal_consistency(self) -> float:
        """
        Measure how stable emotions are over the recent history window.
        Returns a value between 0 (chaotic) and 1 (perfectly stable).
        """
        if len(self._history) < 2:
            return 1.0

        diffs = []
        history = list(self._history)
        for i in range(1, len(history)):
            diff = sum(
                abs(history[i][k] - history[i - 1][k])
                for k in ("valence", "arousal", "dominance")
            )
            diffs.append(diff)

        avg_diff = sum(diffs) / len(diffs)
        # Normalize: max possible diff per step is 6.0 (3 dims, range 2 each)
        consistency = 1.0 - min(avg_diff / 6.0, 1.0)
        return consistency

    def get_emotional_intensity(self) -> float:
        """Return the magnitude of the current emotional state (0 to ~1.73)."""
        return math.sqrt(
            self._current_state["valence"] ** 2
            + self._current_state["arousal"] ** 2
            + self._current_state["dominance"] ** 2
        )

    def get_dominant_emotion(self) -> str:
        """Map current PAD state back to the closest discrete emotion label."""
        best_label = "neutral"
        best_dist = float("inf")
        for label, pad in EMOTION_PAD_MAP.items():
            dist = sum(
                (self._current_state[k] - pad[k]) ** 2
                for k in ("valence", "arousal", "dominance")
            )
            if dist < best_dist:
                best_dist = dist
                best_label = label
        return best_label

    def _ema(self, current: float, target: float) -> float:
        """Exponential moving average step."""
        return current + self._alpha * (target - current)

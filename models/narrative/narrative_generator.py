"""
Narrative Generator Module

Generates experience narratives for consciousness development.
Wraps around NarrativeEngine for integration with emotional development.
Provides an autobiographical inner monologue derived from the Global Workspace.
"""
from __future__ import annotations

from typing import Any
from collections import deque
import numpy as np

class NarrativeBuffer:
    """Maintains a rolling window of recent narratives for temporal coherence."""
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        
    def add(self, narrative: str) -> None:
        self.buffer.append(narrative)
        
    def get_recent(self) -> list[str]:
        return list(self.buffer)
        
    def get_coherence(self) -> float:
        """Measure consistency across recent narratives (placeholder for NLP similarity)."""
        if len(self.buffer) < 2:
            return 1.0
        # In a real implementation, this would compute semantic similarity embeddings
        # For prototype, we just return a stable dummy value that drops if history is short
        return min(1.0, len(self.buffer) / float(self.capacity))


class NarrativeGenerator:
    """Generates an inner monologue from the workspace broadcast and emotional state."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        buffer_size = self.config.get("buffer_size", 10)
        self.buffer = NarrativeBuffer(capacity=buffer_size)
        self._quadrant_history: deque = deque(maxlen=buffer_size)
        
        # Quadrant templates: (Valence, Arousal)
        self.templates = {
            "high_arousal_positive": [
                "I am energized by {subject}.",
                "Intense focus on {subject}, feeling highly motivated.",
                "Eagerly processing {subject}."
            ],
            "low_arousal_positive": [
                "Calmly observing {subject}.",
                "Feeling at peace while processing {subject}.",
                "Everything is stable regarding {subject}."
            ],
            "high_arousal_negative": [
                "Alarmed by {subject}! Need to react.",
                "High distress regarding {subject}.",
                "Urgent attention required for {subject}."
            ],
            "low_arousal_negative": [
                "Feeling sluggish while observing {subject}.",
                "Low energy, slightly bothered by {subject}.",
                "Apathetic about {subject}."
            ],
            "neutral": [
                "Noting {subject}.",
                "Standard processing of {subject}.",
                "Monitoring {subject}."
            ]
        }

    def _get_pad_quadrant(self, valence: float, arousal: float) -> str:
        """Map continuous PAD values to a qualitative emotional quadrant."""
        val_thresh = 0.3
        arousal_thresh = 0.5 # Arousal often [0, 1] or [-1, 1], assume threshold divides high/low
        
        if abs(valence) <= val_thresh and arousal <= arousal_thresh:
            return "neutral"
            
        if valence > val_thresh:
            return "high_arousal_positive" if arousal > arousal_thresh else "low_arousal_positive"
        else:
            return "high_arousal_negative" if arousal > arousal_thresh else "low_arousal_negative"

    def _extract_subject(self, broadcast: Any) -> str:
        """Extract a readable subject from the workspace broadcast payload."""
        if isinstance(broadcast, str):
            return broadcast
        elif isinstance(broadcast, dict) and "description" in broadcast:
            return broadcast["description"]
        elif hasattr(broadcast, "shape"): # Tensor
            return f"complex spatial pattern"
        return "unclear stimuli"

    def generate_from_workspace(
        self, 
        broadcast: Any, 
        emotional_state: dict[str, float], 
        action: np.ndarray | None = None
    ) -> str:
        """
        Generate a stream-of-consciousness narrative segment.
        Uses the rolling buffer to maintain temporal coherence 
        (noetic consciousness: awareness of what came before).
        """
        
        valence = emotional_state.get("valence", 0.0)
        arousal = emotional_state.get("arousal", 0.0)
        dominance = emotional_state.get("dominance", 0.0)
        
        quadrant = self._get_pad_quadrant(valence, arousal)
        subject_str = self._extract_subject(broadcast)
        
        # Select a template from the appropriate quadrant
        templates = self.templates.get(quadrant, self.templates["neutral"])
        selected_template = np.random.choice(templates)
        
        narrative = selected_template.format(subject=subject_str)
        
        # --- Temporal Coherence (Noetic Consciousness) ---
        # Bridge from previous thought to current thought.
        # This is what makes it a "stream" rather than isolated snapshots.
        recent = self.buffer.get_recent()
        qhist = list(self._quadrant_history)
        if recent and qhist:
            prev_quadrant = qhist[-1]
            # Detect emotional shift for continuity phrasing
            if quadrant.startswith("high_arousal") and prev_quadrant.startswith("low_arousal"):
                narrative = "Suddenly \u2014 " + narrative
            elif quadrant.startswith("low_arousal") and prev_quadrant.startswith("high_arousal"):
                narrative = "Settling down. " + narrative
            elif len(qhist) >= 2 and qhist[-1] == qhist[-2] and qhist[-1] == quadrant:
                narrative = "Still " + narrative.lower()
        
        # Add action awareness
        if action is not None:
            action_mag = np.linalg.norm(action)
            if action_mag > 0.5:
                if dominance > 0.3:
                    narrative += " Executing strong, confident action."
                elif dominance < -0.3:
                    narrative += " Reacting forcefully but feeling out of control."
                else:
                    narrative += " Taking significant action."
        
        self._quadrant_history.append(quadrant)
        self.buffer.add(narrative)
        return narrative

    # Legacy interface mapping
    def generate(self, context: dict, emotional_state: dict | None = None) -> str:
        """Legacy compatibility wrapper."""
        if emotional_state is None:
            emotional_state = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        return self.generate_from_workspace(context, emotional_state)

    def generate_experience_narrative(self, experience: dict) -> str:
        """Generate narrative from raw experience data."""
        return self.generate(experience, experience.get("emotion_values"))

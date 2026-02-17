"""
Narrative Generator Module

Generates experience narratives for consciousness development.
Wraps around NarrativeEngine for integration with emotional development.
"""

from typing import Dict, Optional


class NarrativeGenerator:
    """Generates narratives from emotional and experiential context."""

    def __init__(self, config=None):
        self.config = config or {}

    def generate(self, context: Dict, emotional_state: Optional[Dict] = None) -> str:
        """Generate a narrative from context and emotional state."""
        emotion_desc = ""
        if emotional_state:
            valence = emotional_state.get("valence", 0.0)
            arousal = emotional_state.get("arousal", 0.0)
            if valence > 0.3:
                emotion_desc = "positive"
            elif valence < -0.3:
                emotion_desc = "negative"
            else:
                emotion_desc = "neutral"
            if arousal > 0.6:
                emotion_desc = f"high arousal {emotion_desc}"
        return f"[Narrative: {emotion_desc} experience]"

    def generate_experience_narrative(self, experience: Dict) -> str:
        """Generate narrative from raw experience data."""
        return self.generate(experience, experience.get("emotion_values"))

"""
Attention Schema for tracking and managing the agent's attentional focus.
Based on Attention Schema Theory (AST): the brain constructs a simplified
model of its own attention to help control the process of attending.
"""
from __future__ import annotations

import logging
from typing import Any
from collections import deque


class AttentionSchema:
    """
    Maintains a model of the agent's current and recent attentional focus.
    Tracks what the agent is attending to, for how long, and with what intensity.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.history_size = config.get('history_size', 100)
        self.focus_history = deque(maxlen=self.history_size)
        self.current_focus: dict[str, Any] = {}
        logging.info("AttentionSchema initialized.")

    async def update(self, focus_data: dict[str, Any]):
        """Update attention schema with new focus data."""
        self.current_focus = focus_data
        self.focus_history.append(focus_data)

    async def get_overview(self) -> dict[str, Any]:
        """Get cumulative overview of recent attentional focus."""
        if not self.focus_history:
            return {"focus_count": 0, "dominant_modality": None}

        modality_counts: dict[str, int] = {}
        for entry in self.focus_history:
            for key in entry:
                if entry[key] is not None:
                    modality_counts[key] = modality_counts.get(key, 0) + 1

        dominant = max(modality_counts, key=modality_counts.get) if modality_counts else None

        return {
            "focus_count": len(self.focus_history),
            "dominant_modality": dominant,
            "current_focus": self.current_focus,
            "modality_distribution": modality_counts,
        }

    def get_current_focus(self) -> dict[str, Any]:
        """Return the current attentional focus."""
        return self.current_focus

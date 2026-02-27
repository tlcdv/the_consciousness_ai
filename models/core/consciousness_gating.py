"""
Consciousness gating mechanism that controls information flow and adaptation
in the consciousness system. Controls learning rates and meta-memory stability.

Key components:
- Attention-based gating for information flow
- Meta-memory stability tracking
- Controlled adaptation
- Narrator confidence tracking
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class GatingState:
    """Track gating mechanism state."""
    attention_level: float = 0.0
    stability_score: float = 0.0
    adaptation_rate: float = 0.0
    meta_memory_coherence: float = 0.0
    narrator_confidence: float = 0.0


class ConsciousnessGate(nn.Module):
    def __init__(self, config):
        """Sets up gating parameters and neural networks."""
        super().__init__()
        # Support both dict and attribute-style config
        if isinstance(config, dict):
            gating = config.get('gating', {})
            self.attention_threshold = gating.get('attention_threshold', 0.5)
            self.stability_threshold = gating.get('stability_threshold', 0.6)
            self.adaptation_rate = gating.get('base_adaptation_rate', 0.01)
            self.hidden_size = config.get('hidden_size', 128)
        else:
            gating = getattr(config, 'gating', config)
            self.attention_threshold = getattr(gating, 'attention_threshold', 0.5)
            self.stability_threshold = getattr(gating, 'stability_threshold', 0.6)
            self.adaptation_rate = getattr(gating, 'base_adaptation_rate', 0.01)
            self.hidden_size = getattr(config, 'hidden_size', 128)

        # Attention gating.
        self.attention_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

        # Stability gating.
        self.stability_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

        self.state = GatingState()

    def forward(
        self,
        input_state: torch.Tensor,
        meta_memory_context: Optional[Dict] = None,
        narrator_state: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, GatingState]:
        """Processes input through gating networks and updates the gating state."""
        attention_level = self.attention_net(input_state)
        stability_score = self.stability_net(input_state)

        adaptation_rate = self._calculate_adaptation_rate(
            stability_score,
            meta_memory_context
        )

        gated_output = self._apply_gating(
            input_state,
            attention_level,
            stability_score
        )

        self._update_state(
            attention_level,
            stability_score,
            adaptation_rate,
            narrator_state
        )

        return gated_output, self.state

    def _calculate_adaptation_rate(
        self,
        stability_score: torch.Tensor,
        meta_memory_context: Optional[Dict]
    ) -> float:
        """Calculates a learning rate multiplier based on stability and meta-memory."""
        base_rate = self.adaptation_rate
        if meta_memory_context:
            if meta_memory_context.get('stable_patterns'):
                base_rate *= 0.5
            if meta_memory_context.get('novel_experiences'):
                base_rate *= 2.0

        # Multiply by average stability for final rate.
        return base_rate * float(stability_score.mean().item())

    def _apply_gating(
        self,
        input_state: torch.Tensor,
        attention_level: torch.Tensor,
        stability_score: torch.Tensor
    ) -> torch.Tensor:
        """Applies gating logic to the input state based on attention and stability."""
        # Example logic: gate input if attention exceeds threshold.
        mask = (attention_level > self.attention_threshold).float()
        return input_state * mask

    def _update_state(
        self,
        attention_level: torch.Tensor,
        stability_score: torch.Tensor,
        adaptation_rate: float,
        narrator_state: Optional[Dict]
    ) -> None:
        """Updates the gating state with new information."""
        self.state.attention_level = float(attention_level.mean().item())
        self.state.stability_score = float(stability_score.mean().item())
        self.state.adaptation_rate = adaptation_rate
        self.state.meta_memory_coherence = 0.0  # Placeholder; integrate as needed.
        if narrator_state and 'confidence' in narrator_state:
            self.state.narrator_confidence = float(narrator_state['confidence'])
        else:
            self.state.narrator_confidence = 0.0


class ConsciousnessGating:
    """
    Implements an attention control mechanism that decides whether sensory inputs
    trigger enhanced processing based on a gating threshold.
    
    Args:
        config (dict): Contains configuration parameters.
    """
    def __init__(self, config: dict):
        self.config = config
        self.gating_threshold = config.get("gating_threshold", 0.5)

    def update_attention(self, sensory_input: list) -> bool:
        """
        Computes the attention level from a list of sensory measurements and
        determines if it meets the threshold.

        Args:
            sensory_input (list): List of numeric sensory values.

        Returns:
            bool: True if attention level exceeds the threshold, otherwise False.
        """
        if not sensory_input:
            return False
        attention = sum(sensory_input) / len(sensory_input)
        return attention >= self.gating_threshold

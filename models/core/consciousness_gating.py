"""
Consciousness gating mechanism that controls information flow and adaptation
in the consciousness system. Controls learning rates and meta-memory stability.

Key components:
- Attention-based gating for information flow
- Meta-memory stability tracking
- Controlled adaptation
- Narrator confidence tracking
"""
from __future__ import annotations

import torch
import torch.nn as nn
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
        """Sets up gating parameters and neural networks.

        Produces 5 continuous gate values that serve as causal nodes for
        IIT Phi computation and EI measurement. All 5 networks take the
        workspace broadcast as input and produce sigmoid-bounded [0,1] output.
        """
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

        # Coherence network: meta-memory consistency signal.
        self.coherence_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

        # Confidence network: narrator certainty about current state.
        self.confidence_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

        # Broadcast predictor: given current gate state, predict next broadcast.
        # This gives the gate networks a learning signal beyond pure reward,
        # forcing them to develop structured representations of workspace dynamics.
        self.broadcast_predictor = nn.Sequential(
            nn.Linear(5 + self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.state = GatingState()

    def predict_next_broadcast(self, gate_values: torch.Tensor, current_broadcast: torch.Tensor) -> torch.Tensor:
        """Predict next broadcast from current gate state + broadcast.

        Used as an auxiliary loss so gate networks develop structured outputs
        instead of producing near-random values on untrained broadcasts.

        Args:
            gate_values: [B, 5] tensor of current gate outputs
            current_broadcast: [B, hidden_size] current workspace broadcast

        Returns:
            [B, hidden_size] predicted next broadcast
        """
        combined = torch.cat([gate_values, current_broadcast], dim=-1)
        return self.broadcast_predictor(combined)

    def forward(
        self,
        input_state: torch.Tensor,
        meta_memory_context: dict | None = None,
        narrator_state: dict | None = None
    ) -> tuple[torch.Tensor, GatingState]:
        """Processes input through gating networks and updates the gating state."""
        attention_level = self.attention_net(input_state)
        stability_score = self.stability_net(input_state)
        coherence = self.coherence_net(input_state)
        confidence = self.confidence_net(input_state)

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
            coherence,
            confidence,
            narrator_state,
        )

        return gated_output, self.state

    def _calculate_adaptation_rate(
        self,
        stability_score: torch.Tensor,
        meta_memory_context: dict | None
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
        coherence: torch.Tensor,
        confidence: torch.Tensor,
        narrator_state: dict | None = None,
    ) -> None:
        """Updates the gating state with new information."""
        self.state.attention_level = float(attention_level.mean().item())
        self.state.stability_score = float(stability_score.mean().item())
        self.state.adaptation_rate = adaptation_rate
        self.state.meta_memory_coherence = float(coherence.mean().item())
        if narrator_state and 'confidence' in narrator_state:
            self.state.narrator_confidence = float(narrator_state['confidence'])
        else:
            self.state.narrator_confidence = float(confidence.mean().item())


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
            sensory_input (list): list of numeric sensory values.

        Returns:
            bool: True if attention level exceeds the threshold, otherwise False.
        """
        if not sensory_input:
            return False
        attention = sum(sensory_input) / len(sensory_input)
        return attention >= self.gating_threshold

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

        # Gate feedback: projects previous 5 gate values back into hidden space.
        # Creates temporal causal structure matching GATE_CM in iit_phi.py:
        # attention->stability, stability->adaptation, coherence->adaptation,
        # confidence->attention. Without this, all gates are independent MLPs
        # taking the same input, producing near-identical outputs (~0.5).
        self.gate_feedback = nn.Linear(5, self.hidden_size)

        # Buffer for previous gate values (detached, used as conditioning)
        self.prev_gate_values: torch.Tensor | None = None

        self.state = GatingState()

    def forward(
        self,
        input_state: torch.Tensor,
        meta_memory_context: dict | None = None,
        narrator_state: dict | None = None
    ) -> tuple[torch.Tensor, GatingState]:
        """Processes input through gating networks and updates the gating state.

        Gate networks receive enriched input that includes previous gate values,
        creating temporal causal dependencies matching GATE_CM.
        """
        # Enrich input with previous gate values for temporal causal structure
        if self.prev_gate_values is not None:
            feedback = self.gate_feedback(self.prev_gate_values.to(input_state.device))
            enriched = input_state + feedback
        else:
            enriched = input_state

        attention_level = self.attention_net(enriched)
        stability_score = self.stability_net(enriched)
        coherence = self.coherence_net(enriched)
        confidence = self.confidence_net(enriched)

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
        mask = (attention_level > self.attention_threshold).float()
        gated = input_state * mask
        # Guard against NaN/Inf from upstream numerical issues
        return torch.nan_to_num(gated, nan=0.0, posinf=1.0, neginf=0.0)

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
        att_val = float(attention_level.mean().item())
        stab_val = float(stability_score.mean().item())
        coh_val = float(coherence.mean().item())
        conf_val = float(confidence.mean().item())
        if narrator_state and 'confidence' in narrator_state:
            conf_val = float(narrator_state['confidence'])

        self.state.attention_level = att_val
        self.state.stability_score = stab_val
        self.state.adaptation_rate = adaptation_rate
        self.state.meta_memory_coherence = coh_val
        self.state.narrator_confidence = conf_val

        # Store current gate values for temporal feedback on next forward call
        self.prev_gate_values = torch.tensor(
            [att_val, stab_val, adaptation_rate, coh_val, conf_val]
        ).detach()


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

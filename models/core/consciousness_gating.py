"""
Consciousness gating mechanism that controls information flow and adaptation.

Produces 5 continuous gate values (attention, stability, adaptation, coherence,
confidence) that serve as causal nodes for IIT Phi computation and EI measurement.

The gate networks have explicit causal routing matching GATE_CM in iit_phi.py.
Every node lies inside a feedback cycle so the system is irreducible under IIT
(non-zero phi when the empirical TPM has structure):

  attention -> stability        (attention output feeds stability input)
  attention -> coherence        (attention selects what to integrate into the model)
  stability -> adaptation       (stability modulates adaptation rate)
  coherence -> adaptation       (coherence modulates adaptation rate)
  coherence -> confidence       (perceived coherence supports confidence)
  adaptation -> confidence      (high adaptation rate erodes model confidence)
  confidence -> attention       (confidence feeds next step's attention via feedback)

The within-step computation order is: attention, coherence, stability,
adaptation, confidence. Confidence is computed last so it can read the
just-computed adaptation, closing the cycle confidence -> attention -> ...
-> adaptation -> confidence on the cross-step boundary.

forward() returns both a GatingState (float snapshot for logging) and a
differentiable gate_values tensor [B, 5] that preserves gradients for
backpropagation through the phi computation chain.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class GatingState:
    """Track gating mechanism state (float snapshot for logging/metrics)."""
    attention_level: float = 0.0
    stability_score: float = 0.0
    adaptation_rate: float = 0.0
    meta_memory_coherence: float = 0.0
    narrator_confidence: float = 0.0


class ConsciousnessGate(nn.Module):
    def __init__(self, config):
        """Sets up gating parameters and causally connected gate networks.

        Each gate network takes a different input depending on its position
        in the causal graph, so that perturbing one gate's output changes
        downstream gate outputs through real architectural connections.
        """
        super().__init__()
        if isinstance(config, dict):
            gating = config.get('gating', {})
            self.attention_threshold = gating.get('attention_threshold', 0.5)
            self.stability_threshold = gating.get('stability_threshold', 0.6)
            self.base_adaptation_rate = gating.get('base_adaptation_rate', 0.01)
            self.hidden_size = config.get('hidden_size', 128)
            self.ablate_feedback = config.get('ablate_feedback', False)
            self.use_self_vector = config.get('use_self_vector', False)
            self._self_vector_dim = config.get('self_vector_dim', 64)
        else:
            gating = getattr(config, 'gating', config)
            self.attention_threshold = getattr(gating, 'attention_threshold', 0.5)
            self.stability_threshold = getattr(gating, 'stability_threshold', 0.6)
            self.base_adaptation_rate = getattr(gating, 'base_adaptation_rate', 0.01)
            self.hidden_size = getattr(config, 'hidden_size', 128)
            self.ablate_feedback = getattr(config, 'ablate_feedback', False)
            self.use_self_vector = getattr(config, 'use_self_vector', False)
            self._self_vector_dim = getattr(config, 'self_vector_dim', 64)

        # --- Causal gate networks ---
        # Each network takes enriched input PLUS the output of its causal parent.

        # Attention: receives enriched + prev_confidence (confidence->attention loop)
        self.attention_net = nn.Sequential(
            nn.Linear(self.hidden_size + 1, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

        # Stability: receives enriched + attention_output (attention->stability)
        self.stability_net = nn.Sequential(
            nn.Linear(self.hidden_size + 1, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

        # Coherence: receives enriched + attention (attention->coherence)
        self.coherence_net = nn.Sequential(
            nn.Linear(self.hidden_size + 1, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

        # Confidence: receives enriched + coherence + adaptation_raw
        # (coherence->confidence, adaptation->confidence). adaptation_raw is the
        # pre-scaling sigmoid output so its [0, 1] range carries usable signal
        # into the confidence net (the post-scaled adaptation is ~[0, 0.02]).
        self.confidence_net = nn.Sequential(
            nn.Linear(self.hidden_size + 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

        # Adaptation: differentiable scalar from stability + coherence
        # (stability->adaptation, coherence->adaptation)
        self.adaptation_net = nn.Sequential(
            nn.Linear(2, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        # Temporal feedback: projects previous 5 gate values into hidden space.
        # Provides the confidence->attention cross-step connection.
        self.gate_feedback = nn.Linear(5, self.hidden_size)

        # Self-model conditioning (Phase 5 deliverable 3). When enabled, the
        # learned self_vector is projected into hidden space and added to the
        # enriched gate input, so the gate's causal nodes are conditioned on the
        # agent's meta-representation of its own state (gating informed by the
        # self-model, the computational-HOT / cognitive-reality-monitoring idea
        # that metacognition arises from gating in hierarchical RL). Default off
        # keeps the baseline gate path bit-identical.
        self.self_projection = (
            nn.Linear(self._self_vector_dim, self.hidden_size)
            if self.use_self_vector else None
        )

        # Buffer for previous gate values (detached for conditioning only)
        self.prev_gate_values: torch.Tensor | None = None

        self.state = GatingState()

    def reset_episode(self) -> None:
        """Clear cross-episode state. Call at the start of every episode so
        episode N+1's first attention is not conditioned on episode N's last
        gate output."""
        self.prev_gate_values = None
        self.state = GatingState()

    def forward(
        self,
        input_state: torch.Tensor,
        meta_memory_context: dict | None = None,
        narrator_state: dict | None = None,
        self_vector: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, GatingState]:
        """Run causal gate computation. Returns (gated_output, state).

        After calling forward(), access self.last_gate_values_tensor for the
        differentiable [B, 5] tensor (attention, stability, adaptation,
        coherence, confidence) that preserves gradients for phi computation.
        """
        # Temporal feedback from previous step's gate values.
        # Skipped under the ablate_feedback flag so the broadband feedback
        # projection's contribution to gate dynamics can be measured. The
        # narrower confidence -> attention path below stays in either case.
        if self.prev_gate_values is not None and not self.ablate_feedback:
            feedback = self.gate_feedback(self.prev_gate_values.to(input_state.device))
            enriched = input_state + feedback
        else:
            enriched = input_state

        # Self-model conditioning (Phase 5 deliverable 3, default off). Adds the
        # projected self_vector to the enriched input so every gate node is
        # conditioned on the agent's meta-representation of its own state.
        if self.self_projection is not None and self_vector is not None:
            sv = self_vector.to(input_state.device)
            if sv.dim() == 1:
                sv = sv.unsqueeze(0)
            if enriched.dim() == 2 and sv.shape[0] == 1 and enriched.shape[0] > 1:
                sv = sv.expand(enriched.shape[0], -1)
            enriched = enriched + self.self_projection(sv)

        # Extract previous confidence for the confidence->attention connection.
        # On first call, use 0.5 (neutral).
        if self.prev_gate_values is not None:
            prev_conf = self.prev_gate_values[4:5].to(input_state.device)
        else:
            prev_conf = torch.tensor([0.5], device=input_state.device)

        # Expand prev_conf to match batch dimension
        if enriched.dim() == 2:
            prev_conf = prev_conf.unsqueeze(0).expand(enriched.shape[0], -1)

        # --- Causal chain ---
        # Order matters: confidence depends on adaptation, so adaptation
        # must be computed first within the step. The cycle closes across
        # steps via prev_confidence -> attention.
        #
        # 1. Attention <- enriched + prev_confidence (cross-step feedback)
        attention = self.attention_net(torch.cat([enriched, prev_conf], dim=-1))

        # 2. Coherence <- enriched + attention (attention->coherence)
        coherence = self.coherence_net(torch.cat([enriched, attention], dim=-1))

        # 3. Stability <- enriched + attention (attention->stability)
        stability = self.stability_net(
            torch.cat([enriched, attention], dim=-1)
        )

        # 4. Adaptation <- stability + coherence (both feed adaptation).
        # Keep the raw sigmoid output for the confidence net so the [0, 1]
        # signal isn't squashed by the small base_adaptation_rate scaling.
        adaptation_raw = self.adaptation_net(
            torch.cat([stability, coherence], dim=-1)
        )
        adaptation = adaptation_raw * self.base_adaptation_rate * 2.0

        # 5. Confidence <- enriched + coherence + adaptation_raw
        # (coherence->confidence, adaptation->confidence). adaptation_raw
        # closes the cycle that lets pyphi return non-zero phi.
        confidence = self.confidence_net(
            torch.cat([enriched, coherence, adaptation_raw], dim=-1)
        )

        # --- Differentiable gate values tensor (preserves gradients) ---
        gate_values = torch.cat(
            [attention, stability, adaptation, coherence, confidence], dim=-1
        )
        self.last_gate_values_tensor = gate_values

        # --- Gated output ---
        gated_output = self._apply_gating(input_state, attention, stability)

        # --- Update float snapshot for logging/metrics ---
        self._update_state(
            attention, stability, adaptation, coherence, confidence,
            narrator_state,
        )

        return gated_output, self.state

    def _apply_gating(
        self,
        input_state: torch.Tensor,
        attention_level: torch.Tensor,
        stability_score: torch.Tensor
    ) -> torch.Tensor:
        """Applies gating to input based on attention and stability."""
        mask = (attention_level > self.attention_threshold).float()
        gated = input_state * mask
        return torch.nan_to_num(gated, nan=0.0, posinf=1.0, neginf=0.0)

    def _update_state(
        self,
        attention: torch.Tensor,
        stability: torch.Tensor,
        adaptation: torch.Tensor,
        coherence: torch.Tensor,
        confidence: torch.Tensor,
        narrator_state: dict | None = None,
    ) -> None:
        """Snapshots gate values as floats for logging. Also stores detached
        tensor for temporal feedback on the next forward call."""
        att_val = float(attention.mean().item())
        stab_val = float(stability.mean().item())
        adapt_val = float(adaptation.mean().item())
        coh_val = float(coherence.mean().item())
        conf_val = float(confidence.mean().item())
        if narrator_state and 'confidence' in narrator_state:
            conf_val = float(narrator_state['confidence'])

        self.state.attention_level = att_val
        self.state.stability_score = stab_val
        self.state.adaptation_rate = adapt_val
        self.state.meta_memory_coherence = coh_val
        self.state.narrator_confidence = conf_val

        self.prev_gate_values = torch.tensor(
            [att_val, stab_val, adapt_val, coh_val, conf_val]
        ).detach()

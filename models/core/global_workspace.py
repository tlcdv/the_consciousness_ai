from __future__ import annotations

import torch
import numpy as np
import time
from typing import Any
from dataclasses import dataclass
import torch.nn.functional as F

from models.evaluation.iit_phi import IITMetrics
from models.core.qualia_mapper import PhenomenologicalMapper

@dataclass
class WorkspaceMessage:
    """A message broadcast into the global workspace by a specialist module."""
    source: str
    content: Any
    priority: float = 0.5
    timestamp: float = 0.0


@dataclass
class WorkspaceState:
    """Current state of the global workspace"""
    active_content: dict[str, Any]
    access_history: list[dict[str, Any]]
    broadcast_strength: float # Activation level (0.0 - 1.0)
    competition_results: dict[str, float]
    
    # Consciousness metrics
    phi_value: float = 0.0
    is_conscious: bool = False
    focus_topic: str = "idle"
    
    # Phenomenological State (Qualia)
    qualia_vector: np.ndarray = np.zeros(3) # [Intensity, Valence, Complexity]

    # Structured payload from winning module (capsule poses, etc.)
    broadcast_payload: dict[str, Any] | None = None

class GlobalWorkspace:
    """
    Implementation of Global Neuronal Workspace (GNW) for artificial consciousness.
    
    Upgrades:
    1. Sigmoid Ignition (Non-linear Phase Transition)
    2. Recurrent Reverberation (Working Memory)
    3. Synchrony Binding (Multimodal Integration)
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.state = WorkspaceState(
            active_content={},
            access_history=[],
            broadcast_strength=0.0,
            competition_results={},
            phi_value=0.0,
            is_conscious=False,
            focus_topic="idle",
            qualia_vector=np.zeros(3)
        )
        self.specialist_modules = {}
        
        # GNW Parameters
        self.ignition_threshold = config.get("ignition_threshold", 0.6)
        self.ignition_gain = config.get("ignition_gain", 10.0) # Steepness of sigmoid
        self.reverberation_alpha = config.get("reverberation_alpha", 0.7) # Decay rate
        self.max_history = config.get("max_history", 100)
        
        # Dependencies
        self.iit_metrics = IITMetrics()
        self.qualia_mapper = PhenomenologicalMapper()
        
        # AKOrN Oscillatory Binding System
        # 5 sensory/cognitive oscillators. Emotion is a parallel modulator,
        # not a workspace competitor (Tier 2 architecture redesign).
        from models.core.oscillatory_binding import WorkspaceBindingSystem
        num_modules = config.get("num_modules", 5)
        module_names = config.get("module_names", ['vision', 'audio', 'memory', 'body', 'semantic'])
        self.binding_system = WorkspaceBindingSystem(num_modules=num_modules, iterations=5)
        self.binding_system.register_modules(module_names)
        
        # Affective Modulator (Tier 2): emotion modulates bids + threshold
        self.affective_modulator = None
        # ConsciousnessGate: when set, phi is computed from gate state instead
        # of the deprecated bid-based proxy
        self.consciousness_gate = None
        
    def register_specialist(self, name: str, module: Any) -> None:
        """Register a specialist cognitive module"""
        self.specialist_modules[name] = module
    
    def run_competition(self,
                        inputs: dict[str, Any],
                        goal_vector: torch.Tensor,
                        bids: dict[str, float] | None = None,
                        payloads: dict[str, Any] | None = None,
                        pad_state: dict[str, float] | None = None,
                        interoceptive_state: dict[str, float] | None = None) -> tuple[dict[str, Any], dict[str, float]]:
        """
        Run GNW competition with Non-linear Ignition, Reverberation, and AKOrN Binding.

        Args:
            inputs: Legacy payload dictionary
            goal_vector: Homeostasis target
            bids: Explicit scalar bid values to enter competition (replaces polling if provided)
            payloads: The semantic content associated with the bids
            pad_state: Current PAD emotion {"valence", "arousal", "dominance"} for the
                affective modulator. When None and a modulator is registered, no
                modulation is applied (no silent failure: the caller is expected to
                pass this explicitly when the modulator is used).
            interoceptive_state: Optional homeostatic drives {"energy", "fatigue", "damage"}.
                Passed through to AffectiveModulator.modulate.
        """
        if bids is None or payloads is None:
            # Legacy mode: Poll registered modules
            bids = {}
            payloads = {}
            for name, module in self.specialist_modules.items():
                if hasattr(module, 'evaluate_salience'):
                    content, bid = module.evaluate_salience(inputs)
                    bids[name] = bid
                    payloads[name] = content
                    
        # Provide fallback content dict name for the rest of the function
        contents = payloads
        
        # 1b. Affective Modulation (Tier 2)
        # If a modulator is present and the caller passed pad_state, apply the
        # valence field to bids and adjust the ignition threshold. pad_state is
        # an explicit parameter so missing affect data fails loudly via a None
        # check here instead of via the silent magic-attribute fallback that
        # used to live in this block.
        if (self.affective_modulator is not None
                and hasattr(self.affective_modulator, 'modulate')
                and pad_state is not None):
            bids, adjusted_threshold = self.affective_modulator.modulate(
                bids, pad_state, interoceptive_state=interoceptive_state,
            )
            self.ignition_threshold = adjusted_threshold
        
        # 2. Oscillatory Binding (AKOrN - ICLR 2025)
        # Replaces the heuristic: If Vision and Audio > 0.5, multiply by 1.2
        # Now uses Kuramoto oscillators. Modules that synchronize get boosted bids.
        bound_bids, sync_order_parameter = self.binding_system.bind_bids(bids)
        self.last_sync_R = sync_order_parameter
        self.last_sync_R_tensor = self.binding_system.last_sync_R_tensor

        # 3. Calculate Input Energy (Max Bound Bid)
        input_energy = max(bound_bids.values()) if bound_bids else 0.0
        
        # 4. Non-linear Ignition (Sigmoid)
        # S(x) = 1 / (1 + e^(-k(x - theta)))
        # Phase transition from subconscious (low) to conscious (high)
        ignition_val = 1.0 / (1.0 + np.exp(-self.ignition_gain * (input_energy - self.ignition_threshold)))
        
        # 5. Reverberation (Recurrence)
        # New State = Alpha * Old State + (1-Alpha) * New Input
        # This gives the workspace "memory" (Working Memory)
        current_strength = (self.reverberation_alpha * self.state.broadcast_strength) + \
                           ((1.0 - self.reverberation_alpha) * ignition_val)
        
        self.state.broadcast_strength = current_strength
        self.state.competition_results = bound_bids
        
        # 6. Determine Consciousness (Threshold Check on Reverberated State)
        self.state.is_conscious = current_strength >= self.ignition_threshold
        
        # 7. Select Winners (If Conscious)
        winners = []
        if self.state.is_conscious:
            winners = self._resolve_competition(bound_bids)
            
        # 8. IIT & Qualia Calculation (Only if Conscious)
        if self.state.is_conscious:
            # Create Abstract Activation Tensor
            workspace_tensor = self._bids_to_tensor(bound_bids)

            # Calculate Phi from ConsciousnessGate causal states (preferred)
            # or fall back to proxy if no gate is attached
            if self.consciousness_gate is not None:
                gate_input = workspace_tensor.unsqueeze(0) if workspace_tensor.dim() == 1 else workspace_tensor
                # Pad or truncate to gate hidden_size
                hs = self.consciousness_gate.hidden_size
                if gate_input.shape[-1] < hs:
                    gate_input = F.pad(gate_input, (0, hs - gate_input.shape[-1]))
                elif gate_input.shape[-1] > hs:
                    gate_input = gate_input[..., :hs]
                _, gate_state = self.consciousness_gate(gate_input)
                phi_result = self.iit_metrics.compute_phi_from_gate_state(gate_state)
                phi = phi_result.phi
            else:
                phi = self.iit_metrics.compute_phi_proxy(workspace_tensor)

            # Binding increases information integration
            phi += (sync_order_parameter * 0.1)

            self.state.phi_value = phi
            
            # Map to Qualia (Phenomenology)
            qualia = self.qualia_mapper.map_state(workspace_tensor, goal_vector)
            self.state.qualia_vector = qualia.to_vector()
            
            # Broadcast: attach structured payload from winning modules
            broadcast_content = {}
            structured_payload = {}
            for winner in winners:
                payload = contents[winner]
                if isinstance(payload, dict):
                    broadcast_content.update(payload)
                    # Preserve structured data (capsule poses, etc.) per winner
                    structured_payload[winner] = payload
                else:
                    broadcast_content[winner] = payload
                    structured_payload[winner] = payload

            self.state.active_content = broadcast_content
            self.state.broadcast_payload = structured_payload
            self.state.focus_topic = f"Processing: {', '.join(winners)} (Sync: {sync_order_parameter:.2f})"
            
            # History
            self.state.access_history.append({
                'content': broadcast_content,
                'strength': current_strength,
                'winners': winners,
                'phi': phi,
                'qualia': self.state.qualia_vector.tolist(),
                'timestamp': time.time()
            })
             # Trim history if needed
            if len(self.state.access_history) > self.max_history:
                self.state.access_history = self.state.access_history[-self.max_history:]

            return broadcast_content, bids
        else:
            # Subconscious Processing
            self.state.focus_topic = "Idle / Subconscious"
            self.state.phi_value = 0.0
            self.state.qualia_vector = np.zeros(3)
            self.state.broadcast_payload = None
            return {}, bids
    
    def _resolve_competition(self, bids: dict[str, float]) -> list[str]:
        """Determine winners (Winner-Take-Most)."""
        if not bids: return []
        sorted_bids = sorted(bids.items(), key=lambda x: x[1], reverse=True)
        # Return top K or all above threshold
        return [k for k, v in sorted_bids if v >= self.ignition_threshold * 0.8] # Soft threshold

    def _bids_to_tensor(self, bids: dict[str, float]) -> torch.Tensor:
        """Convert bids to tensor for IIT/Qualia mapping."""
        slots = 8
        data = torch.zeros(slots)
        vals = list(bids.values())
        for i, v in enumerate(vals):
            if i < slots: data[i] = v
        return data

    def get_unity_metrics(self) -> tuple[float, bool, str, list[float]]:
        """
        Export metrics for Unity Bridge.
        Returns: (Phi, IsConscious, FocusTopic, QualiaVector)
        """
        return (
            self.state.phi_value,
            self.state.is_conscious,
            self.state.focus_topic,
            self.state.qualia_vector.tolist()
        )

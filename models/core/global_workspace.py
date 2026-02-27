import torch
import numpy as np
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import torch.nn.functional as F

from models.evaluation.iit_phi import IITMetrics
from models.core.qualia_mapper import QualiaMapper

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
    active_content: Dict[str, Any]
    access_history: List[Dict[str, Any]]
    broadcast_strength: float # Activation level (0.0 - 1.0)
    competition_results: Dict[str, float]
    
    # Consciousness metrics
    phi_value: float = 0.0
    is_conscious: bool = False
    focus_topic: str = "idle"
    
    # Phenomenological State (Qualia)
    qualia_vector: np.ndarray = np.zeros(3) # [Intensity, Valence, Complexity]

class GlobalWorkspace:
    """
    Implementation of Global Neuronal Workspace (GNW) for artificial consciousness.
    
    Upgrades:
    1. Sigmoid Ignition (Non-linear Phase Transition)
    2. Recurrent Reverberation (Working Memory)
    3. Synchrony Binding (Multimodal Integration)
    """
    
    def __init__(self, config: Dict):
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
        self.qualia_mapper = QualiaMapper()
        
        # New: AKOrN Oscillatory Binding System
        # 4 sensory/cognitive oscillators. Emotion is now a parallel modulator,
        # not a workspace competitor (Tier 2 architecture redesign).
        from models.core.oscillatory_binding import WorkspaceBindingSystem
        self.binding_system = WorkspaceBindingSystem(num_modules=4, iterations=5)
        self.binding_system.register_modules(['vision', 'audio', 'memory', 'body'])
        
        # Affective Modulator (Tier 2): emotion modulates bids + threshold
        self.affective_modulator = None
        
    def register_specialist(self, name: str, module: Any) -> None:
        """Register a specialist cognitive module"""
        self.specialist_modules[name] = module
    
    def run_competition(self, 
                        inputs: Dict[str, Any], 
                        goal_vector: torch.Tensor, 
                        bids: Optional[Dict[str, float]] = None, 
                        payloads: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Run GNW competition with Non-linear Ignition, Reverberation, and AKOrN Binding.
        
        Args:
            inputs: Legacy payload dictionary
            goal_vector: Homeostasis target
            bids: Explicit scalar bid values to enter competition (replaces polling if provided)
            payloads: The semantic content associated with the bids
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
        # If a modulator is present, apply valence field to bids and
        # adjust ignition threshold before competition.
        if self.affective_modulator is not None and hasattr(self.affective_modulator, 'modulate'):
            pad_state = self.affective_modulator._current_pad_state \
                if hasattr(self.affective_modulator, '_current_pad_state') \
                else getattr(self, '_current_pad_state', {})
            if pad_state:
                bids, adjusted_threshold = self.affective_modulator.modulate(bids, pad_state)
                self.ignition_threshold = adjusted_threshold
        
        # 2. Oscillatory Binding (AKOrN - ICLR 2025)
        # Replaces the heuristic: If Vision and Audio > 0.5, multiply by 1.2
        # Now uses Kuramoto oscillators. Modules that synchronize get boosted bids.
        bound_bids, sync_order_parameter = self.binding_system.bind_bids(bids)
        
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
            
            # Calculate Phi (Using Proxy for now, will upgrade to Empirical later)
            phi = self.iit_metrics.compute_phi_proxy(workspace_tensor)
            
            # Bonus: The binding itself increases information integration
            # We add a fraction of the Kuramoto synchronization parameter to Phi
            phi += (sync_order_parameter * 0.1)
            
            self.state.phi_value = phi
            
            # Map to Qualia (Phenomenology)
            qualia = self.qualia_mapper.map_state(workspace_tensor, goal_vector)
            self.state.qualia_vector = qualia.to_vector()
            
            # Broadcast
            broadcast_content = {}
            for winner in winners:
                broadcast_content.update(contents[winner])
            
            self.state.active_content = broadcast_content
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
            return {}, bids
    
    def _resolve_competition(self, bids: Dict[str, float]) -> List[str]:
        """Determine winners (Winner-Take-Most)."""
        if not bids: return []
        sorted_bids = sorted(bids.items(), key=lambda x: x[1], reverse=True)
        # Return top K or all above threshold
        return [k for k, v in sorted_bids if v >= self.ignition_threshold * 0.8] # Soft threshold

    def _bids_to_tensor(self, bids: Dict[str, float]) -> torch.Tensor:
        """Convert bids to tensor for IIT/Qualia mapping."""
        slots = 8
        data = torch.zeros(slots)
        vals = list(bids.values())
        for i, v in enumerate(vals):
            if i < slots: data[i] = v
        return data

    def get_unity_metrics(self) -> Tuple[float, bool, str, List[float]]:
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

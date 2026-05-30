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

        # Broadcast assembly mode (Phase A of the 2026-05-17 Phi-1 retest plan).
        # Legacy default 'winner_take_all' iterates winners and merges their
        # payloads via .update(); this decouples broadcast content from
        # AKOrN sync_R because the winning module's payload is just its own
        # representation, not an integration.
        # 'attention_weighted' computes broadcast as a softmax-weighted sum of
        # all eligible module payloads, with weights derived from bound_bids.
        # Phi computed on this broadcast is structurally downstream of sync_R.
        self.broadcast_mode = config.get("broadcast_mode", "winner_take_all")
        self.attention_temperature = config.get("attention_temperature", 0.5)
        self.attention_floor = config.get("attention_floor", 0.05)
        # workspace_dim is the target tensor size for fusion. Payload tensors
        # (tectum, audio, semantic) are typically 256-D; the bid-to-tensor
        # mapping uses only 8 slots and is unrelated.
        self.workspace_dim = config.get("workspace_dim", 256)

        # Phase B of 2026-05-19 plan: AKOrN-modulated cross-attention on
        # module content tensors. Consumes pairwise phase coherence from
        # AKOrN's bind_bids and uses it to gate cross-attention logits so
        # synchronized module pairs share content during fusion. Opt-in
        # because Phase A's attention_weighted fusion is the prior step
        # tested; Phase B adds content-level binding on top.
        self.enable_content_binding = config.get("enable_content_binding", False)
        self.binding_attention = None
        if self.enable_content_binding:
            from models.core.binding_attention import BindingAttention
            self.binding_attention = BindingAttention(
                payload_dim=self.workspace_dim,
                hidden_dim=config.get("content_binding_hidden_dim", 64),
            )
        
        # Dependencies
        self.iit_metrics = IITMetrics()
        self.qualia_mapper = PhenomenologicalMapper()
        
        # Oscillatory Binding System.
        # 5 sensory/cognitive oscillators. Emotion is a parallel modulator,
        # not a workspace competitor (Tier 2 architecture redesign).
        #
        # binding_mechanism selects between:
        #   "akorn"   (default, AKOrN ICLR 2025): abstract N-sphere phases
        #             detached from content. Original Tier 1 implementation.
        #   "komplex" (Phase B-alt of 2026-05-19 plan): KomplexNet-style
        #             scalar per-module phases woven multiplicatively into
        #             content via weave_content(). The structural hypothesis
        #             Phase B-alt tests is that AKOrN's separation of phase
        #             from content is the reason Phi-1 keeps failing.
        num_modules = config.get("num_modules", 5)
        module_names = config.get("module_names", ['vision', 'audio', 'memory', 'body', 'semantic'])
        self.binding_mechanism = config.get("binding_mechanism", "akorn")
        if self.binding_mechanism == "komplex":
            from models.core.complex_binding import ComplexBindingSystem
            self.binding_system = ComplexBindingSystem(
                num_modules=num_modules,
                iterations=config.get("binding_iterations", 5),
                eta=config.get("komplex_eta", 0.1),
                desync_eps=config.get("komplex_desync_eps", 0.01),
            )
        elif self.binding_mechanism == "akorn":
            from models.core.oscillatory_binding import WorkspaceBindingSystem
            self.binding_system = WorkspaceBindingSystem(num_modules=num_modules, iterations=5)
        else:
            raise ValueError(
                f"Unknown binding_mechanism '{self.binding_mechanism}'. "
                f"Expected 'akorn' or 'komplex'."
            )
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

        # 2b-alt. Phase B-alt content weaving (2026-05-19 plan, komplex only)
        # KomplexNet weaves the per-module scalar phase directly into the
        # content tensor: content_m <- content_m * cos(theta_m - theta_global).
        # Modules whose phase aligns with the mean field keep magnitude;
        # antiphase modules get sign-flipped; orthogonal modules get
        # suppressed. The hypothesis: phi-on-broadcast tracks sync_R because
        # the binding signal and the content signal are the same signal.
        if (self.binding_mechanism == "komplex"
                and hasattr(self.binding_system, 'weave_content')):
            contents = self.binding_system.weave_content(contents)

        # 2b. Phase B content-level binding (2026-05-19 plan, akorn only)
        # When --enable-content-binding is set, the module payload tensors
        # themselves are passed through AKOrN-modulated cross-attention BEFORE
        # the broadcast-assembly fusion. Synchronized module pairs share
        # content via coherence-gated attention; desynced pairs do not. This
        # addresses failure modes 2 and 3 from the 2026-05-17 diagnosis
        # (AKOrN binds phases not content; reentrant feedback updates bids
        # not content). The downstream Phase A fusion (broadcast_mode =
        # attention_weighted) then operates on these already-bound payloads,
        # so phi-on-broadcast is now downstream of BOTH bid-weighting (Phase A)
        # AND content-weighting (Phase B) by sync_R.
        #
        # NOTE: BindingAttention is meaningful only for AKOrN because komplex
        # already does content-level weaving via weave_content above. If both
        # are active simultaneously the weaving runs first and BindingAttention
        # then re-modulates the woven content; not a recommended config.
        if self.binding_attention is not None:
            coherence = self.binding_system.get_pairwise_coherence()
            if coherence is not None:
                module_order = self.binding_system.module_names
                bound_contents = self.binding_attention(
                    contents, coherence, module_order,
                )
                # Replace each module's payload tensor with the bound version.
                # If the original payload was a dict (e.g., vision payload with
                # capsule_poses), update the "tensor" key in-place to preserve
                # structured data; if it was a raw tensor, replace directly.
                for name, bound_tensor in bound_contents.items():
                    orig = contents.get(name)
                    if isinstance(orig, dict):
                        # Preserve structured payload (capsule data etc.);
                        # only update the tensor field
                        new_payload = dict(orig)
                        new_payload["tensor"] = bound_tensor
                        contents[name] = new_payload
                    else:
                        contents[name] = bound_tensor

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

            # Calculate Phi from ConsciousnessGate causal states. The legacy
            # compute_phi_proxy fallback was REMOVED because it called
            # _extract_subsystem_state which writes 4-tuple states (topk(4))
            # into iit_metrics.state_history, while compute_phi_from_gate_state
            # writes 5-tuple states. The mixed history then has rows of two
            # different arities; build_empirical_tpm skips the 4-tuples via
            # `if len(state_t) != num_nodes: continue`, leaving the TPM with
            # only ~33 transitions out of 200, and pyphi correctly returned 0.
            # If no gate is attached, phi is reported as 0.0 and no state is
            # written to history. Attach a ConsciousnessGate to get real phi.
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
                phi = 0.0

            # Phi is reported as the actual IIT result. The previous
            # `phi += sync_order_parameter * 0.1` line was the duplicate
            # of the train_rlhf.py:402 identity that 2026-04-27 commit
            # 06f96db removed; it survived here, polluting state.phi_value
            # whenever anything read the workspace state directly. Phi and
            # sync_R are independent metrics; their correlation is a real
            # prediction to test, not a built-in identity.
            self.state.phi_value = phi
            
            # Map to Qualia (Phenomenology)
            qualia = self.qualia_mapper.map_state(workspace_tensor, goal_vector)
            self.state.qualia_vector = qualia.to_vector()
            
            # Broadcast assembly. Two modes per Phase A of the 2026-05-17 plan.
            broadcast_content = {}
            structured_payload = {}
            if self.broadcast_mode == "attention_weighted":
                # Phase A path: softmax-weighted fusion of all eligible modules.
                # Eligibility is by AKOrN bound_bid (post-binding amplitude),
                # NOT by the legacy ignition_threshold winners. This makes the
                # broadcast a true integration across modules whenever their
                # post-binding bids are non-trivial, so phi-on-broadcast is
                # structurally downstream of sync_R.
                eligible = {m: contents[m] for m in contents
                            if bound_bids.get(m, 0.0) >= self.attention_floor}
                if eligible:
                    scores = torch.tensor(
                        [bound_bids[m] for m in eligible],
                        dtype=torch.float32,
                    ) / max(self.attention_temperature, 1e-6)
                    weights = F.softmax(scores, dim=0)
                    fused = self._fuse_tensor_payloads(
                        eligible, weights, self.workspace_dim,
                    )
                    broadcast_content = {
                        "_fused": fused,
                        "_weights": {m: float(w) for m, w in zip(eligible.keys(), weights)},
                    }
                    # Preserve structured per-module data (capsule poses, etc.)
                    structured_payload = {m: contents[m] for m in eligible}
            else:
                # Legacy winner-take-all path. Default, unchanged behavior.
                for winner in winners:
                    # A module can win ignition (high bound bid) without having
                    # supplied a content payload: bids come from bound_bids
                    # while contents come from payloads, and e.g. the body
                    # channel bids but provides no tensor. Such a winner
                    # contributes no broadcast content, so skip it rather than
                    # raising KeyError.
                    payload = contents.get(winner)
                    if payload is None:
                        continue
                    if isinstance(payload, dict):
                        broadcast_content.update(payload)
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

    def _payload_to_tensor(self, payload: Any, target_dim: int) -> torch.Tensor:
        """Extract a 1-D tensor of shape [target_dim] from a module payload.

        Payloads come in three flavors:
          - dict with "tensor" key: use that tensor (Phase A common case)
          - raw torch.Tensor: use directly
          - dict without "tensor" key: return zero vector (graceful degradation)

        Pad or truncate to target_dim along the last axis. Batched tensors
        ([B, D]) are reduced to [D] by mean over the batch dim so the
        fusion produces a single broadcast vector per step.
        """
        if isinstance(payload, dict):
            tensor = payload.get("tensor")
            if not isinstance(tensor, torch.Tensor):
                return torch.zeros(target_dim)
        elif isinstance(payload, torch.Tensor):
            tensor = payload
        else:
            return torch.zeros(target_dim)

        # Reduce batched tensors to 1-D
        if tensor.dim() > 1:
            tensor = tensor.mean(dim=tuple(range(tensor.dim() - 1)))

        # Pad or truncate to target_dim
        if tensor.shape[-1] < target_dim:
            tensor = F.pad(tensor, (0, target_dim - tensor.shape[-1]))
        elif tensor.shape[-1] > target_dim:
            tensor = tensor[..., :target_dim]
        return tensor

    def _fuse_tensor_payloads(
        self,
        eligible: dict[str, Any],
        weights: torch.Tensor,
        target_dim: int,
    ) -> torch.Tensor:
        """Weighted sum of payload tensors per Phase A spec.

        Args:
            eligible: dict mapping module-name to payload (dict-or-tensor)
            weights: 1-D tensor of shape [len(eligible)] summing to 1.0
            target_dim: dimensionality of the output fused tensor

        Returns:
            1-D tensor [target_dim] = sum_i(weights[i] * payload_i_as_tensor)
        """
        tensors = torch.stack([
            self._payload_to_tensor(p, target_dim) for p in eligible.values()
        ], dim=0)
        # tensors: [N, target_dim]; weights: [N]; output: [target_dim]
        return (weights.unsqueeze(-1) * tensors).sum(dim=0)

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

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KuramotoLayer(nn.Module):
    """
    Oscillatory binding via Kuramoto dynamics on the N-sphere.

    Inspired by Löwe et al., "Artificial Kuramoto Oscillatory Neurons",
    ICLR 2025. Coupling uses the tangent-plane projection from the
    N-sphere Kuramoto model: for each oscillator i, the pull from j is
    projected onto the tangent plane at i before accumulation.
    """
    def __init__(self, 
                 num_oscillators: int, 
                 dimensions: int = 2,
                 coupling_strength: float = 1.0,
                 natural_frequency_std: float = 0.1,
                 dt: float = 0.1):
        """
        Args:
            num_oscillators: Number of specialist modules/features to bind
            dimensions: N-dimensional sphere for the oscillators (2 = standard phase circle)
            coupling_strength: Global K parameter for how strongly oscillators pull each other
            natural_frequency_std: Variance of initial natural frequencies
            dt: Integration time step
        """
        super().__init__()
        self.num_oscillators = num_oscillators
        self.dimensions = dimensions
        self.K = coupling_strength
        self.dt = dt
        
        # Natural frequencies (omega) for each oscillator
        # In multi-dimensional Kuramoto, this is a skew-symmetric matrix generator
        self.natural_frequencies = nn.Parameter(
            torch.randn(num_oscillators, dimensions, dimensions) * natural_frequency_std
        )
        # Ensure skew-symmetry for valid rotation matrices
        with torch.no_grad():
            self.natural_frequencies.copy_(
                self.natural_frequencies - self.natural_frequencies.transpose(-1, -2)
            )

        # Learnable coupling weights between specific oscillators
        self.coupling_weights = nn.Parameter(
            torch.ones(num_oscillators, num_oscillators) / num_oscillators
        )
        
    def init_phases(self, batch_size: int = 1) -> torch.Tensor:
        """Initialize random phases on the N-sphere"""
        # [batch, num_oscillators, dimensions]
        phases = torch.randn(batch_size, self.num_oscillators, self.dimensions, device=self.coupling_weights.device)
        return F.normalize(phases, p=2, dim=-1)
        
    def forward(self, 
                phases: torch.Tensor, 
                amplitudes: torch.Tensor | None = None,
                iterations: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the discrete Kuramoto update.
        
        Args:
            phases: Current oscillator phases [B, N, D]
            amplitudes: Input bid values/strengths from modules [B, N]
            iterations: Number of discrete integration steps
            
        Returns:
            Updated phases [B, N, D] and synchronization order parameter R [B]
        """
        B, N, D = phases.shape
        device = phases.device
        
        if amplitudes is None:
            amplitudes = torch.ones(B, N, device=device)
            
        # Ensure valid initial state
        current_phases = F.normalize(phases, p=2, dim=-1)
        
        # Enforce skew-symmetry on natural frequencies during forward pass
        omega = self.natural_frequencies - self.natural_frequencies.transpose(-1, -2)
        
        for _ in range(iterations):
            # [B, N, N] interaction matrix = Amplitudes * LearnableWeights
            interaction_strength = torch.einsum('bi,ij,bj->bij', amplitudes, self.coupling_weights, amplitudes)

            # Tangent-plane projection: for each oscillator i, project phase_j
            # onto the tangent plane at phase_i before computing the pull.
            # proj_j = phase_j - <phase_j, phase_i> * phase_i
            # This is the correct N-sphere Kuramoto coupling.
            # dot_ij[b,i,j] = <phase_i, phase_j>
            dot_ij = torch.einsum('bid,bjd->bij', current_phases, current_phases)
            # projected[b,i,j,d] = phase_j[b,j,d] - dot_ij[b,i,j] * phase_i[b,i,d]
            projected = current_phases.unsqueeze(1) - dot_ij.unsqueeze(-1) * current_phases.unsqueeze(2)
            # pull[b,i,d] = sum_j interaction[b,i,j] * projected[b,i,j,d]
            pull = torch.einsum('bij,bijd->bid', interaction_strength, projected)

            # Natural rotation (omega * phase)
            rotation = torch.einsum('ndd,bnd->bnd', omega, current_phases)

            # Update: phase + dt * (rotation + K * pull)
            dp = self.dt * (rotation + self.K * pull)
            current_phases = current_phases + dp
            current_phases = F.normalize(current_phases, p=2, dim=-1)
            
        # Calculate Kuramoto order parameter R (synchronization level)
        # R = ||(1/N) * sum_i (amplitude_i * phase_i)||
        mean_field = torch.einsum('bn,bnd->bd', amplitudes, current_phases) / N
        synchronization_R = torch.norm(mean_field, p=2, dim=-1)

        # Phase B of the 2026-05-19 plan: expose the pairwise phase-coherence
        # matrix `dot_ij[b,i,j] = <phase_i, phase_j>` for downstream
        # content-level binding (BindingAttention). Values are cosines of
        # the angle between oscillator phase vectors (in [-1, 1] for D=2,
        # generally in [-1, 1] for D-sphere). The `dot_ij` inside the
        # iteration loop above is overwritten each step; here we recompute
        # it once on the FINAL phases, which is the state Phase B uses to
        # modulate content cross-attention. Detached because downstream
        # consumers use it as a gating signal, not for gradient flow.
        self.last_pairwise_coherence = torch.einsum(
            'bid,bjd->bij', current_phases, current_phases
        ).detach()

        return current_phases, synchronization_R

class WorkspaceBindingSystem(nn.Module):
    """
    Wrapper to apply AKOrN binding to the Global Workspace competition.
    Replaces the hardcoded scalar synchrony multiplier.
    """
    def __init__(self, num_modules: int, iterations: int = 5):
        super().__init__()
        self.num_modules = num_modules
        self.iterations = iterations
        self.kuramoto = KuramotoLayer(num_oscillators=num_modules, dimensions=2)
        
        # We need to maintain the phase state across workspace steps
        self.register_buffer('current_phases', None)
        self.module_names: list[str] = []
        
    def register_modules(self, names: list[str]):
        """Map module names to oscillator indices"""
        self.module_names = names
        assert len(names) == self.num_modules, "Mismatch between mapped names and oscillator count"
        
    def reset_state(self):
        """Reset phases for a new episode/sequence"""
        self.current_phases = None
        
    def bind_bids(self, bids: dict[str, float]) -> tuple[dict[str, float], float]:
        """
        Take scalar bids, run Kuramoto synchronization, and boost bids
        that synchronize with the global mean field.
        """
        device = next(self.parameters()).device
        
        if not self.module_names:
            # First run initialization if not explicitly registered
            self.module_names = list(bids.keys())
            
        # Prepare amplitudes vector matching the registered order
        amps = torch.zeros(1, self.num_modules, device=device)
        for i, name in enumerate(self.module_names):
            amps[0, i] = bids.get(name, 0.0)
            
        # Initialize phases if needed
        if self.current_phases is None:
            self.current_phases = self.kuramoto.init_phases(batch_size=1)
            
        # Run AKOrN binding dynamics
        new_phases, sync_R = self.kuramoto(self.current_phases, amplitudes=amps, iterations=self.iterations)
        
        # Update persistent state
        self.current_phases = new_phases.detach()
        
        # Calculate how bound each module is to the whole
        # By taking dot product with the mean field
        mean_field = torch.mean(new_phases, dim=1, keepdim=True) # [1, 1, D]
        # Alignment is [-1, 1], shifted to [0, 1]
        alignment = F.cosine_similarity(new_phases, mean_field.expand_as(new_phases), dim=-1) 
        alignment_score = (alignment + 1.0) / 2.0 # [1, N]
        
        # Apply binding boost:
        # Heavily bound modules (alignment > 0.8) get their bids boosted
        # This replaces the hardcoded `if vision > 0.5 and audio > 0.5: mult by 1.2`
        # Now it's an emergent property of the oscillator dynamics.
        bound_bids = {}
        alignment_flat = alignment_score[0].tolist()
        
        for i, name in enumerate(self.module_names):
            orig_bid = bids.get(name, 0.0)
            align_val = alignment_flat[i]
            
            # Boost formula: up to 1.5x boost for perfect synchronization
            boost_factor = 1.0 + (0.5 * align_val)
            bound_bids[name] = orig_bid * boost_factor
            
        # Store differentiable tensor for gradient-based optimization
        self.last_sync_R_tensor = sync_R

        return bound_bids, sync_R.item()

    def get_pairwise_coherence(self) -> torch.Tensor | None:
        """Pairwise phase coherence [B, N, N] from the last bind_bids call.

        Returns the cosine-similarity matrix of oscillator phase vectors at
        the end of the most recent Kuramoto iteration. Values are in [-1, 1]:
        +1 = fully synchronized pair, 0 = orthogonal, -1 = antiphase. Used by
        Phase B's BindingAttention to gate cross-attention keys so that
        synchronized module pairs literally share content.

        Returns None if bind_bids has not been called yet (no phase state).
        """
        return getattr(self.kuramoto, 'last_pairwise_coherence', None)

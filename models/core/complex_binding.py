"""KomplexNet-style content-level binding (Phase B-alt of 2026-05-19 plan).

Replaces AKOrN's abstract-phase binding (`oscillatory_binding.py`) with a
mechanism where phase is woven directly into module content tensors. The
core hypothesis the Phi-1 retest under Phase B-alt tests:

  "If module phases are EMBEDDED in the content vectors themselves (not
  attached as separate abstract oscillator states like AKOrN), then phi
  computed on the resulting complex-valued content will structurally
  track sync_R, because the binding signal and the content signal are
  the same signal."

Mathematical source: Muzellec et al. (2025), "Enhancing deep neural
networks through complex-valued representations and Kuramoto
synchronization dynamics", arxiv 2502.21077, MIT-licensed at
github.com/S4b1n3/KomplexNet. This is a clean-room implementation of
the paper's Equation 1 (amplitude-modulated Kuramoto with global
desynchronization term) adapted to our 5-module workspace use case (5
modules, scalar phase per module, not spatial CNN).

Differences from AKOrN (`oscillatory_binding.py`):
  - AKOrN: phases live on a 2D N-sphere as abstract oscillator state;
    coupling via tangent-plane projection.
  - KomplexNet (this module): scalar phase per module in [0, 2π);
    coupling via amplitude-modulated sine with explicit desync term.
  - AKOrN: phases never touch content tensors.
  - KomplexNet: phases woven multiplicatively into content. Each module's
    content tensor gets a per-module phase factor cos(θ_m). When phases
    sync, all modules' content scales similarly; when desynced, they scale
    differently. The content carries the binding signature.

The license decision in `docs/decisions/2026_05_19_komplexnet_license.md`
records that the upstream code is MIT and we are not vendoring (the math
is free per CC BY 4.0 paper distribution, and our module structure
differs enough from KomplexNet's CNN context that a clean-room
implementation is cleaner than a port).

Interface (drop-in replacement for `WorkspaceBindingSystem.bind_bids`):
  - `bind_bids(bids: dict[str, float]) -> (bound_bids: dict[str, float], sync_R: float)`
  - `get_pairwise_coherence() -> [1, N, N]` (cosine of pairwise phase differences)
  - `weave_content(payloads: dict) -> dict` (NEW: multiplies content by per-module phase factor)
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


class ComplexBindingSystem(nn.Module):
    """KomplexNet-style binding with per-module scalar phases woven into content.

    Adapts KomplexNet's Equation 1 from spatial CNN to a 5-module workspace:
      d θ_m / dt = η * sum_n [(r[m, n] - ε) * sin(θ_n - θ_m) * tanh(a_n)]

    Where r is a learnable coupling matrix, ε is a global desync parameter,
    η is the integration gain, and a_n is the amplitude (bid) of module n.

    Phases are scalar (single angle per module), not on the N-sphere as in
    AKOrN. The Kuramoto order parameter is computed as:
      R = || (1/N) * sum_n a_n * (cos(θ_n), sin(θ_n)) ||
    which is the standard amplitude-weighted mean-phase magnitude.

    Phases are woven into content via multiplicative gating: the post-binding
    content of module m is `content_m * cos(θ_m - θ_global)` where θ_global
    is the mean field phase. Modules whose phase aligns with the mean field
    keep their full content magnitude; antiphase modules get sign-flipped
    (content × negative cosine), and orthogonal modules get suppressed.
    This is the structural coupling that AKOrN lacks.
    """

    def __init__(
        self,
        num_modules: int,
        iterations: int = 5,
        eta: float = 0.1,
        desync_eps: float = 0.01,
    ):
        super().__init__()
        self.num_modules = num_modules
        self.iterations = iterations
        # Learnable coupling matrix r[m, n] for amplitude-modulated coupling.
        # Initialized uniform 1/N (same as AKOrN's coupling_weights), so the
        # default behavior is symmetric all-to-all coupling.
        self.coupling = nn.Parameter(
            torch.ones(num_modules, num_modules) / num_modules
        )
        # Global desync term ε from Equation 1. Small positive value pushes
        # phases apart in the absence of strong coupling signal.
        self.desync_eps = nn.Parameter(torch.tensor(desync_eps))
        # Phase update gain η from Equation 1.
        self.eta = nn.Parameter(torch.tensor(eta))
        # Persistent phase state across competition steps. Scalar phase per
        # module, in [0, 2π). Stored as a buffer so it survives `.to(device)`.
        self.register_buffer("current_phases", None)
        # Module name to oscillator index mapping
        self.module_names: list[str] = []
        # Cached pairwise coherence and sync_R for downstream consumers
        self.last_pairwise_coherence: torch.Tensor | None = None
        self.last_sync_R_tensor: torch.Tensor | None = None

    def register_modules(self, names: list[str]) -> None:
        """Map module names to oscillator indices (parallel to AKOrN)."""
        self.module_names = names
        assert len(names) == self.num_modules, (
            f"Module count mismatch: registered {len(names)} names but "
            f"system has {self.num_modules} oscillators"
        )

    def reset_state(self) -> None:
        """Reset phases for a new episode/sequence (parallel to AKOrN)."""
        self.current_phases = None

    def _init_phases(self, device: torch.device) -> torch.Tensor:
        """Initialize random phases uniformly in [0, 2π)."""
        return torch.rand(self.num_modules, device=device) * 2 * math.pi

    def bind_bids(
        self, bids: dict[str, float]
    ) -> tuple[dict[str, float], float]:
        """Take scalar bids, run KomplexNet-style Kuramoto, return boosted bids.

        Drop-in replacement for `WorkspaceBindingSystem.bind_bids`. The phase
        state is updated in place; downstream consumers can pull the new
        phases via `get_module_phases()` and use them to weave content.
        """
        device = next(self.parameters()).device
        if not self.module_names:
            # First run initialization if not explicitly registered
            self.module_names = list(bids.keys())

        # Prepare amplitudes vector matching the registered module order
        amps = torch.zeros(self.num_modules, device=device)
        for i, name in enumerate(self.module_names):
            amps[i] = bids.get(name, 0.0)

        # Initialize phases if needed
        if self.current_phases is None:
            self.current_phases = self._init_phases(device)

        # Run KomplexNet Kuramoto iterations.
        # Equation 1: d θ_m = η * sum_n (r[m,n] - ε) * sin(θ_n - θ_m) * tanh(a_n)
        phases = self.current_phases
        for _ in range(self.iterations):
            # Pairwise phase differences: diff[m, n] = θ_n - θ_m
            diff = phases.unsqueeze(0) - phases.unsqueeze(1)  # [N, N]
            # Amplitude modulation via tanh(a_n) — saturating, in [0, 1) for
            # positive bids; matches KomplexNet Equation 1.
            amp_mod = torch.tanh(amps).unsqueeze(0)  # [1, N]
            # Coupling term per (m, n) pair
            coupling_term = (
                (self.coupling - self.desync_eps)
                * torch.sin(diff)
                * amp_mod
            )  # [N, N]
            # Sum over n for each m, scaled by η
            d_phases = self.eta * coupling_term.sum(dim=1)  # [N]
            phases = phases + d_phases

        # Update persistent state (detached so gradients don't accumulate
        # across competition steps; the parameters self.coupling, self.eta,
        # self.desync_eps stay differentiable for the workspace_optimizer).
        self.current_phases = phases.detach()

        # Compute Kuramoto order parameter R = amplitude-weighted mean phase magnitude
        # Mean field: (1/N) * sum_n a_n * (cos θ_n, sin θ_n)
        mean_real = (amps * torch.cos(phases)).sum() / self.num_modules
        mean_imag = (amps * torch.sin(phases)).sum() / self.num_modules
        sync_R = torch.sqrt(mean_real ** 2 + mean_imag ** 2 + 1e-12)
        self.last_sync_R_tensor = sync_R

        # Cache pairwise coherence cos(θ_m - θ_n) for downstream consumers
        # (BindingAttention etc.). Shape [1, N, N], detached.
        coh = torch.cos(phases.unsqueeze(0) - phases.unsqueeze(1))
        self.last_pairwise_coherence = coh.unsqueeze(0).detach()

        # Cache global mean-field phase for the weaving step
        mean_phase = torch.atan2(mean_imag, mean_real)
        self._last_mean_phase = mean_phase.detach()

        # Boost bids per AKOrN convention: alignment with mean field in [0, 1]
        # determines boost factor in [1.0, 1.5].
        alignment = torch.cos(phases - mean_phase)  # [-1, 1]
        alignment_score = (alignment + 1.0) / 2.0  # [0, 1]
        bound_bids: dict[str, float] = {}
        for i, name in enumerate(self.module_names):
            orig_bid = bids.get(name, 0.0)
            boost_factor = 1.0 + 0.5 * float(alignment_score[i].item())
            bound_bids[name] = orig_bid * boost_factor

        return bound_bids, float(sync_R.item())

    def get_pairwise_coherence(self) -> torch.Tensor | None:
        """Pairwise phase coherence [1, N, N] from the last bind_bids call.

        Same interface as `WorkspaceBindingSystem.get_pairwise_coherence`,
        so Phase B's BindingAttention works unchanged with this module.
        Values in [-1, 1] (cosines of phase differences).
        """
        return self.last_pairwise_coherence

    def get_module_phases(self) -> torch.Tensor | None:
        """Per-module scalar phases [N] from the last bind_bids call.

        Used by `weave_content` to multiply content tensors. Returned phases
        are detached (gradient does not flow back through phase state across
        competition steps).
        """
        return self.current_phases

    def weave_content(self, payloads: dict[str, Any]) -> dict[str, Any]:
        """Multiply each module's content tensor by cos(θ_m - θ_global).

        This is the KEY ADAPTATION of KomplexNet to our 5-module workspace:
        the per-module phase is multiplicatively woven into the content
        tensor. Modules whose phase aligns with the global mean field keep
        their full content magnitude (cosine near +1); antiphase modules
        get sign-flipped (cosine near -1); orthogonal modules get suppressed
        (cosine near 0).

        The hypothesis: phi computed on the woven content will track sync_R,
        because high sync_R means most modules have cosine near +1 (content
        kept), low sync_R means content is scrambled (some kept, some flipped,
        some suppressed). The resulting content distribution variance IS the
        sync_R signal made measurable.

        Preserves payload structure: dict payloads keep their non-tensor
        fields; the "tensor" or "_fused" key gets multiplied by the phase
        factor.
        """
        if self.current_phases is None or self._last_mean_phase is None:
            return payloads
        phases = self.current_phases
        mean_phase = self._last_mean_phase
        # Per-module multiplicative factor: cos(θ_m - θ_global) ∈ [-1, 1]
        factors = torch.cos(phases - mean_phase)  # [N]
        out: dict[str, Any] = {}
        for i, name in enumerate(self.module_names):
            if name not in payloads:
                continue
            factor = float(factors[i].item())
            payload = payloads[name]
            if isinstance(payload, dict):
                new_payload = dict(payload)
                if isinstance(new_payload.get("tensor"), torch.Tensor):
                    new_payload["tensor"] = new_payload["tensor"] * factor
                if isinstance(new_payload.get("_fused"), torch.Tensor):
                    new_payload["_fused"] = new_payload["_fused"] * factor
                out[name] = new_payload
            elif isinstance(payload, torch.Tensor):
                out[name] = payload * factor
            else:
                out[name] = payload
        return out

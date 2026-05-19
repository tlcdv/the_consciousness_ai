"""AKOrN-modulated cross-attention for content-level binding (Phase B).

Phase B of the 2026-05-19 plan (~/.claude/plans/let-s-plan-the-next-misty-parasol.md).
Addresses failure modes 2 and 3 from the 2026-05-17 diagnosis:
  - AKOrN binds phases of scalar bids, not content tensors
  - Reentrant feedback updates bids only, not content

Standard cross-attention:
    attn = softmax(Q K^T / sqrt(d)); out = attn @ V

Phase B modulation: pre-multiply the attention logits by the pairwise phase
coherence matrix from AKOrN. Synchronized module pairs (coherence near +1)
get larger logits and thus more attention weight; desynchronized pairs
(coherence near -1 or 0) are downweighted. The result: post-binding content
of module i is a weighted blend of all modules j whose phases are coherent
with i's phase. Synchronized modules literally share content; desynced
ones don't.

Theory: this makes phi computed on the resulting content structurally
downstream of AKOrN sync_R, because the content variation is now a
function of the phase coherence matrix that sync_R summarizes.

This module does NOT replace AKOrN. It consumes the pairwise coherence
matrix already computed inside KuramotoLayer.forward() (cached at
`last_pairwise_coherence` per the Phase B1 edit) and applies it as a
multiplicative gate on attention logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BindingAttention(nn.Module):
    """AKOrN-modulated cross-attention over module payload tensors.

    Consumes the [B, N, N] pairwise phase coherence from AKOrN and uses it
    to gate cross-attention logits so synchronized module pairs share
    content during fusion.
    """

    def __init__(self, payload_dim: int = 256, hidden_dim: int = 64):
        super().__init__()
        self.payload_dim = payload_dim
        self.hidden_dim = hidden_dim
        self.W_q = nn.Linear(payload_dim, hidden_dim)
        self.W_k = nn.Linear(payload_dim, hidden_dim)
        self.W_v = nn.Linear(payload_dim, hidden_dim)
        self.W_out = nn.Linear(hidden_dim, payload_dim)
        self.scale = hidden_dim ** -0.5

    def forward(
        self,
        payloads: dict[str, torch.Tensor],
        coherence: torch.Tensor,
        module_order: list[str],
    ) -> dict[str, torch.Tensor]:
        """Apply coherence-modulated cross-attention to bind module content.

        Args:
            payloads: dict mapping module name -> tensor. Each tensor may be
                [payload_dim], [1, payload_dim], or [B, payload_dim].
                Tensors not at payload_dim are padded/truncated.
            coherence: [B, N, N] pairwise phase coherence from AKOrN. Values
                in [-1, 1] (cosines of phase-vector angles).
            module_order: list of module names. The order MUST match the
                oscillator order in AKOrN (so coherence[i, j] is the
                coherence between module module_order[i] and module_order[j]).

        Returns:
            dict mapping module name -> [1, payload_dim] bound content tensor.
            Each output is a coherence-weighted blend of all module payloads.
        """
        # Filter module_order to only those modules that have payloads
        present = [m for m in module_order if m in payloads]
        if not present:
            return {}

        # Stack payloads in present-order. _to_1d_payload returns shape
        # [payload_dim]; stack along dim=0 gives [N_present, payload_dim];
        # unsqueeze(0) adds the batch dim => [1, N_present, payload_dim].
        normalized = [self._to_1d_payload(payloads[m]) for m in present]
        stacked = torch.stack(normalized, dim=0).unsqueeze(0)

        # If the coherence matrix is for the FULL N modules but only a subset
        # are present, restrict coherence to the present-set rows/cols.
        if coherence.shape[1] != len(present):
            present_indices = [module_order.index(m) for m in present]
            idx = torch.tensor(present_indices, device=coherence.device)
            # coherence[:, idx, :][:, :, idx]
            coherence = coherence.index_select(1, idx).index_select(2, idx)

        # Standard QKV projection
        Q = self.W_q(stacked)  # [1, N, hidden_dim]
        K = self.W_k(stacked)
        V = self.W_v(stacked)

        # Convert coherence [-1, 1] to a non-negative mask [0, 1] via
        # (coh + 1) / 2. Then use it as a BIAS on the attention logits in
        # log-space: log(mask + eps) ranges from log(eps) (coh=-1, very
        # negative -> zero attention) to log(1) = 0 (coh=+1, no penalty).
        # This guarantees:
        #   coh = +1 (synced)   -> log-bias = 0       -> full QK attention
        #   coh =  0 (no info)  -> log-bias = log(0.5) -> moderate damping
        #   coh = -1 (antiphase)-> log-bias = log(eps) -> zero attention
        # And critically: with all-equal coherence (uniform mask), the log
        # bias is uniform too, so softmax produces uniform attention (every
        # module gets equal weight from every other), matching the Plan
        # agent's original intent for the "no information" case.
        coh_mask = ((coherence.detach() + 1.0) / 2.0).clamp(min=1e-8, max=1.0)
        logits = torch.einsum('bid,bjd->bij', Q, K) * self.scale
        biased = logits + torch.log(coh_mask)
        attn = F.softmax(biased, dim=-1)

        # Apply attention
        out = torch.einsum('bij,bjd->bid', attn, V)
        bound = self.W_out(out)  # [1, N_present, payload_dim]

        # Expose attention weights for testing and diagnostics (detached).
        # Critical for the synthetic-drive gate test: attn[i, j] should be
        # high when coherence[i, j] is high, regardless of whether the
        # downstream W_q/W_k/W_v/W_out projections happen to wash out
        # the differentiation. Untrained random projections can mask the
        # design's effect at the OUTPUT level; the ATTENTION WEIGHTS are
        # the directly-modulated quantity.
        self.last_attention = attn.detach()

        # Return as dict keyed by module name. Each bound output is the
        # coherence-weighted blend of all eligible module payloads.
        return {m: bound[:, i] for i, m in enumerate(present)}

    def _to_1d_payload(self, payload) -> torch.Tensor:
        """Coerce a payload (dict-or-tensor) to shape [payload_dim].

        Accepts:
          - dict with "tensor" key (the common case from global_workspace)
          - dict with "_fused" key (from Phase A fusion output)
          - dict without either key -> returns zero vector (graceful)
          - raw tensor of shape [payload_dim], [1, payload_dim], or [B, payload_dim]

        Pads or truncates the last axis to payload_dim if mismatched.
        Reduces batched tensors to 1-D by mean over leading dims.
        """
        if isinstance(payload, dict):
            tensor = payload.get("tensor")
            if not isinstance(tensor, torch.Tensor):
                tensor = payload.get("_fused")
            if not isinstance(tensor, torch.Tensor):
                return torch.zeros(self.payload_dim)
        elif isinstance(payload, torch.Tensor):
            tensor = payload
        else:
            return torch.zeros(self.payload_dim)

        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() > 1:
            # Reduce leading dims to a single sample
            tensor = tensor.mean(dim=tuple(range(tensor.dim() - 1)))
        # Now tensor is 1-D
        if tensor.shape[0] < self.payload_dim:
            tensor = F.pad(tensor, (0, self.payload_dim - tensor.shape[0]))
        elif tensor.shape[0] > self.payload_dim:
            tensor = tensor[: self.payload_dim]
        return tensor

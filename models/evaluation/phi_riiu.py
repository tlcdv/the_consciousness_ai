"""
RIIU-style differentiable phi estimator (sliding-window SVD).

Computes phi as the Frobenius-norm residual of a low-rank approximation of the
empirical covariance matrix over a sliding window of activations:

    phi = ||Sigma - U_r (U_r^T Sigma)||_F / (||Sigma||_F + eps)

Sigma is the mean-centered sample covariance, normalized to scale-invariance.
U_r are the top-r left singular vectors of Sigma. A high phi means most of
Sigma's variance is NOT captured by the top-r principal components, i.e. the
signal is high-rank and "integrated" across many dimensions.

This is an alternative phi pathway parallel to the pyphi-based pipeline in
`iit_phi.py`. The pyphi pipeline builds a binary-state TPM and runs MIP, which
in the 2026-05-14 ablation campaign collapsed to a fixed point of ~6.5e-05
across all 5 architectural variants. RIIU operates on continuous activations
with an online SVD, sidestepping the binarization and MIP bottlenecks.

The core `AutoPhiSurrogate.forward` block is vendored from
https://github.com/ReFractals/RIIU (Apache-2.0). The `RIIUPhi` wrapper around
it manages a ring buffer and exposes a push/compute/reset API matching the
project's sliding-window integration pattern. The Apache-2.0 LICENSE is
preserved at `third_party/RIIU_LICENSE`.

References:
    N'guessan, G. L. R., and Karambal, I. (2025). The Reflexive Integrated
    Information Unit: A Differentiable Primitive for Artificial Consciousness.
    arxiv:2506.13825. https://arxiv.org/abs/2506.13825

    Upstream code: https://github.com/ReFractals/RIIU/blob/main/riiu.py
    Upstream license: Apache-2.0 (see third_party/RIIU_LICENSE).

Modifications from upstream:
    - Wrapped `AutoPhiSurrogate.forward` in a class that owns a ring buffer.
    - Added `push`, `compute_value`, `reset`, `is_warm` to match the project's
      step-wise integration pattern in `scripts/training/train_rlhf.py`.
    - Cast to float32 inside the SVD path for numerical stability across
      mixed-precision broadcasts.
    - Added explicit NaN guard at the wrapper level so the training loop sees
      0.0 instead of NaN when SVD fails on degenerate input.
"""
from __future__ import annotations

from collections import deque
from typing import Union

import torch
import torch.nn as nn


class AutoPhiSurrogate(nn.Module):
    """Truncated-SVD residual phi estimator. Vendored from RIIU (Apache-2.0).

    Operates on a batch of samples z of shape [N, D] where N is the window
    length and D is the activation dimension. Returns a scalar tensor.
    """

    def __init__(self, rank: int = 16, eps: float = 1e-5):
        super().__init__()
        self.rank = rank
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.size(0) <= 1:
            return torch.tensor(0.0, device=z.device)

        z_c = z - z.mean(0, keepdim=True)
        cov = (z_c.T @ z_c) / (z_c.size(0) + self.eps)
        cov = cov / (cov.abs().mean() + 1e-9)

        if not torch.isfinite(cov).all() or torch.isnan(cov).any():
            return torch.tensor(0.0, device=z.device)

        try:
            U, _, _ = torch.linalg.svd(cov, full_matrices=False)
        except RuntimeError:
            return torch.tensor(0.0, device=z.device)

        U_r = U[:, : self.rank]
        proj = U_r @ (U_r.T @ cov)
        numer = torch.norm(cov - proj, p="fro")
        denom = torch.norm(cov, p="fro") + 1e-9
        return numer / denom


class RIIUPhi:
    """Sliding-window phi estimator wrapping `AutoPhiSurrogate`.

    The training loop calls `push(z)` once per step with a 1-D activation
    vector. After `rank + 1` pushes (`is_warm` becomes True), `compute_value`
    returns a scalar float phi. The buffer is FIFO with capacity `window`.

    Usage:
        riiu = RIIUPhi(dim=256, rank=16, window=64, device="cpu")
        for step in range(num_steps):
            riiu.push(broadcast.detach())
            if riiu.is_warm:
                phi = riiu.compute_value()

    The buffer stores DETACHED copies of the input. `compute()` returns a
    differentiable scalar tensor when the most recent push retained grad; the
    project's current use is reward-shaping (detached float), so the
    differentiable path exists but is unused for now.
    """

    def __init__(
        self,
        dim: int,
        rank: int = 16,
        window: int = 64,
        eps: float = 1e-8,
        device: Union[str, torch.device] = "cpu",
    ):
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if window <= rank:
            raise ValueError(
                f"window ({window}) must exceed rank ({rank}) for the residual "
                f"to be defined"
            )
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        self.dim = dim
        self.rank = rank
        self.window = window
        self.device = torch.device(device)
        self._surrogate = AutoPhiSurrogate(rank=rank, eps=eps).to(self.device)
        self._buffer: deque[torch.Tensor] = deque(maxlen=window)

    def push(self, z: torch.Tensor) -> None:
        """Append a sample to the sliding window.

        Accepts a 1-D tensor of shape [dim] or a 2-D tensor of shape [B, dim].
        For 2-D input, rows are averaged before appending (one sample per push).
        The stored copy is detached and cast to float32.
        """
        if z.ndim == 2:
            z = z.mean(dim=0)
        if z.ndim != 1:
            raise ValueError(
                f"push expects 1-D [dim] or 2-D [B, dim], got shape {tuple(z.shape)}"
            )
        if z.shape[0] != self.dim:
            raise ValueError(
                f"push expects dim={self.dim}, got {z.shape[0]}"
            )
        self._buffer.append(z.detach().to(self.device, dtype=torch.float32))

    @property
    def is_warm(self) -> bool:
        """True when at least `rank + 1` samples are buffered."""
        return len(self._buffer) >= self.rank + 1

    def compute(self) -> torch.Tensor:
        """Return phi as a scalar tensor. Returns 0.0 if not warm."""
        if not self.is_warm:
            return torch.tensor(0.0, device=self.device)
        z = torch.stack(list(self._buffer), dim=0)
        phi = self._surrogate(z)
        if not torch.isfinite(phi):
            return torch.tensor(0.0, device=self.device)
        return phi

    def compute_value(self) -> float:
        """Return phi as a detached Python float. Returns 0.0 if not warm."""
        return float(self.compute().detach().cpu().item())

    def reset(self) -> None:
        """Clear the sliding window. Call between episodes if cross-episode
        contamination of the covariance estimate must be avoided. For the
        current training loop the window is 64 and episodes are 200 steps, so
        a single window straddles the episode boundary by design; calling
        reset is optional."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

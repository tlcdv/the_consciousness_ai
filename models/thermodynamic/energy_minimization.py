"""
Free energy relaxation for a linear Gaussian generative model.

Model: s ~ N(m, Pi_s^-1) and o | s ~ N(W s, Pi_o^-1). Under the Laplace
approximation the variational free energy of a belief mean mu is, up to constants,

    F(mu) = 1/2 (o - W mu)^T Pi_o (o - W mu) + 1/2 (mu - m)^T Pi_s (mu - m)

The solver integrates the gradient flow d mu/dt = -grad F(mu) with explicit Euler or
Heun steps. The minimum is the exact posterior mean, so the result can be checked in
closed form. This is numerical relaxation in software; it is not a claim about how
any physical substrate minimizes free energy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch


def _check_symmetric(name: str, matrix: torch.Tensor) -> None:
    if not torch.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric")


@dataclass
class LinearGaussianModel:
    """Linear Gaussian generative model with precision (inverse covariance) matrices."""

    weights: torch.Tensor
    prior_mean: torch.Tensor
    prior_precision: torch.Tensor
    obs_precision: torch.Tensor

    def __post_init__(self) -> None:
        _check_symmetric("prior_precision", self.prior_precision)
        _check_symmetric("obs_precision", self.obs_precision)

    def hessian(self) -> torch.Tensor:
        """H = W^T Pi_o W + Pi_s, the constant Hessian of F."""
        W = self.weights
        return W.T @ self.obs_precision @ W + self.prior_precision


@dataclass
class RelaxationTrajectory:
    """Belief means and free energies at every step, including the start (n_steps + 1 rows)."""

    mu: torch.Tensor
    free_energy: torch.Tensor


def free_energy(mu: torch.Tensor, obs: torch.Tensor, model: LinearGaussianModel) -> torch.Tensor:
    """Laplace free energy F(mu), up to constants."""
    obs_error = obs - model.weights @ mu
    prior_error = mu - model.prior_mean
    return 0.5 * obs_error @ model.obs_precision @ obs_error + 0.5 * prior_error @ model.prior_precision @ prior_error


def free_energy_gradient(mu: torch.Tensor, obs: torch.Tensor, model: LinearGaussianModel) -> torch.Tensor:
    """grad F(mu) = -W^T Pi_o (o - W mu) + Pi_s (mu - m)."""
    obs_error = obs - model.weights @ mu
    return -model.weights.T @ model.obs_precision @ obs_error + model.prior_precision @ (mu - model.prior_mean)


def stable_step_bound(model: LinearGaussianModel) -> float:
    """Largest stable step for Euler and Heun on this flow: 2 / lambda_max(H)."""
    return 2.0 / torch.linalg.eigvalsh(model.hessian()).max().item()


def _euler_step(mu, obs, model, step):
    return mu - step * free_energy_gradient(mu, obs, model)


def _heun_step(mu, obs, model, step):
    slope = -free_energy_gradient(mu, obs, model)
    predicted = mu + step * slope
    return mu + 0.5 * step * (slope - free_energy_gradient(predicted, obs, model))


_STEPPERS: Dict[str, Callable] = {"euler": _euler_step, "heun": _heun_step}


def relax_free_energy(
    mu0: torch.Tensor, obs: torch.Tensor, model: LinearGaussianModel, step: float, n_steps: int, method: str
) -> RelaxationTrajectory:
    """Integrate d mu/dt = -grad F(mu) from mu0 for n_steps fixed steps."""
    if method not in _STEPPERS:
        raise ValueError(f"method must be one of {sorted(_STEPPERS)}, got {method!r}")
    bound = stable_step_bound(model)
    if step >= bound:
        raise ValueError(f"step {step} is at or above the stability bound {bound} (2 / lambda_max)")
    mus = [mu0]
    for _ in range(n_steps):
        mus.append(_STEPPERS[method](mus[-1], obs, model, step))
    stacked = torch.stack(mus)
    energies = torch.stack([free_energy(mu, obs, model) for mu in mus])
    return RelaxationTrajectory(mu=stacked, free_energy=energies)

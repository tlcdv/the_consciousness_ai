"""
Tests for the free energy relaxation solver (models/thermodynamic/energy_minimization.py).

The generative model is linear Gaussian: s ~ N(m, Pi_s^-1), o | s ~ N(W s, Pi_o^-1).
Under the Laplace approximation, up to constants,

    F(mu) = 1/2 (o - W mu)^T Pi_o (o - W mu) + 1/2 (mu - m)^T Pi_s (mu - m)

Expected values are closed form, never copied from a run:

  * grad F(mu) = H mu - c, with H = W^T Pi_o W + Pi_s and c = W^T Pi_o o + Pi_s m.
  * The unique minimum is the exact posterior mean mu* = H^-1 c.
  * The gradient flow d mu/dt = -grad F has the exact solution
    mu(t) = mu* + expm(-H t) (mu0 - mu*).
  * Explicit Euler on this flow is stable only for step < 2 / lambda_max(H).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.thermodynamic.energy_minimization import (  # noqa: E402
    LinearGaussianModel,
    free_energy,
    free_energy_gradient,
    relax_free_energy,
    stable_step_bound,
)


def _two_d_model() -> LinearGaussianModel:
    return LinearGaussianModel(
        weights=torch.tensor([[1.0, 0.5], [-0.3, 2.0]], dtype=torch.float64),
        prior_mean=torch.tensor([0.2, -0.4], dtype=torch.float64),
        prior_precision=torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=torch.float64),
        obs_precision=torch.tensor([[4.0, 0.0], [0.0, 0.5]], dtype=torch.float64),
    )


def _hessian_and_target(model: LinearGaussianModel, obs: torch.Tensor):
    W, Po, Ps = model.weights, model.obs_precision, model.prior_precision
    hessian = W.T @ Po @ W + Ps
    target = W.T @ Po @ obs + Ps @ model.prior_mean
    return hessian, target


OBS = torch.tensor([1.5, -0.7], dtype=torch.float64)
MU0 = torch.tensor([3.0, 3.0], dtype=torch.float64)


class TestFreeEnergy:
    def test_gradient_matches_autograd(self):
        model = _two_d_model()
        mu = torch.tensor([0.4, -1.1], dtype=torch.float64, requires_grad=True)

        free_energy(mu, OBS, model).backward()

        assert torch.allclose(free_energy_gradient(mu.detach(), OBS, model), mu.grad, atol=1e-12)

    def test_gradient_vanishes_at_exact_posterior_mean(self):
        model = _two_d_model()
        hessian, target = _hessian_and_target(model, OBS)
        posterior_mean = torch.linalg.solve(hessian, target)

        gradient = free_energy_gradient(posterior_mean, OBS, model)

        assert torch.allclose(gradient, torch.zeros(2, dtype=torch.float64), atol=1e-12)

    def test_asymmetric_precision_is_rejected(self):
        with pytest.raises(ValueError, match="symmetric"):
            LinearGaussianModel(
                weights=torch.eye(2),
                prior_mean=torch.zeros(2),
                prior_precision=torch.tensor([[1.0, 0.2], [0.0, 1.0]]),
                obs_precision=torch.eye(2),
            )


class TestRelaxation:
    @pytest.mark.parametrize("method", ["euler", "heun"])
    def test_converges_to_exact_posterior_mean(self, method):
        model = _two_d_model()
        hessian, target = _hessian_and_target(model, OBS)
        step = 0.5 * stable_step_bound(model)

        trajectory = relax_free_energy(MU0, OBS, model, step, n_steps=2000, method=method)

        assert torch.allclose(trajectory.mu[-1], torch.linalg.solve(hessian, target), atol=1e-8)

    def test_free_energy_never_rises_below_stability_bound(self):
        model = _two_d_model()
        step = 0.5 * stable_step_bound(model)

        trajectory = relax_free_energy(MU0, OBS, model, step, n_steps=200, method="euler")

        assert torch.all(trajectory.free_energy[1:] <= trajectory.free_energy[:-1] + 1e-12)

    def test_heun_tracks_exact_flow_better_than_euler(self):
        model = _two_d_model()
        hessian, target = _hessian_and_target(model, OBS)
        posterior_mean = torch.linalg.solve(hessian, target)
        step, n_steps = 0.2 * stable_step_bound(model), 10
        exact = posterior_mean + torch.matrix_exp(-hessian * step * n_steps) @ (MU0 - posterior_mean)

        euler = relax_free_energy(MU0, OBS, model, step, n_steps, method="euler").mu[-1]
        heun = relax_free_energy(MU0, OBS, model, step, n_steps, method="heun").mu[-1]

        assert (heun - exact).norm() < (euler - exact).norm()

    def test_step_above_stability_bound_is_rejected(self):
        model = _two_d_model()
        with pytest.raises(ValueError, match="stab"):
            relax_free_energy(MU0, OBS, model, 1.01 * stable_step_bound(model), 10, "euler")

    def test_unknown_method_is_rejected(self):
        model = _two_d_model()
        with pytest.raises(ValueError, match="method"):
            relax_free_energy(MU0, OBS, model, 0.01, 10, method="rk4")

    def test_trajectory_records_every_step(self):
        model = _two_d_model()

        trajectory = relax_free_energy(MU0, OBS, model, 0.01, n_steps=7, method="euler")

        assert trajectory.mu.shape == (8, 2)
        assert trajectory.free_energy.shape == (8,)

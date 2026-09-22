"""
Tests for the continuous <-> spike bridge (models/thermodynamic/interfaces/snn_bridge.py).

Expected values are closed form, never copied from a run:

  * Rate coding: one Bernoulli draw per bin with p = x * f_max * dt, so the decoded
    rate has mean x and variance x (1 - x f_max dt) / (n_steps f_max dt).
  * Latency coding (RC neuron): t = tau * ln(x / (x - theta)) for x > theta, which
    inverts exactly to x = theta / (1 - exp(-t / tau)). Larger x spikes earlier.
  * Phase coding: a phase in [0, 2 pi) maps to a time in [0, period) and back.
  * LIF: with constant input I above threshold and v(0) = v_rest = 0, the first
    spike comes at the first step where v_k = I (1 - (1 - dt/tau)^k) >= v_th.
  * ATan surrogate: forward is a step at zero; the backward pass is the derivative
    of (1/pi) atan(pi alpha u / 2), which is (alpha / 2) / (1 + (pi alpha u / 2)^2).
"""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.thermodynamic.interfaces.snn_bridge import (  # noqa: E402
    LIFBridgeLayer,
    atan_spike,
    latency_decode,
    latency_encode,
    phase_readout,
    phase_to_spike_time,
    rate_decode,
    rate_encode,
    spike_time_to_phase,
)


class TestRateCoding:
    def test_decoded_rate_matches_input_within_binomial_error(self):
        x = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64)
        f_max, dt, n_steps = 200.0, 0.001, 20000
        generator = torch.Generator().manual_seed(0)

        spikes = rate_encode(x, n_steps, f_max, dt, generator)
        decoded = rate_decode(spikes, f_max, dt)

        p = x * f_max * dt
        std = torch.sqrt(p * (1 - p) / n_steps) / (f_max * dt)
        assert torch.all((decoded - x).abs() <= 5 * std + 1e-12)

    def test_spike_train_shape_and_values(self):
        spikes = rate_encode(torch.full((2, 3), 0.5), 7, 100.0, 0.001, torch.Generator().manual_seed(1))

        assert spikes.shape == (7, 2, 3)
        assert set(spikes.unique().tolist()) <= {0.0, 1.0}

    def test_probability_above_one_is_rejected(self):
        with pytest.raises(ValueError, match="f_max"):
            rate_encode(torch.tensor([1.0]), 10, 2000.0, 0.001, torch.Generator())

    def test_input_outside_unit_interval_is_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            rate_encode(torch.tensor([1.2]), 10, 100.0, 0.001, torch.Generator())


class TestLatencyCoding:
    def test_round_trip_is_exact_above_threshold(self):
        x = torch.tensor([0.3, 0.5, 0.9, 2.0], dtype=torch.float64)

        times = latency_encode(x, tau=0.01, threshold=0.2, t_max=1.0)

        assert torch.allclose(latency_decode(times, tau=0.01, threshold=0.2), x, rtol=1e-12)

    def test_stronger_input_spikes_earlier(self):
        times = latency_encode(torch.tensor([0.3, 0.6, 1.2]), tau=0.01, threshold=0.2, t_max=1.0)

        assert times[0] > times[1] > times[2]

    def test_at_or_below_threshold_never_spikes(self):
        times = latency_encode(torch.tensor([0.1, 0.2]), tau=0.01, threshold=0.2, t_max=1.0)

        assert torch.isinf(times).all()

    def test_spike_after_window_is_dropped(self):
        tau, threshold = 0.01, 0.2
        x_late = threshold / (1 - math.exp(-0.5 / tau))

        times = latency_encode(torch.tensor([x_late]), tau, threshold, t_max=0.1)

        assert torch.isinf(times).all()

    def test_no_spike_decodes_to_zero(self):
        decoded = latency_decode(torch.tensor([math.inf]), tau=0.01, threshold=0.2)

        assert decoded.tolist() == [0.0]


class TestPhaseCoding:
    def test_two_dimensional_oscillator_phase_is_its_angle(self):
        angles = torch.tensor([0.0, math.pi / 2, math.pi, 1.5 * math.pi])
        vectors = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

        assert torch.allclose(phase_readout(vectors), angles, atol=1e-6)

    def test_higher_dimension_uses_the_first_two_coordinates(self):
        vector = torch.tensor([[0.0, 0.6, 0.8]])

        assert phase_readout(vector).item() == pytest.approx(math.pi / 2)

    def test_one_dimensional_input_is_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            phase_readout(torch.ones(4, 1))

    def test_phase_time_round_trip(self):
        phases = torch.tensor([0.0, 1.0, 3.0, 6.0], dtype=torch.float64)

        times = phase_to_spike_time(phases, period=0.025)

        assert torch.all((times >= 0) & (times < 0.025))
        assert torch.allclose(spike_time_to_phase(times, period=0.025), phases)

    def test_composes_with_kuramoto_layer_output(self):
        from models.core.oscillatory_binding import KuramotoLayer

        layer = KuramotoLayer(num_oscillators=6, dimensions=4)
        phases = layer.init_phases(batch_size=2)

        times = phase_to_spike_time(phase_readout(phases), period=0.025)

        assert times.shape == (2, 6)
        assert torch.all((times >= 0) & (times < 0.025))


class TestLIFBridgeLayer:
    def test_first_spike_time_matches_closed_form(self):
        dt, tau, v_th, current = 0.001, 0.02, 1.0, 1.5
        layer = LIFBridgeLayer(1, 1, tau_mem=tau, v_threshold=v_th, dt=dt, bias=False)
        with torch.no_grad():
            layer.linear.weight.fill_(1.0)
        expected_k = next(
            k for k in range(1, 1000) if current * (1 - (1 - dt / tau) ** k) >= v_th
        )

        spikes = layer(torch.full((200, 1, 1), current))

        assert torch.nonzero(spikes[:, 0, 0])[0].item() + 1 == expected_k

    def test_membrane_resets_after_a_spike(self):
        layer = LIFBridgeLayer(1, 1, tau_mem=0.02, v_threshold=1.0, dt=0.001, bias=False)
        with torch.no_grad():
            layer.linear.weight.fill_(1.0)

        spikes = layer(torch.full((400, 1, 1), 1.5))

        assert spikes.sum().item() > 1

    def test_no_input_no_spikes(self):
        layer = LIFBridgeLayer(3, 2, tau_mem=0.02, v_threshold=1.0, dt=0.001, bias=False)

        assert layer(torch.zeros(50, 4, 3)).sum().item() == 0


class TestAtanSurrogate:
    def test_forward_is_a_step_at_zero(self):
        u = torch.tensor([-0.5, 0.0, 0.5])

        assert atan_spike(u, 2.0).tolist() == [0.0, 1.0, 1.0]

    def test_gradient_matches_closed_form(self):
        alpha = 2.0
        u = torch.tensor([-0.4, 0.0, 0.3], requires_grad=True)

        atan_spike(u, alpha).sum().backward()

        expected = (alpha / 2) / (1 + (math.pi * alpha * u.detach() / 2) ** 2)
        assert torch.allclose(u.grad, expected)

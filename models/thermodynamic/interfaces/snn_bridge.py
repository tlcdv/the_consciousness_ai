"""
Bridge between continuous tensors and spike representations.

Three codes and one spiking layer:

  * Rate coding. One Bernoulli draw per time bin with P(spike) = x * f_max * dt, the
    Poisson process in the limit dt -> 0. Decoding divides the spike count by the
    expected count at x = 1.
  * Latency coding (time to first spike) with the RC neuron form
    t = tau * ln(x / (x - theta)) for x > theta, as in snnTorch spikegen.latency.
    It inverts exactly: x = theta / (1 - exp(-t / tau)). An input at or below
    theta, or one whose spike would fall after t_max, does not spike (time inf).
  * Phase coding. The oscillators of KuramotoLayer are unit vectors on a sphere
    ([..., D]). The phase readout is the angle in the plane of the first two
    coordinates; for D = 2 that is the oscillator's own angle, for D > 2 it is a
    stated projection. A phase maps to a spike time within one reference period.
  * LIFBridgeLayer. A linear map into leaky integrate and fire neurons,
    tau dv/dt = -(v - v_rest) + I, spike and reset at v >= v_threshold, with an
    ATan surrogate gradient. This is the layer the NIR export writes out.
"""
from __future__ import annotations

import math

import torch
from torch import nn

TWO_PI = 2.0 * math.pi


def rate_encode(
    x: torch.Tensor, n_steps: int, f_max_hz: float, dt_s: float, generator: torch.Generator
) -> torch.Tensor:
    """Spike train [n_steps, *x.shape] of 0/1, one Bernoulli draw per bin."""
    if torch.any((x < 0) | (x > 1)):
        raise ValueError("rate coding needs inputs in [0, 1]")
    if f_max_hz * dt_s > 1:
        raise ValueError(f"f_max * dt = {f_max_hz * dt_s} > 1: one bin cannot hold that rate")
    p = (x * f_max_hz * dt_s).expand(n_steps, *x.shape)
    return (torch.rand(p.shape, generator=generator, dtype=p.dtype) < p).to(x.dtype)


def rate_decode(spikes: torch.Tensor, f_max_hz: float, dt_s: float) -> torch.Tensor:
    """Estimate x from a spike train [n_steps, ...]: mean spikes per bin / (f_max * dt)."""
    return spikes.mean(dim=0) / (f_max_hz * dt_s)


def latency_encode(x: torch.Tensor, tau: float, threshold: float, t_max: float) -> torch.Tensor:
    """First-spike times; inf where x <= threshold or the spike would come after t_max."""
    safe = torch.where(x > threshold, x, torch.full_like(x, threshold + 1.0))
    times = tau * torch.log(safe / (safe - threshold))
    no_spike = (x <= threshold) | (times > t_max)
    return torch.where(no_spike, torch.full_like(times, math.inf), times)


def latency_decode(times: torch.Tensor, tau: float, threshold: float) -> torch.Tensor:
    """Invert latency_encode. No spike (inf) decodes to 0, the sub-threshold convention."""
    finite = torch.isfinite(times)
    safe = torch.where(finite, times, torch.ones_like(times))
    decoded = threshold / (1 - torch.exp(-safe / tau))
    return torch.where(finite, decoded, torch.zeros_like(decoded))


def phase_readout(oscillators: torch.Tensor) -> torch.Tensor:
    """Angle in [0, 2 pi) of each oscillator vector [..., D], taken in its first two coordinates."""
    if oscillators.shape[-1] < 2:
        raise ValueError(f"phase readout needs oscillator dimension at least 2, got {oscillators.shape[-1]}")
    angles = torch.atan2(oscillators[..., 1], oscillators[..., 0])
    return torch.remainder(angles, TWO_PI)


def phase_to_spike_time(phases: torch.Tensor, period: float) -> torch.Tensor:
    """Spike time in [0, period) aligned to a reference oscillation cycle."""
    return torch.remainder(phases, TWO_PI) / TWO_PI * period


def spike_time_to_phase(times: torch.Tensor, period: float) -> torch.Tensor:
    """Inverse of phase_to_spike_time for times in [0, period)."""
    return times / period * TWO_PI


class _AtanSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, alpha):
        ctx.save_for_backward(u)
        ctx.alpha = alpha
        return (u >= 0).to(u.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (u,) = ctx.saved_tensors
        slope = (ctx.alpha / 2) / (1 + (math.pi * ctx.alpha * u / 2) ** 2)
        return grad_output * slope, None


def atan_spike(u: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """Step at u = 0 forward; derivative of (1/pi) atan(pi alpha u / 2) backward."""
    return _AtanSpike.apply(u, alpha)


class LIFBridgeLayer(nn.Module):
    """Linear map into leaky integrate and fire neurons, stepped with explicit Euler."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau_mem: float,
        v_threshold: float,
        dt: float,
        v_rest: float = 0.0,
        v_reset: float = 0.0,
        bias: bool = True,
        surrogate_alpha: float = 2.0,
    ):
        super().__init__()
        if dt >= tau_mem:
            raise ValueError(f"dt {dt} must be smaller than tau_mem {tau_mem} for a stable Euler step")
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.tau_mem, self.v_threshold, self.dt = tau_mem, v_threshold, dt
        self.v_rest, self.v_reset, self.surrogate_alpha = v_rest, v_reset, surrogate_alpha

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs [T, B, in_features] -> spikes [T, B, out_features]."""
        v = torch.full((inputs.shape[1], self.linear.out_features), self.v_rest, dtype=inputs.dtype)
        spikes = []
        for current in self.linear(inputs):
            v = v + self.dt / self.tau_mem * (-(v - self.v_rest) + current)
            spike = atan_spike(v - self.v_threshold, self.surrogate_alpha)
            v = torch.where(spike > 0, torch.full_like(v, self.v_reset), v)
            spikes.append(spike)
        return torch.stack(spikes)

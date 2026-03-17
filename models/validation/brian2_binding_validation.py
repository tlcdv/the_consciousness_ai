"""
Brian2 Biological Validation for AKOrN Oscillatory Binding

Translates AKOrN (Artificial Kuramoto Oscillatory Neurons) parameters into
a Brian2 spiking neural network with equivalent Kuramoto dynamics. Compares
synchronization curves from both simulators to validate that the PyTorch
implementation faithfully reproduces biologically plausible oscillator
synchronization.

Brian2 is an optional dependency. All public functions degrade gracefully
when it is not installed.

References:
    - Löwe et al., "Artificial Kuramoto Oscillatory Neurons", ICLR 2025
    - Stimberg et al., "Brian 2, an intuitive and efficient neural simulator",
      eLife 2019
    - Kuramoto, "Chemical Oscillations, Waves, and Turbulence", Springer 1984
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch

try:
    import brian2
    from brian2 import (
        NeuronGroup, Synapses, StateMonitor,
        ms, second, radian, Hz,
        defaultclock, seed as brian_seed,
    )
    BRIAN2_AVAILABLE = True
except ImportError:
    BRIAN2_AVAILABLE = False

from models.core.oscillatory_binding import KuramotoLayer


@dataclass
class SyncCurve:
    """Time series of the Kuramoto order parameter R."""
    times: np.ndarray          # shape [T], time points in seconds
    order_parameter: np.ndarray  # shape [T], R values in [0, 1]
    phases: np.ndarray         # shape [T, N], oscillator phases at each time
    source: str                # "akorn" or "brian2"


@dataclass
class ValidationResult:
    """Comparison result between AKOrN and Brian2 synchronization curves."""
    akorn_curve: SyncCurve
    brian2_curve: SyncCurve | None
    correlation: float                # Pearson correlation of R curves
    max_deviation: float              # max |R_akorn - R_brian2|
    mean_deviation: float             # mean |R_akorn - R_brian2|
    final_r_akorn: float
    final_r_brian2: float | None
    passed: bool                      # True if correlation > threshold
    method: str                       # "brian2" or "skipped"
    message: str = ""


def _extract_akorn_natural_frequencies(kuramoto: KuramotoLayer) -> np.ndarray:
    """
    Extract effective natural frequencies from the AKOrN skew-symmetric
    frequency matrices. For 2D oscillators (phase on circle), the angular
    velocity equals the off-diagonal element of the skew-symmetric matrix.
    For higher dimensions, we take the Frobenius norm as a scalar proxy.
    """
    omega_matrices = kuramoto.natural_frequencies.detach().cpu().numpy()
    # omega_matrices shape: [N, D, D]
    N, D, _ = omega_matrices.shape

    if D == 2:
        # 2D: skew-symmetric [[0, -w], [w, 0]], natural freq = w
        freqs = omega_matrices[:, 1, 0]
    else:
        # Higher D: use Frobenius norm / sqrt(2) as angular speed proxy
        freqs = np.array([
            np.linalg.norm(omega_matrices[i]) / math.sqrt(2)
            for i in range(N)
        ])
    return freqs


def _extract_akorn_coupling_matrix(kuramoto: KuramotoLayer) -> np.ndarray:
    """Extract the learnable coupling weight matrix from AKOrN."""
    return kuramoto.coupling_weights.detach().cpu().numpy()


def run_akorn_simulation(
    kuramoto: KuramotoLayer,
    initial_phases: np.ndarray,
    amplitudes: np.ndarray,
    total_steps: int = 100,
) -> SyncCurve:
    """
    Run AKOrN forward pass step by step, recording R at each step.

    Args:
        kuramoto: The KuramotoLayer instance (parameters define the network)
        initial_phases: [N, D] array of initial phases on the unit sphere
        amplitudes: [N] array of oscillator amplitudes (bid strengths)
        total_steps: Number of discrete integration steps

    Returns:
        SyncCurve with per-step order parameter and phases
    """
    device = kuramoto.coupling_weights.device
    N = kuramoto.num_oscillators
    D = kuramoto.dimensions
    dt = kuramoto.dt

    phases_t = torch.tensor(initial_phases, dtype=torch.float32, device=device).unsqueeze(0)
    amps_t = torch.tensor(amplitudes, dtype=torch.float32, device=device).unsqueeze(0)

    times = []
    r_values = []
    all_phases = []

    for step in range(total_steps):
        t = step * dt
        times.append(t)

        # Record current state
        with torch.no_grad():
            mean_field = torch.mean(phases_t, dim=1)  # [1, D]
            R = torch.norm(mean_field, p=2, dim=-1).item()
        r_values.append(R)

        if D == 2:
            phase_angles = torch.atan2(phases_t[0, :, 1], phases_t[0, :, 0]).cpu().numpy()
        else:
            phase_angles = phases_t[0, :, 0].cpu().numpy()
        all_phases.append(phase_angles)

        # Single integration step
        with torch.no_grad():
            phases_t, _ = kuramoto(phases_t, amplitudes=amps_t, iterations=1)

    return SyncCurve(
        times=np.array(times),
        order_parameter=np.array(r_values),
        phases=np.array(all_phases),
        source="akorn",
    )


def build_brian2_network(
    num_oscillators: int,
    natural_frequencies: np.ndarray,
    coupling_strength: float,
    coupling_matrix: np.ndarray,
    amplitudes: np.ndarray,
    initial_phases: np.ndarray,
    dt_ms: float = 1.0,
) -> tuple:
    """
    Build a Brian2 network with Kuramoto dynamics equivalent to AKOrN.

    The Brian2 model uses NeuronGroup with the standard Kuramoto equation:
        dTheta/dt = omega_i + (K/N) * sum_j [w_ij * a_j * sin(Theta_j - Theta_i)]

    where:
        omega_i = natural frequency of oscillator i
        K = global coupling strength
        w_ij = pairwise coupling weight (from AKOrN's coupling_weights)
        a_j = amplitude of oscillator j (from AKOrN's bid amplitudes)
        Theta_i = phase of oscillator i

    Args:
        num_oscillators: Number of oscillators N
        natural_frequencies: [N] array of angular velocities (rad/s)
        coupling_strength: Global K parameter
        coupling_matrix: [N, N] pairwise weights
        amplitudes: [N] bid amplitudes
        initial_phases: [N] initial phases in radians
        dt_ms: Integration timestep in milliseconds

    Returns:
        tuple of (NeuronGroup, Synapses, StateMonitor) Brian2 objects

    Raises:
        RuntimeError: If Brian2 is not installed
    """
    if not BRIAN2_AVAILABLE:
        raise RuntimeError(
            "Brian2 is not installed. Install with: pip install brian2"
        )

    N = num_oscillators

    defaultclock.dt = dt_ms * ms

    eqs = '''
    dTheta/dt = omega + (K_val/N_val)*coupling : radian
    omega : radian/second (constant)
    coupling : radian/second
    amp : 1 (constant)
    K_val : 1/second (constant)
    N_val : 1 (constant)
    '''

    oscillators = NeuronGroup(N, eqs, method='euler')
    oscillators.Theta = initial_phases * radian
    oscillators.omega = natural_frequencies * radian / second
    oscillators.amp = amplitudes
    oscillators.K_val = coupling_strength / second
    oscillators.N_val = N

    # Synapses with weighted Kuramoto coupling
    syn_eqs = 'w : 1 (constant)'
    syn_on_pre = ''
    coupling_code = 'coupling_post = w * amp_pre * sin(Theta_pre - Theta_post) / second : radian/second (summed)'

    connections = Synapses(
        oscillators, oscillators,
        model='w : 1 (constant)',
        on_pre=None,
    )
    # Replace with summed variable approach
    connections = Synapses(
        oscillators, oscillators,
        coupling_code,
    )
    connections.connect()

    # Set pairwise weights from AKOrN coupling matrix
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            connections.w[idx] = coupling_matrix[i, j]

    mon = StateMonitor(oscillators, 'Theta', record=True)

    return oscillators, connections, mon


def run_brian2_simulation(
    num_oscillators: int,
    natural_frequencies: np.ndarray,
    coupling_strength: float,
    coupling_matrix: np.ndarray,
    amplitudes: np.ndarray,
    initial_phases: np.ndarray,
    duration_seconds: float = 1.0,
    dt_ms: float = 1.0,
    random_seed: int = 42,
) -> SyncCurve:
    """
    Run a Brian2 Kuramoto simulation and return the synchronization curve.

    Args:
        num_oscillators: Number of oscillators
        natural_frequencies: [N] natural angular velocities
        coupling_strength: Global K
        coupling_matrix: [N, N] pairwise weights
        amplitudes: [N] bid amplitudes
        initial_phases: [N] initial phase angles (radians)
        duration_seconds: Total simulation time
        dt_ms: Timestep in milliseconds
        random_seed: Brian2 random seed

    Returns:
        SyncCurve from Brian2 simulation

    Raises:
        RuntimeError: If Brian2 is not installed
    """
    if not BRIAN2_AVAILABLE:
        raise RuntimeError(
            "Brian2 is not installed. Install with: pip install brian2"
        )

    brian_seed(random_seed)

    oscillators, connections, mon = build_brian2_network(
        num_oscillators=num_oscillators,
        natural_frequencies=natural_frequencies,
        coupling_strength=coupling_strength,
        coupling_matrix=coupling_matrix,
        amplitudes=amplitudes,
        initial_phases=initial_phases,
        dt_ms=dt_ms,
    )

    brian2.run(duration_seconds * second)

    # Extract phases and compute order parameter at each timestep
    theta_all = mon.Theta[:]  # shape [N, T]
    times_brian = mon.t[:] / second  # convert to plain seconds
    T = theta_all.shape[1]
    N = num_oscillators

    r_values = np.zeros(T)
    for t_idx in range(T):
        phases_t = theta_all[:, t_idx]
        x = np.cos(phases_t)
        y = np.sin(phases_t)
        r_values[t_idx] = np.sqrt(np.mean(x)**2 + np.mean(y)**2)

    # Phase snapshots
    all_phases = theta_all.T  # [T, N]

    return SyncCurve(
        times=np.array(times_brian),
        order_parameter=r_values,
        phases=np.array(all_phases),
        source="brian2",
    )


def _interpolate_to_common_times(
    curve_a: SyncCurve,
    curve_b: SyncCurve,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate both curves to a common time grid for comparison.
    Uses the curve with fewer points as the reference grid.
    """
    if len(curve_a.times) <= len(curve_b.times):
        common_times = curve_a.times
        r_a = curve_a.order_parameter
        r_b = np.interp(common_times, curve_b.times, curve_b.order_parameter)
    else:
        common_times = curve_b.times
        r_a = np.interp(common_times, curve_a.times, curve_a.order_parameter)
        r_b = curve_b.order_parameter
    return common_times, r_a, r_b


def validate_binding(
    kuramoto: KuramotoLayer,
    amplitudes: np.ndarray | None = None,
    total_steps: int = 100,
    brian2_duration: float = 1.0,
    brian2_dt_ms: float = 1.0,
    correlation_threshold: float = 0.85,
    random_seed: int = 42,
) -> ValidationResult:
    """
    Run both AKOrN and Brian2 simulations with equivalent parameters and
    compare their synchronization dynamics.

    The validation passes if the Pearson correlation between the two
    order parameter curves exceeds correlation_threshold.

    Args:
        kuramoto: Configured KuramotoLayer instance
        amplitudes: [N] bid amplitudes (default: uniform ones)
        total_steps: AKOrN integration steps
        brian2_duration: Brian2 simulation duration in seconds
        brian2_dt_ms: Brian2 timestep in ms
        correlation_threshold: Minimum acceptable correlation
        random_seed: Seed for reproducible initial conditions

    Returns:
        ValidationResult with comparison metrics
    """
    N = kuramoto.num_oscillators
    D = kuramoto.dimensions

    if amplitudes is None:
        amplitudes = np.ones(N)

    # Generate shared initial conditions
    rng = np.random.RandomState(random_seed)
    if D == 2:
        # Random angles on the unit circle
        angles = rng.uniform(0, 2 * math.pi, size=N)
        init_phases_nd = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # [N, 2]
        init_phases_1d = angles
    else:
        # Random points on N-sphere
        init_phases_nd = rng.randn(N, D)
        norms = np.linalg.norm(init_phases_nd, axis=-1, keepdims=True)
        init_phases_nd = init_phases_nd / norms
        # Project to 1D for Brian2 (use first two components as angle)
        init_phases_1d = np.arctan2(init_phases_nd[:, 1], init_phases_nd[:, 0])

    # Extract AKOrN parameters
    natural_freqs = _extract_akorn_natural_frequencies(kuramoto)
    coupling_matrix = _extract_akorn_coupling_matrix(kuramoto)
    K = kuramoto.K if isinstance(kuramoto.K, float) else float(kuramoto.K)

    # --- Run AKOrN ---
    akorn_curve = run_akorn_simulation(
        kuramoto=kuramoto,
        initial_phases=init_phases_nd,
        amplitudes=amplitudes,
        total_steps=total_steps,
    )

    # --- Run Brian2 (if available) ---
    if not BRIAN2_AVAILABLE:
        warnings.warn(
            "Brian2 not installed. Skipping biological validation. "
            "Install with: pip install brian2",
            UserWarning,
        )
        return ValidationResult(
            akorn_curve=akorn_curve,
            brian2_curve=None,
            correlation=float('nan'),
            max_deviation=float('nan'),
            mean_deviation=float('nan'),
            final_r_akorn=akorn_curve.order_parameter[-1],
            final_r_brian2=None,
            passed=False,
            method="skipped",
            message="Brian2 not installed",
        )

    brian2_curve = run_brian2_simulation(
        num_oscillators=N,
        natural_frequencies=natural_freqs,
        coupling_strength=K,
        coupling_matrix=coupling_matrix,
        amplitudes=amplitudes,
        initial_phases=init_phases_1d,
        duration_seconds=brian2_duration,
        dt_ms=brian2_dt_ms,
        random_seed=random_seed,
    )

    # --- Compare ---
    common_times, r_akorn, r_brian2 = _interpolate_to_common_times(
        akorn_curve, brian2_curve
    )

    deviation = np.abs(r_akorn - r_brian2)
    max_dev = float(np.max(deviation))
    mean_dev = float(np.mean(deviation))

    # Pearson correlation
    if np.std(r_akorn) < 1e-10 or np.std(r_brian2) < 1e-10:
        # One curve is flat, correlation undefined
        corr = 1.0 if max_dev < 0.1 else 0.0
    else:
        corr = float(np.corrcoef(r_akorn, r_brian2)[0, 1])

    passed = corr >= correlation_threshold

    return ValidationResult(
        akorn_curve=akorn_curve,
        brian2_curve=brian2_curve,
        correlation=corr,
        max_deviation=max_dev,
        mean_deviation=mean_dev,
        final_r_akorn=akorn_curve.order_parameter[-1],
        final_r_brian2=brian2_curve.order_parameter[-1],
        passed=passed,
        method="brian2",
        message=f"Correlation={corr:.4f}, threshold={correlation_threshold}",
    )


def translate_akorn_params(kuramoto: KuramotoLayer) -> dict:
    """
    Extract all AKOrN parameters in a plain dict for inspection or
    manual Brian2 network construction.

    Returns:
        dict with keys: num_oscillators, dimensions, coupling_strength,
        natural_frequencies, coupling_matrix, dt
    """
    return {
        "num_oscillators": kuramoto.num_oscillators,
        "dimensions": kuramoto.dimensions,
        "coupling_strength": kuramoto.K if isinstance(kuramoto.K, float) else float(kuramoto.K),
        "natural_frequencies": _extract_akorn_natural_frequencies(kuramoto),
        "coupling_matrix": _extract_akorn_coupling_matrix(kuramoto),
        "dt": kuramoto.dt,
    }

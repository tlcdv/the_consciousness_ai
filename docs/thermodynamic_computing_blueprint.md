# Thermodynamic computing blueprint

This document describes the experimental package `models/thermodynamic/`, the maths it
implements, the hardware it could map to, and the protocol for one question.

**The question.** Do the verified vector representations of this architecture
(`obs_map -> z_state -> tectum_content`) survive physical stochastic transduction?
That is the substrate independence thesis. Nothing here claims that thermodynamic
computing produces consciousness, raises IIT Phi, or improves any indicator.

**Status.** Software emulation on a CPU. No physical device has been used. All three
metrics defined below are UNPROVEN instruments in `docs/instrument_inventory.md`, and
0 instruments are TRUSTED. No result from this package is reported. A single seed is a
hypothesis, and a comparative claim needs at least 3 seeds.

**Isolation.** Production code never imports `models.thermodynamic`. The test
`tests/test_substrate_isolation.py` checks this on the import graph. Every optional
library (`thrml`, `nir`, `nirtorch`, `brian2`) is imported on call, skipped in tests when
missing, and kept out of `requirements.txt`. Any experimental telemetry goes behind a
default-off flag, and the baseline stays bit-identical when the flag is off.

## 1. Mathematical formalisms

Each formula below is implemented in the file named beside it.

### 1.1 Poisson rate code (`interfaces/snn_bridge.py`)

For an input x in [0, 1], each time bin of width dt holds one Bernoulli draw with

    P(spike) = x * f_max * dt

This is the Poisson process in the limit dt -> 0. The decoder is the mean spike count
per bin divided by f_max * dt. It is unbiased, and its variance falls as 1 / T for T bins.
The product f_max * dt must not exceed 1.

### 1.2 Time to first spike with an RC neuron (`interfaces/snn_bridge.py`)

For a neuron with time constant tau, threshold theta, and constant input x > theta, the
membrane voltage crosses theta at

    t = tau * ln( x / (x - theta) )

The inverse is exact:

    x = theta / ( 1 - exp(-t / tau) )

An input at or below theta, or a spike later than t_max, does not spike. The encoder
returns infinity and the decoder returns 0 for it.

### 1.3 Phase to spike time for AKOrN oscillators (`interfaces/snn_bridge.py`)

The oscillators of `KuramotoLayer` are unit vectors. The phase readout is the angle of
the first two coordinates, `atan2(o_1, o_0)` mapped to [0, 2 pi). For dimension 2 this
is the oscillator angle. For larger dimensions it is a stated projection and not the
full state. A phase maps to a spike time inside one reference period P:

    t = phase / (2 pi) * P

and the inverse is phase = 2 pi * t / P.

### 1.4 Leaky integrate and fire layer with a surrogate gradient (`LIFBridgeLayer`)

The layer is a linear map into neurons that follow

    tau * dv/dt = -(v - v_rest) + I

It uses the explicit Euler step v <- v + (dt / tau) * ( -(v - v_rest) + I ), and it
requires dt < tau. A neuron spikes at v >= v_threshold and resets to v_reset. The
forward spike is a step function. The backward pass uses the derivative of the arctangent
surrogate

    d s / d u = (alpha / 2) / ( 1 + (pi * alpha * u / 2)^2 ),  u = v - v_threshold

`tests/test_brian2_matched_validation.py` compares the spike trains of this layer with
Brian2. At the same step size the trains are identical. Against a Brian2 run at a step
10 times smaller, spike counts agree within 1 spike plus 5 percent.

### 1.5 P-bit update and chromatic block Gibbs sampling (`p_bit_emulator.py`)

A p-bit m_i in {-1, +1} follows the rule of Camsari et al. (Physical Review X 7, 031014,
2017):

    m_i = sgn( tanh(beta * h_i) - r_i ),  h_i = b_i + sum_j J_ij m_j,  r_i ~ U(-1, 1)

so P(m_i = +1) = (1 + tanh(beta * h_i)) / 2. The Hamiltonian is

    H(m) = - sum_{i<j} J_ij m_i m_j - sum_i b_i m_i

and the stationary distribution is P(m) proportional to exp(-beta * H(m)). Two coupled
p-bits must not update at the same time, because a simultaneous update does not sample
this distribution. The sampler colors the coupling graph and updates one color class per
step. One sweep updates each class once. For n <= 20 spins, `exact_boltzmann` enumerates
all states, and the tests compare sampled statistics with it. The parameter beta is a
dimensionless gain. It is not the physical temperature of any chip.

### 1.6 Laplace free energy relaxation (`energy_minimization.py`)

For the model s ~ N(m, Pi_s^-1) and o | s ~ N(W s, Pi_o^-1), the Laplace free energy of a
belief mean mu is, up to constants,

    F(mu) = 1/2 (o - W mu)^T Pi_o (o - W mu) + 1/2 (mu - m)^T Pi_s (mu - m)

The solver integrates d mu / dt = -grad F(mu) with Euler or Heun steps. The Hessian is
H = W^T Pi_o W + Pi_s, and both integrators are stable below a step of 2 / lambda_max(H).
The fixed point is the exact posterior mean (W^T Pi_o W + Pi_s)^-1 (W^T Pi_o o + Pi_s m),
and the tests check against it. This is numerical relaxation in software. It is not a
claim about how a physical device minimizes free energy.

### 1.7 Metrics (all UNPROVEN)

| Name | Definition | Computed by |
|---|---|---|
| `pbit_energy` | H(m) of section 1.5 | `p_bit_emulator.ising_energy` |
| `thermodynamic_entropy` | Heat part of entropy production per sweep, beta * (E_t - E_{t+1}), in units of k_B. It sums to beta * (E_0 - E_T). It is not the total entropy production. | `scripts/analysis/probe_thermodynamic_cost.py` |
| `fep_free_energy` | F(mu) of section 1.6 | `energy_minimization.free_energy` |

## 2. Hardware mapping bounds

This section states what each target needs from the software. Hardware performance
figures are not stated here, because none was measured in this project and none has been
checked against a primary source for this document. Check a vendor source before
citing any figure.

| Target | What maps | What does not map |
|---|---|---|
| Extropic Z1 (CMOS p-bits) | Section 1.5 maps directly. The emulator uses a 16 neighbor grid as a stand-in, because the public material gives the degree and not the layout. The optional adapter `interfaces/thrml_adapter.py` wraps Extropic's open source `thrml` library, which runs on JAX. | Dense couplings. A representation of dimension D needs a sparse graph, so a dense Hebbian coupling matrix must be pruned or embedded. The profiler uses dense couplings, so its settling results do not carry over until that step is tested. |
| Normal Computing (SPU, thermodynamic linear algebra) | Section 1.6 maps to a linear solve, because the posterior mean is the solution of H mu = W^T Pi_o o + Pi_s m. A device that relaxes a linear system to equilibrium would produce it. | Anything nonlinear. The Laplace model here is linear Gaussian. |
| SpiNNaker 2 and Loihi 2 | `LIFBridgeLayer` exports to the Neuromorphic Intermediate Representation (Pedersen et al., Nature Communications 15, 8122, 2024) as input, affine, LIF, output. Toolchain support for a given chip must be checked at the time of use. | `KuramotoLayer`, because NIR has no oscillator primitive. Phase coding needs a hand mapping. |

## 3. Substrate independence protocol

The cost profiler `scripts/analysis/probe_thermodynamic_cost.py` ran four pass/fail gates (2026-10-10). They gave
UNTESTABLE, FAILED, a PASS and a failed replication of that PASS
(`docs/results/thermodynamic_transduction_2026_10.md` and the three documents that follow it). All four used a ridge readout
with an untuned penalty, which understated the class information. Those verdicts are records of that probe. They are not
statements about the representations. The gate series is superseded by
`scripts/analysis/probe_thermodynamic_fidelity.py`, which measures a dose response and not a pass line
(`docs/results/thermodynamic_fidelity_2026_10.md`). Its findings are these. All four streams carry the class when the readout is
tuned. The budget needed through a stochastic spike channel depends on the stream and on the model. The p-bit channel needs
fewer events than the Poisson channel. The cost is set by where the class information sits in the variance spectrum
of the representation. Survival at infinite budget follows from an unbiased decoder and is not a finding.

**Stages (dose-response protocol).**

1. Record vectors from trained checkpoints, on several stimulus draws per checkpoint. The probe records `tectum_content`, the
   workspace broadcast, `obs_map` and `z_state`. The last two are pooled 4 by 4 spatially. Vectors larger than 256 dimensions
   are reduced by training-fold PCA to 256 components, and all vectors are z-scored on the training rows.
2. Fit the reference readout, a ridge readout of the stimulus class on x = sigmoid(z), with the penalty chosen by nested cross
   validation grouped by trial. A fixed untuned penalty understated the class information in every stream and must not be used.
3. Send held-out vectors through a stochastic channel at a ladder of event budgets T per dimension. The channels are a Poisson
   rate code and independent p-bits. Decode, then apply the same readout fitted on clean vectors.
4. Report retained information rho(T) = (A_T - 1/6) / (A_clean - 1/6) with a trial bootstrap interval, and the budgets T90 and T95
   at which it reaches 0.90 and 0.95. Report per checkpoint. Do not pool checkpoints.
5. Pre-state what is reported before the data run, and keep pass lines out of it. The earlier pass/fail gates were unstable at the
   margins they used.

**Controls and checks.**

- Shuffled labels must bring the reference readout to chance. The permutation null was used in the earlier gates.
- An unbiased decoder gives rho -> 1 as T grows, so survival at infinite budget is not a finding. The finding is the budget.
- A sensitivity check on the readout penalty is needed. At model 44, `z_state` and `tectum_content` are readable only with a nearly
  unregularised readout. Their class information sits in low-variance directions, which a white-noise channel removes first.
- The Poisson spike probability scale (0.2) and the sigmoid map are assumptions. State them with every result.

**Reading rules.**

- A drop in `pbit_energy` shows only that a sampler descended. It says nothing about the stimulus.
- `fep_free_energy` falls for any fitted model, so a fall is not evidence about the architecture.
- Energy estimates use assumed constants. They are assumptions and not measurements. No energy per event has been measured here.
- A result about transduction does not change any rubric entry or the instrument count.

**Variance-equalised encoding (2026-10-10).** Rotating each vector into its training principal axes and scaling the axes before the
channel cut the events needed to reach a stated accuracy by 16 to 256 times where both encodings reach it, at equal numbers of
dimensions (`docs/results/thermodynamic_encoding_2026_10.md`). The saving stayed the same when the readout was trained on the channel's
own noisy outputs (`docs/results/thermodynamic_matched_readout_2026_10.md`). It is still an exploratory finding. The cost of the transform is
not counted, and the encoded vectors of `workspace_broadcast` have a lower clean ceiling.

**Cost of the transform (2026-10-10).** Counting the multiply-adds of the pre-processing changes the picture
(`docs/results/thermodynamic_transform_cost_2026_10.md`). For the raw 256-D streams the dense rotation pays for itself only if one
multiply-add costs less than about 60 channel events. A sourced reference scenario (a 45 nm digital multiply-add of about 4.6 pJ, against a
modelled cell energy of about 2 fJ) gives a ratio near 2300, where it does not pay. For the larger streams the scaling folds into the PCA
matrix at no cost, but that projection dominates the total energy. So the equalisation is a lever on fidelity per channel event, and a lever
on total energy only where the pre-processing is cheap.

**Open experiments, in order.** Rank-truncated equalisation for the raw streams. A sweep of the
equalisation strength on held-out draws. A noisy attractor memory across the delay phase. The policy in the loop.

## 4. Files

| Path | Content |
|---|---|
| `models/thermodynamic/p_bit_emulator.py` | Section 1.5 |
| `models/thermodynamic/energy_minimization.py` | Section 1.6 |
| `models/thermodynamic/interfaces/snn_bridge.py` | Sections 1.1 to 1.4 |
| `models/thermodynamic/interfaces/nir_export.py` | NIR export of the LIF layer |
| `models/thermodynamic/interfaces/thrml_adapter.py` | Optional wrapper for `thrml` |
| `models/thermodynamic/interfaces/hardware_stub.py` | Deterministic stand-in driver for tests |
| `models/validation/brian2_binding_validation.py` | Spike train check against Brian2 (optional dependency) |
| `scripts/analysis/probe_thermodynamic_cost.py` | Offline cost profile with the superseded pass/fail gates |
| `scripts/analysis/probe_thermodynamic_fidelity.py` | Dose-response probe for the stochastic channel |
| `scripts/analysis/probe_thermodynamic_encoding.py` | Variance-equalised encoding arms on the cached recordings |
| `scripts/analysis/probe_thermodynamic_matched.py` | The same arms with a readout trained on channel outputs |
| `scripts/analysis/probe_thermodynamic_transform_cost.py` | Operation counts of the transform and the break-even energy ratio |
| `tests/test_substrate_isolation.py` | Import graph guard |

# Thalamic-gating instruments: integration record (2026-07-28)

**This is an instrument-integration record, not a scientific verdict.** It follows the
precedent of `ce2_svd_integration_2026_07.md`: the wiring is verified, the measures are
validated on constructed inputs, and no claim is made about the agent. Every number below
comes from an UNTRAINED tectum on a SINGLE seed and is a smoke test. None of it may be
cited as a result.

Motivation, sources and the alignment audit: `docs/thalamic_gating_evidence.md`.

## What was built

| Component | Path | Status |
|---|---|---|
| PCI_LZ measure | `models/evaluation/perturbational_complexity.py` | new, 26 tests |
| Coupling measures (PLV, phase TE, PAC) | `models/evaluation/coupling_measures.py` | new, 26 tests |
| Near-threshold DMTS knob | `simulations/environments/dmts_env.py` | new params, 16 tests |
| PCI probe | `scripts/analysis/probe_pci.py` | new, runs end to end |
| Ordering probe | `scripts/analysis/probe_workspace_ordering.py` | new, runs end to end |
| PCI placeholder | `models/evaluation/consciousness_metrics.py` | RETIRED |

The retirement matters on its own. `PerturbationTester.calculate_pci_approximation`
returned `np.random.rand() * 10.0` from a method named after a metric. It now raises with a
pointer to the real implementation. Both live call sites already caught exceptions; the
third (`consciousness_monitor.py`) was guarded to record unavailability rather than a
number.

## Two deviations from the published methods, both deliberate

**1. PCI normalization.** Casali et al. divide the Lempel-Ziv count by `L*H/log2(L)`, the
asymptotic complexity of a random source at the OBSERVED activity level. That term goes to
zero as the response gets sparse, so the ratio diverges. Measured during development on a
32-channel, 64-step response with 4 significant entries out of 2048, it returns **1.32**,
ranking a dying local response ABOVE a fully differentiated one. This module normalizes by
`L/log2(L)` instead, the maximum-entropy asymptote for the matrix SIZE, and reports
`pci_casali` alongside so the published quantity stays recoverable. The inversion is pinned
by a regression test (`test_casali_normalization_diverges_on_sparse_responses`) so that if
it ever stops holding, the deviation gets revisited rather than silently kept.

The consequence is stated rather than hidden: `pci` retains a dependence on how much of the
substrate responded. That is wanted here. Casali divides activity level out because in
TMS/EEG the number of significant sources is confounded by stimulator intensity and
electrode montage; here the perturbation magnitude and channel set are fixed by the caller
across every comparison, so the confound is absent and spatial spread is the integration
half of the measure.

**2. No frequency units.** The agent has no millisecond clock. Every band in
`coupling_measures.py` is in cycles per step and carries no Hz reading. The published bands
(28 Hz, 11-17 Hz, 2-8 Hz) are NOT translated, and a test asserts the module docstring still
carries that warning so a refactor cannot quietly drop it.

## Causal response obtained without snapshot/restore

`probe_pci.py` runs two rollouts from the same seed with the same fixed action stream, one
clean and one with a single impulse. Determinism makes them bit-identical up to the impulse,
so everything after it is the causal effect and nothing else. The probe asserts this every
run: measured pre-impulse divergence was **0.000e+00** in every trial reported below. If it
were not, the probe prints FAILED and refuses the numbers.

The action stream is fixed on purpose. If the policy were allowed to respond, the impulse
would change the observations and the measurement would be of behaviour rather than of the
internal causal response.

## Smoke-test observations (UNTRAINED tectum, single seed, NOT results)

Configuration: `--env dmts --seed 42 --trials 3 --perturb-step 20 --response-window 40
--magnitude 20.0`, perturbing the RSSM recurrent state.

| read site | default config (discrete / final) | fixed config (continuous / all_levels) |
|---|---|---|
| rssm (control) | pci 0.0619, sd 0.0000 | pci 0.0590, sd 0.0042 |
| gate (primary) | pci 0.0000 | pci 0.0000 |
| broadcast (exploratory) | pci 0.0000 | pci 0.0000 |

The control site is what makes those zeros readable. A separate diagnostic confirmed the
impulse propagates: perturbing `h_state` produced downstream `h_state` differences decaying
18.62, 14.68, 11.57, 7.77, 4.64, 2.56, 1.28, 0.68 over eight steps, while `tectum_content`
moved by at most **3.7e-09**, which is float32 noise. The perturbation propagates through
the recurrence and dies before the workspace.

**This is a smoke test and not evidence about the architecture,** because on an untrained
tectum the workspace content is constant to float32 epsilon in both configurations
(measured `tectum_content` temporal std 1.17e-08). With no baseline fluctuation there is no
scale against which significance can be defined, so a zero is expected regardless of what
the architecture does. The 2026-07 signature ablation that found responding markers used
TRAINED checkpoints. The same probe must be run against `--load-tectum` before any of this
means anything.

`probe_workspace_ordering.py` reports the same precondition explicitly, refusing to print a
coupling number when the signals are flat:

| signal | default config std | fixed config std | usable (fixed) |
|---|---|---|---|
| vision | 0.000e+00 | 1.384e-08 | no |
| broadcast | 7.132e-11 | 5.472e-02 | yes |
| sync_R | 9.765e-03 | 7.337e-03 | yes |
| gate_attention | 6.276e-07 | 6.744e-05 | no |
| gate_coherence | 0.000e+00 | 1.098e-04 | yes |

In the default configuration only `sync_R` varies, so the probe returns "the ordering
question cannot be asked" rather than a number. In the fixed configuration three signals
clear the floor and no pair showed a directional asymmetry beyond the circular-shift null.

One degeneracy to record before anyone reads that table: `broadcast <-> gate_coherence` gave
PLV exactly **1.0000**, because in this harness the gate is a deterministic function of the
broadcast norm. That is the same signal measured twice, not a finding. It does confirm the
PLV estimator behaves at the ceiling.

## Pre-stated gates for the real runs

To be met on TRAINED checkpoints, three seeds (42/43/44), before anything is reported:

- **PCI PASS** = PCI is non-constant across conditions where phi, EI and CE 2.0 are
  bit-identical or floor-pinned. **KILL** = PCI is also constant, in which case the honest
  output is that the agent's causal response is stereotyped at both sites, reported as such
  and not patched.
- **PCI is uninterpretable** whenever the control site reads 0.0. The probe says so in its
  own output rather than leaving it to the reader.
- **Near-threshold PASS** = some `--sample-contrast` value puts the per-trial decodable
  fraction strictly between floor and ceiling. **KILL** = it jumps from chance to ceiling
  with no intermediate band, meaning this agent has no near-threshold regime and the
  paradigm does not transfer.
- **Ordering** has no pass/fail. Asymmetry present would make a thalamic hub redundant;
  absent, it is the gap a hub would fill. Both outcomes feed the Stage 3 decision, which is
  the owner's.

## Deferred, and why

`probe_divergence_onset.py` (the translation of Fang et al. Figures 2D and 2F, measuring
WHERE in the stage chain aware and unaware trials first separate) is **not built**. It needs
trained near-threshold runs to split trials against, and none exist yet: the contrast knob
shipped in this same change. Building it now would mean shipping a probe that cannot be
validated on any available data, which is the accretion pattern this project has paid for
before. It is the natural next step once one near-threshold training run exists.

## Honesty notes

- Every number here is single seed on an untrained network. Nothing is replicated.
- The published PCI scale does not transfer. The human conscious/unconscious cutoff near
  0.31 is meaningless against these values and must never be quoted alongside them.
- No Butlin indicator moves. No consciousness claim is made or implied.
- Two of the three source papers are preprints and are labelled as such wherever cited.

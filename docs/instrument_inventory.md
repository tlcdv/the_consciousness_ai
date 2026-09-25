# Instrument inventory

**Nothing is TRUSTED. Of 17 instrument entries, 0 are trusted, 4 are unproven, 1 is kept
under a narrowed claim, 6 are retired, and 6 are not running.** As of 2026-09-13 PCI is SETTLED and it FAILED the
content clause. Its reading rule exists (rule 8, a 42x to 60x empty gap in the gate
baseline across probe seeds) and the gate on `gate3_s42` does respond, on 81 of 120 probe
seeds. But the response does not depend on the stimulus: eta-squared 0.060960 against a
null MEAN of 0.061989, and the rssm control fails the same test. PCI cannot become
TRUSTED. And PCI does NOT join the project's scalar-reduction pattern: the 5x60 response
MATRIX does not decode the stimulus either (`pci_matrix_content_2026_09.md`), so there is
no content in the perturbational response for a scalar to discard. PCI fails differently
from the other seven, and the count of scalar-reduction cases stays at 7.

This is the inventory the instrument effort has been working toward. Each entry states a
status, the evidence, and what would change it. Every number here was loaded from disk or
from a named verdict document.

Status meanings, from the seven-clause acceptance bar:

- **TRUSTED**: meets all six clauses.
- **UNPROVEN**: could be sound; a required demonstration is missing. Not retired. No
  verdict may rest on it until the demonstration exists.
- **RETIRED**: carries a defect that a demonstration cannot fix.
- **NOT RUNNING**: logs a constant because its module is disabled. No verdict is possible
  either way, and this is not the same as broken.

Last updated 2026-09-12 (PCI row; everything else 2026-09-02). Run evidence is `runs/bcast_s4{2,3,4}`: 3 seeds, 40 episodes,
8000 steps each, dmts, `--enable-audio --enable-mock-semantic`.

## The finding that organizes this inventory

**Every vector representation tested carries the stimulus. Every scalar tested does not.**

| Quantity | Type | Decodes 6-class `sample_shape` |
|---|---|---|
| `obs_map` | vector | yes, ~1.0 |
| `kl_map` | vector | yes, 0.84 / 0.71 / 0.76 |
| 256-D broadcast | vector | yes, 0.76 / 0.69 / 0.77 |
| vision bid, 4 unsupervised reductions | scalar | no |
| sync_R | scalar | no |
| `phi` | scalar | no |
| `broadcast_mag` | scalar | no |

Sources: `b1_continuous_latent_2026_07.md`, `klmap_phase_information_2026_08.md`,
`broadcast_geometry_2026_08.md`, `bid_reduction_candidates_2026_08.md`,
`sync_r_content_2026_09.md`, `scalar_content_2026_09.md`.

The content is present in the architecture. Every point that reduces one of those vectors
to a single number discards it, and every instrument in this inventory reads a single
number. This is an empirical pattern at seven cases. It is not a theorem, and it does not
prove that no scalar could carry the content.

**PCI is the one measured case that does NOT fit this pattern, and it is informative.**
Its 5x60 response MATRIX does not decode the shape either, at 0.2096 against a majority
rate of 0.2099 (`pci_matrix_content_2026_09.md`). So PCI does not fail by reduction. It
fails because a response to a RANDOM impulse reports the dynamics at that moment and not
what the system is representing. That distinction is worth keeping: the pattern above is
about reductions of representations, and a perturbational response is not a
representation.

## Clause 7, content sensitivity, adopted 2026-09-12

The first six clauses do not test whether an instrument's INPUT carries the thing the
instrument claims to measure. Clause 7 does, and it was added because of `phi`.

`phi` shows why that matters. It passes clause 1 (it writes a `phi_method` sentinel), it
passes the strict non-degeneracy bar with 1128 to 1267 distinct values, and it was the one
instrument to satisfy clause 3 by moving under an intervention. The bar as written does
not reject it. It is nonetheless measured contentless at 3 seeds.

**Clause 7, CONTENT SENSITIVITY:** an instrument's input must be shown to carry the
quantity the instrument claims to track, or the instrument is UNPROVEN. The
control already exists and is cheap: eta-squared or decode accuracy against a permutation
null that shuffles labels across trials.

Adopted by the owner 2026-09-12 and written into the acceptance bar. It does NOT require
an instrument to track the stimulus: `phi` does not claim to. It requires the input to be
shown to carry whatever the instrument does claim. A contentless reading alone gives
UNPROVEN; RETIRED needs a contentless reading plus an identified structural cause that no
demonstration can fix.

## Inventory

### Unproven (4), plus 1 kept under a narrowed claim

PCI is listed here because its evidence belongs beside the others. It is NOT unproven:
its content clause is settled negatively and closed. It is kept for the narrower
question it does answer, and its row states that question.

| Instrument | Why unproven | What would change it |
|---|---|---|
| **PCI** `perturbational_complexity.py` <br>**RELABELLED 2026-09-13: a PROPAGATION CHECK, not a content measure** | **What PCI measures on this architecture: whether a perturbation propagates from the recurrent state to a named site, and how complex the resulting response is.** That much it does reliably: the control discriminates at every checkpoint, determinism is 0.000e+00 on every trial, and the gate response replicates on 81 of 120 admitted probe seeds. **What it does NOT measure is content, and that is now permanent, not pending.** It may never be cited as evidence for a consciousness indicator, never against the published human 0.31 scale, and never as a measure of integration. Cite it only for the question it answers: did the impulse arrive. Reading rules 1 to 8 apply to every citation. **FAILED THE CONTENT CLAUSE 2026-09-13** (`pci_content_2026_09.md`). 120 probe seeds on `gate3_s42`, phase held constant at DELAY on all 120 so no result can be a phase reading, 6-class `sample_shape`, 2000-permutation null shuffled across seeds. Gate raw response eta2 **0.060960** against null p95 0.134895 and null MEAN **0.061989**: the measured value is BELOW what random labels give on average. Gate `pci` 0.113845 vs p95 0.131807. The rssm CONTROL fails too (0.078322 vs p95 0.088844), so the failure belongs to the MEASURE and not to the site. Null means match the theoretical (k-1)/(n-1) to three decimals, so the test is calibrated. What works: rule 8 gives the reading rule (gate baseline is bimodal across probe seeds with a 42x to 60x EMPTY gap; the same 4 seeds are noisy on independently trained checkpoints), and the gate on `gate3_s42` responds on 81 of 120 admitted seeds at 6.87 to 24.42 times its own baseline with a live control and determinism 0.000e+00 (`pci_probe_seed_null_2026_09.md`). `gate3_s43` and `gate3_s44` respond on 0 of 20, at 0 to 8 ulp of float32, so they are genuinely unreachable. The default 1e-4 floor sits 40x ABOVE the quiet gate baseline and silenced the real response, which confirms rules 1 and 2. Attenuation is at ONE stage, the tectum forward from `h_state` to `tectum_content` (`pci_gate_attenuation_2026_09.md`); `vision_bid` is 1.0000000000 with zero response, and the gate sigmoids sit at 0.485 to 0.498. Gate PCI is coarsely quantized: 6 values in steps of 0.027429. **RETRACTED: `pci_gate_saturation_2026_09.md`.** **The matrix test is DONE and it also FAILED** (`pci_matrix_content_2026_09.md`): decoding 6-class `sample_shape` from the response, 5-fold CV against a 200-permutation null, gate continuous 5x60 reads 0.2096 against a majority rate of 0.2099 and a null p95 of 0.2463; binary 0.1478; rssm 64x60 0.1083, below uniform chance from overfitting. A PCA-reduced retest closes the p>>n objection and confirms it: 1 of 11 cells crosses p95, at 10 components only, absent at 5 and 20, which is a multiple-comparison artefact and is recorded as not a finding. Binarization separately discards about 90 percent of varying features (gate 139 live to 13), which is worth knowing but is not where content is lost. | **No further work is planned on PCI as an instrument.** Its content clause is settled negatively by two independent methods and cannot be reopened by a better floor, a better null, or a better normalization. It is kept because the propagation question is worth answering and nothing else answers it. The remaining optional passes (colour and size labels on the saved `mat120_s42.npz`) would only broaden a negative. |
| **phi** `iit_phi.py` (pyphi) | **Downgraded 2026-09-02.** Contentless at 3 seeds: eta2 0.027118 / 0.008728 / 0.025970 against null p95 0.055994 / 0.062348 / 0.065874, below the null mean at 2 of 3 seeds (`scalar_content_2026_09.md`). Not degenerate: 1128 to 1267 distinct values, modal share 0.25 to 0.44 percent. Computed at only 1599 of 8000 steps; the rest carry the last value forward. Gated by `is_conscious`, which is pinned. | Show that whatever `phi` is computed over carries content, or restate what `phi` is a measure of on this architecture. |
| **Coupling measures** `coupling_measures.py` (PLV, phase transfer entropy, PAC) | Smoke-tested only, never run on a trained checkpoint. Values are in cycles per step and carry no Hz grounding, so no published frequency band may be cited against them (clause 6, usage). | Run on a trained checkpoint, and settle whether a cycles-per-step value can be cited at all. |
| **Wave detection** `wave_detection.py` (phase gradient directionality, phase singularities) <br>**UNPROVEN, added 2026-09-25** | Tests pin a plane wave, noise, one spiral, the shuffle null and refusal of a constant field. First run on the tectum h state (`tectum_wave_detection_2026_09.md`): PGD above the shuffle null at 3 of 3 capfix seeds, and the h-zeroed control shows the gradient survives without the carried state at 2 of 3, so PGD against the shuffle null alone reads input layout as a wave. Units are cells per step. | Must always be read with the h-zeroed control. Needs a case where it moves under an intervention that creates real propagation before any verdict can rest on it. |
| **Ignition salience** | Alive but very small: 471 distinct values, std 2.385e-04 around a mean of 2.181e-06 (seed 42, measured 2026-09-02). Its saturation is what pins `is_conscious`. Never content-tested. | Content-test it, and decide whether the saturating transform is the defect. |

### Retired (6)

| Instrument | Defect | Clause |
|---|---|---|
| **EI** `effective_information.py` | Deprecated in favour of CE 2.0. Gate-level EI was bit-identical (0.031178) in every window of every run, reproduced exactly by a single-state trajectory. Separately, every EI number was computed on a TPM estimated from OBSERVED transitions where the theory specifies an INTERVENTIONAL one. | 1, 6 |
| **`ce2_ratio`** `causal_emergence_svd.py` | The cross-level ratio is confounded by state-space cardinality, and the source proposes no such comparison. The instrument's within-level readings stay individually available; the ratio does not. | 4, 6 |
| **`emergent_complexity`** | Reports 112 where the constructed answer is 1, at full coverage (`ce2_complexity_estimation_2026_08.md`). | 2 |
| **sync_R** | Once the Kuramoto oscillators converge, sync_R equals the arithmetic mean of the bid vector, verified against a closed form that raises on mismatch. `reset_state()` is never called from the training loop, so they do converge. The modal value 0.450108000 is identical to 9 decimals at all 3 seeds and holds 48.5 / 83.8 / 85.5 percent of steps. Contentless at 3 seeds (`sync_r_content_2026_09.md`). | 1, and clause 7 |
| **`broadcast_mag`** | `broadcast.norm()`. Contentless at 3 seeds, and 39.7 / 43.4 / 41.7 percent pinned at its modal value. The 256-D broadcast it summarizes decodes shape at 0.69 to 0.77, so the norm demonstrably discards content that is present. This was PREDICTED in `broadcast_geometry_2026_08.md` before it was measured. | 1, and clause 7 |
| **`is_conscious`** | Pinned at 1 at 7979 of 8000 steps (99.7 percent, seed 42, measured 2026-09-02), by a saturated ignition gate, and it still gates `phi`. A field whose name makes a claim, reporting 1 on nearly every step. Owner decision open at planning #13. | 1 |

Separately and already done: **14 placeholder metrics were RETIRED 2026-07-29** for
returning numbers they never computed. Each is now a `raise`, pinned by
`tests/test_retired_placeholders.py` including a tokenizing tripwire over
`models/evaluation/`. They are not re-listed individually here.

### Not running (6)

Verified at seed 42 on 2026-09-02: each is exactly 0.0 with 1 distinct value across all
8000 steps, because its module is disabled. **No verdict is possible on these and none
should be claimed.**

- `phi_riiu` and its three channels (`phi_riiu_broadcast`, `phi_riiu_tectum`,
  `phi_riiu_audio`)
- The five Levin metrics (`levin_bioelectric_complexity`,
  `levin_morphological_adaptation`, `levin_collective_intelligence`,
  `levin_goal_directed`, `levin_basal_cognition`)
- `self_pred_mse` and `self_pred_skill`
- The PAD `dominance` channel, structurally zero
- `match_head_loss` and `match_head_acc`, from machinery the project decided not to build
- `recon_loss`, outside the instrument set

### The levels the instruments read

Not instruments, but the inventory is unreadable without them.

| Level | State | Evidence |
|---|---|---|
| gate, 5 nodes | 4 alive, 1 dead. `gate_attention` 130, `gate_stability` 128, `gate_coherence` 135, `gate_confidence` 122 distinct values, std 4.1e-05 to 6.9e-05. `gate_adaptation` has 2 distinct values and std 1.043e-07, below the 1e-6 dead bar. | measured 2026-09-02, seed 42; `gate_binning_2026_07.md` |
| workspace 3-tuple | Alive, badly discretized. Components 167 to 1245 distinct values; the binning resolves 4 or 5 states of 8; bins 0 and 1 are structurally unreachable. | `workspace_state_variance_2026_08.md` |
| 256-D broadcast | Alive and carries content. 95 percent of its variance sits in 2 to 4 of 256 dimensions, on a fixed offset 7 to 20 times larger. | `broadcast_geometry_2026_08.md` |
| module bids | Degenerate. Vision pinned at exactly 1.000000000; the winner never changes. | `workspace_competition_2026_08.md`, `workspace_bids_live_2026_08.md` |

## What this inventory does NOT do

- **It promotes no indicator.** The rubric stays 3 IMPLEMENTED, 11 PARTIAL, 14 total, and
  the clock does not move.
- **It repairs nothing.** Every entry above is a status, not a change. No flag was added
  and no instrument was modified in writing this.
- **It does not settle CE 2.0's within-level survival.** That is planning #7 and it is
  open.
- **It does not re-examine RPT-1, GWT-1 and GWT-3**, which were awarded under an older
  reading of the mechanisms. That is a real open question and a separate job.
- **It says nothing about the C1 competence wall.** The policy does not learn the task.
  That is a separate and unfixed problem, and no instrument work touches it.

## The honest summary

Six sessions of instrument work produced six diagnoses and zero trusted instruments. The
reason is now visible and it is not a property of any single instrument: the instruments
read scalars, and the scalars in this architecture do not carry the content that its
vector representations do.

PCI is the only entry with a working control and a demonstrated discrimination. It is also
the only one that supplies its own variation instead of reading a spontaneous trajectory.
That is the one lead this inventory leaves standing, and planning #15 already asks the
matching question.

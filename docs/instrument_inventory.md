# Instrument inventory

**Nothing is TRUSTED. Of 16 instrument entries, 0 are trusted, 4 are unproven, 6 are
retired, and 6 are not running.** As of 2026-09-13 PCI has six reading rules and three
trained-gate checkpoints, and CANNOT measure the gate in either direction: one floor
silences the substrate, the other admits its noise. It needs a null, not a threshold.

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

### Unproven (4)

| Instrument | Why unproven | What would change it |
|---|---|---|
| **PCI** `perturbational_complexity.py` | **PCI CANNOT CURRENTLY MEASURE THE GATE AT ALL, IN EITHER DIRECTION** (`pci_reading_rules_2026_09.md`, 2026-09-13). The default floor of 1e-4 sits ABOVE the gate's own PCI-window fluctuation of 1.2e-05 to 1.2e-04, so every gate zero read at the default carries no information. A per-site relative floor was implemented and FAILED its own noise control: over 40 synthetic seeds it returns non-zero from PURE NOISE in 29 of 40, mean 0.083, max 0.192, and the real gate readings it produced (0.0439 and 0.0603 on `gate3_s42`) sit inside that range. So neither floor works: absolute silences the substrate, relative admits its noise. This is a fact about the SUBSTRATE, whose fluctuation is too close to numerical noise for any fixed threshold. What still stands: the rssm CONTROL is alive and discriminating at every checkpoint (0.0471 to 0.1631 across three trained gates), determinism 0.0e+00 throughout, and `pci_trained_2026_08.md` was RE-CHECKED at 1e-6 and its 0.0000 holds. Rules 1 to 5 stand; rule 6 says a gate PCI needs a NULL, not a threshold. | Build the null: score the probe with no impulse and use that distribution as the floor. Same probe, one argument. Until then cite no gate-level PCI under any floor setting. |
| **phi** `iit_phi.py` (pyphi) | **Downgraded 2026-09-02.** Contentless at 3 seeds: eta2 0.027118 / 0.008728 / 0.025970 against null p95 0.055994 / 0.062348 / 0.065874, below the null mean at 2 of 3 seeds (`scalar_content_2026_09.md`). Not degenerate: 1128 to 1267 distinct values, modal share 0.25 to 0.44 percent. Computed at only 1599 of 8000 steps; the rest carry the last value forward. Gated by `is_conscious`, which is pinned. | Show that whatever `phi` is computed over carries content, or restate what `phi` is a measure of on this architecture. |
| **Coupling measures** `coupling_measures.py` (PLV, phase transfer entropy, PAC) | Smoke-tested only, never run on a trained checkpoint. Values are in cycles per step and carry no Hz grounding, so no published frequency band may be cited against them (clause 6, usage). | Run on a trained checkpoint, and settle whether a cycles-per-step value can be cited at all. |
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

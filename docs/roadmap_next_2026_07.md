# Forward roadmap (2026-07): a gated decision tree for the next sessions

This document plans the next several sessions. It is a decision tree, not a fixed
schedule. The project's own record is that most single-bet chapters end at a kill gate
(reconstruction x2, value-equivalent world model, latent-identity supervision), so every
step below states its cost, its pre-stated pass/kill gate, and where each outcome
redirects. It extends `docs/roadmap.md` (Phases 5 and 6); it does not replace it. The
success axis is unchanged: consciousness signatures (the Butlin rubric in
`docs/consciousness_indicators_butlin.md`) and causal efficacy on DMTS/WCST (the
pre-registered section 13 test in `docs/preregistered_predictions.md`), never task reward.

## Status 2026-07-06: A1/A2 done, B0 closed, B1 is a PASS (the RSSM wall broke; it moved downstream)

First execution pass on this roadmap:

- **A1 (EI floor correction): done** (`46d7464`). The instrument is corrected; the honest
  reading is the micro/gate level is frozen everywhere while the macro/workspace level has
  real structure, so the old 12x ratio was macro structure over a Laplace floor. See
  `docs/results/instrument_repair_2026_07.md`.
- **A2 (ignition): done, not fixable at the gate** (`0a87d9b`). The gate signal is
  phase-invariant (sample-vs-delay |d| < 0.06); the saturation is a content problem, not a
  threshold problem. Same doc, A2 section, plus `scripts/analysis/probe_ignition_signal.py`.
- **B0 (ceiling-test weight sweep): done** (`46d7464`). Weights 1/10/100 all keep the CE at
  the chance floor and z_state at chance; the FAILED discrete-latent verdict is airtight.
- **B1 (continuous latent): PASS** (`docs/results/b1_continuous_latent_2026_07.md`). A
  default-off continuous Gaussian latent makes z_state decode identity (0.71 to 1.0 vs
  chance 0.167), confirmed structural by a reward-only control. The discrete gumbel latent
  was the wall. Single seed, so it is a hypothesis pending >= 3 seeds; the mode stays
  default-off. The collapse MOVED downstream: capsule_poses and tectum_content are still at
  chance, so the new locus is the capsule routing. Task reward stayed flat (RL wall separate).

Next per the tree: replicate the continuous result at >= 3 seeds (gate before any default
change), and/or open a B2-like thread on the capsule stage (the new identity locus). C1
(RL wall) remains independent. These are owner decisions.

## Where we are (all verified on disk, 2026-07-05)

Two blockers are now characterized to the point of being actionable:

1. **Perception collapse, root isolated to the latent.** The reward-trained RSSM discards
   stimulus identity at `obs_map -> z_state`. This is now robust to reward, frame
   reconstruction, obs_map reconstruction, a value-equivalent world model, and direct
   supervised CE (`docs/results/latent_identity_ceiling_2026_07.md`). The confirmed wall
   is the discrete gumbel-softmax categorical latent itself (or its batch-1 online
   optimization), not the training objective. The one lever not yet pulled is the latent
   representation.
2. **Signature instruments largely degenerate** (`docs/results/signature_assessment_2026_07.md`).
   EI is a Laplace-floor artifact, GNW ignition is saturated, sync_R is objective-invariant,
   and RIIU/Levin/self-prediction report zeros from default-off code. Most dials would not
   register agent improvement even if it happened. phi is the one objective-sensitive signal
   and is near zero.

A third, older constraint stands and must not be forgotten: **the RL policy did not learn
DMTS even when identity was offline-decodable** (2026-06-14/15). Fixing perception is
necessary but not sufficient for section 13; the learning/credit-assignment wall is a
separate front.

## Two independent workstreams

The blockers are separable, so the roadmap runs two tracks that do not depend on each
other. Track A needs little or no compute and improves the mission's actual scoring
instruments regardless of the agent. Track B is the architectural bet on perception. Track
C is downstream and gated on both.

- **Track A: instrument repair.** Make the signatures measure something before we measure
  them. Independent of perception.
- **Track B: the latent representation bet.** The last live lever on the perception collapse.
- **Track C: competence, then the pre-registered tests.** Only reachable if B succeeds AND
  the RL wall is addressed.

Recommended cadence: run Track A first or alongside Track B's gate, because A is cheap,
de-risked, and moves the mission axis directly. Do not start Track C until its
prerequisites are met.

---

## Track A: instrument repair (low / zero compute, do first)

The signature assessment already wrote each instrument's requirement to become measurable.
This track implements those, each default-off or as a corrected metric, each with a test
that the fix changes a degenerate output on existing run data (no training needed to
validate the math).

### A1. EI floor-bias correction (zero new compute)

- **Problem:** `ei_gates = 0.031178` in every window of every run is the exact
  constant-trajectory Laplace floor for 243 states at window 10000; the ~12x emergence
  ratio is the ratio of two floors.
- **Do:** subtract the constant-trajectory baseline for the given (window length, state
  count) in `models/evaluation/effective_information.py`, or resolve the gate binning so
  the micro level actually transitions. Add a unit test that a constant trajectory yields
  corrected-EI 0 (not the floor) and that a synthetic transitioning trajectory yields > 0.
- **Gate:** re-run `report_signatures.py` over the five existing runs. PASS if corrected EI
  is no longer bit-identical across windows and the emergence ratio is no longer a pure
  state-count artifact. This is a math fix, so it either corrects the bias or it does not;
  no seeds needed.
- **Redirect on fail:** if the gates genuinely never transition (saturated micro states),
  that is itself a finding about the workspace dynamics and feeds Track B's motivation.

### A2. Ignition selectivity, real version (zero to low compute)

- **Problem:** `is_conscious == input_energy >= EMA(input_energy)` saturates once input
  energy stabilizes; the 2026-06-21 fix survives only as an early transient.
- **Do:** diagnose read-only first (does any candidate ignition signal on the existing runs
  dip below its running average on task-relevant events, e.g. DMTS sample onset vs delay?).
  Only if a discriminating signal exists, wire it. If none does, the honest output is that
  ignition cannot discriminate on this agent's near-constant workspace content, which is the
  same low-variation root, and that is reported, not patched cosmetically.
- **Gate:** PASS = a defensible per-step conscious/quiet split that tracks task phase on
  existing data. Do not tune parameters to chase a prettier ratio (explicitly flagged as
  the cosmetic trap in the 2026-06-21 log).

### A3. Enable the dormant measurement modules for real (low compute)

- **Problem:** RIIU phi, all five Levin metrics, and self-prediction are all-zero because the
  modules are default-off in the training path.
- **Do:** decide per module whether it is measured going forward, and if so run one seed-42
  DMTS + one dark_room run with the relevant flags on (`--enable-riiu`, Levin wiring per
  Phase 5 deliverables). This is the Phase 5 "activate dormant Levin modules" deliverable,
  now with the honest precondition that their inputs vary enough to be non-degenerate.
- **Gate:** PASS = the module produces a non-constant signal that differs between DMTS and
  dark_room. FAIL = still degenerate, report as such; do not list zeros as measurements.

### A4. Instrument-repair report

- Consolidate A1-A3 into `docs/results/instrument_repair_2026_07.md` (degenerate-before /
  corrected-after tables), update the Butlin rubric only where a now-valid measurement
  justifies a status move, add a roadmap status entry. Commit as tlcdv.

**Track A exit criterion:** at least the EI and ignition instruments produce
non-degenerate, defensible outputs, so that any future agent change can be seen. This is
mission progress on its own terms even if Track B never succeeds.

---

## Track B: the latent representation bet (the last perception lever)

This is the architectural decision the ceiling test surfaced. It is a real bet with an
uncertain payoff and it touches the RSSM core, so it is staged with a cheap gate that
reuses the tested ceiling-test harness.

### B0. Close the ceiling-test caveat first (cheap, ~2 serial runs)

Before committing to an architecture change, remove the one honest hole in the FAILED
verdict: loss weight and LR were not swept.

- **Do:** re-run `--enable-latent-id` at weight 10 and weight 100 (and optionally 10x LR on
  the head/latent path), DMTS 100 ep seed 42, serial, verify by episode-row count.
- **Gate:** if the supervised CE still cannot leave the chance floor at 100x pressure, the
  discrete-latent verdict is airtight and B1 is justified. If a higher weight suddenly makes
  z_state decode on the probe, the wall was optimization strength, not the latent, which
  redirects to tuning rather than a rewrite (and would be a major, welcome surprise).
- **Cost:** two ~50 min runs plus two probes. Do this before any architecture work.

### B1. Continuous / higher-capacity RSSM latent (the bet)

- **Do:** add a default-off latent variant to the RSSM core (`models/core/`, the
  `SensoryTectum` RSSM): a continuous Gaussian latent (Dreamer-style with a KL) or a
  higher-capacity categorical, selectable by flag, baseline bit-identical when off.
- **Cheap gate (reuse B0 harness):** with the new latent, rerun the SAME supervised ceiling
  test (`--enable-latent-id`), one seed-42 DMTS run + collapse-locus probe.
  - PASS = z_state decodes identity well above chance (toward obs_map ~1.0). The latent can
    carry identity; the discrete latent was the wall. Proceed to B2.
  - KILL = z_state still at chance under direct supervision. The wall is deeper than the
    latent representation (candidate: the batch-1 online regime, or the encoder->RSSM
    interface). Report FAILED; escalate the regime question to the owner. Do NOT build a
    label-free objective on a latent that fails its supervised ceiling.
- **Why the ceiling test gates it:** direct supervision is the maximum identity pressure; if
  a latent cannot pass it, no self-supervised objective on that latent will.

### B2. Label-free identity objective (only if B1 passes)

- **Do:** replace the privileged-label ceiling head with a self-supervised contrastive /
  InfoNCE objective on the passing latent, so identity is preserved without env labels
  (the shippable mechanism, not a probe). Default off.
- **Gate:** collapse-locus probe z_state above chance without labels, then **>= 3 seeds**
  before any default-on change (hard rule). Only after 3-seed confirmation does this become
  a default of the perception pipeline.

**Track B exit criterion:** a perception pipeline whose latent provably carries stimulus
identity across seeds, OR a clean, characterized FAILED verdict that the wall is the
training regime rather than the representation. Either is a legitimate mission output.

---

## Track C: competence, then the pre-registered tests (gated on B + the RL wall)

Do not start until Track B has a latent that carries identity AND the RL learning wall is
confronted. Running section 13 before this only confirms "blocked", which the record
already shows; its thresholds stay pre-registered and untouched.

### C1. Confront the RL / credit-assignment wall (separate front)

- **Context:** the policy did not learn DMTS even with identity offline-available
  (2026-06-14/15). This is a distinct blocker from perception. The docs-lane resource map
  points at relational RL, temporal value transport, and meta-RL as the aligned literature
  (`docs/aligned_external_resources.md`, the DMTS-wall tier).
- **Do:** diagnose first (is the failure exploration, credit assignment across the delay, or
  policy capacity?) with a read-only or cheap probe before building. Then pick the single
  cheapest mechanism the diagnosis points to. Default off, FAILED-first, staged.
- **Gate:** the consciousness agent reaches above-chance DMTS choice accuracy on a held-out
  trial split. Until this passes, C2 is blocked.

### C2. Pre-registered substrate-independence test (section 13)

- **Only when** the agent can enter the DMTS/WCST diagnostic regime (C1 passed) and the
  repaired instruments (Track A) can measure the signatures.
- **Do:** settle the section-13 prerequisites (goal_directed embeddings or run on
  `collective_intelligence` alone; the DMTS/WCST self-monitoring phase windows; train-or-not
  for the Levin modules), then run the test on the consciousness agent vs the DQN baseline.
- **Gate:** the pre-registered thresholds and signs, unchanged. Report the indicator moves
  only where a measured signature justifies it. This is the strongest evidence the project
  can produce: that the integrated machinery is causally useful on consciousness-demanding
  tasks.

---

## Cross-cutting: consolidation, not accretion

The 2026-06-09 audit and the 2026-06-21 self-review both flagged that `train_rlhf.py` has
accreted ~84 enable_/ablate_ flag references, a graveyard of default-off failed experiments.
Two disciplines apply across all tracks:

- **Before any new mechanism, run the zero-compute diagnosis that could kill it** (the
  ceiling test and the PCA variance probe are the templates). This is the lesson the
  2026-06-21 self-review paid for.
- **When a track closes, the active-inference unification** (`docs/active_inference_unification.md`)
  is the sanctioned consolidation direction: collapse the ~6 hand-wired objectives toward a
  single expected-free-energy principle, rather than adding a seventh. This is Phase 6 and is
  the coherent home for a successful Track B latent + world model.

## Suggested ordering across sessions

This is a suggestion; the gates decide the actual path.

1. **Next session:** Track A1 + A2 (instrument repair, cheap, high mission value) and B0
   (close the ceiling-test caveat) in parallel, since B0 is compute-bound and A is
   compute-light.
2. **Then:** decide Track B1 with the B0 sweep in hand (owner call on the architecture bet).
   Run A3/A4 to finish the instrument track.
3. **If B1 passes:** B2 (3 seeds), and begin C1 diagnosis in parallel.
4. **If B1 kills:** the perception lever is exhausted; the mission output becomes the full
   characterization (Track A repaired instruments + the collapse verdicts) plus a decision
   with the owner on whether to attack the RL/regime wall (C1) directly or consolidate
   toward active inference.
5. **Track C2 (section 13)** only after C1 passes and Track A instruments are valid.

## What does NOT get reopened

- Phi-1 (closed across 9 runs, 4 architectures, 2 formulations).
- More match-head machinery.
- A non-biological control encoder (the architecture is biology-first by design).
- Reconstruction-family perception fixes (exhausted; identity is a low-variance direction).
- Section-13 threshold revision (pre-registered).

## Owner decision points (do not auto-proceed past these)

- **B1:** committing to the continuous-latent architecture change (after seeing B0).
- **C1:** which RL/credit-assignment mechanism to build, after the diagnosis.
- **Any default-on change:** requires >= 3 seeds first (hard rule).

# Architecture assessment (2026-06-09): overengineering, smaller delta, elegance, coherence

Read-only assessment prompted by four direct questions: is the project
overengineered (is there a simpler way), is there a smaller delta that buys most of
the benefit, is there a more elegant way, and is it architecturally coherent. It is
a companion to `architecture_audit_2026_05_31.md`. Every number was loaded from disk
on the day of writing. No code was changed to produce the analysis; the only code
change shipped alongside it is a navigation comment block in `train_rlhf.py` (see
the closing note).

## Evidence (verified from disk, not from the self-report)

- `scripts/training/train_rlhf.py` = 1700 lines, 6 optimizers (tectum,
  reward-predictor, workspace, RND, self-vector, control-repr), and ~8 distinct
  `backward()` paths, three of them sharing one graph through `retain_graph=True`
  (lines 1023, 1052, 1093). A tangled autograd graph is a coherence smell.
- 46 CLI flags, of which 18 are experiment toggles (9 `--ablate-*`, 9 `--enable-*`).
- 95 files under `models/`, 70 test files, 25 markdown docs, 14 results write-ups.
- `architecture_audit_2026_05_31.md` verdict: integrity legit, architecture has
  "real coherence debt." `active_inference_unification.md` already designs the
  simpler objective.
- Capability baseline (`docs/results/experiment_comparison.md`): dark_room DQN
  last-100 reward ~92.0 vs consciousness agent ~12.95.
- Phi-1 tested across 9 runs / 4 architectures / 2 phi formulations, all FAILED the
  r > 0.4 prediction.

## Q1. Is it overengineered? Is there a simpler way?

Yes, and the consolidation is already underway. Three measured signals:

1. The training loop is a 1700-line script optimizing ~6 partly-reward-coupled
   objectives through 6 optimizers and a `retain_graph=True` graph. Several terms
   (reward-predictor MSE, phi-delta reward, sync_R*reward, RND) were each added to
   patch a separate symptom. That is accretion, not design. This is the strongest
   overengineering signal, stronger than the raw flag count.

2. The 18 experiment toggles are a navigability problem, not dead weight. This
   correction matters: an earlier draft called them "frozen archaeology," and a
   reference check disproved that. Most are live default-off experiments
   (`--enable-self-vector`, `--enable-riiu`, `--ablate-existence-bias`, and others),
   and the legacy `--ablate-gate-*` flags are referenced by 4 results docs, a live
   analysis script, and the campaign runner, so they are repro-locked, not
   neglected. The real cost is that a reader cannot tell at a glance which toggle is
   a live experiment, which is a settled default kept as a repro alias, and which is
   dormant. The fix is to write that status down (done in this pass), not to delete.

3. Five ways to measure consciousness, one weak agent. phi/pyphi, EI, Levin
   metrics, RIIU, self_pred_skill. The 2026-05-31 audit (section 2) found only
   phi-delta and sync_R are even thinly wired into learning; the rest are
   measurement-only or instantiated-but-inert.

The simpler way is the one the project's own docs point to: collapse the loss stack
under active inference (`active_inference_unification.md`) and finish the
consolidation the audit began (P1/P2 done, P3/P4 done as mechanism, P5 open).

## Q2. Is there a smaller delta that buys most of the benefits?

Yes, and it is not where recent effort went. The benefit wanted is an agent whose
behaviour shows measurable consciousness signatures. The binding constraint on that
is agent competence and perception, not the count of metrics. dark_room sits at
12.95 vs DQN 92.0; WCST triggers 0-1 rule changes in 60 episodes, so it never enters
the self-monitoring regime. Phi-2, EI-2/3, IM-1/2/3, and WCST self-monitoring all
read INCONCLUSIVE for one shared reason: the subject cannot perform the task the
signature needs. The instrumentation outweighs the thing instrumented.

So the smallest delta is to stop adding measurement and make one agent competent on
one task where a signature is already measurable. The audit names that task:
navigation, the single place the self-vector validates (residual self-prediction
skill +0.35). In the roadmap's own order:

- (a) Run the Phase 5 "perception-supports-the-tasks" probe (cheap, already
  specified). It says whether the front-end can support DMTS/WCST at all.
- (b) If it cannot, implement stage 1 only of the active-inference unification:
  amortize the RSSM ELBO the project already has and replace RND with an epistemic
  term. That single stage removes 2 of the 6 hand-wired losses, is the principled
  fix for the perception bottleneck the 2026-06-02 localization flagged, and makes
  learning intrinsic as the README already claims.

One cheap probe plus one of three active-inference stages buys most of the
coherence-plus-capability payoff. The smaller delta is explicitly not "another
binding mechanism" or "another phi formulation." Both were tried, to a robust
negative result.

## Q3. Is there a more elegant way?

Yes, and it is already designed. `active_inference_unification.md` collapses the
~6 objectives into two: variational free energy for perception and learning,
expected free energy for action, with reward demoted from the learning signal to a
preference prior. One principle subsumes the accreted terms. It also fixes a live
contradiction: the README says the agent explores to reduce prediction error "not
to accumulate external reward," yet 4 of the 6 current objectives key off external
reward. Active inference removes that contradiction by construction (RND becomes the
principled epistemic term; phi-delta is dropped as a training driver, which also
removes the circularity of rewarding the very quantity the project is trying to
observe emerge).

A point the docs understate: elegance here means fewer signals that are causally
load-bearing, with fewer signals that are epiphenomenal. The audit's P4 (integrate
EI/Levin into the causal loop, or demote them honestly to diagnostics) is the
elegance move at the conceptual level, and it was resolved as "diagnostics." That is
the right call. Keep pruning measurement that does not drive behaviour.

## Q4. Is it architecturally coherent?

Split verdict, both halves stated plainly.

- As a thesis: coherent, and that is rare. Functionalist Emergentism plus
  Feinberg-Mallatt's six features plus Rouleau-Levin substrate independence is a
  genuine, mappable research program. Components correspond to real theoretical
  commitments. Few projects in this space have that spine.
- As implemented: partially coherent, with the debt the audit already named. A
  small coherent core (perception, workspace, action, with phi/sync_R thinly wired
  in) sits under measurement-only modules, a learning objective that contradicts the
  stated philosophy, a self-model that is causally inert as wired, and a base agent
  too weak to exercise the tasks.

The deepest incoherence is structural, not cosmetic. Functionalist Emergentism
requires the integrated/conscious state to be causally efficacious, non-
epiphenomenal. Most of the "consciousness" stack is measurement that does not drive
behaviour. The thesis is undercut by its own implementation. The fix is the audit's
P3/P4 plus active inference: make the integrated signal select action (via expected
free energy), so the conscious state earns its causal keep.

One thing underwrites all of the above: the project is legit. The empirical record
is honest, FAILED-first, ~658-678 tests passing, no fabrication. The problem is
architectural debt, not self-deception. Debt is payable and the foundation is sound.

## Synthesis: the one root cause

The recent stretch added measurement and binding variants to a subject that cannot
yet perform. Five consciousness meters, one ~13-reward agent. That inversion
explains the overengineering (Q1), the missed smaller delta (Q2), the lost elegance
(Q3), and the implementation-level incoherence (Q4) at once. Re-order the work: make
the substrate causally able, then let the signatures read something real.

## What this assessment recommends

- Now (tier 1, shipped with this doc): document the flag status inline so the
  1700-line script is navigable. No deletion, no behaviour change, the reproducible
  record stays intact.
- Next (tier 2, the real smaller delta): the Phase 5 perception probe, then
  active-inference stage 1 only. Default-off, FAILED-first, at least 3 seeds before
  any default flip.
- Later (tier 3, scheduled, not a side change): P5 agent competence plus the staged
  remainder of active inference.

## What not to do

- Do not reopen Phi-1. It is closed at 9 runs by decision.
- Do not add a sixth or seventh consciousness meter before the agent can perform.
- Do not do the full active-inference re-architecture in one pass. Stage it.
- Do not introduce a non-biological control encoder. It breaks the thesis.

## Closing note on the shipped change

The only code change made with this assessment is a comment block above the
experiment-flag group in `scripts/training/train_rlhf.py`, grouping the 18 toggles
into live experiments, settled-default-with-legacy-alias, and dormant. It is
comments only; training behaviour is bit-identical and the test suite is unchanged.

## Addendum (2026-06-10): the perception-decodability probe sharpens Q2 and Q4

Numbers re-verified from disk: `train_rlhf.py` is 1738 lines, 6 optimizers, 5
`backward()` calls with 3 sharing one graph via `retain_graph=True` (lines 1023,
1052, 1093), 47 argparse flags of which 18 are experiment toggles (9 `--ablate-*`,
9 `--enable-*`), 95 files under `models/`, 73 test files, 15 results docs. The
structural picture in the body stands; only the counts drifted, and "~8 backward
paths" above is corrected to the precise 5 calls / 3 shared-graph.

The perception-decodability probe (`docs/results/perception_decodability_2026_06_09.md`),
run after this assessment was first written, does not overturn any of the four
verdicts. It converts two of them from argument into measurement.

**Q2 (smaller delta) is now precise, not just directional.** The body recommended
"make one agent competent on one task, starting with the Phase 5 perception probe."
That probe has run. It localizes the competence bottleneck to a single stage: a
linear decode of stimulus identity is perfect off the spatial map (`pixels ->
obs_map` is lossless, decode = 1.000 for shape/color/size/count) and collapses to the
majority-class baseline off the compressed tectum representation (`obs_map ->
tectum_content` decode = 0.13 to 0.60, each within 0.01 of chance). The policy and
the workspace both read `tectum_content`/`broadcast`, so they operate on a
representation that has already discarded the stimulus. Training DMTS for 100
episodes did not recover it (reward flat at ~-39, decode still at chance). This makes
the smaller delta both sharper and cheaper than "active-inference stage 1 = rebuild
the encoder," which the probe shows targets the wrong stage:

- Cheapest test: route the policy at `obs_map` via the existing `--policy-input
  spatial` tap (added 2026-06-02), which is already decodable, and measure whether
  the agent can enter the DMTS/WCST regime. No new objective, no new module.
- Principled fix: a reconstruction / variational-free-energy objective on the
  RSSM/capsule collapse stage, because reconstructing the frame forces the 256-D
  latent to keep shape/color/count. The current tectum objectives (reward-prediction
  MSE + TDANN) ask for neither.

Both gated, default-off, FAILED-first, >= 3 seeds before any default flip. The
body's caveat that the smaller delta is "explicitly not another binding mechanism or
another phi formulation" is unchanged.

**Q4 (coherence) gains a concrete mechanical instance of the deepest incoherence the
body named.** The abstract claim was: Functionalist Emergentism needs the integrated
state to be causally efficacious, yet most of the consciousness stack is measurement
that does not drive behaviour. The probe shows a sharper version: the conscious
bottleneck (the GNW broadcast) is not merely under-wired into learning, it is
operating on content the upstream collapse has already degraded to chance. The
workspace cannot integrate a stimulus identity it never receives. The fix is the
body's (P3/P4 plus active inference), now with a named first target: the `obs_map ->
tectum_content` handoff, not the encoder and not the policy.

Net: the four verdicts hold. Q2 and Q4 are now backed by a measurement rather than an
inference, and the recommended next step is unchanged in direction and cheaper in
form: the spatial-tap test before any active-inference build.

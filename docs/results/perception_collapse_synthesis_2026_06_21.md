# Perception-collapse investigation: synthesis and decision (2026-06-21)

This consolidates the 2026-06-21 session (confirmatory re-probe + the full R1 effort) into
one finding and one decision. Every number is from the per-experiment verdict docs cited
below, all loaded from disk this session.

## What was established (FAILED-first)

1. **The collapse is real and at the RSSM, trained and untrained (CONFIRMED).** obs_map
   decodes stimulus identity at ~1.0; z_state, capsule_poses, and tectum_content sit at
   chance, on both an untrained and a reward-trained tectum.
   [collapse_locus_trained_2026_06_21.md](collapse_locus_trained_2026_06_21.md).

2. **R1 via reconstruction FAILED, twice.** Reconstructing the frame from the RSSM latent
   (R1-pixels, recon_loss -15x) and reconstructing the dense identity-rich obs_map from
   the latent (R1-obs_map, recon_loss -1300x to MSE 2e-4) both trained their loss down and
   neither made the latent decode identity.
   [collapse_locus_wmrecon_2026_06_21.md](collapse_locus_wmrecon_2026_06_21.md),
   [collapse_locus_wmobs_2026_06_21.md](collapse_locus_wmobs_2026_06_21.md).

3. **The mechanism is CONFIRMED: identity is a low-variance direction in obs_map.** The
   top-20 PCs of obs_map capture 96.3% of the variance but decode shape at only 0.512;
   shape fully decodes (0.976) only with ~50 PCs reaching 99.9% variance
   (`scripts/analysis/probe_obsmap_variance.py`). An MSE-minimizing code keeps the
   high-variance bulk and discards the low-variance identity direction, which is why every
   reconstruction objective trains its loss down without putting identity into the latent,
   and why a third MSE target (DINOv2 features) would fail the same way.

## What this means for the architecture (a mission-relevant finding, not just a failed fix)

The GNW-style integration pathway (RSSM world model + capsule composition into a 256-D
workspace) **preserves high-variance gestalt structure and discards low-variance
discriminative identity**, and this is robust to adding a reconstruction objective at the
exact locus of the loss. This is partly BY DESIGN (a limited-capacity conscious workspace
is the GWT bottleneck, Butlin GWT-2), and the low-variance analysis makes the cost of that
design concrete: fine stimulus identity, which DMTS/WCST need, is exactly the kind of
low-variance detail the bottleneck sheds. This is a real, characterized property of the
consciousness architecture, reported honestly.

## The decision (pivot, with reasons)

**Stop the perception-fix-via-reconstruction track. Do not iterate more reconstruction
targets.** The reasons, in order:

1. **The fix family is exhausted with a confirmed mechanism.** Two targets, both FAILED,
   for a reason (low-variance identity vs MSE) that predicts further MSE targets fail too.

2. **Even a perception fix may not unblock the tasks (the second wall).** The 2026-06-14/15
   DMTS arc established that the RL policy did not learn the match even when identity was
   offline-available (~0.74). So task competence is blocked on BOTH perception and RL
   credit assignment. Fixing only perception is not sufficient.

3. **The causal-efficacy test is therefore blocked on agent competence.** The pre-registered
   substrate-independence test ([preregistered_predictions.md](../preregistered_predictions.md)
   section 13) SI-1 requires the consciousness agent to have a task-accuracy ADVANTAGE over
   DQN on self-monitoring-demanding DMTS/WCST phases. The agent cannot do those tasks at
   all, so SI-1 would FAIL on a non-functional agent. Running section 13 now would confirm
   "blocked", not test the hypothesis. Its thresholds are NOT revised; it stays pre-registered
   and UNTESTED until an agent can enter the regimes.

4. **Reading #2 says judge by consciousness signatures, not task reward.** The mission is
   not served by an open-ended representation-learning campaign against a by-design
   bottleneck.

## What is achievable now (the honest next direction)

Two materially different forward paths remain. This doc records the decision to NOT
auto-pursue the speculative one, and to set the achievable one as next.

- **Path B (a real architectural investment, NOT chosen here):** an objective that directly
  pressures the low-variance identity (a contrastive / InfoNCE loss on the latent, a
  continuous higher-capacity latent, or both). The confirmed mechanism points a contrastive
  objective at the right target (it pressures the discriminative direction regardless of
  variance), so this is now a motivated bet rather than a guess. But it is a multi-front
  build (perception AND the RL second wall), with uncertain payoff, and it is the kind of
  open-ended effort the project's discipline says to confirm before starting.

- **Path C (achievable signature measurement, recommended next):** assess the consciousness
  signatures that do NOT require task competence, on the current agent, against the Butlin
  rubric ([consciousness_indicators_butlin.md](../consciousness_indicators_butlin.md)):
  structural recurrence (RPT-1), workspace bottleneck and broadcast (GWT-1..4), EI causal
  emergence ([effective_information.py](../../models/evaluation/effective_information.py)),
  the Levin intrinsic metrics, phenomenological mapping. Report the causal-efficacy
  indicators (the DMTS/WCST substrate-independence test) honestly as BLOCKED on agent
  competence, with the blocker now characterized (low-variance identity discarded by the
  integration bottleneck, robust to reconstruction). This is the honest mission output and
  needs no speculative build.

The `--enable-wm-recon` mechanism (both targets) and `probe_obsmap_variance.py` stay in the
repo as documented negatives / analysis tools. The decision between Path B and a focused
Path C measurement is the user's; this session's evidence recommends not opening Path B
without an explicit decision, given its doubly-uncertain payoff.

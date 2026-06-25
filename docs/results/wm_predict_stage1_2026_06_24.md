# Value-equivalent world model, Stage 1 working-memory validation (2026-06-24): FAILED

**FAILED.** A value-equivalent (MuZero / Dreamer-without-decoder) world-model objective on
the RSSM latent, trained with per-trial BPTT through the DMTS delay, did NOT make the RSSM
recurrent state `h_state` retain the held sample across the delay. Tested in two
configurations (latent-only reward head, then action-conditioned reward head); the WM loss
trained down in both, but `delay h_state` stayed at chance, indistinguishable from the
reward-only baseline. This is the pre-stated KILL gate: the discrete gumbel-softmax
categorical RSSM latent is the wall. Per the staging discipline, the effort stops here and
is escalated rather than iterated further.

All numbers below were loaded from disk this session
(`scripts/analysis/probe_wm_leakage_free.py`, leakage-free, one record per trial, seed 42,
n=280; `runs/wmpredict_trained/`, `runs/wmpredict_trained2/`, `runs/collapse_trained/`).

## The hypothesis and the build

The model-based reframe (`docs/results/collapse_locus_wmobs_2026_06_21.md`,
`perception_collapse_synthesis_2026_06_21.md` and the literature on DreamerV3 / MuZero)
held that the RSSM never holds working memory because it is trained to predict reward
single-step, not as a generative/value-equivalent world model with temporal credit. Stage 1
(`--enable-wm-predict`, commit `873960e` + `f851c3d`): action-condition the RSSM, train the
prior/posterior KL as a balanced loss with free bits, predict reward + continue from the
RSSM latent, and accumulate a per-trial buffer so ONE BPTT backward at the choice phase
sends the delayed-reward gradient back through the delay to the sample step. No decoder, so
it structurally avoids the reconstruction-discards-identity failure of R1.

Falsifiable gate: if the objective works, `delay h_state` (the recurrent state during the
blank delay, when the stimulus is off-screen) should decode the held sample above chance.

## Two configurations, both trained, both FAILED

| config | WM loss (first->last half) | reward (first10->last10) |
|--------|---------------------------:|-------------------------:|
| latent-only reward (`runs/wmpredict_trained`) | 1.223 -> 1.150 (882 flushes, min 1.000) | -11.08 -> -12.83 |
| action-conditioned reward (`runs/wmpredict_trained2`) | 1.169 -> 1.161 (897 flushes, min 1.000) | -11.08 -> -11.70 |

The WM loss trained down and bottomed near the 1.0 KL free-bits floor in both, so the
FAILED result is NOT an untrained-head artifact. The reward stayed flat and negative: the
agent does not learn DMTS (Stage 1 has no imagination actor; the policy is still the
model-free Go/No-Go), so the choice-phase reward signal is weak. That is an honest
confound, but the action-conditioned reward head was the principled fix for it and it also
failed.

## Decisive result: delay h_state stays at chance (chance 0.167, majority ~0.19/0.22)

| arm | sample obs_map (control) | delay h_state shape | delay h_state color |
|-----|-------------------------:|--------------------:|--------------------:|
| wm-predict v2 (action-conditioned) | 0.988 / 1.000 | 0.214 | 0.214 |
| wm-predict v1 (latent-only reward) | (same control) | 0.095 | 0.155 |
| reward-only (`runs/collapse_trained`) | 0.988 / 1.000 | 0.190 | 0.238 |
| untrained init | 0.976 / 1.000 | 0.250 | 0.214 |

The on-screen `sample obs_map` control decodes at ~1.0 in every arm, so the probe is valid.
`delay h_state` is at chance in every arm: the wm-predict configurations (0.21 and 0.10-0.16)
are not above the reward-only baseline (0.19-0.24) or the majority class (0.19-0.22). The
external `mem_slot @choice` latch decodes the sample at ~0.89-0.98 in all arms, but that is
the separately-engineered obs_map memory, not the RSSM.

## Verdict: the KILL gate

Per the pre-stated Stage-1 gate: "KILL if the losses train down but `delay h_state` stays
at chance: the discrete gumbel-softmax STE categorical latent is the wall." Both
configurations meet this exactly. The value-equivalent objective with BPTT through the delay
does not make the discrete RSSM latent encode and retain the sample. The effort stops; it is
not iterated further.

## Honest scope and the remaining lever

- Single seed (42) per configuration. The effect is clean (chance across all arms,
  control at ~1.0), but it is a confirmation within seed 42, not a multi-seed law.
- Confound: the weak DMTS reward signal (the agent acts near-randomly at the choice). The
  action-conditioned reward head was the principled correction and also failed, which
  reduces but does not fully eliminate this confound.
- The mechanism (BPTT + action-conditioning + KL) is correct and the loss trains; what does
  not happen is the discrete latent encoding the low-information sample identity (the same
  low-variance-identity finding that defeated R1, now at the recurrent-state level).

The remaining lever is a **continuous / higher-capacity RSSM latent** (replacing the
gumbel-softmax categorical), paired with an identity-pressuring objective. That is a
separate, larger architectural bet with its own uncertain payoff, and it is escalated to the
project owner rather than pursued silently. The `--enable-wm-predict` mechanism stays in the
codebase, default off, as a documented negative result alongside `--enable-wm-recon`.

## Where this leaves the perception/competence thread

Across this session: R1 reconstruction (2 targets) FAILED to put stimulus identity into the
RSSM latent; the value-equivalent world model (2 reward designs) FAILED to put working
memory into the RSSM recurrent state. Both point at the same wall: the discrete RSSM latent
does not encode low-variance task-relevant detail, with the current objectives. The
causal-efficacy evidence on DMTS/WCST remains blocked on this. The honest options are a
continuous-latent architectural bet, or measuring the consciousness signatures that do not
require task competence (per the 2026-06-02 reading-#2 decision and the Butlin rubric).

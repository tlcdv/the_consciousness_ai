# The reversal does not read the existence drive

Date: 2026-10-31. Rules: `docs/decisions/2026_10_31_reversal_drive_off.md`, pushed before any run.

**The reversal gate cannot test the existence drive at these runs.** The records of the movement are byte for byte the same with the drive on and off. So the full gate of about 2.6 hours (stage C) did not run, by the stated rule. This is a test of identity, and it is not a new reversal result. No score changed. AE-1 stays PARTIAL, and its score is the owner's decision.

## Stage A: same revision, drive on and off

Seed 999 (discarded), 6 episodes, revision `e117bbe` with no local changes. Both runs have 7 lines in `episodes.csv` and 1201 lines in `metrics.csv`.

| File | Drive on | Drive off | Rule |
|---|---|---|---|
| `value_gate.json` | `306eefc3edbb5c81fce2eb0d86ba3583` | same | Holds |
| `tectal_orienting.json` | `fcfd560d1f91f45c73557c603f75c1d3` | same | Holds |
| `map_binding.json` | `84020d18804031719023c62745c2ec59` | same | Holds |
| `metrics.csv` | `8fbcb1ad74e28631204d8033bd8ef82b` | `1608ff686983d74cd3292fff9c2da4ec` | Expected to differ, reported only |

The manifests of the runs record `on` and `off`. So the flag took effect, and 12 columns of `metrics.csv` differ.

| Column of `metrics.csv` | Rows that differ, of 1200 |
|---|---|
| `ignition_salience` | 1198 |
| `sync_r` | 662 |
| `phi` | 500 |
| `reward` | 268 |
| `gate_attention` | 34 |
| `gate_coherence` | 18 |
| `broadcast_mag`, `gate_confidence` | 17 each |
| `gate_stability` | 19 |
| `is_conscious`, `gate_adaptation`, `bid_winner` | 16 each |

## Stage B: one full run, drive off, against the gate run

Seed 701, 60 episodes, `--dark-room-food-switch 29`, revision `e117bbe`, 23:14 end, about 25 minutes. The run has 61 lines in `episodes.csv` and 12001 lines in `metrics.csv`. The reference is `runs/rv_switch_s701`, with the drive on, at an older revision.

| File | Reference | Drive off | Rule |
|---|---|---|---|
| `value_gate.json` | `ba464e91176fc8036aead3c4cdd6fa5e` | same | Holds, this is the required md5 |
| `tectal_orienting.json` | `392ee4f4a5873b37eee5ab79f203ae5c` | same | Holds |
| `map_binding.json` | `7994c99fc8aeba0ebbdcf5e39fde702c` | same | Holds |
| `metrics.csv` | `2752358f446218ac351e78cf65dfefca` | `74e8ca8e0e54f090f531e692e10c8689` | Reported only |
| `episodes.csv` | `0e21a61305093abee8c2feda3050c359` | `10f799ec711db0aa8eb77ad54263007c` | Reported only |

Stage B compares two revisions and two drive settings. Because the three records are the same, the extra run with the drive on at the current revision was not needed. For `metrics.csv` and `episodes.csv` this comparison does not separate the drive from the code. Stage A does that for the short run only.

## Limits

- This shows one short run at seed 999 and one full run at seed 701. The result is deterministic, so I claim no seed replication. Seeds 702 and 703 did not run with the drive off. For them the identity is a reading of the code.
- The R1 to R3 numbers of seed 701 in `docs/results/reversal_2026_10.md` hold with the drive off by identity. R1 to R3 at 3 of 3 seeds with the drive off are not measured.
- The frozen and scrambled controls of the value gate did not run.
- The identity covers the movement only. The drive still changes other parts of the agent, as the differing columns show. The drive is present in the agent, and this behaviour does not read it.
- The logs of both stage A runs carry the warning that stage 2 appraisal failed, so emotion is reflex only. So this result describes the agent with that failure in place. I did not test the drive with a working appraisal.
- By my reading of the code, the drive could reach the movement through the reward bonuses in the value gate, if they cross the threshold of 0.5. In these runs the records show no such change. A room with other reward values could give another result.
- Stage A ran with `--record-episodes none`. Stage B used the default recording, the same as the reference run.

## What this changes

If behaviour in this gate does not read the drive, then it does not come from a homeostatic struggle. Public text that describes the reversal in those terms needs a review. That decision is the owner's. A repeat on fresh seeds would be a replication of the reversal and not a test of the drive, so that is also the owner's decision.

## Where the drive stops, added 2026-11-08

Read-only. A read of the code at revision `1a15d75` and of the stage A records. No run was trained. The md5 of the three records and of both `metrics.csv` files of `runs/drv_a_on_s999` and `runs/drv_a_off_s999` were checked again and match the tables above.

**The drive cannot reach the movement in these runs, and no flag of the present gates changes that.** It is cut at three points.

1. **The room.** The battery starts at 1.0 in each episode and falls by 0.001 per step (`simulations/environments/simple_visual_env.py:265`). Food does not restore it. In 200 steps it stays at 0.8 or more, so the parts that need low energy never act. The body bid rises from 0.05 to 0.15 only when energy is below 0.4 (`scripts/training/train_rlhf.py:1204`), and the energy valence term is 0 while energy is above 0.5 (`models/emotion/affective_modulator.py:121`). `bid_body` is 0.05 at 1200 of 1200 steps in both stage A runs, and at 12000 of 12000 steps in `runs/rv_switch_s701`, which used the drive on.
2. **The movement path.** With `--tectal-orienting value`, `_wire_value_gate` calls the policy first and then replaces its action with the motor map output, or with no movement when there is no target. The reward shaping terms that the drive gates (`models/emotion/reward_shaping.py:200` and `:206`) and the self-vector (`models/self_model/self_representation_core.py:427`) reach only the policy.
3. **The gate threshold.** The only input of the value gate that the drive can change is the reward, through the intrinsic bonus `0.5 * delta_phi + 0.1 * curiosity` (`train_rlhf.py:2003`). The gate counts a reward above 0.5 as food and any other reward as no food (`models/core/value_gate.py:67-75`). The `reward` column of `metrics.csv` is the value that the gate reads (`train_rlhf.py:2046`).

| Stage A, drive on against drive off | Value |
|---|---|
| Steps where the reward differs | 268 of 1200 |
| Largest difference | 0.000878 |
| Largest reward at or below 0.5, both runs | -0.004971 |
| Smallest reward above 0.5, both runs | 0.998937 |
| Steps on different sides of 0.5 | 0 of 1200 |

The smallest gap to the threshold, 0.498937, is 568 times the largest difference. So the gate saw the same outcomes, learned the same values and made the same choices. At revision `1a15d75` phi is off by default, so the phi term of the bonus is 0 and only curiosity is left. Curiosity is set to 0 at steps with a positive room reward.

What the drive still changes: with the drive off the affective modulator gives no affect (`affective_modulator.py:113`). That changes the bid modulation, the winner at 16 steps, and with it the broadcast, phi and curiosity, which are the differing columns of stage A.

A gate that tests the drive would need a need that changes inside an episode, a resource that restores it, and a choice that the need can move. A primary reward in homeostatic reinforcement learning is an outcome that reduces a physiological need (Keramati and Gutkin, eLife 3, e04811, 2014, doi 10.7554/eLife.04811). In larval zebrafish, hunger shifts the choice from avoidance to approach and recruits more prey-responsive neurons in the tectum (Filosa et al., Neuron 90, 596, 2016, doi 10.1016/j.neuron.2016.03.014). In this agent the need state reaches neither the value of the outcome nor the tectal selection. A room signal that lowers `energy` is a noxious channel under rule E6, and rule E2 allows no self-preservation drive or reward term, so such a gate needs an owner decision under the ethics framework. The competing goals gate uses two foods and the drive off instead (`docs/decisions/2026_11_08_competing_goals_gate.md`).

Limits: the threshold numbers come from one short run of 6 episodes, and the result is deterministic, so no seed replication is claimed. The claim that the binder's input does not depend on the drive is a reading of the code, supported by the identity of the stage A and stage B records. How the self-model computes fatigue with the drive on was not read. That changes only the size of the affect path, and the cut at the gate threshold does not depend on it.

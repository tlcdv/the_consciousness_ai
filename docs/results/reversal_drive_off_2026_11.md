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

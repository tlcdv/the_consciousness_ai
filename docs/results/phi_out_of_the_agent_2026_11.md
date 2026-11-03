# Phi out of the agent. The default is unchanged, the off path runs, and on one seed the rewards barely move

Date: 2026-11-03. Decision document and rules: `docs/decisions/2026_11_03_phi_out_of_the_agent.md`,
pushed at commit 3549da9 before any run, with one sentence reworded at d4d8b12, also before any run. All
runs used Python 3.12.10 and the Gate B7 command (the recorded argv of `runs/f2_py312_s73`). The runs of
this branch ran at d4d8b12 with no local changes. No indicator moves, and no score changes.

## Verdict

1. **I1 holds. With the default `--phi-control on`, the agent is unchanged.** A checkout of main at
   a1dae12 and this branch gave the same `metrics.csv`, md5 `9ac225a5b6a68625da4721b838f1f136`
   (`runs/phi_i1_main_s1298` and `runs/phi_i1_branch_s1298`, 2 episodes, seed 1298).
2. **I2 holds. The off path runs.** `runs/phi_i2_off_s1298` ended with 2 rows in `episodes.csv`, and all
   400 recorded steps have `phi_method` "off" with phi recorded as NaN. The loop sets "off" only in the
   branch that calls no pyphi, so every step took that branch.
3. **P, reported without a verdict. On one seed, removing phi changed the path and barely changed the
   rewards.** The actions of the two arms first differ at episode 0, step 11. The rewards of the episodes
   stay within 1.012 in episode 0 and within 0.01 in episodes 1 to 9. One seed is a hypothesis.

## Numbers of P

`runs/phi_p_on_s1299` and `runs/phi_p_off_s1299`, 10 episodes each, seed 1299.

| Episode | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Total reward, phi on | 47.581 | -1.983 | -1.995 | -1.999 | -2.000 | -1.990 | -1.996 | 0.029 | -1.999 | -1.999 |
| Total reward, phi off | 46.569 | -1.983 | -1.996 | -2.000 | -2.000 | -2.000 | -1.996 | 0.028 | -1.998 | -1.997 |

The mean of the last 5 episodes is -1.5911 with phi on and -1.5923 with phi off. In the on arm, pyphi
computed phi at 399 steps, and phi was carried forward at 1600 steps. Its largest value was 0.0106 and
its mean 0.00158, so it scaled exploration down to 0.894 at the most.

## What this means

The flag does what the decision document says, and it leaves every earlier run reproducible. On the one
seed measured, phi was a small influence on the agent. It changed which actions were taken, and not the
rewards the agent reached. Whether `--phi-control off` becomes the default is the owner's decision. If it
does, the README, the website and the logo stop presenting phi, and runs from then on are not
comparable with runs before, except through `--phi-control on`.

## Limits

One seed of 10 episodes for P, and one configuration, the dark room of Gate B7 with the drive on. The
memory values of the off path were chosen to match what phi gave in practice, which the decision
document explains.

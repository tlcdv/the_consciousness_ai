# The direct route to the workspace: design B stops after its cheap checks FAILED

Date: 2026-10-31. No tectum, no workspace, no training run of the agent, and no model code changed. Rules: the
docstring of `scripts/analysis/check_route_learnability.py`, each stage pushed before its run (commits 312bfe1,
a1929b1, af77350, 5d4163d).

## Verdict

FAILED. With the real binder selecting the cells, a plain actor-critic did not learn to use the route, at any of 6
seeds, in two versions of the check. By the pre-stated rule, design B of
`docs/decisions/2026_10_31_workspace_link_design.md` stops, and no live run follows.

Two findings stand. With a perfect selection, a policy can use a place code, and cutting it removes the behaviour.
The real binder holds its selection under smooth movement and loses it under the policy's exploration noise.

I built the checks because a decode gate of the route would pass by construction. They cost about 21 minutes of one
CPU thread. A live test would have needed the build and about 2.6 hours of machine time.

## Stages, in the order they ran

Reading: light first in episodes 21 to 40, within the first 60 steps, chance about 0.5. The rule at each stage asks
for at least 0.8 and at least 0.2 above the blank input.

| Stage | Input | Seeds | Rule | Route | Blank | Verdict |
|---|---|---|---|---|---|---|
| First rule | Truth of the room | 1001 | C1 on the direction control | 1.00 | 0.40 | C1 FAILED (direction 0.55), run stopped |
| Follow-up | Truth of the room | 1002 to 1006 | F1 | 0.95, 0.80, 0.90, 0.75, 0.90 | 0.40, 0.30, 0.30, 0.40, 0.45 | FAILED, 4 of 5 |
| Follow-up, cut | Truth of the room | 1002 to 1006 | F2 | see below | | PASSED, 5 of 5 |
| Binder | Group of the real binder | 1007 to 1009 | B1 | 0.40, 0.40, 0.50 | 0.30, 0.55, 0.40 | FAILED, 0 of 3 |
| Binder, cut | Group of the real binder | 1007 to 1009 | B2 | | | FAILED, 1 of 3 |
| Low noise | Group of the real binder, exploration noise starts at 0.1 | 1010 to 1012 | N1 | 0.65, 0.40, 0.40 | 0.40, 0.45, 0.30 | FAILED, 0 of 3 |
| Low noise, cut | The same | 1010 to 1012 | N2 | | | FAILED, 0 of 3 |

**My errors in these rules.** I stated that a failure of C1 would mean that the policy cannot learn even from a
perfect input. The route arm learned at that seed, so the direction arm was a weak control, and I did not find why it
fails. A smoke run on a discarded seed caught two errors before any counted run. My first reading, the share of steps
in the food, gave 0.82 in one untrained episode, and I had the wrong learning rate. The check changed three times
after results. Each change was pushed before its run, on fresh seeds, and the low noise check was stated as the last.

## The cut: a causal result, in a simulation only

The perfect place code for the first 30 episodes, then the blank input. Episodes 31 to 40, seeds 1002 to 1006.

| Arm | Light first | Share of steps in the food |
|---|---|---|
| Route | 1.00, 0.80, 1.00, 0.90, 0.90 | 0.971, 0.711, 0.929, 0.587, 0.838 |
| Route cut | 0.50, 0.30, 0.40, 0.50, 0.30 | 0.039, 0.039, 0.010, 0.150, 0.037 |
| Blank | 0.50, 0.20, 0.20, 0.40, 0.30 | 0.123, 0.042, 0.003, 0.020, 0.118 |

The input here comes from the truth of the room. It says nothing about the workspace of the agent, because no
workspace ran.

## Why the binder check failed

The share of steps with a group on the light, 20 episodes per motion, no policy, seeds 991 to 993, reported only. The
agent holds still for 13 steps in all motions.

| Motion | Step 13 | Steps 14 to 30 | Steps 31 to 60 |
|---|---|---|---|
| Still | 1.00, 1.00, 0.95 | 1.00, 0.99, 1.00 | 1.00, 1.00, 1.00 |
| Smooth, toward the light, noise 0.1 | 1.00, 1.00, 0.95 | 1.00, 1.00, 1.00 | 1.00, 1.00, 1.00 |
| Jitter, noise 1.0 | 1.00, 1.00, 0.95 | 0.75, 0.71, 0.64 | 0.45, 0.61, 0.54 |

The earlier gates moved the agent with the smooth reflex, and the binding held there. The binder compares timing at
fixed places, and a large random movement shifts the object across the map between steps. That is my reading of the
cause, and I did not test the mechanism.

Lower noise did not rescue the check. With noise that starts at 0.1, the group was present at only 0.45 to 0.56 of
steps, against 0.92 to 0.93 before, and the group on the light at 0.53 to 0.64. I did not test why. So the two needs
pull against each other in this test. The binder needs small movements, and this policy with this input did not learn
with them.

## What this shows, and what it does not

- The workspace link cannot be tested in this way today. A route is useful only with a selection that survives the
  policy's own movement.
- This does not refute design B. No live run took place, and these checks used one policy, one learning rate and the
  raw reward.
- The value gate, the reversal and the motor map stand. They use the smooth reflex and not this policy.
- The binder's weakness under movement adds to `docs/results/rhythm_binding_harder_rooms_2026_10.md`. It was not known
  before, because no earlier gate moved the agent with large noise.
- No indicator moves. GWT-2 stays PARTIAL.

## Runs

Python 3.12.10. Output in `runs/_audit_followup/oct13/` (local, gitignored): `route_learnability.out`,
`route_followup.out`, `route_binder.out`, `binder_motion.out`, `route_low_noise.out`, with their `.json` files.

```bash
python -m scripts.analysis.check_route_learnability --follow-up --output-json route_followup.json
```

```bash
python -m scripts.analysis.check_route_learnability --binder --output-json route_binder.json
```

```bash
python -m scripts.analysis.check_route_learnability --binder-low-noise --output-json route_low_noise.json
```

```bash
python -m scripts.analysis.diagnose_binder_motion
```

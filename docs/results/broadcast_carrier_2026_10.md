# The recorded winner is not the module whose vector the policy receives. With the merge that makes them agree, competition FAILED at 6 of 6 seeds

**With `--broadcast-merge top_winner`, where the policy receives the vector of the module
with the highest bid, vision takes 0.990 to 1.000 of ignited steps at 6 of 6 seeds.** The
same seeds with the default merge gave 0.564 to 0.846 and passed the competition criteria
of Gate B7 and Gate B8. The prediction that this would occur was written before the runs
and PASSED. The competition between the bids that the gates measured needs the default
merge.

**With the default merge, the policy receives the vector of the recorded winner at 0.025
to 0.095 of ignited steps.** At the other steps two modules pass the threshold, the record
names the one with the higher bid as the winner, and the broadcast holds the vector of the
one with the lower bid. This holds in all 39 recorded dark room runs with the default
`--broadcast-merge legacy` that were measured. In the 9 recorded runs with
`--broadcast-merge top_winner` the share is 1.000.

**Consequence.** Every dark room gate from Gate B to Gate B8 judges which module has the
highest bid. No gate judges which vector the workspace broadcasts. A statement such as
"hearing won 0.30 of ignited steps" is true of the bids. At most of those steps the policy
received the vision vector.

No default is changed and no indicator moves. This document also records why the GWT-2
indicator is not proposed for a re-score.

## The question

Gate B6 and Gate B7 show that the module with the highest bid changes between vision and
hearing (`docs/results/dark_room_senses_2026_09.md`). The GWT-2 indicator lost its
IMPLEMENTED status on 2026-08-17 because the winner did not change
(`docs/consciousness_indicators_butlin.md`). The question was if the new gates are evidence
for the indicator. To answer it, the selected module must be the module whose information
the workspace holds. That was not measured before.

## What the code does

1. `GlobalWorkspace._resolve_competition` admits every module whose bound bid is at least
   0.8 of the ignition threshold, strongest first. It does not admit one module only.
2. The training loop records the first of them as the winner
   (`scripts/training/train_rlhf.py`, `winner=workspace.state.winners[0]`).
3. With `--broadcast-merge legacy`, the default, each admitted module writes its payload
   over the payload before it. The broadcast tensor is then the one of the weakest admitted
   module that has a payload (`models/core/global_workspace.py`, lines 116 to 121). The
   help text of `--broadcast-merge` states this.
4. With `--broadcast-merge top_winner` the broadcast keeps the tensor of the strongest
   admitted module.

The mechanism was known and has a flag. How often two modules are admitted was not known.

## Observation on recorded runs

Read-only probe, `python -m scripts.analysis.probe_broadcast_carrier --runs <run folders>`,
pinned by `tests/test_broadcast_carrier_probe.py`. It compares the recorded broadcast
vector with the recorded vision and audio vectors, element by element, at each ignited
step. It does not rebuild the broadcast from the bids.

| Runs | Merge | Two or more modules admitted | Broadcast is the recorded winner's vector | Broadcast is the audio vector |
|---|---|---|---|---|
| Gate B4, seeds 57 to 59 | legacy | 0.940 to 0.964 | 0.036 to 0.060 | 0.450 to 0.868 |
| Seeds 57 to 59, moved layouts | legacy | 0.958 to 0.968 | 0.032 to 0.042 | 0.602 to 0.951 |
| Gate B5, seeds 60 to 62 | legacy | 0.941 to 0.971 | 0.029 to 0.059 | 0.600 to 0.805 |
| Gate B6, seeds 63 to 72 | legacy | 0.932 to 0.975 | 0.025 to 0.068 | 0.526 to 0.951 |
| Gate B7, seeds 73 to 82 | legacy | 0.905 to 0.974 | 0.026 to 0.095 | 0.486 to 0.752 |
| Gate B8, seeds 83 to 92 | legacy | 0.948 to 0.974 | 0.026 to 0.052 | 0.478 to 0.838 |
| Seeds 57 to 59, moved layouts | top_winner | 0.961 to 0.971 | 1.000 at all 3 | 0.002 to 0.039 |

All shares are of ignited steps, 1284 to 1909 per run. The Gate B8 row was added after
those runs ended. The other rows were in the first version of this document. In no run is the broadcast equal to
neither vector, and in no run is it all zeros at an ignited step.

Three readings of the table.

1. Two or more modules are admitted at 0.905 to 0.975 of ignited steps, and vision and
   hearing are both among them at every such step. With the default merge, the steps with
   one admitted module are all steps where vision is alone.
2. With the default merge the policy receives the audio vector at about half to nearly all
   ignited steps. At Gate B6 seed 68 the record gives vision 0.996 of ignited steps, and
   the policy received the audio vector at 0.951 of them.
3. The task criterion of the gates reads "hearing has the highest bid more often in
   episodes where the light is out of view". With the default merge, a step where hearing
   has the highest bid and vision is also admitted is a step where the policy receives the
   vision vector.

This is a description of recorded runs. It does not show what the agent would do with
another merge.

## What the earlier `top_winner` runs showed

The audit of 2026-10-03 ran seeds 57 to 59 with `--broadcast-merge top_winner`
(`docs/results/audit_gpu_checks_2026_10.md`). Vision took 0.998, 0.998 and 0.961 of ignited
steps and the task link was absent. Those runs had `--learned-valence` with the linear
boost. The lock-in probe on them gives a mean learned value of vision of 4.502, 0.881 and
-0.080. So at seed 59 vision took 0.961 with a learned value near 0. The lock with
`top_winner` is therefore not explained by the learned value alone.

## The check, stated before it is run

Code at the revision of the commit that adds this document. Two arms of 3 seeds. Each run
is paired with a gate run of the same seed that differs in the merge flag only.

- **Arm T1.** The Gate B7 command (no `--learned-valence`) with
  `--broadcast-merge top_winner`. Seeds 73, 74 and 75. Output `runs/carrier_t1_s<seed>`.
- **Arm T2.** The Gate B8 command (`--learned-valence --valence-boost saturating`) with
  `--broadcast-merge top_winner`. Seeds 83, 84 and 85. Output `runs/carrier_t2_s<seed>`.

Check that the merge was in use. `share_winner_vector` of the probe must be 1.000 in every
run. A run where it is not is void.

**Prediction P3, arm T1.** The vision share of ignited steps, as the gate probe computes
it, is 0.95 or more at 2 or more of the 3 seeds. The basis is seed 59 above. If the vision
share is below 0.95 at 2 or more seeds, P3 FAILED, and the competition between the bids
holds when the policy receives the vector of the module with the highest bid.

Arm T2 has no prediction. It is reported only. The gate probe output of both arms is
reported in full. Three seeds do not make a gate.

Rules. One run per seed and arm. A run with fewer than 10 episodes or 2000 steps on disk is
run again with the same seed, and that is reported. No other rerun.

What P3 means. P3 PASSED says that the competition seen in the gates needs the default
merge, where the policy receives the vector of the module with the lower bid. P3 FAILED
says that the competition does not need it at these seeds.

## Result of the check, 2026-10-03. P3 PASSED, and competition FAILED at 6 of 6 seeds

Run at revision `831541f`, after the section above was committed. Each run has 10 episodes
and 2000 steps on disk. No run was repeated. In every run the broadcast is the recorded
winner's vector at 1.000 of ignited steps, so the merge was in use. In arm T2 every
recorded step holds the boost rule `saturating`.

| | T1, seed 73 | T1, seed 74 | T1, seed 75 | T2, seed 83 | T2, seed 84 | T2, seed 85 |
|---|---|---|---|---|---|---|
| Vision share with `top_winner` | **1.000** | **0.999** | **0.997** | **0.996** | **0.996** | **0.990** |
| Vision share of the paired gate run, default merge | 0.564 | 0.787 | 0.736 | 0.672 | 0.846 | 0.762 |
| (1) runner-up share with `top_winner`, at least 0.05 | **0.000** | **0.001** | **0.003** | **0.004** | **0.004** | **0.010** |
| (2) silence with `top_winner`, below 0.50 | 0.302 | 0.267 | 0.237 | 0.195 | 0.194 | 0.169 |
| (4) task, rho per seed with `top_winner` | not defined | -0.059 | -0.180 | 0.261 | 0.254 | 0.329 |
| Two or more modules admitted | 0.973 | 0.970 | 0.939 | 0.959 | 0.956 | 0.965 |

Bold marks a failed criterion or a share at or above the kill limit of 0.95. The paired
gate runs are Gate B7 for arm T1 and Gate B8 for arm T2. At seed 73 hearing never has the
highest bid, so the rho of that seed is not defined.

**P3 PASSED.** In arm T1 the vision share is 0.95 or more at 3 of 3 seeds, and the
prediction asked for 2.

Arm T2, reported only. The vision share is 0.990 or more at 3 of 3 seeds. The mean learned
value of vision in those runs is 0.075, 0.574 and 0.230, so the learned valence is not the
cause here.

The task criterion over 30 episodes. Arm T1 gives a pooled rho of -0.172 with a one-sided
p of 0.754. Arm T2 gives 0.238 with a p of 0.066. Neither passes.

Hearing is still admitted at 0.939 to 0.973 of ignited steps in these runs. It passes the
threshold, and its bid is below the vision bid.

With the 3 audit runs, vision took 0.961 or more of ignited steps in 9 of 9 runs with
`top_winner`, over 3 configurations.

## Why GWT-2 is not proposed for a re-score

What the source asks. Butlin et al. (2023, section 2.2.3) derive GWT-2 from two conditions.
The workspace has a smaller capacity than the modules that feed it. And, in their words,
there is "an attention mechanism that selects information from the modules" for
representation in the workspace. Section 2.2.1 describes the same thing as a contest for
entry that the stronger representation wins. State dependence of the selection is a
separate indicator, GWT-4. The authors judge example systems from their architecture and
say that the Perceiver "arguably" has GWT-1 and GWT-2.

What the project rule asks. An indicator is IMPLEMENTED when there is empirical evidence
that the system has the property. Code that holds the mechanism is not that evidence
(`docs/consciousness_indicators_butlin.md`, the rule of 2026-08-11).

Why the gates are not that evidence.

1. The gates measure the module with the highest bid. The property is about the
   information the workspace holds. With the default merge the two differ at about 0.95 of
   ignited steps, and the workspace holds the information of the module with the lower
   bid.
2. With the merge that makes the two agree, vision took 0.961 or more of ignited steps in
   9 of 9 runs, over 3 configurations. That is the state the indicator was demoted for on
   2026-08-17.
3. The admission rule does not limit the workspace to one module. Two modules pass it at
   0.905 to 0.975 of ignited steps. The single broadcast tensor is a result of the merge
   order.
4. Gate B6, the configuration with the learned valence, FAILED on competition at 3 of 10
   seeds. Gate B7 and Gate B8 passed with two other configurations. All three use flags
   that are off by default, and all three use the default merge.

What evidence a re-score would need. These are conditions, and each is a decision for the
project owner.

- A merge where the broadcast holds the vector of the module with the highest bid.
- A gate, written before its runs, that passes competition at every seed with that merge.
- The configuration named in the rubric row, because the default configuration is the one
  measured on 2026-08-17.

## What this establishes

- In 39 recorded runs with the default merge, the policy receives the vector of the
  recorded winner at 0.025 to 0.095 of ignited steps.
- The gates from Gate B to Gate B8 are statements about bids.
- At 6 seeds, in 2 configurations, the only change from the default merge to `top_winner`
  moves the vision share from 0.564 to 0.846 up to 0.990 to 1.000. The competition between
  the bids needs the default merge at these seeds.

## What this does NOT establish

- Not that the bids carry no information. The bid competition is measured and it varies
  with the task (Gate B6 and Gate B7).
- Not that the default merge is wrong for the agent. Which vector the policy should
  receive is a design decision.
- Not what the agent does with `top_winner` at 10 seeds. Only 3 seeds per arm are run.
- Not why vision takes every step with `top_winner`. The cause is open. The learned value
  is excluded as the cause in arm T1, which runs without it.
- Nothing about affect or experience.

## Decisions for the project owner

1. The default of `--broadcast-merge`. It was already an open decision after the audit.
2. The wording of public texts that say a module "won" a step. With the default merge the
   accurate statement is that the module had the highest bid.
3. If a gate with `top_winner` at 10 fresh seeds is run, and with which configuration.
   After the result above, the cheaper next step is a diagnosis of why the hearing bid
   stays below the vision bid when the policy receives the vision vector.

## Reproduce

```
python -m scripts.analysis.probe_broadcast_carrier --runs runs/gate_b6_s63 ... runs/gate_b6_s72
python -m scripts.analysis.probe_broadcast_carrier --runs runs/gate_b7_s73 ... runs/gate_b7_s82
python -m scripts.analysis.probe_valence_lockin --runs runs/gate_b4_top_winner_s57 \
    runs/gate_b4_top_winner_s58 runs/gate_b4_top_winner_s59
python -m scripts.analysis.probe_gate_b2 --runs runs/carrier_t1_s73 runs/carrier_t1_s74 runs/carrier_t1_s75
python -m scripts.analysis.probe_gate_b2 --runs runs/carrier_t2_s83 runs/carrier_t2_s84 runs/carrier_t2_s85
python -m scripts.analysis.probe_broadcast_carrier --runs runs/carrier_t1_s73 ... runs/carrier_t2_s85
python -m scripts.analysis.probe_broadcast_carrier --runs runs/gate_b8_s83 ... runs/gate_b8_s92
```

The run commands of the two arms are the Gate B7 and the Gate B8 commands in
`docs/decisions/2026_09_16_precision_weighted_bids.md` with `--broadcast-merge top_winner`.

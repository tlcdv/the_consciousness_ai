# The path from the workspace to the movement: designs compared, for the owner's decision

Date: 2026-11-15. Status: no design chosen, no code, no run. This document reads the project's result
documents and the code of the path. It compares seven designs and proposes one. The owner decides.

No indicator moves. The rubric and the clock stay as they are. Nothing here is a claim about consciousness.

## The conclusion, stated first

1. My proposal is design P. The policy reads the broadcast and chooses one of the candidates that the
   tectum has bound. The motor map of the tectum then orients to that candidate, as it does today. So the
   workspace decides which object, and the tectum decides where it is.
2. Design P closes all three cuts with one change of the interface. It adds no relay module, so the
   decision to build no thalamus stays. It does not change which modules compete, so the text of GWT-1
   stays.
3. Its first check is read-only on saved runs and can fail. It asks if the recorded broadcast at the choice
   step holds which candidate is the light. If it does not, design P stops before any code.
4. A repair of the policy alone (design R) is the cheaper build. By earlier results it is likely to fail
   as a test of behaviour, because the broadcast does not hold the direction of an object. I state this as
   my reading and not as a measurement of design R.
5. Design E, no link, stays a real option. It keeps the behaviour in the tectum loop, and the causal test
   of the workspace does not become possible in this room.
6. Design P also serves the open question of the agency indicator AE-1. In it the policy learns the
   choice, in place of a table of two values that the design sets.

## What is measured, and what each design must respect

| Measured | Source | What a design must do |
|---|---|---|
| In 11 of 12 replayed gate runs the action mean of the policy does not depend on the recorded broadcast. One run is undecided | `docs/results/workspace_action_2026_11.md`, check B | Show a ratio of check B of 0.5 or more before any gate of behaviour |
| The relay layer is in no loss term. It changed in 0 of 172 runs | `docs/results/action_core_relay_defect_2026_10.md` | Train the last layer, or take the output before it |
| "The relay layer alone is the cut" is EXCLUDED. The go signal moves by 0.004 to 0.095 over the steps, below the exploration noise of 0.103 to 0.156 | `workspace_action_2026_11.md`, check E | Not assume that a trained relay layer is enough |
| With the default merge the policy receives the vector of the recorded winner at 0.025 to 0.095 of ignited steps. With `top_winner`, vision takes 0.990 to 1.000 of ignited steps | `docs/results/broadcast_carrier_2026_10.md` | Say which vector the policy needs, and with which merge it arrives |
| The direction of the selected object is not readable from the workspace content or the broadcast | `docs/results/tectal_attention_2026_10.md`, `docs/results/content_map_sensitivity_2026_11.md` | Not need the direction from the broadcast |
| The path from the map to the content has three separate problems. They are sampling noise at the latent, capsules near zero or saturated, and a projection bias | `content_map_sensitivity_2026_11.md` | Not depend on a repair of that path, or name it as a first step |
| The real binder holds its selection under smooth movement and loses it under the exploration noise of a policy. A direct route FAILED at 6 of 6 seeds | `docs/results/route_learnability_2026_10.md` | Keep the movement smooth, so keep the motor map as the source of the movement |
| With a perfect place code, a plain actor-critic uses it, and a cut of the code removes the behaviour | `route_learnability_2026_10.md`, the cut | Include a cut of the broadcast as the test of use |
| The winner of the workspace as a priority term on the tectal map is EXCLUDED offline. It changed 14 of 300 choices, each toward the loudest band | `docs/results/workspace_priority_check_2026_11.md` | Use the content of the broadcast, and not only the name of the winning sense |
| Hearing tells the louder tone at AUC 1.0. Vision does not tell the two objects apart | `docs/results/content_relation_2026_11.md` | Expect the useful content in the audio vector |
| The value gate chooses a band by a softmax over salience plus a weight times a learned value. Its table of values is outside the workspace and outside the policy | `models/core/value_gate.py` | Say what replaces the table, or what stays |

## The three cuts, read from the code

| Cut | Code | What I read |
|---|---|---|
| 1. The merge | `models/core/global_workspace.py`, `broadcast_merge` | `legacy` (default) lets each weaker winner overwrite the vector, so the broadcast holds the vector of the weakest admitted module. `top_winner` keeps the strongest |
| 2. The policy | `models/self_model/action_selection_core.py`, `BasalGanglia.forward` and `ActionSelectionCore.update_policy` | The action is `thalamic_relay(go x (1 - no-go) x (1 - brake))`. The loss reads the go, no-go and brake outputs and the value. It does not read the relay layer |
| 3. The motor map | `scripts/training/train_rlhf.py`, `_wire_tectal_orienting` and `_wire_value_gate` | The policy is called first, so the random streams stay. Then the command of the motor map replaces the action. With idle `still`, a step with no target gives no movement |

A second defect of the policy, from my reading of the code. It is not measured. The loss moves the go
signal toward the recorded action when the advantage is positive, and away from it when negative. The
recorded action is the output after the relay layer plus noise, and the go signal is the value before
the relay layer. With a relay layer that keeps its random start, these are two different spaces. So a
relay layer that is added to the loss does not by itself make the go target correct. Check E agrees with
this in one respect. The pathways before the relay follow the broadcast very little after training.

The flag `--policy standard` selects a plain actor-critic head that exists in the code. The only result
document that names it is `docs/results/agent_competence_diagnosis_2026_06_01.md`. No dark room gate
used it.

## The designs

| Design | What changes | Cuts it closes | First check that can fail | Against, and risk | Cost |
|---|---|---|---|---|---|
| R. Repair the policy only | Behind a flag, the relay layer enters the loss or is removed, and the go target is set in one space. The motor map is off, or idle `policy` | 2 | The ratio of check B on a replay, after a short training run | The policy must then move the agent from the broadcast, and the broadcast does not hold the direction of an object. The exploration noise breaks the binder. I expect the ratio to pass and the behaviour to fail | Low to build, a gate to run |
| M. Repair the policy and use `top_winner` | Design R with the merge that sends the vector of the strongest module | 1 and 2 | The same | Vision takes 0.990 to 1.000 of ignited steps with this merge, so the competition that Gates B6 and B7 measured is lost. The vision vector does not tell the objects apart | Low |
| **P. The policy chooses, the tectum orients (proposed)** | Behind a flag, the choice among the bound candidates comes from the policy, which reads the broadcast. The motor map orients to the chosen candidate. The table of the value gate is not used in this mode | 1, 2 and 3 | Read-only on saved runs. From the recorded broadcast at the choice step, read which candidate is the light | The choice is a new interface, so it is a designed wire for the action. The learning is in the policy. It shows use of the audio content, and not of a visual object. See "What a passed gate would not show" | Medium |
| W. The winner of the workspace as a priority term | A term on the choice of the value gate when hearing wins | 3, in part | Done. EXCLUDED offline on 2026-11-11 | It can only favour the loudest band, so it adds nothing to the salience | None now |
| G. The policy acts when the tectum has no target | A room phase with no sound and no light in view. The policy moves the agent from what the broadcast held before | 2, and 3 for those steps | The ratio of check B, then a cut | It needs a memory of a place in the broadcast, and the place is not readable there today. A new room option. The exploration noise problem returns | High |
| U. Repair the path from the map to the content | Design A of `docs/decisions/2026_10_31_workspace_link_design.md` | None of the three. It is the path before the workspace | A decode of the direction in training mode | Three changes that cannot be told apart. It changes the path behind all earlier decodes. It is a condition of R, M and G, and not of P | High |
| E. No link | Nothing. The record says that the workspace is outside the behaviour in this room | None | None | GWT-2 and GWT-3 get no new evidence. The causal test of the workspace is not possible here | None |

## Why P

1. **It asks only for what the broadcast can hold today.** R, M and G need the direction or the place of
   an object in the broadcast. Two gates and one diagnosis found that it is not there. P needs the identity
   of a tone. Hearing tells the louder tone at AUC 1.0, and the audio vector is often the vector that the
   default merge sends. I did not measure that it is there at the choice step. That is the first check.
2. **It keeps the movement that the binder needs.** The motor map moves the agent with its smooth reflex.
   The policy makes one choice per episode, as the value gate does now, so its exploration is a choice and
   not a jitter of the movement.
3. **Its test of use is a cut.** With the broadcast replaced by a blank or by the broadcast of another
   episode, the choice must fall to chance. That is a small form of the workspace ablation test, in one
   room.
4. **It adds no module.** No relay module and no new competitor. The basal ganglia part of the policy
   already has go and no-go channels, and one channel for each candidate is a small change of its use.
5. **One build serves two open questions.** The link from the workspace to behaviour, and the choice that
   AE-1 asks the policy to learn.

## Why a constant policy must not be able to pass

If the candidates reach the policy in a fixed order, a policy that ignores its input can learn "take the
second one". Then a cut of the broadcast changes nothing, and the gate shows no use of the workspace. So
the gate needs a condition in which the correct choice differs between episodes, and the difference is
in the broadcast.

The rooms of the value gate and of the reversal have such a condition. With
`--dark-room-decoy-level 0.5,3.0` the level of the second tone changes at each reset
(`simulations/environments/simple_visual_env.py`), so the louder tone is the tone of the light in one
episode and the tone of the decoy in the next. With the candidates in order of salience, the place of
the correct candidate then changes between episodes. Two points need a rule before any run. The levels
come in a fixed order, so the episode number predicts the louder tone, and a policy with a memory across
episodes could use that. The working memory of the policy is never reset in the live loop
(`workspace_action_2026_11.md`, limits of check B). The rules must name a control for both.

## The steps of design P, as a draft and not yet the rules

| Step | What | Stops the design if |
|---|---|---|
| P0. Read-only, saved runs | From the recorded `broadcast` at the choice step, read which candidate is the light. Whole episodes held out, a null with shuffled labels, and the same read from `audio_content` as the upper reference. The runs are the value gate, reversal and competing goals runs, which record both vectors for each step | The read is at its null at 3 of 3 seeds. Then the broadcast does not hold the content, and the options are E, or the merge, or U |
| P1. Offline, no agent | A choice head trained on the recorded broadcasts and the recorded outcomes of some episodes, tested on the others. Both the go and no-go head of the project and the plain head | No head learns the choice above its null |
| P2. Build, flag off by default | The choice mode behind a flag. Bit identity with the flag off, by the md5 of the records of the movement of a fixed seed. `run_episode` and `init_components` are not refactored | The identity fails |
| P3. Gate, fresh seeds, 3 or more | Criteria of the form of L1 to L3 for the learned choice, the ratio of check B at 0.5 or more, and the cut. An arm with the table of the value gate, as it is today, is the reference | Any criterion fails. The verdict is then FAILED |

Cost by the run times on record. P0 and P1 are minutes on one CPU thread. A gate of 3 seeds and 2 or 3
arms of 60 episodes is some hours of machine time, one run at a time. I did not measure a run of this
design.

## What a passed gate would not show

- It would not show that the workspace holds information about a seen object. The useful content is
  expected in the audio vector.
- It would not show that the policy can move the agent. The motor map still makes the movement.
- It would not show the competition between modules that GWT-2 asks for. With the default merge the
  broadcast often holds the vector of the weaker module, and the rules must state which merge the gate
  uses and why.
- It would not move GWT-2, GWT-3 or AE-1 by itself. Each score stays the owner's decision.
- It would say nothing about consciousness.

## Ethics framework

Each design passes the mechanical test of rule E6. No design adds an input channel, and no signal lowers
`energy`, raises `fatigue` or raises `damage`. No design adds a self-preservation term (rule E2). Every
run declares the existence drive (rule E1), and I propose `off`, as in the competing goals gate.

## Indicator text that each design touches

| Design | GWT-1 | GWT-2 | GWT-3 | AE-1 | The decision to build no thalamus |
|---|---|---|---|---|---|
| R, M | No change | M changes which vector is sent, so the row needs a note | New evidence if a gate holds | No change | Stays |
| P | No change | The row needs a note on the merge that the gate uses | New evidence if the cut holds | New evidence if the policy learns the choice | Stays |
| G | No change | No change | New evidence if a gate holds | No change | Stays |
| U | No change | No change | No change | No change | Stays |
| E | No change | No change | No change | No change | Stays |

## Limits of this document

- No biology source was read for it. The designs are compared by the project's own measurements. A
  reading of the sources on action selection by the basal ganglia among targets of the tectum is a step
  before the rules of a gate, if the owner chooses P. The selection rule of Gurney, Prescott and Redgrave
  (2001) was read at abstract level only for an earlier check
  (`docs/results/workspace_priority_check_2026_11.md`).
- The second defect of the policy loss is a reading of the code. I ran nothing.
- The statement that the audio vector is often the vector of the broadcast comes from
  `broadcast_carrier_2026_10.md`, which measured the Gate B runs. I did not measure it at the choice step
  of the value gate runs. Step P0 does.
- "I expect design R to fail as a test of behaviour" is a prediction from three earlier results. It is not
  a result.
- The cost figures are estimates from the run times of earlier gates.

## Decisions for the owner

1. The design. It is one of P, R, M, G, U and E.
2. If P, a yes to step P0 as the next piece of work. It is read-only and trains nothing. Its rules are
   pushed before it runs.
3. If P, which head makes the choice in the build, the go and no-go head of the project or the plain
   head. My proposal is to decide after step P1, which compares both.

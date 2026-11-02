# Stage 2 appraisal, the change of the body state, and a signal of realness. Designs for the owner, no code

Date: 2026-11-02. Status: design only. The owner asked on 2026-11-02 for three things in the
emotion path. Stage 2 of the appraisal must run, so that dominance is no longer 0.0 at every step.
The change of the body state caused by the broadcast (`delta_proto_self` in
`docs/damasio_self_hierarchy.md`) must be computed. A signal of realness must mark content that
the senses and the body confirm. The owner also decided that reading and cheap checks come before
any build. This document compares the designs. The checks are in
`scripts/analysis/probe_core_self_appraisal.py`, whose rules were pushed before its first run.

## What the code does today

| Fact | Source |
|---|---|
| Stage 2 has never run. `evaluate_emotion` calls `map_state(broadcast)`, `map_state` also needs a goal vector, and the `TypeError` is caught. Every step so far used the reflex valence and a dominance of 0.0. A test pins this as an open owner decision | `scripts/training/train_rlhf.py:962`, `tests/test_emotion_appraisal.py` |
| With the goal passed, `map_state` cuts both vectors to the shorter length. The goal has 3 values, so the valence is the cosine of the first 3 of the 256 broadcast values with the goal. The intensity is `tanh` of the broadcast length | `models/core/qualia_mapper.py:64` to `:81` |
| The broadcast is a fixed vector of length about 0.39 with a movement of 5 to 15 percent of that size. Measured on DMTS runs, not on the dark room | `docs/results/broadcast_geometry_2026_08.md` |
| The workspace already calls `map_state` with the goal at each competition. Its input there is the tensor of the bound bids, so its stored valence compares the first three bids with the goal | `models/core/global_workspace.py:403`, `:439` |
| The goal is the literal `[1.0, -1.0, 1.0]`. No document states what its three values mean | `scripts/training/train_rlhf.py:1314` |
| Dominance has two readers. The modulator raises every bid by dominance times 0.05. The shaped reward adds dominance times 0.05, and the code names this term part of the "stay in control" survival drive | `models/emotion/affective_modulator.py:186`, `models/emotion/reward_shaping.py:197` |
| The ethics framework lists that reward term as a path of the existence drive. It adds 0.0 today only because Stage 2 does not run | `docs/ethics_framework.md`, table of the existence drive |
| The body state is `energy`, `fatigue` and `damage`. With the drive on, `energy` is a copy of the battery. With the drive off, the battery reaches nothing, and the internal depletion is harmless only because nothing reads it | `models/self_model/self_representation_core.py:186`, `docs/ethics_framework.md` |
| No prediction of the body state runs. The self-vector module predicts the next values of its features, `energy`, `fatigue` and `damage` among them, but it is off in the dark room runs of the value gate, and with the drive off it sees neutral values | `runs/vg_learn_s601/config.json` (`enable_self_vector` false), `models/self_model/self_representation_core.py:426` |
| A learned valence exists behind `--learned-valence`. It is a temporal difference critic over the module bids, trained on the external task reward only, and it changes the salience of the bids. It is not connected to Stage 2 | `models/emotion/learned_valence.py`, `scripts/training/train_rlhf.py:2655` |
| The action core's value estimate is recorded at each step as `value_estimate`. The movement network of that core ignores its input (relay weights changed in 0 of 172 runs) | `scripts/training/session_capture.py:123`, `docs/results/action_core_relay_defect_2026_10.md` |

## What the sources say

Each row says how much was read in this session.

| Source | Read | Finding |
|---|---|---|
| Seth, Suzuki and Critchley, Front. Psychol. 2, 395 (2012) | Main text, through a summary of the first 100,000 of 210,855 characters of the page | Presence, the sense that self and world are real, comes from the suppression of informative interoceptive prediction error. An uninformative prediction with no error does not give presence. Depersonalization and derealization follow from imprecise interoceptive predictions. A comparator for agency sits above the one for presence. The model has no equations |
| Barrett and Simmons, Nat. Rev. Neurosci. 16, 419 (2015), author manuscript | Main text, through a summary of the first 100,000 of 170,048 characters | Agranular visceromotor cortex sends predictions of the body state, and the mid and posterior insula compute the errors, weighted by precision. What is felt is mostly the prediction. Errors update the prediction, drive action, or change attention. No equations, and valence is not a named variable |
| Moors, Ellsworth, Scherer and Frijda, Emotion Review 5, 119 (2013) | Main text, the sections on appraisal variables, from the published file | The core appraisal variables are goal relevance, goal congruence, certainty, coping potential (control or power) and agency. Appraisal is one cause of the feeling, and it shapes the motivational, bodily and motor parts of the emotion |
| Gatus, Jamieson and Stevenson, Front. Hum. Neurosci. 16, 744487 (2022) | Main text, through a summary of the first 100,000 of 173,670 characters | Derealization and depersonalization are read as interoceptive predictions that become imprecise relative to the other senses. The order of the two is a theoretical proposal and is not tested |
| Damasio, as summarized in `docs/damasio_self_hierarchy.md` | Project document | The core self maps an object and the change of the body state it causes. Without a body, the simulated (as if) body loop is the only one available |
| Seth and Tsakiris (2018), Solms and Friston (2018), as summarized in `docs/affective_consciousness_solms_panksepp.md` | Project document, not read again | Emotion and selfhood as prediction of the body state |

Three points follow.

1. In appraisal theory, valence comes from goal congruence, the value of the event for the
   agent's concerns. Dominance belongs to coping potential, the agent's control over the event.
   Neither is the cosine of three coordinates, and neither is the length of the broadcast.
2. In both predictive accounts, the change of the body state that counts is the change the
   content predicts, compared with the change that then arrives. A large change that was not
   predicted is error, and error lowers presence. So a realness signal that rises with the size of
   the change has the wrong sign.
3. Presence needs an informative prediction of the body state. This agent runs none today. So no
   design can compute presence before such a prediction exists.

## Designs for the appraisal

| Design | What it does | Basis in the sources | Ethics framework | Cost | Cheap check |
|---|---|---|---|---|---|
| A. Pass the goal (the requested fix) | `map_state(broadcast, [1, -1, 1])` in Stage 2 | None. It reads 3 arbitrary coordinates and a length | Dominance above 0 switches on the survival reward term with the drive on. In my reading that counts as adding the term under rule E2, which needs a framework revision | Smallest | CC1 |
| B. Predicted change of the body state | A small read of the broadcast predicts the change of `energy` over the next window. Valence is the predicted gain. The core self change is the predicted change against the change that arrives | Damasio (core self, simulated body loop), Barrett and Simmons, Seth | It reads `energy`, which is the existence drive. With the drive off it must read nothing, so it would be zero. This is the known tension between body needs and rule E3 | A learned read and its tests | CC2, a necessary condition only |
| C. Learned value of the content | Valence is a value learned from the external task reward. Two learners exist. `LearnedValence` holds one value per module, and `ValueGate` one value per heard tone | Goal congruence learned from outcomes (appraisal theory). The learned valence cites Feinberg and Mallatt and the temporal difference error. The value gate PASSED its gate, 56 of 60 against 28 of 60 (`docs/results/value_gate_2026_10.md`) | Reads the task reward, not the body, as `LearnedValence` already does under rule E2. Works with the drive off | Wiring only | CC1 reads the recorded `value_estimate` of the action core as a stand-in. The two learners above are not recorded at each step in these runs |
| D. Keep Stage 2 off | Correct the docstring and the row of the Damasio document | Not applicable | None | None | None |

Dominance. No source maps coping potential to a quantity this agent computes. Any value above 0
also starts the survival term of the reward with the drive on. My proposal is that dominance stays
0.0 in every design above until a design for control exists.

## Designs for the realness signal

| Design | What it does | Basis | Problem |
|---|---|---|---|
| R1. The requested formula | sigmoid of a1 times the cosine of observed and predicted features, plus a2 times the size of the body change, minus b | None for the signs. a1, a2 and b have no source | It rises with the size of the body change, the opposite of Seth and colleagues, and it uses the outer senses where the model uses the body |
| R2. Precision-weighted match of the body prediction | Realness is high when an informative prediction of the body state matches what arrives | Seth, Suzuki and Critchley (2012) | Needs the prediction of design B first |
| R3. No single number | Record the predicted change, the change that arrives, and their difference, apart | The project finding that every vector tested carries the stimulus and every scalar tested does not (`docs/instrument_inventory.md`) | Not a signal the agent can use. A record for later analysis |

The bar of the Metzinger document applies to R2. A new signature must not repackage an existing
quantity. Prediction error of the outer senses exists already (`kl_map`). Only the error of the
body prediction would be new.

## The cheap checks

Both read recorded sessions of the value gate runs, `runs/vg_learn_s601` to `s603`, 3 seeds of 40
episodes of 200 steps, with the drive on. The vectors were recorded in live runs, which train, so
the module mode of the check is the mode of the runs. The full rules are in the probe docstring.

- CC1 asks if design A carries anything about the next food. Its valence and dominance are
  computed with the real `map_state` from each recorded broadcast. The recorded
  `value_estimate` is read beside it as a stand-in for design C, and the recorded reflex
  valence is the baseline.
- CC2 asks if the broadcast predicts the next food at all. If it does not, design B has nothing
  to learn from.

## What I recommend, and what could make it wrong

I give the recommendation after the checks, in `docs/results/core_self_appraisal_2026_11.md`. Before
the checks my reading is this. Design A has no basis in the sources and it starts a survival
term. Design C is the cheapest grounded design and it holds with the drive off. Design B follows
the core self most closely, it costs more, and it is part of the existence drive. A pass of CC1
would not give design A a basis. It would show that three coordinates happen to carry the signal.

## Decisions for the owner

1. Which appraisal design to build, if any. A and B need a decision under rules E2 and E3.
2. Whether dominance stays 0.0.
3. Whether the realness signal waits for a prediction of the body state (R2), or is recorded as
   parts (R3), or is not built.

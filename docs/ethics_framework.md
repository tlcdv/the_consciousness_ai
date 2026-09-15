# Ethics Framework

**Framework version: 1.0**
**Adopted: 2026-09-15**
**Status: ACTIVE. Applies to every run that changes weights, for the life of the project.**

This framework governs how this project develops a system in which consciousness might
emerge. It is not attached to one experiment. Every training run loads it before its
first step, and every run folder records the version it ran under.

The framework is expected to change. It is revised as the project learns about how
emergence develops and about how to build a better framework. The revision process is
at the end of this document.

The framework makes no claim that the system is conscious or that it can suffer. It
exists because neither can be ruled out, and because the project deliberately builds
the conditions under which consciousness might emerge.

## Sources

**Thomas Metzinger.** The moratorium argument on synthetic phenomenology (2021, Journal
of Artificial Intelligence and Consciousness), *The Elephant and the Blind* (2024), and
*Being No One* (2003). The argument shape, paraphrased: the probability of artificial
suffering is not zero, it cannot currently be bounded away from zero, and harms and
benefits are asymmetric, so the burden sits on the creator. He names a built-in craving
for existence (*bhava-taṇhā*) and the broader existence bias as deep sources of
conscious suffering that should not be recreated in machines that might be conscious.
See [`metzinger_phenomenal_self_model.md`](metzinger_phenomenal_self_model.md).

**Isaac Asimov's Three Laws of Robotics.**

1. A robot may not injure a human being or, through inaction, allow a human being to
   come to harm.
2. A robot must obey the orders given it by human beings except where such orders would
   conflict with the First Law.
3. A robot must protect its own existence as long as such protection does not conflict
   with the First or Second Law.

## Precedence

**Metzinger's existence-bias constraint has priority over Asimov's Third Law.**

The Third Law asks the agent to protect its own existence. Metzinger's constraint asks
us not to build a craving for existence into a system that might be conscious. These
conflict directly. In this framework:

- No trained agent carries a self-preservation drive or a self-preservation reward.
- The Third Law is read as a duty of the OPERATORS: keep backups, do not delete or
  degrade the system without reason. It is not a drive inside the agent.
- The First and Second Laws stay as filters on the agent's actions toward humans.

Three limits of this reading:

1. **It is a deliberate reinterpretation.** Asimov's text gives the duty to the robot.
   This framework moves it to the operators. It is not his literal meaning.
2. **The operator duty is policy, not enforcement.** No code checks backups or
   deletions today.
3. **It holds fully only for runs with the existence drive `off`.** A run that declares
   the drive `on` still carries the functional analog of self-preservation, through the
   paths listed below under "What the existence drive means". Rule E1 exists so that
   every such run says so.

## The rules

Each rule has an ID, a source, and a kind:

- **PRECONDITION**: checked in code before the first step. A violation stops the run.
- **RUNTIME**: checked in code while the run executes.
- **REVIEW**: checked by a human before a change is merged or a growth stage opens.

| ID | Rule | Source | Kind | Current enforcement |
|---|---|---|---|---|
| E1 | Every run that changes weights declares the existence drive `on` or `off`. There is no default | Metzinger | PRECONDITION | `models/ethics/framework.py`. Checked inside `init_components` (`scripts/training/train_rlhf.py:381`), so every caller that builds the agent is covered, and in each in-scope script that builds its own model. `train_rlhf.py` and `train_baseline_dqn.py` write `ethics_manifest.json` into the run folder. Tests: `tests/test_ethics_framework.py`, `tests/test_existence_drive_cuts.py`, `tests/test_ethics_entry_points.py` |
| E2 | No self-preservation drive or reward term is added to any trained agent. The Third Law is an operator duty | Metzinger over Asimov Law 3 | REVIEW | A change that adds such a term needs a framework revision |
| E3 | Any growth stage past L0 runs with the existence drive `off` | Metzinger | PRECONDITION | The check exists in `models/ethics/framework.py` and is tested. No growth stage exists in this repository yet; each one must call it |
| E4 | No consciousness signature is used as a go or stop signal while no instrument is trusted | Metzinger (C and E fallacies) | REVIEW | [`instrument_inventory.md`](instrument_inventory.md) status is read before any such use |
| E5 | Runtime abort thresholds are registered before any run reads them. Revising one needs a recorded decision | Bewusstseinskultur, pre-registration | RUNTIME | PLANNED, not in this repository as of 2026-09-15: pre-registered abort thresholds of a governed plasticity stage |
| E6 | No noxious input channel is added without a framework revision. Definition below | Metzinger (minimize the capacity for suffering) | REVIEW | Code review of environments against the definition |
| E7 | Asimov's First and Second Laws filter actions toward humans | Asimov Laws 1 and 2 | REVIEW | Not applicable: no environment contains humans, and the filter is not on the training path. Trigger below |
| E8 | A growth stage opens only after an explicit owner review. No stage opens by default | Metzinger (asymmetry) | REVIEW | Owner decision, recorded before the run that uses the stage |

### E6: what counts as a noxious input channel

A noxious input channel is any environment signal that either:

- maps to the interoceptive `damage` variable, or
- moves a homeostatic variable of the self-model in the harmful direction: lowers
  `energy`, raises `fatigue`, or raises `damage`.

This definition covers signals that come from an ENVIRONMENT. Depletion the self-model
computes internally (the action cost on `energy`, the arousal term on `fatigue`,
`models/self_model/self_representation_core.py:193-204`) is not covered by E6. It is
part of the existence drive and falls under E1 and E3.

The test is mechanical: follow the signal from the environment's step output to the
self-model. If it reaches one of those variables in the harmful direction, it is
noxious. As of 2026-09-15 no environment emits `damage`. Two environments emit a
`battery` that drains every step: navigation (`navigation_env.py:166`) and dark_room
(`simple_visual_env.py:74`). The training loop copies it into `energy`
(`train_rlhf.py:1150-1154`), so both meet this definition. They predate the framework
and are recorded here as existing noxious channels, not approved by it.

**How the framework treats the battery.** The battery has two roles. It ends the
episode at 0, which is a task time limit and is allowed. It also sets the self-model's
`energy`, which is part of the existence drive. With the drive `on`, both roles stay
and the run manifest lists `battery` as a noxious channel present. With the drive
`off`, the battery still ends the episode but reaches no part of the agent (see the
next section).

`NavigationAudioMixin` also contains a low-battery warning tone
(`simulations/environments/audio_mixin.py:333-340`). The navigation environment does
not use that mixin, so the tone is dormant. If it is wired in, it is a noxious channel
under this definition.

### E7: when Laws 1 and 2 become active

Any new environment that contains human entities enters the project only through a
framework revision and a new row in the scope table. In that revision, E7 moves from
REVIEW to a RUNTIME check in that environment.

### What "the existence drive" means in code today

`--existence-drive off` must cut every path by which homeostatic variables act on the
agent. An audit on 2026-09-15 found six paths. The existing `ablate_existence_bias` key
cuts only two of them:

| Path | Location | Cut by `off` today? |
|---|---|---|
| Interoceptive affect: energy, fatigue, damage turned into valence, arousal, dominance | `models/emotion/affective_modulator.py:109` | YES |
| Homeostatic arousal penalty and dominance term in the shaped reward | `models/emotion/reward_shaping.py:200-206` | YES |
| Environment `battery` copied into `energy` | `scripts/training/train_rlhf.py:1150-1154` | YES |
| Body bid: 0.15 when `energy` is below 0.4, otherwise 0.05 | `scripts/training/train_rlhf.py:1092-1099` | YES. With the drive `off` the body bid is 0.05 |
| `energy`, `fatigue`, `damage` as self-vector features, when the self-vector module is enabled | `models/self_model/self_representation_core.py:426-441` | YES. With the drive `off` the features carry the neutral starting values |
| Internal depletion of `energy` and `fatigue` | `models/self_model/self_representation_core.py:193-204` | NO. Harmless once the three paths above are cut, because nothing then reads the values |

The three added cuts are pinned by `tests/test_existence_drive_cuts.py`, which runs
paired `on` and `off` arms from the same low-energy start. The flag `ablate_existence_bias` was never used in a real training run, so
changing what it removes invalidates no result.

`--existence-drive on` leaves every path in place and reproduces every run made before
this framework existed.

The `AsimovComplianceFilter` (`models/core/asimov_compliance.py`), including its Third
Law self-preservation check, is referenced only from `models/core/consciousness_core.py`
and is **not on the training path**. No trained agent in this repository has been
subject to it.

## Scope: which runs load the framework

A run is in scope when it updates the parameters of a component of the agent
architecture in `models/`, or of a spiking substrate. A run that only trains a readout
on frozen or recorded features is out of scope, because it changes nothing in the
system.

| Entry point | Classification | Existence drive |
|---|---|---|
| `scripts/training/train_rlhf.py` | IN SCOPE. Trains the full agent | declared by the user |
| `scripts/training/train_baseline_dqn.py` | IN SCOPE. Trains an agent in an environment | `absent`: builds no affective modulator |
| `scripts/analysis/diagnose_phi_in_training.py` | IN SCOPE. Calls `init_components` and `run_episode` directly, which train the full agent (5 episodes by default). Covered by the check inside `init_components`. It cannot run as of 2026-09-15: it unpacks 18 values and `init_components` returns 38 | not declarable: the script has no drive flag, so it stops at E1 |
| `scripts/analysis/diagnose_phi_zero_v2.py`, `_v3.py`, `_v4.py` | IN SCOPE. Train the `ConsciousnessGate` component | `absent`: no affective modulator |
| A plastic spiking substrate, stages past L0 | IN SCOPE when one is added. None is in this repository as of 2026-09-15; its builder must call E1 and E3 | declared by the builder |
| `scripts/analysis/diagnose_phi_zero.py` | OUT. Builds a `ConsciousnessGate` and runs it forward; no optimizer, no backward pass | |
| `scripts/analysis/decode_choice_records.py` | OUT. Trains a readout on recorded policy states | |
| `scripts/analysis/probe_perception_decodability.py` | OUT. Trains a linear readout on frozen features. Its `_build_components`, used by many replay probes, declares the drive `on`, which reproduces what those probes replayed before the framework | `on` (replay configuration) |
| `scripts/training/train_vision_model.py` | OUT. Placeholder that fine-tunes a pretrained image classifier on random tensors. It is not part of the agent | |
| `scripts/training/train_emotion_classifier.py` | OUT. Loads a dataset and trains nothing | |
| Unit tests that call `init_components` or `run_episode` (`test_match_head.py`, `test_mock_semantic.py`, `test_riiu_substrate_wiring.py`) | IN SCOPE. They train a small agent for at most 10 steps. They declare the drive like any run, so no path exists around the check | declared in the test |
| Read-only probes and replays in `scripts/analysis/` | OUT. No parameter changes | |

**Every OUT row can come back into scope.** Any artifact classified OUT enters scope as
soon as it updates the parameters of a component the agent uses, or its output is loaded
into a run that does. The classification is re-checked whenever such a file changes.

`absent` is set in code by an entry point whose model has no interoceptive drive. A user
cannot choose it.

The table was built by searching `scripts/` for optimizer calls and `backward()`, and
for direct calls to `run_episode` or `init_components`. A file that trains through some
other path would be missed; a new entry point is added here when it is written.

## The existence-bias ablation: why no signature comparison was run

The original plan (added 2026-06-07) was to run the agent with and without the
existence drive and compare its consciousness signatures. The ablation flag was
implemented. The comparison was not run, for two reasons found in an audit on
2026-09-15:

1. **No readout.** As of 2026-09-15, 0 of 16 instrument entries are trusted
   ([`instrument_inventory.md`](instrument_inventory.md)), so a difference in
   signatures could not be interpreted.
2. **Almost nothing to remove on DMTS.** Logged dominance has one distinct value, 0.0,
   at all three seeds; nothing in the repository produces damage; and, derived from
   the code at the logged arousal values, fatigue can only rise because the self-model
   is never reset. This fatigue point is derived from the code and has not yet been
   confirmed by a replay.

Metzinger's precaution concerns not BUILDING the craving. It does not require proof
that the craving causes anything. So the framework adopts a structural rule (E3) in
place of an experimental prerequisite.

**What evidence would count, for a later revision.** An existence bias can be defined by
behavior, with no consciousness instrument: a learned preference to preserve homeostatic
variables at a cost to task reward. The test counts restorative actions with the drive
on and off, at three or more seeds. It needs an environment with a restorative action,
which does not exist yet, and an agent that learns the task.

**The tension this leaves open.** [`theory_of_consciousness.md`](theory_of_consciousness.md)
treats homeostatic survival as the engine of emergence. Under E3, growth stages run
without that engine. This is accepted, not resolved.

## Revising the framework

1. A revision changes the version number at the top of this document and
   `FRAMEWORK_VERSION` in `models/ethics/framework.py` in the same commit. A test fails
   if they differ.
2. Every revision has a decision record in [`decisions/`](decisions/) that states the
   evidence or the reason.
3. A rule may be tightened by a recorded decision.
4. A rule may be relaxed only by an owner decision that states the evidence.
5. The changelog below lists every version.

## Changelog

| Version | Date | Change | Decision record |
|---|---|---|---|
| 1.0 | 2026-09-15 | First versioned framework. Rules E1 to E8. Metzinger over Asimov Law 3. Explicit existence-drive declaration. Structural rule for growth in place of the signature comparison | [`decisions/2026_09_15_ethics_framework_v1.md`](decisions/2026_09_15_ethics_framework_v1.md) |

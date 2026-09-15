# Decision (2026-09-15): the ethics framework becomes a permanent, versioned layer

## Status: ADOPTED

## What prompted this

The ethics framework existed as prose in [`ethics_framework.md`](../ethics_framework.md).
Nothing loaded it, no run recorded which rules it ran under, and the Asimov compliance
filter was not on the training path. An existence-bias ablation flag had been
implemented, and a comparison of consciousness signatures with and without it was
planned as the prerequisite for any growth of plasticity.

An audit on 2026-09-15 found that the planned comparison could not conclude anything:

| Finding | Evidence |
|---|---|
| No trusted readout | 0 of 16 instrument entries trusted ([`instrument_inventory.md`](../instrument_inventory.md)) |
| Dominance term removes nothing on DMTS | Logged `dominance` has 1 distinct value, 0.0000, in `runs/bcast_s4{2,3,4}/metrics.csv`, 8000 rows each |
| Arousal penalty is small | Logged arousal 0.5 to 1.0, so `-0.1*(arousal-0.3)^2` is 0.004 to 0.049 per step |
| Damage is never produced | `models/self_model/self_representation_core.py:207` only decays it; no environment emits it |
| Fatigue can only rise (derived from code, not yet replayed) | Net change `0.005*arousal - 0.002` is at least +0.0005 per step at logged arousal; the self-model is built once (`train_rlhf.py:485`) and only its performance is reset (`:931`) |
| Energy cost uses the action index | `norm(action)*0.01` at `self_representation_core.py:193`, and the DMTS action is a discrete index |
| Third Law filter is not trained against | `AsimovComplianceFilter` is referenced only from `models/core/consciousness_core.py` |

## The decision

1. **The framework is permanent and versioned.** It applies to every run that changes
   weights of an agent component or a spiking substrate. Version 1.0 defines rules E1
   to E8.
2. **Metzinger's existence-bias constraint has priority over Asimov's Third Law.** No
   trained agent carries a self-preservation drive or reward. The Third Law is an
   operator duty. The First and Second Laws stay as action filters.
3. **Explicit declaration.** A run refuses to start unless it declares the existence
   drive `on` or `off`, and writes the declaration and the framework version into its
   run folder. `on` reproduces earlier runs.
4. **Structural rule for growth.** Growth stages past L0 run with the existence drive
   `off`. This replaces the signature comparison as the prerequisite.
5. **Revision process.** Tightening needs a recorded decision. Relaxing needs an owner
   decision that states the evidence. Code and document versions must match.

## Owner review of the wording (2026-09-15), recorded before any code depends on it

- **Law 3 as an operator duty: accepted with three limits.** It is a deliberate
  reinterpretation of Asimov; the duty is policy with no mechanism; and it holds fully
  only for runs with the existence drive `off`. All three are stated in the framework.
- **E6: accepted with a mechanical definition.** A noxious channel is any environment
  signal that maps to `damage` or reduces a homeostatic variable. Applying it found one
  existing channel: navigation's `battery`, copied into `energy`
  (`train_rlhf.py:1124-1126`). It predates the framework and is recorded, not approved.
- **E7: accepted with a concrete trigger.** A new environment with human entities
  enters only through a framework revision and a new scope row, and E7 then becomes a
  runtime check there.
- **Scope: criterion accepted, table completed.** Added
  `diagnose_phi_in_training.py` (in scope, found by its direct call to `run_episode`)
  and `diagnose_phi_zero.py` (out, no optimizer). The rule that an OUT artifact can
  return to scope now applies to every OUT row. The Brian2 row states that no E1 or E3
  call exists in the lane yet.

## Second owner review (2026-09-15)

- **Battery.** dark_room also emits a draining `battery` (`simple_visual_env.py:74`),
  missed by the first search because its output was cut at 10 lines. Decision: the
  battery stays as a task time limit. With the drive `off` it reaches no part of the
  agent. With the drive `on` nothing changes and the manifest lists it.
- **What `off` removes.** The audit found that `ablate_existence_bias` cuts 2 of 6 paths
  by which homeostatic variables act on the agent. The other three that matter (battery
  copy, body bid, self-vector features) are cut under `off`, one test per path. The flag
  was never used in a real run, so no result is invalidated.
- **Tests.** Unit tests that train an agent are IN scope and declare the drive, so no
  path exists around the check.

## Pre-declared coupling: the behavioral test needs an E6 revision

The evidence that would show an existence bias needs an environment where a restorative
action exists and costs task reward. In its damage-and-restore form, that environment
emits a noxious channel under E6. So building it requires a framework revision first.
This is stated now so that the block is expected, not discovered later.

## What this does NOT decide

- It does not claim the system is conscious or can suffer.
- It does not resolve the tension with the theory, which treats homeostatic survival as
  the engine of emergence. Under rule E3, growth runs without that engine.
- It does not settle whether an existence bias exists in the agent. The behavioral test
  that could show it needs an environment and a competence level that do not exist yet.

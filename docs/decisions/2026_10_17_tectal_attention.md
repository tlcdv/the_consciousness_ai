# The tectal attention gain: the tectum's selection decides what the workspace holds

Date: 2026-10-17. Status: rules written before any training run. The owner chose this step and this
design on 2026-10-17.

## Why

GWT-2 asks for a workspace of smaller capacity than its modules, and "an attention mechanism that
selects information from the modules" (Butlin et al. 2023). The rubric scores it PARTIAL: the
bottleneck holds, and the selective attention half is measured false. With the default merge the
broadcast holds the vector of the module with the lower bid; with `--broadcast-merge top_winner`
vision takes 0.961 or more of ignited steps (`docs/results/broadcast_carrier_2026_10.md`).

The tectum does select. It binds each of two objects to its tone (`docs/results/rhythm_binding_gate_2026_10.md`)
and learns which one gives food (`docs/results/value_gate_2026_10.md`, `docs/results/reversal_2026_10.md`).
That selection reaches movement through the motor map, but not the workspace: the binding
multiplies the vision and audio bids by one shared factor, and the module with the highest bid
changed at 0.0000 of the steps. This test asks whether the tectum's selection can decide which
object's information the workspace holds.

## Sources

| Source | Read | Statement used here |
|---|---|---|
| Lovejoy and Krauzlis, Nat. Neurosci. 13 (2010) | Abstract | Inactivating the superior colliculus impaired which signals informed perceptual judgments when distracters competed, with no orienting movement: the colliculus is "a bottleneck in the covert selection of signals" |
| Müller, Philiastides and Newsome, PNAS 102, 524 (2005) | Abstract | Microstimulation of a collicular site improved detection at the matching place, with the eyes still |
| Fernandes et al., Neuron 109, 805 (2021) | Abstract | In zebrafish, a loop between the tectum and the nucleus isthmi selects among competing stimuli and boosts the signals of the winner; ablation disrupts stimulus and action selection |
| Knudsen, Eur. J. Neurosci. 33, 1961 (2011) | Abstract and introduction | A midbrain network (optic tectum and isthmic nuclei) encodes the priorities of competing stimuli in a map of space and controls spatial attention |
| Reynolds and Heeger, Neuron (2009), doi 10.1016/j.neuron.2009.01.002 | Model section | The attention field multiplies the stimulus drive before normalization; it is 1 except at the attended place, where its gain is above 1 |

## Designs compared

| Design | What changes | Does the workspace then hold the selected object? |
|---|---|---|
| **A gain on the tectum's own map at the selected cells, then normalization (chosen)** | A forward hook on the topographic map | Yes, if the gain survives the RSSM and the capsules (checked below) |
| Bias the module bids by each sense's evidence for the selected object | Bids | No: the payload is still the whole scene, and in this room both senses carry both objects |
| Objects, not modules, compete for entry | The workspace | Yes, by construction, but it changes the module structure of GWT-1 |
| A thalamic relay between tectum and workspace | A new module | Yes, against the decision to build no thalamus |

## The mechanism

- `models/core/tectal_attention.py`: the attention field is 1 everywhere and γ on the cells of the
  group; the map is multiplied by it and rescaled to its total absolute activity before. The rescale
  is one global pool, a stated simplification of the model's spatial suppressive field.
- `--tectal-attention gain --tectal-attention-gain γ`: a forward hook on the tectum's topographic
  map applies the field of the group that the map binding selected at the previous step. The
  binding runs after the map in the same forward pass, so the gain follows the selection by one
  step, as a feedback loop does. No group, or no binding record since the last reset of the room,
  leaves the map unchanged. Record: `tectal_attention.json`.
- `--tectal-orienting-covert`: the motor map still selects and records its target, and the agent
  does not move. Without movement, the place of the selected object cannot reach the workspace
  through the view.

## Checks before any training run

**Baseline on recorded runs** (reported only): at the value gate's choice step, before the agent
moves, the chosen object's direction decoded from the recorded content scored -0.177 to 0.219 and
from the broadcast -0.221 to 0.180 over 9 runs; with the directions shuffled, -0.072 to 0.138.
Score: the mean cosine between the decoded and the true direction, chance 0.

**Offline diagnosis, PASSED.** The covert room, the cochlea and the rhythm binder, the trained
tectum of `runs/vg_learn_s601` (episode 39) in eval mode with no training; 40 episodes of 60 steps
per sequence; rule written in the script before its first run. Score as above, from the tectum's
`workspace_content`, with folds that hold whole episodes out.

| Sequence | No gain | γ 1.5 | γ 2 | γ 3 |
|---|---|---|---|---|
| 521 (pipeline) | -0.132 | 0.850 | 0.854 | 0.904 |
| 522 (pipeline) | -0.082 | 0.876 | 0.880 | 0.887 |
| 523 (pipeline) | -0.132 | 0.950 | 0.962 | 0.971 |
| 524 (held out) | 0.118 | 0.970 | | |
| 525 (held out) | -0.202 | 0.917 | | |
| 526 (held out) | 0.087 | 0.977 | | |

The rule took the smallest γ with a score of at least 0.5 at all pipeline sequences while the
no-gain score stays at or below 0.2: **γ = 1.5**. It then held at each held-out sequence.

## Gate

Seeds 801, 802 and 803, 40 episodes, one run at a time. The Gate B7 flags with
`--dark-room-decoy-sound --dark-room-decoy-level 0.5,3.0 --dark-room-decoy-visible
--dark-room-start-between --dark-room-pulse --dark-room-sound-direction off --map-binding rhythm
--tectal-orienting group --tectal-orienting-idle still --tectal-orienting-covert
--broadcast-merge top_winner`, and each arm's flags:

| Arm | Flags |
|---|---|
| link | `--tectal-attention gain --tectal-attention-gain 1.5` |
| no link | none |

The binder reads the raw frames and the sound, and the agent does not move, so both arms of a seed
see the same frames and make the same selections. Read by `scripts/analysis/probe_tectal_attention.py`.

| Check | Rule, at 3 of 3 seeds |
|---|---|
| A1 the workspace content carries the selection | link arm, score from `tectum_content` ≥ 0.5 |
| A2 the broadcast carries it | link arm, score from the broadcast at ignited steps ≥ 0.5 |
| A3 the link causes it | link above no link for both vectors, both no-link scores ≤ 0.2, and a permutation test with the directions shuffled across episodes gives p < 0.05 for both link scores |

PASS when A1 to A3 hold. FAIL when one check fails at 2 or more seeds. Reported only: the share of
ignited steps each module has the highest bid (vision is expected near 1.0 with `top_winner`), the
vision bid with and without the gain, the steps with the gain, and whether the selections are the
same in both arms. A control with all new flags off must give the md5
`7825766f42d3d85b032f8028c94dc1e0` for `metrics.csv` of seed 73.

## Limits, stated before the runs

- The gain is a designed mechanism. The gate measures whether the selection reaches the broadcast
  through the trained tectum and capsules, in live runs that train.
- The selection here is by salience (the louder object). Selection by learned value would be a
  test of GWT-4 (state-dependent attention), later.
- The contest between modules is not solved: with `top_winner` vision is expected to have the
  highest bid at nearly every ignited step.
- The policy cannot use the broadcast (`docs/results/action_core_relay_defect_2026_10.md`), so this
  is not a behavioural claim.
- A passed gate does not move GWT-2 by itself. The score is the owner's decision.

## Note added after the runs (2026-10-31)

The gate FAILED, A1 to A3 at 3 of 3 seeds (`docs/results/tectal_attention_2026_10.md`). The rules
above are unchanged. The offline diagnosis above ran the tectum in evaluation mode, where the
continuous latent is its mean, and the live runs use a sample of it. In training mode the same check
with the same weights gives -0.015 for γ 1.5 on sequence 521, against 0.850 in evaluation mode. So the
offline diagnosis did not test the condition of the gate. An offline check now runs in the same module
mode as its gate.

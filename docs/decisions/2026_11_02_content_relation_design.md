# A relation of content for the workspace binding. The present test bed cannot test it

Date: 2026-11-02. Status: design only, no code. The owner asked on 2026-11-02 for a binding in
which the coupling of two modules depends on the agreement of their contents, and decided the
same day that it comes as a new choice of the existing `--binding-relation`, after reading and
cheap checks.

## What exists

| Part | State | Source |
|---|---|---|
| A second sound with no visual object | `--dark-room-decoy-sound` | `simulations/environments/simple_visual_env.py` |
| Coupling by a relation r in [0, 1], factor 2r - 1 on the pair of vision and audio | `--binding-relation place` and `overlap` | `models/core/oscillatory_binding.py:173`, `models/core/global_workspace.py:172` |
| Agreement of place | Gate FAILED, G1 and G3 at 2 of 3 seeds | `docs/results/binding_place_relation_2026_10.md` |
| Binding by timing on the map | Gate PASSED, then FAILED in harder rooms | `docs/results/rhythm_binding_gate_2026_10.md`, `docs/results/rhythm_binding_harder_rooms_2026_10.md` |

## Why the present test bed cannot test a relation of content

Found by reading the code, before any run.

1. The two sources differ in pitch. The light sounds at 440 Hz and the decoy at 660 Hz
   (`simulations/environments/audio_mixin.py:290`, `:296`).
2. The visible decoy is drawn with the colour and size of the light, so only the tone tells the
   two objects apart (`simulations/environments/simple_visual_env.py:97`).
3. A step counts as related when the light tone is the louder one at the agent
   (`scripts/analysis/probe_binding_selectivity.py:7`).

So with the hidden decoy, the label follows from the sound alone. A learned relation that reads
the louder pitch passes, and vision can add nothing to it. With the visible decoy, the vision of the
two objects is the same, so no relation of content can tell which object a sound belongs to. In
both cases the outcome of the check that was planned (CC3) is fixed before it runs, so it was not
run.

## What the sources say

| Source | Read | Finding |
|---|---|---|
| Yu, Rowland and Stein, J. Neurosci. 30, 4904 (2010) | Main text, through a summary of the page | Cats reared in the dark had collicular neurons that answered both senses and did not integrate them. Exposure to a light and a noise together, at one place, started integration. The first integrating neuron came after about 3,600 presentations, and 90 percent of the neurons whose fields held the exposed place integrated at the last stage. The same presentations given apart gave 9.6 percent. The learning held best at the exposed place and carried over to other places more weakly. No reward was needed. The authors propose a rule of the Hebb type |
| Wallace and Stein, J. Neurophysiol. 97, 921 (2007) | Abstract, through secondary summaries | Animals reared with a light and a sound together but at different places had collicular neurons that integrated pairs at different places. The rules of integration are set by early experience |
| Chen and Spence, Front. Psychol. 8, 445 (2017) | Main text in part, the published file | The belief that two signals belong to one object (the unity assumption) can come from learned knowledge. Arbitrary pairs of a sound and a picture help each other only when they occur together reliably. A learned coupling prior encodes that two features go together, and not that they belong to one object |
| Parise and Ernst, Nat. Commun. 7, 11543 (2016) | Full text, read for `docs/decisions/2026_10_15_rhythm_binding.md` | A detector of the correlation between the senses explains causal inference and judgements of synchrony |

Three points follow.

1. In the midbrain, which senses go together is learned from signals that occur together in
   time and place. A fixed comparison of two contents, with no experience, has no basis there.
2. A learned relation encodes a correlation. In this room the only content that differs between
   the two sources is the pitch, so a learned relation would read the pitch.
3. To test a relation of content, the two objects must differ in a visual feature that goes
   with their tone, and some episodes must break the usual pairing, so that neither pitch alone
   nor place alone gives the answer.

## Designs

| Design | What it does | Basis | Can it be tested now |
|---|---|---|---|
| B1. Cosine in a fixed shared space (the requested form) | Project each content onto a unit sphere and use the cosine as r | None. Vision and hearing have no shared space without experience | No. It is also expected to be noise, because the projection would be random |
| B2. Coupling learned from co-occurrence | Learn which visual features go with which sounds from steps where they occur together, and use the learned agreement as r | Yu and colleagues (2010), Wallace and Stein (2007), the coupling prior in Chen and Spence (2017) | No. It needs the test bed below |
| B3. Agreement of place | The existing `place` relation | Stein and Meredith, the spatial rule | Done. FAILED |
| B4. Agreement of timing | The existing binding by rhythm | Parise and Ernst (2016) | Done. PASSED, then FAILED in harder rooms |

## The test bed that B2 would need

- A room option, off by default, that draws the decoy with a different colour from the light.
- A room option, off by default, that swaps the two tones in some episodes, like the existing
  `food_switch`, so that pitch alone does not give the label.
- Sound with no direction, which exists (`sound_direction` false), so that place alone does not
  give the label either.
- A read-only probe on short recorded runs that asks if a relation learned on some episodes
  separates the pairs that belong together on other episodes, with vision shuffled as a control.

My estimate is about one day for the two room options, their tests and the probe, before any run.
That estimate is not measured.

## Recommendation

Do not build B1. Build B2 only after the owner approves the test bed above, because without it no
result can be read. The binding in harder rooms by timing, which is already planned, stays the
first binding step. One thing could make this wrong. A relation of content could still help where
place and timing are both missing, and this room cannot show that until the test bed exists.

## Decision for the owner

1. Approve the test bed for B2, or close the relation of content for now.

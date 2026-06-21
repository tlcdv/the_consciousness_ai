# Affective Consciousness: Solms, Panksepp, and the Homeostatic Engine

*Companion to [feinberg_mallatt_approach.md](feinberg_mallatt_approach.md) (which
describes the limbic "affective track") and to
[active_inference_unification.md](active_inference_unification.md). The project's stated
engine is emergence from emotional homeostasis, but its affective grounding has been
thin (Mehrabian PAD, Keramati-Gutkin). This doc supplies the neuroscience that engine
actually models and connects it to the planned active-inference objective. It is an
original synthesis with citations, not a reproduction of the sources.*

Primary sources:
- Solms, M. (2021), *The Hidden Spring: A Journey to the Source of Consciousness*. Norton.
- Solms, M. & Friston, K. (2018), "How and why consciousness arises: Some considerations
  from physics and physiology," *Journal of Consciousness Studies* 25(5-6):202-238.
- Panksepp, J. (1998), *Affective Neuroscience*. Oxford University Press.
- Panksepp, J. & Biven, L. (2012), *The Archaeology of Mind*. Norton.
- Seth, A. (2021), *Being You*; Seth & Tsakiris (2018), "Being a Beast Machine," *TiCS*
  22(11):969-981. (The predictive-processing bridge.)

## Why this matters to the project

The README's core claim is that behavior emerges from an internal struggle for emotional
homeostasis, and the roadmap's Phase 6 wants to fold the training objective into one
free-energy principle. Solms supplies the theory that makes those two statements the same
statement: affect is the felt side of homeostatic free-energy regulation. Panksepp
supplies the subcortical circuit map for the specific drives the affective core
implements. Seth supplies the predictive-processing formalism that links interoception to
the self-model. Together they turn the affective core from a set of hand-chosen
coefficients into a grounded model.

## Solms: consciousness is affective, homeostatic, and free-energy-based

Solms locates the source of consciousness in the upper brainstem and argues four linked
points: consciousness is fundamentally affective (it feels like something to be in a
state); affect is the subjective side of homeostasis (how the organism is doing relative
to its viable bounds); this is formalized by the Free Energy Principle (the organism acts
to keep prediction error, that is expected uncertainty, low); and so affect is, in his
framing, the felt experience of free-energy dynamics. Decreases in expected uncertainty
register as pleasure, increases as unpleasure.

For the project, the load-bearing point is the bridge: **the homeostatic reward the
agent already optimizes and the free-energy objective the Phase 6 plan proposes are two
descriptions of one thing.** Solms-Friston is the citation that licenses treating
`models/emotion/reward_shaping.py` (homeostatic PAD reward) and the EFE objective in
[active_inference_unification.md](active_inference_unification.md) as the same engine
seen from two sides, rather than two competing objectives to reconcile.

## Panksepp: the subcortical drive systems

Panksepp's cross-species affective neuroscience identifies seven primary-process
emotional systems generated in subcortical structures (periaqueductal gray,
hypothalamus, and related circuits): SEEKING, FEAR, RAGE, LUST, CARE, PANIC/GRIEF, and
PLAY. These are raw affects, present before any cortical elaboration, which fits the
Feinberg-Mallatt claim that the affective track is ancient and subcortical.

The project implements a compressed version of this with PAD plus a small set of drives.
The mapping:

| Panksepp system | Our component |
|---|---|
| SEEKING (curiosity, foraging, anticipation; mesolimbic dopamine) | RND curiosity (`models/core/rnd_curiosity.py`), the exploration drive |
| FEAR, RAGE | THREAT module set in `models/emotion/affective_modulator.py` |
| Homeostatic affect (energy, fatigue, damage) | interoceptive PAD generation, `interoceptive_to_pad` |
| Positive social/approach affects | APPROACH module set |

SEEKING is the most useful single mapping. It names the biological system the project's
curiosity bonus stands in for, and it cautions against treating exploration as a bolt-on:
in Panksepp's account SEEKING is a primary affect, which is consistent with the Phase 6
plan folding curiosity into the epistemic term of expected free energy rather than a
separate RND reward.

## Seth: interoceptive inference and the bodily self

Seth recasts emotion and selfhood as interoceptive predictive inference: feelings are the
brain's best guess about the causes of signals from inside the body, and the basic sense
of being a self is grounded in control-oriented prediction of bodily state (the "beast
machine"). This is the predictive-processing form of the project's interoceptive PAD loop
and self-model, and it is the cleanest bridge from the affective core to the
active-inference direction.

| Seth's claim | Our component |
|---|---|
| Emotion as interoceptive inference | interoceptive PAD generation feeding the affective modulator |
| Selfhood grounded in bodily regulation | `models/self_model/self_representation_core.py`, body schema + interoception |
| Predictive control of the body | the embodiment-affect loop, Phase 6 active inference |

## What this changes in how we describe the project

- The affective core stops being "PAD with tuned coefficients" and becomes a model of
  homeostatic affect with a named theoretical basis (Solms) and named circuit referents
  (Panksepp).
- The Phase 6 active-inference unification gains its affective justification: minimizing
  expected free energy is, in Solms' reading, the formal description of the homeostatic
  feeling the agent is already built around. This is an argument for coherence, not a new
  result.
- Curiosity / RND gets a primary-process identity (SEEKING), which supports moving it
  into the epistemic value term rather than keeping it as a separate bonus.

## Honest caveats and tensions

- Solms and Panksepp both insist on a **biological substrate** and would not grant that a
  PyTorch agent feels anything. The project's response is the functionalist /
  substrate-independence argument in
  [rouleau_levin_substrate_independence.md](rouleau_levin_substrate_independence.md), plus
  Metzinger's discipline in [metzinger_phenomenal_self_model.md](metzinger_phenomenal_self_model.md)
  that a measured affective signature is never an existence proof.
- **Affect-first vs sensory-first** is a live debate. The project does not pick a winner:
  it implements the sensory track (tectum) and the affective track (modulator) in
  parallel, per Feinberg-Mallatt.
- These are **theories**. They motivate the design and the Phase 6 framing. They do not
  score a signature, and nothing follows from alignment alone. Any change they suggest
  (for example folding curiosity into EFE) is gated and tested FAILED-first.
- **Ethical tension to keep in view.** Making the survival drive more biologically
  faithful (Panksepp's SEEKING, Solms' homeostatic affect) raises the stakes of the
  concern Metzinger flags: building a craving for existence into a possibly conscious
  machine. This is the rationale for the planned default-off existence-bias ablation. See
  [ethics_framework.md](ethics_framework.md) and
  [metzinger_phenomenal_self_model.md](metzinger_phenomenal_self_model.md).

## References

- Solms, M. (2021). *The Hidden Spring*. Norton.
- Solms, M. & Friston, K. (2018). How and why consciousness arises. *J. Consciousness
  Studies* 25(5-6):202-238.
- Panksepp, J. (1998). *Affective Neuroscience*. Oxford University Press.
- Panksepp, J. & Biven, L. (2012). *The Archaeology of Mind*. Norton.
- Seth, A. (2021). *Being You*. Faber/Dutton. Seth, A. & Tsakiris, M. (2018). Being a
  beast machine. *Trends in Cognitive Sciences* 22(11):969-981.
- See also [feinberg_mallatt_approach.md](feinberg_mallatt_approach.md),
  [active_inference_unification.md](active_inference_unification.md),
  [damasio_self_hierarchy.md](damasio_self_hierarchy.md).

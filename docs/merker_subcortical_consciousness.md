# Merker: Subcortical (Tectum-First) Consciousness

*Companion to [feinberg_mallatt_approach.md](feinberg_mallatt_approach.md). Where
Feinberg-Mallatt give the evolutionary and architectural blueprint for tectum-first
consciousness, Björn Merker gives an independent clinical and comparative case for the
same claim. This doc summarizes his argument in our own words and maps it onto the
project. It does not reproduce his text.*

Primary source: Merker, B. (2007), "Consciousness without a cerebral cortex: A challenge
for neuroscience and medicine," *Behavioral and Brain Sciences* 30:63-81.

## Why this matters to the project

The whole architecture rests on a single load-bearing claim: that the seat of basic
consciousness is the midbrain (optic tectum / superior colliculus), not the cortex. The
project currently sources that claim from Feinberg-Mallatt alone. Merker is a second,
independent line of evidence for it, from a different field (clinical neurology and
comparative ethology rather than neuroevolution). A cornerstone with two independent
supports is sturdier than one with a single support. That is the value here, not a new
mechanism.

## Merker's argument, in brief

1. **The conscious state is organized in the upper brainstem, not the cortex.** Merker
   argues the midbrain and basal diencephalon integrate sensory, motor, and
   motivational information into the single coherent point of view that a mobile animal
   needs to act. The cortex elaborates the *contents* of consciousness, but the *state*
   itself, the basic "being a subject in a world," is constituted subcortically.

2. **The integration-for-action argument.** A single body with a single set of muscles,
   in a single world, pursuing a single set of needs, has to commit to one action at a
   time. Merker frames the midbrain as solving this with three coupled selections:
   where to look or go (target selection), what to do (action selection), and how much
   it matters (motivational ranking). Their joint solution is a unified behavioral
   reality model. This is a functional argument: consciousness is the format that makes
   coherent action by a unified agent possible.

3. **The clinical evidence.** Children with hydranencephaly (born with little or no
   cerebral cortex) and decorticate animals still show goal-directed, affectively
   expressive, environmentally responsive behavior. Merker reads this as evidence that
   the conscious state survives without cortex, which is hard to reconcile with strictly
   cortico-centric theories.

## How it maps to our architecture

| Merker's claim | Our component |
|---|---|
| Midbrain integrates aligned sensory maps into one reality model | `models/core/sensory_tectum.py` (topographic fusion, RSSM) |
| Target selection + action selection | Global Workspace competition + basal-ganglia Go/No-Go action selection |
| Motivational ranking of stimuli | `models/emotion/affective_modulator.py` valence field over bids |
| A single body in a single world (unified point of view) | self-model self-other boundary, body schema on the tectum grid |
| Cortex elaborates contents, not the state | Qwen2-VL semantic pathway as an optional elaborator, not the seat |

The mapping is close because the project was already built tectum-first. Merker mostly
adds confidence and a sharper functional rationale (integration-for-action) for why the
selection-plus-ranking design in the workspace and affective modulator is the right
shape.

## What it adds beyond Feinberg-Mallatt

- A **clinical** evidence base (hydranencephaly, decortication) that Feinberg-Mallatt do
  not lean on.
- The **selection triangle** (target, action, motivation) as an explicit functional
  account of what the midbrain is *for*, which lines up with our workspace-plus-modulator
  split.
- A framing of consciousness as the **format required for unified action**, which is
  congenial to the project's emphasis on the consciousness machinery being causally
  useful for behavior (the substrate-independence test in
  [preregistered_predictions.md](preregistered_predictions.md) section 13).
- An explicit **ego-center / first-person pivot**, developed further in Merker's 2013
  follow-up. The conscious frame is organized around an implicit self, which lines up with
  the project's self-model and self-other boundary rather than a passive image.

## Honest caveats and tensions

- Merker's thesis is **contested**. Many researchers hold that the cortex is required at
  least for the specific contents of human conscious experience. The project does not
  need the strong reading. We use Merker as architectural corroboration that the *state*
  can be organized subcortically, which is what the tectum-first design assumes.
- Merker argues about **biological brains**. The leap to a PyTorch re-instantiation is
  carried by the functionalist / substrate-independence argument in
  [rouleau_levin_substrate_independence.md](rouleau_levin_substrate_independence.md), not
  by Merker.
- This is a **theory**, not a validated result for our system. It motivates the design;
  it does not score any signature. Per project discipline, no claim follows from
  alignment alone.

## References

- Merker, B. (2007). Consciousness without a cerebral cortex: A challenge for
  neuroscience and medicine. *Behavioral and Brain Sciences* 30:63-81 (PMID 17475053).
- Merker, B. (2013). The efference cascade, consciousness, and its self: naturalizing the
  first person pivot. *Frontiers in Psychology* 4:501. (Develops the ego-center /
  first-person framing.)
- Feinberg, T.E. & Mallatt, J. (2016). *The Ancient Origins of Consciousness*. MIT Press.
- See also [feinberg_mallatt_approach.md](feinberg_mallatt_approach.md) and
  [rouleau_levin_substrate_independence.md](rouleau_levin_substrate_independence.md).

# The link from the tectum to the workspace: designs compared, for the owner's decision

Date: 2026-10-31. Status: no design chosen, no code, no run. This document reads the project documents
and the sources, and compares five designs. Paraphrase and citation only.

## The conclusion, stated first

1. My proposal is design B, a direct route from the selected cells of the tectum map to the workspace,
   beside the capsule path. The owner decides. It reverses the decision to build no thalamic route, so it
   needs an explicit yes.
2. Design B can pass a decode gate by construction, because it writes the place into the vector. So a pass
   would show that the selection is available in the workspace. It would not show that anything uses it.
   Only the policy repair and a later ablation can test use.
3. The sources do not support the idea behind the failed gain, that an attention gain travels up to the
   next stage. Two read here point against it.
4. One source reading supports design E, with no link at all. It is a real option, and it would keep GWT-2
   at PARTIAL.

## What the measurements require

From `docs/results/content_map_sensitivity_2026_11.md` and `docs/results/tectal_attention_2026_10.md`.

| Measured | What a design must do |
|---|---|
| Sampling noise of standard deviation about 1 enters at the latent, and `cont_logvar` does not move | Avoid the sample, or set its size |
| The capsules are near zero or saturated, and upper levels are near zero | Avoid the capsules, or change their scale in both states |
| In the covert runs the content is the projection bias | Have no fixed offset that hides the signal |
| At seeds 602 and 603 the gain moves the content by 0.4 to 1.2 percent, and the direction stays unreadable | Carry the place explicitly, because a larger effect alone does not do it |
| The covert runs stay near their starting weights, with a constant reward of -0.01, and the tectum's main loss is the prediction of reward | Not depend on training that the covert room does not give |
| Vision has the highest bid at 1.000 of ignited steps with `top_winner` | Say what still competes |

## Project documents read

`docs/sensory_selection_evidence.md` (its option C moves the competition to places),
`docs/thalamic_gating_evidence.md` (a mechanism is the owner's call and follows measurement, which has now
happened), `docs/merker_subcortical_consciousness.md`, `docs/feinberg_mallatt_approach.md` sections 2 and 3,
`docs/results/b1_continuous_latent_2026_07.md`, `docs/decisions/2026_10_17_tectal_attention.md`, and the
training code of the tectum.

## Sources, with reading depth

| Source | Depth | Statement used here |
|---|---|---|
| Berman and Wurtz, J. Neurosci. 31, 373 (2011) | Abstract, methods, first part of the results | Macaque. Inferior pulvinar neurons on the path from the superficial colliculus to area MT have receptive fields and surrounds like collicular visual neurons. They did not clearly show the attentional enhancement seen in the colliculus |
| Zénon and Krauzlis, Nature 489, 434 (2012) | Main text through a text extract | Two macaques. Collicular inactivation cut hit rates from 53 to 57 percent to about 10 to 15 percent in the affected field. The attention modulation in MST stayed at 0.075 to 0.071, and in MT at 0.061 to 0.057. The authors conclude that the colliculus acts through other processes and do not name one route |
| Bogadhi et al., bioRxiv 841155 (published in Neuron 2021) | Preprint main text through an extract, published version not read | Two macaques. Collicular inactivation cut the share of object-selective neurons in the floor of the superior temporal sulcus from 30.3 to 17.9 percent, with spike rates unchanged. The route through the pulvinar is inferred and was not tested |
| Bloch et al., eLife 9, e54945 (2020) | First half through an extract | In teleost fish a path runs from the tectum through the preglomerular nucleus to the pallium. Its cells come mostly from the midbrain, and the authors argue that it is not homologous with the amniote path. So it evolved at least twice |
| Merker, Front. Psychol. 4, 501 (2013) | First third through an extract, theory paper | The colliculus is a priority gate that selects one target. Phenomenal space is more likely among its ascending thalamic targets, with the dorsal pulvinar as candidate. Its inhibition is proposed as constraint satisfaction and not as winner-take-all. No measurement |
| Knudsen, Trends Neurosci. 41, 789 (2018) | Read for the project review of 2026-10-13, not read again | A midbrain network selects across space. The selected signal goes on to the thalamus and forebrain. It selects across places, and not across senses |
| Carandini and Heeger, Nat. Rev. Neurosci. 13, 51 (2012) | First half, with the abstract | Responses are divided by the summed activity of a pool. This gives saturation and keeps tuning across contrast, and it can give winner-take-all competition. It spans invertebrates to mammals, and its circuits differ |
| Sabour, Frosst and Hinton, arXiv 1710.09829 | Abstract | The length of a capsule's vector stands for the existence of an entity |

Not read in the main text. Krauzlis, Lovejoy and Zénon (2013), Fernandes et al. (2021), Lovejoy and Krauzlis
(2010). All experiments are in animals, none in an artificial agent, and the step to this agent is my reading.

**What the sources say together, by my reading.** Collicular activity changes what later stages code (Bogadhi).
The routes that exist carry a place-tuned signal (Berman and Wurtz, Bloch). Those routes need not carry the
attention gain (Berman and Wurtz), and a colliculus deficit does not need a change of gain in visual cortex
(Zénon and Krauzlis). So the failed gain was the less supported design, and I should have seen this before
the gate of 2026-10-17. The sources of that gate were read mostly at abstract level.

## The designs

| Design | What changes | What it does for the three causes | Support | Against, and risk | Cost |
|---|---|---|---|---|---|
| A. Repair the capsule path | Divisive normalization in place of squash, the posterior mean in the path, no projection bias | Treats all three, with three changes that cannot be told apart | Carandini and Heeger for the normalization. Keeps the nested hierarchy that Feinberg and Mallatt require | Seeds 602 and 603 show that a larger effect does not carry the place. It changes the path behind all earlier decodes. Needs tuning, so it invites a search for a pass | High |
| **B. Direct route from the selected cells (proposed)** | Under a default-off flag, the vision payload becomes a fixed projection with no bias of two parts. One is a place code of the selected cells, and the other is the mean map feature at those cells | Avoids all three. The capsule path stays for the bid and for earlier comparisons | Berman and Wurtz, Bloch, Knudsen, Bogadhi. Fits a place code from the map | Passes a decode by construction. It is a designed wire and not an emergent one. Reverses the no-thalamus decision. Vision still takes every ignited step, so the contest between modules stays open | Low |
| C. Places compete in the workspace | Candidates are places on the fused map, with global inhibition and adaptation. The audio module becomes a forebrain route or goes | Avoids all three, and also answers what competes | The strongest fit with Knudsen and Merker, and option C of the review of 2026-10-13 | Changes the module structure behind GWT-1. Large, and it touches `global_workspace.py` | High |
| D. A relay module with its own dynamics | A new pulvinar or zona incerta module with lateral inhibition | Avoids all three | Merker 2013, as theory only | No measurement asks for the dynamics. It adds parts before a simple route has been tried. It is the hub that the project chose not to build | High |
| E. No link | Keep behaviour in the tectum loop, record the workspace as outside it, and go to the action path | Treats none, and asks if it needs to | Zénon and Krauzlis point to routes other than sensory gain. Merker's triangle puts selection in the midbrain | GWT-2 stays PARTIAL. The mission's causal test of the workspace does not become possible | None now |

## Why B, and what would make it a fair test

B is the cheapest design that removes the three measured causes at once, so it tests one question. Does a
selection that is written into the workspace change what the rest of the agent can do? A and C can follow if
B shows that it is worth the cost. E stays the honest result if it does not.

A gate for B, as a draft and not yet the rules. Fresh seeds, 40 episodes, training mode, the covert room, with
the route and without it.

| Check, draft | Why |
|---|---|
| The broadcast at ignited steps holds the direction of the selected object, score at least 0.5, at 3 of 3 seeds | The check that failed before. It is nearly guaranteed here, so it is only a check of the wiring |
| Matched frames. Where the same layout is seen and the louder tone selects the other object, the vector follows the selection at 3 of 3 seeds | The stimulus is held and the selection varies, the contrast that `docs/thalamic_gating_evidence.md` says no result here has used |
| Control. With the group replaced by the cells of the other object, the vector follows those cells | Shows that the route reads the selection and nothing else |
| Bit identity with the flag off, by the md5 of seed 73 | Project rule |

An offline check comes first, in training mode, on at least 3 weight sets from both run groups. The rules
would be pushed before any run.

Two open points for the build. The vision payload is set inside `run_episode`, and `reward_predictor` reads the
same tensor. So the route must not change what the reward predictor learns from without saying so. I would
keep the capsule content as that input, by the smallest change behind the flag.

## What a passed gate would not show

It would not show that the workspace acts on behaviour, because the movement network ignores its input
(`docs/results/action_core_relay_defect_2026_10.md`). It would not move GWT-2 by itself, and the score stays the
owner's decision. It would say nothing about consciousness.

## Note added the same day: design B stopped

The owner chose design B. Before any build, cheap checks asked if a policy can use the route. With the real binder selecting the cells, they FAILED at 6 of 6 seeds, so design B stopped and no live run took place (`docs/results/route_learnability_2026_10.md`). A draft gate above is not in force.

# Content relation check. The identity condition of B2 FAILED, because vision does not tell the two objects apart

Date: 2026-11-02. Design document: `docs/decisions/2026_11_02_content_relation_design.md`, design B2.
Rules: the docstring of `scripts/analysis/probe_content_relation.py`. Version 1 was pushed at commit
60f32aa and stopped at its smoke run by its own rule. Version 2 was pushed at commit cb874c0 (19:31 UTC),
approved by the owner, before any counted run. The three counted runs ran at that commit with no local
changes, one at a time, from 19:31 to 20:04 UTC. The probe took 218 seconds on Python 3.12.10. Output:
`runs/_audit_followup/nov02/relation_v2_counted.json`, md5 `ad190f857117ba9e7b8df0a335883b0d`.
No indicator moves, and no score changes.

## Verdict

1. **FAILED. Vision does not tell the light from the decoy at 0 of 3 seeds (V1).** The decoy was drawn
   pale cyan (200, 255, 255) and the light pale yellow (255, 255, 200), of almost equal luminance. The
   ridge read of `tectum_content` did not beat the read of the episode number alone at any seed. At seed
   1101 it was below chance. So the identity condition of B2 FAILS by the stated rule, and the next step
   that rule names, a relation of content in this agent, has no visual identity to use.
2. **Hearing tells which tone is louder, at AUC 1.0000 at 3 of 3 seeds (A1).** A1 holds at seeds 1101 and
   1103. At seed 1102 its null also reached 1.0000, so it does not hold there by the rule.
3. **Seed 1102 shows a weakness of my VOID rule.** The light was the seen object in only 2 of its 20
   episodes, so its reads are degenerate (0.0000 and 1.0000). The rule makes a seed VOID only when no
   held-out fold shows both objects, and that did not cover this case. The verdict does not depend on it.
   Seeds 1101 and 1103 fail V1 on their own, and two failures make FAIL.
4. **The two-choice test, reported only, is below its null at 3 of 3 seeds.** That fits point 1.

## Numbers

From `relation_v2_counted.json`. Whole episodes are held out in 5 folds, and each AUC is computed inside
its fold and averaged. The null gives each episode the labels of another episode, 20 times.

| | s1101 | s1102 | s1103 |
|---|---|---|---|
| Counted steps, light seen and decoy seen | 561, 511 | 139, 820 | 445, 847 |
| Episodes with each object seen, light and decoy | 5, 10 | 2, 7 | 8, 8 |
| V1, vision. AUC, null p95, episode number AUC | 0.4572, 0.9409, 0.7595 | 0.0000, 1.0000, 0.0000 | 0.7919, 0.7839, 0.7971 |
| A1, hearing. AUC, null p95, episode number AUC | 1.0000, 0.9172, 0.7595 | 1.0000, 1.0000, 0.0000 | 1.0000, 0.7853, 0.7971 |
| Two-choice, reported. Reading, null p95 | 0.4063, 0.6447 | 0.2500, 0.7531 | 0.5371, 0.7802 |

## Version 1 and its smoke run

Version 1 asked if a learned relation tells related from unrelated steps, where related means that the
seen object is the louder source. Its smoke run, seed 1199 with 2 episodes, had two of the four cells
empty (102 and 198 steps against 0 and 0), so it stopped by its own rule. In the counted runs all four
cells have steps:

| Cells, object seen and louder source | s1101 | s1102 | s1103 |
|---|---|---|---|
| light, light | 561 | 139 | 445 |
| light, decoy | 355 | 172 | 419 |
| decoy, light | 104 | 83 | 508 |
| decoy, decoy | 511 | 820 | 847 |

So the smoke run of 2 episodes did not show the cells of 20 episodes. Version 1's question could be
asked on new runs with new rules. It needs vision to tell the objects apart first, which point 1 shows
it does not.

## Runs

The Gate B7 command (the recorded argv of `runs/f2_py312_s73`) with `--episodes 20 --existence-drive off
--dark-room-decoy-sound --dark-room-decoy-visible --dark-room-decoy-colour 200,255,255
--dark-room-decoy-level 0.5,3.0 --dark-room-sound-direction off`, seeds 1101, 1102 and 1103, in
`runs/cr_s1101` to `runs/cr_s1103`. Each run has 20 rows in `episodes.csv` and 20 episode folders with
vectors. Each `ethics_manifest.json` records commit cb874c0, no tracked changes and the drive off, and
each `session.json` records the decoy colour 200,255,255 and the sound direction off.

## What this points to

The frames enter the agent in colour. The first layer of the retinotopic encoder has 3 input channels
(`models/core/retinotopic_encoder.py:30`). So the difference of hue is lost after the input, and where it
is lost was not measured. The content map diagnosis found that the capsule stage either reduces the
content to the fixed bias of `workspace_proj` (covert runs) or saturates, so that 95 percent of the
content's motion lies in 1 or 2 dimensions (learning runs, `docs/results/content_map_sensitivity_2026_11.md`).
That may be where the hue is lost. I cannot verify that from this check.

## Recommendation, and what could make it wrong

Close B2 for now and keep the two room options, which are off by default. If the owner wants to go on,
the cheapest next check is a read-only layer table of the hue difference, as the content map probe made
for the gain. It would read the same frame with a pale yellow and with a pale cyan object at the encoder,
the map, the latent and `tectum_content`, in training mode, on saved weights. That takes minutes. One
thing could make the recommendation wrong. A stronger difference of colour or of luminance could pass V1,
but then the two objects would also differ in brightness, which the tectum already reads, and the test
would no longer ask about content alone.

## Limits

Open loop and offline. Three seeds of 20 episodes, one colour pair and one room. The relation was a read
of the vectors, not a mechanism in the agent.

# Stage 2 appraisal checks. Design A and design B both FAILED, and the first checks FAILED their own design

Date: 2026-11-02. Design document: `docs/decisions/2026_11_02_core_self_appraisal_design.md`.
Rules: the docstring of `scripts/analysis/probe_core_self_appraisal.py`, pushed at commit 1d5e0dd
(13:12 UTC) before any run. The smoke data changed twice after that, at commits 1b70cc6 and
92272d1, each pushed before its own run, and no counted setting changed. The counted run started at
13:15 UTC at commit 92272d1 and took 37 seconds on Python 3.12.10. Output:
`runs/_audit_followup/nov02/appraisal_counted.json`, md5 `bb4e246095fa0fb62dce318e078e9f6b`.
No indicator moves, and no score changes.

## Verdict

0. **The follow-up CC2b FAILED, so design B fails its necessary condition.** B2b held at 0 of 3
   seeds. The owner approved this follow-up after the first result, and its rules were pushed at
   commit 03eb4ed (13:33 UTC) before its run. Its smoke run changed once, at commit 3e84cfa, also
   before its run. Details are in the section on the follow-up below.
1. **Both first checks FAILED their own design.** The label FOOD_AHEAD never changes inside an episode in
   these runs. Its negative steps are whole episodes with no food at all. They are episodes 0, 2 and
   4 at seed 601, episodes 0, 1, 2 and 4 at seed 602, and episode 0 at seed 603. In every other episode, food
   comes within 60 steps at every counted step. So the label marks 1 to 4 early episodes without
   food. It does not mark the moments before food. I
   did not check the structure of the label against the records before the rules were pushed. That
   was my error.
2. **Design A FAILED CC1 by its rule.** A2 held at 0 of 3 seeds. The valence of design A was lower,
   not higher, in the episodes with food. Because of point 1, this shows only that the valence of
   design A does not separate the early episodes without food in the stated direction.
3. **The PASS of design B in CC2 is not evidence.** By the rule, B2 held at 3 of 3 seeds. But the
   read separates at most 4 episodes from the rest. At seed 603 it is below chance (AUC 0.3837), and
   it passes only because its null is below chance as well (p95 0.3077). With one episode without
   food, the fold that holds that episode trains on one label only, so its scores are constant, and
   pooling the folds ranks them above the rest. The pooled AUC mixes folds with different
   intercepts, so its size cannot be read at the other two seeds either.

## Numbers

From `appraisal_counted.json`. AUC is for FOOD_AHEAD over the counted steps. The null swaps the
labels of whole episodes, 200 times for the fixed streams and 20 times for the read.

| | s601 | s602 | s603 |
|---|---|---|---|
| Counted steps, positive and negative | 5180, 420 | 5040, 560 | 5460, 140 |
| `phen_valence` AUC, null p95 (A2) | 0.4608, 0.5770 | 0.2728, 0.6271 | 0.4374, 0.8604 |
| `a_dominance` abs(AUC - 0.5), null p95 (A3) | 0.1059, 0.0911 | 0.0391, 0.1092 | 0.1018, 0.1827 |
| `a_dominance` distinct values (A1, sanity) | 1414 | 1666 | 1632 |
| `reflex` AUC, null p95 | 0.0515, 0.5415 | 0.0521, 0.5547 | 0.0535, 0.5220 |
| `a_valence` AUC, null p95 | 0.0572, 0.6046 | 0.0462, 0.6399 | 0.0329, 0.8039 |
| `value_estimate` AUC, null p95 | 0.8898, 0.7087 | 0.9829, 0.7172 | 0.9948, 0.9395 |
| Broadcast read AUC, null p95, step index AUC (B2) | 0.7215, 0.6368, 0.2838 | 0.8527, 0.6498, 0.3889 | 0.3837, 0.3077, 0.0897 |

A3 held at 1 of 3 seeds (601). The value estimate of the action core separates the episodes most
clearly. The episodes without food are among the first of each run, so any quantity that changes
while the agent learns could separate them too. I did not measure how the value estimate changes
over the episodes, so this column cannot be read as a property of design C.

The negative counts are 3, 4 and 1 times 140, the counted steps of one episode. A short diagnostic
read of the same records, after the run, gave the episode numbers in point 1. It is not a check.

## A finding outside the checks

**The Stage 1 valence is -1.0 at every step that follows a food step.** Counted on the same records:
6833 of 6833 such steps at seed 601, 6643 of 6643 at 602, and 7181 of 7181 at 603. The cause is in
the code. Both calls of `evaluate_emotion` before the action pass an environment reward of 0.0
(`scripts/training/train_rlhf.py:1146` and `:1532`), so the reward change is the negative of the last
reward, and a last reward of 1.0 gives a valence of -1.0. The valence of 1146 is the PAD state that
the settle loop passes to the modulator (`:1315`). With a negative valence the modulator raises the
bids of the modules in its threat set, unless `--learned-valence` is on
(`models/emotion/affective_modulator.py:181`). So while the agent receives food, its bids are
modulated as if by a threat. This is a fact about the code that ran in every recorded run.

**Checked the same day, at the owner's request. The inverted valence changes the bids and does not
change the leader, at 3 of 3 seeds.** Rules in `scripts/analysis/probe_valence_after_food.py`, pushed
at commit d388965 before any run. The smoke pass then reproduced the recorded bids at only 4.3
percent of its steps, because the last settle cycle starts from the settle loop's final bids and
not from the raw bids. The probe was corrected to read the final bids and pushed at commit 5b088d2,
before the counted run. The counted run took 12 seconds. Output:
`runs/_audit_followup/nov02/valence_counted.json`, md5 `eaab77cfd7d00fad24c2d70f097945af`.

| | s601 | s602 | s603 |
|---|---|---|---|
| Steps after a food step | 6833 | 6643 | 7181 |
| Recorded bids reproduced by the real modulator (R0) | 1.0000 | 1.0000 | 1.0000 |
| Share of those steps with another leader at valence 0.0 (V2) | 0.0 | 0.0 | 0.0 |
| Mean rise of the audio bid and of the body bid | 0.1199, 0.1210 | 0.1203, 0.1209 | 0.1198, 0.1209 |
| Mean rise of the vision bid | 0.0604 | 0.0674 | 0.0630 |

The rise is at most 0.15, the valence gain, and the bids stop at 1.0. Vision, audio and body are all
in the threat set, so they rise together and their order does not change. By the rule stated before
the run, the defect is recorded and no change of the default is proposed. The limits are those of
the probe. The check covers one settle cycle, it reads the leader of the modulated bids and not
the winner after binding, and it does not measure the other readers of this valence (the memory
query, the self-model and the logged `valence` in `metrics.csv`).

## Smoke passes

The first smoke choice, episodes 0 and 1 of `runs/ta_nolink_s801`, was VOID (0 food steps), and that
run has no food step in any of its 40 episodes. The second, episodes 0 to 4 of `runs/s2c_s125`, was
VOID (39 positive steps). The third, all 20 episodes of `runs/s2c_s125`, reached every check. Its
numbers are not counted.

## The follow-up CC2b

Rules in the probe docstring, section FOLLOW-UP CC2b. Before the rules, I read only when the first
food comes in each counted episode. It comes at steps 12 to 21 in all 112 food episodes, with a
median of 14. So only the steps before the first food count, a step is positive when the first
food comes within 7 steps (half of 14), the episodes without food are left out, and the AUC is
computed inside each held-out fold. The counted run started at 13:35 UTC at commit 3e84cfa and took
23 seconds. Output: `runs/_audit_followup/nov02/appraisal_followup_counted.json`, md5
`0ee0404c8199fda2bc5aca6767da4974`.

| | s601 | s602 | s603 |
|---|---|---|---|
| Food episodes, positive and negative steps | 37, 259, 271 | 36, 252, 269 | 39, 273, 307 |
| Broadcast read, mean fold AUC | 0.6905 | 0.6585 | 0.6468 |
| Null p95, 20 reads with swapped labels | 0.6540 | 0.6083 | 0.6237 |
| Step index read, mean fold AUC | 0.9666 | 0.9707 | 0.9625 |

B2b needs the broadcast read above both the null and the step index read. It is above the null at 3
of 3 seeds and below the step index at 3 of 3 seeds, so B2b holds at 0 of 3 and design B FAILS its
necessary condition. The broadcast carries something about the coming food, and much less than the
time since the start of the episode. The first food comes at nearly the same step in every
episode, so the step index is a strong control. The rules said so before the run, and the rule
stands.

The smoke run of the follow-up was first `runs/s2c_s125`, VOID at its run (4 food episodes), and
then `runs/rv_stay_s701`, which reached the read. Its numbers are not counted.

Not run, and not planned without a decision. A read of the broadcast together with the step index,
against the step index alone, would ask the same question more directly. It is another read, so it
would be a new check with new rules.

## What this means for the designs

- Design A stays rejected. It has no basis in the sources, it FAILED its rule within the limit of
  point 1, and with the drive on it would switch on a survival term of the reward.
- Design B is removed for this broadcast. It FAILED its necessary condition in the follow-up.
- Design C was not checked. The only recorded stand-in grows with learning.
- The realness signal R2 needs the prediction of design B, so it has no basis in this agent now.
- The relation of content cannot be tested on the present test bed
  (`docs/decisions/2026_11_02_content_relation_design.md`), so its check was not run.

## Recommendation, and what could make it wrong

1. Do not build design A or design B.
2. Keep Stage 2 off (design D) unless the owner wants design C, which reads the learned value of the
   task reward and holds with the drive off.
3. Keep dominance at 0.0 (owner decision of 2026-11-02).
4. Leave the Stage 1 valence after food as it is. It does not change the leader in these runs, and a
   change of it would change every training number. Any document that reads `valence` from
   `metrics.csv` reads this inverted signal, which should be said where such a value is cited.

What could make this wrong. Design B was tested only on the broadcast of these three runs, where
the light is reached at nearly the same step every time. A room where the time to food varies more
could give another result. The other readers of the valence after food were not measured.

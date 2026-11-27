# What the broadcast swap of step P3 predicts, for a head with two slots

Date: 2026-11-24. Status: rule text written before any code and before any run. It is a proposed
addition to requirement 6 of step P3 and to note N10 in
`docs/decisions/2026_11_22_workspace_choice_mode.md`. It is not a rule of step P3. The rules of step P3,
with their thresholds, stay the owner's decision. No indicator moves with this text. Nothing here is a
claim about consciousness.

## The source

He and Wu 2026, "Does the Workspace Do the Work? A Causal Protocol for Verifying Global-Workspace
Computation in a Language Model", arXiv 2611.26018. A preprint, not peer reviewed. I read sections 3 to 7
and most of appendices E to H in the HTML version. I did not read appendices A to D or appendix I.

What I took from it.

1. Task accuracy and an ablation of the broadcast show only that information flows through the channel.
   They do not show what the reader does with it (section 1, section 3.2).
2. The central test transplants one message of one module, taken from a donor input. The donor is chosen
   from the task, not from the outputs of the model, so that three computations predict three different
   answers. Ignoring the message predicts the native answer. Relaying the donor's answer predicts the
   donor's answer. Recombining the message with what the reader already holds predicts a third answer
   (section 3.2, table 1). The three predictions must differ in each counted case.
3. Two controls. A transplant of the module's own message changes nothing (self replacement,
   appendix F.4). A transplant at a site that nothing reads changes nothing (section 3.2).
4. On small networks built to relay or to recombine, the test gave the built computation in 4,096 of 4,096
   cases (section 3.2). Trained models gave all three outcomes (section 5.5). These numbers come from a
   language model setting. None of them carries over to this agent.

## Why the swap of note N10 cannot use the three answers as written

The choice head of step P2 gives two values, one for slot 0 and one for slot 1. Slot 0 is the candidate
with the higher salience (`models/core/choice_head.py` lines 3 to 7 and 40 to 43, and the rules of step
P2, table "The choice mode", row "Slots"). The room has two candidates. So each swap has two possible
answers, and a third answer that differs from the other two cannot exist. A test that waits for "neither
of the two" would never see it.

## What a head with two slots can still separate

The head reads the broadcast and gives a slot. The slot becomes a band through the salience order of the
recipient episode, which the head does not read from the broadcast. This gives three hypotheses that
predict a band.

| Hypothesis | What the head does with the donor broadcast | Predicted band |
|---|---|---|
| IGNORE | Nothing. The choice is the native choice | The target band of the recipient |
| BAND | It relays the donor's target as a band | The target band of the donor |
| SLOT | It reads which slot held the target in the donor, and applies that slot to the salience order of the recipient. This is a recombination of the message with what the recipient holds | The band at that slot in the recipient's order |

The prediction of IGNORE assumes that the native choice is the target. The rules of step P3 decide how
episodes with a wrong native choice are counted.

## The donor classes

Each donor is the vector of the same module from another episode, as in note N10. Each swap belongs to one
class, set from the targets and the salience orders before the choice is read.

| Class | Donor target | Donor salience order | IGNORE | BAND | SLOT |
|---|---|---|---|---|---|
| C1 | Other band | Same as the recipient | own | other | other |
| C2 | Other band | Reversed | own | other | own |
| C3 | Same band | Reversed | own | own | other |
| C0, self replacement | Same band | Same as the recipient | own | own | own |

"own" is the target band of the recipient. "other" is the other band.

Rules.

1. **One class alone cannot separate the three hypotheses.** In C1, BAND and SLOT predict the same band. In
   C2, IGNORE and SLOT do. In C3, IGNORE and BAND do. Only the classes together give three different
   rows.
2. **Note N10 as written mixes C1 and C2.** It takes a donor "with the other target" and says nothing of
   the salience order. With the decoy levels in a random order (requirement 2), both classes occur. SLOT
   then predicts "other" in C1 and "own" in C2, and the pooled result looks like a weak BAND. So each swap
   records its class, and the choice is reported for each class, never only pooled.
3. **C1 and C2, each reported apart, are enough to separate the three.** C3 is added as a second contrast.
   It is the only class in which SLOT predicts a band that IGNORE and BAND do not, so a SLOT result does
   not rest on one class.
4. **C0 is the sham.** All three hypotheses predict the native choice. A change of the choice in C0 is a
   disruption by a foreign vector, not a use of content. It is the self replacement control of the paper.
5. **The blank cut of requirement 6 cannot separate BAND from SLOT.** Both predict a fall to chance when
   the content is removed. Only the classes above separate them.
6. **The ladder of requirement 7.** A choice that follows BAND or SLOT supports, at most, the level that
   the broadcast moves the choice. IGNORE in every class counts against that level. The level that the
   choice needs the broadcast still needs requirement 8.
7. **What this text does not set.** No threshold, no count of swaps per class and no pass rule. Those
   belong to the rules of step P3.

## The helper

`scripts/analysis/swap_predictions.py` gives the predicted band of each hypothesis for a recipient and a
donor, the class of a swap, and a check that a set of swaps separates the three hypotheses. It reads no
run and writes no number. It is not imported by the training script, so no flag is needed and a run of
the agent does not change. `tests/test_swap_predictions.py` holds its tests, written before it.

## Limits

1. The paper is a language model study with one message per module. The agent has two candidates and a
   head with no input other than the broadcast.
2. SLOT is one form of recombination. A head can mix the two in a way that none of the three rows
   predicts. The rules of step P3 decide how a mixed result is labelled.
3. The classes assume that both episodes have the same two bands as candidates. A swap in which they
   differ is not in any class.
4. I did not read appendices A to D or appendix I of the paper.

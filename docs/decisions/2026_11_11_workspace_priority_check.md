# Workspace priority on the tectal map: rules of the offline check

Date: 2026-11-11. Status: rules written before the check script exists and before any run of the
check. No model is trained. No gate runs. No default changes. The mechanism is not written.

## The question

The design under test keeps the tectal map as the shared space. The workspace adds a priority to a
candidate group that the map already holds. No place enters the broadcast. The offline check asks one
question of saved runs. Would a priority term from the workspace winner change the choice of the value
gate, and is that term more than a copy of the salience that the gate already reads?

The check is a correlation on saved runs. It is not an ablation, and it cannot show that the workspace
acts on behaviour.

## The runs

Six saved runs, in two groups. Each run has `value_gate.json`, `map_binding.json` and one `steps.jsonl`
for each episode.

| Group | Runs | Seeds | Episodes in each run |
|---|---|---|---|
| Value learning | `vg_learn_s601`, `vg_learn_s602`, `vg_learn_s603` | 601, 602, 603 | 40 |
| Food switch | `rv_switch_s701`, `rv_switch_s702`, `rv_switch_s703` | 701, 702, 703 | 60 |

A read of the records before these rules confirmed that the fields exist. Each step has the candidate
groups by band, the salience by band, the winner of the workspace and the ignition state. Each episode
has the choice of the gate with its salience and its priorities. That read looked at field names, at
counts and at the two controls below. It did not look at the winner at any choice step.

## Definitions, all fixed here

1. **Choice step.** The first step of an episode where the binding record has a band and at least one
   group with cells. This is the rule of `_wire_value_gate` in `scripts/training/train_rlhf.py`.
2. **Order inside a step.** The tectum writes the binding record, then the workspace competes, then the
   action is selected. So the winner of the choice step exists before the gate chooses.
3. **Workspace term.** The saved runs record the winner as a sense and not as a band. So the term is
   `term(band) = 1` when the workspace is ignited at the choice step, the winner is `audio`, and the
   band is the loudest band of that step. In all other cases `term(band) = 0`. A `vision` winner gives
   no term, because both candidate groups are seen objects and the sense does not tell them apart.
4. **Weight.** The score of a band becomes its recorded priority plus `1.0 x term`. The weight 1.0 is
   the salience of the loudest tone. It is the one free value of this check. Weights 0.5 and 2.0 are
   printed as a report only and have no part in the verdict.
5. **Predicted choice.** For each episode, the gate's softmax at temperature 0.25 over the scores of
   that episode (recorded priority plus weighted term), with the random draw that the gate used. The draws come from a generator with the
   gate's own seed, the run seed plus 1000003, one draw for each episode. The values of earlier
   episodes stay as recorded, so each episode is a one-step change and not a new history.

## The rules

| Rule | Measure | Holds when |
|---|---|---|
| 1 | Share of episodes where the predicted choice with the term differs from the recorded choice | 0.05 or more |
| 2 | R squared of an ordinary least squares fit, with intercept, of the term on the salience. The rows are the candidate bands of all episodes of the run | Below 0.90 |

Both rules are read for each run.

## Controls with exact expected values

| Control | Expected value | On a mismatch |
|---|---|---|
| A. Replay with weight 0. The predicted choice equals the recorded choice | Every episode of every run, 40 of 40 or 60 of 60 | VOID |
| B. The salience of the gate record equals the salience of the binding record at the choice step | Exact equality in every episode | VOID |
| C. The values are computed again from the recorded outcomes with the gate's update rule, from 0, at the recorded learning rate. The priorities `salience + 2 x value` equal the recorded priorities | Largest difference at most 1e-9 | VOID |
| D. The fit of rule 2, given `2 x salience + 1` in place of the term | R squared 1 within 1e-9 | VOID |
| E. The fit of rule 2, given a term that is 0 for every row | The script raises and prints no number | VOID if a number comes out |

The script raises on a mismatch. It does not print a verdict after a failed control.

## What the verdicts mean

- **SUPPORTED.** All controls pass, and rules 1 and 2 both hold at 3 of 3 seeds in each group. It means
  that in these runs the winner of the workspace would change the gate's choice often enough to be
  measured, and the term is not a linear copy of the salience. This permits a rules file for a live
  gate. It does not show an effect on behaviour.
- **EXCLUDED.** All controls pass, and SUPPORTED is not reached. The report states which rule failed, at
  which seeds and in which group. It means that in this room the workspace winner gives the gate nothing
  that it can use at this weight, or it gives a copy of the salience. The option with no link stays in
  place for this room.
- **VOID.** A control fails, a field is missing, or an episode count differs from the table above. No
  verdict on the design follows.

## Limits known before the run

1. **Rule 2 is weak for a binary term.** The term is 1 only for the loudest band, and only when audio
   wins. Let q be the share of episodes with an audio winner at the choice step. If the quieter
   salience did not vary, R squared would equal q / (2 - q). That value reaches 0.90 only when q is
   above 0.947. So rule 2 can fail only when audio wins at nearly every choice step. A pass of rule 2
   is weak evidence. The script prints q for each run so that the reader can see this.
2. **The term cannot favour the quieter band.** The recorded winner is a sense. The only band that an
   audio winner can point to, from the records, is the loudest band. So the band that gets the term is
   set by the salience. The workspace adds only the fact that audio won at that step.
3. **One free weight.** A different weight gives a different count for rule 1.
4. **One step.** The choice is made once in each episode, at one step. The winner at that one step is
   the full workspace input to the check.
5. **Not independent groups.** Both groups use the same room, binder and gate. The food switch runs
   differ in the reward rule and in length.
6. **Sources.** The selection rule of Gurney, Prescott and Redgrave (2001) was read at abstract level
   only. This check does not use that rule. It uses the gate's own softmax.
7. **A run with no audio winner at any choice step.** The term is then 0 in every row. The fit of rule 2
   has no value, the script prints "undefined", and rule 2 does not hold for that run. Rule 1 gives 0.
   This clause was added on 2026-11-11 after the first push of this file and before the script ran.

## After the check

The verdict goes in a dated file in `docs/results/`, with the commands and the output. A live gate
needs its own rules file and the owner's approval.

# KILL. De-saturating the vision bid is achievable and does not produce competition

**KILL, on the pre-stated gate's condition (ii), checkpoint arbitrariness.** Stage 1
and Stage 2 of the workspace-bid repair do not proceed. No model code was changed and
no training was run.

**Two results, and the second is the one worth having.**

1. **The pre-`tanh` KL scale varies 8,079x across three independently trained
   checkpoints of the same configuration**: mean 2.1251e+03, 4.5509e+06, 1.7168e+07.
   Nothing in the loss constrains it. So any reduction with a FIXED divisor has no
   defined operating point, and gives a different constant winner per training seed.
2. **De-saturation is solvable and it is not sufficient.** The self-normalizing
   reductions de-saturate reliably on all three checkpoints. The winner still never
   changes. **0 of 24 candidate and opponent combinations** produce a changing winner
   at all three checkpoints.

**So the bid formula is not the blocker.** GWT-2's selective-attention half is blocked
by two other things, named below, and neither is reachable from
`sensory_tectum.py:456`.

No indicator moves. The rubric stays 3 IMPLEMENTED, 11 PARTIAL of 14. The clock does
not move.

## Method

`scripts/analysis/probe_bid_counterfactual.py`, read-only. Six candidate vision-bid
reductions replayed through the REAL `reentrant.settle` into the real
`GlobalWorkspace`, on `runs/gate3_s4{2,3,4}`, against four opponent configurations
drawn from measured live values. 8 episodes per checkpoint, 400 warm-up steps
discarded, 1200 settled steps analysed.

Free parameters were fixed in the module docstring before any winner number was read:
`EMA_ALPHA` 0.01, `SD_FLOOR` 1e-9, `TOPK_FRACTION` 0.01.

**Harness fidelity passed at all three checkpoints.** S0 under the `bcast` opponent
set, which reproduces the live configuration of `runs/bcast_s4{2,3,4}`, gives a vision
share of 1.000 of ignited steps with silence at 0.000, against a published live result
of vision 7922 and semantic 57 of 7979 ignited steps. The converged `sync_R` reads
exactly 0.450000 against a published modal 0.450108.

## Three harness defects found and fixed before any number was believed

Recorded because each would have produced a confident wrong answer.

**The Kuramoto phase init was unseeded.** The same candidate on identical input gave a
vision share of 0.946 and then 0.490. Phases decide near-ties at
`oscillatory_binding.py:190-199`, so every difference between candidates would have
been phase noise. Now seeded identically for every chain.

**An empty winner was being counted as a winner.** `winners` is only populated when
`is_conscious` (`global_workspace.py:307-309`). Counting the empty string let a bid
that SILENCES the workspace look like a changed winner. Silence is now excluded from
the share and reported separately.

**A single `run_competition` call per step is not the live harness.** The first version
called it once and failed fidelity at a vision share of 0.375. `reentrant.settle`
calls it up to five times per step, which updates the ignition baseline EMA five times
as often. Fixed by replaying through the real settle loop.

**And a warm-up transient.** Over the first quarter of a run the workspace is silent on
0.608 of steps with a vision share of 0.790; from the second quarter on, `sync_R` is
converged and the figures are 0.000 and 1.000. 400 steps are now discarded.

## Result 1: the KL scale is a free parameter of training

| Checkpoint | mean | min | max |
|---|---|---|---|
| `gate3_s42` | **2.1251e+03** | 1.4847e+03 | 2.1371e+03 |
| `gate3_s43` | **4.5509e+06** | 6.8499e+03 | 4.5947e+06 |
| `gate3_s44` | **1.7168e+07** | 8.5182e+03 | 1.7311e+07 |

**8,079x between the smallest and largest.** Same architecture, same flags, same
number of episodes, three training seeds.

S1, `tanh(kl/N)`, shows the consequence directly in its own output:

| Checkpoint | S1 range | distinct values in 1200 steps |
|---|---|---|
| `gate3_s42` | 0.0057 to 0.0082 | **484** |
| `gate3_s43` | 0.0261 to 1.0000 | **3** |
| `gate3_s44` | 0.0325 to 1.0000 | **2** |

One fixed divisor gives a live, varying, de-saturated bid on one checkpoint and a
near-binary saturated one on the other two. That is the arbitrariness, and it is why
condition (ii) fired: S1 under `memfix` gives a constant winner of memory, vision,
vision across the three.

## Result 2: self-normalizing reductions work, and it changes nothing

| Candidate | distinct values, s42 / s43 / s44 |
|---|---|
| S0 `tanh(sum)`, baseline | 1 / 1 / 1 |
| S1 `tanh(mean)`, fixed divisor | 484 / 3 / 2 |
| **S2** `sigmoid(mean z)` | **1196 / 1199 / 1195** |
| **S4** `sigmoid(top 1% z)` | **1200 / 1197 / 1198** |
| **S5** `sigmoid(z of sum)` | **1200 / 1200 / 1200** |

S2, S4 and S5 de-saturate on every checkpoint, consistently, with no tuned constant.
The saturation problem at `sensory_tectum.py:456` is solvable.

And the winner still does not change:

**3 of 72 checkpoint/candidate/opponent rows have more than one winner at all**, and in
every one of those the runner-up takes 0.0033 to 0.0065 of ignited steps, against a
pre-stated bar of 0.05. The minimum runner-up share across all three checkpoints is
**0.000 for every one of the 24 combinations.**

## Why, and this is the part that matters

**The opponents are constants.** Audio is 0.0, memory is 0.1 or 0.6, body is 0.15,
semantic is 0.0 or 1.0. A varying vision bid crossing a constant is a threshold
crossing, not a competition. Whichever side of the constant the bid sits on, the same
module wins every step.

**Lowering the vision bid does not transfer the win. It silences the workspace.** Of
the 24 combinations, 13 are silent on more than half of steps and 11 are silent on
1.000 of them. The mechanism is `global_workspace.py:293-307`: ignition requires
`input_energy >= EMA(input_energy)`. When one constant module dominates, the baseline
converges onto it exactly, the workspace sits on the ignition boundary, and any
downward fluctuation turns it off. The workspace does not hand the win to the
runner-up. It stops igniting.

**So GWT-2's selective attention is blocked by the constant bids and by the ignition
rule, not by the bid formula.** No change at `sensory_tectum.py:456` reaches either.

## What this establishes

- A fixed-divisor bid reduction is disqualified on this architecture, measured at 3
  checkpoints.
- The saturation itself is fixable by self-normalization, with no tuned constant.
- Fixing it does not produce competition, so the planned Stage 1 and Stage 2 would
  have spent about two days and three hours of training to reach a negative that
  cost 25 minutes of read-only replay.

## What this does NOT establish

- **It does not say competition is unreachable**, only that it is unreachable from the
  vision bid alone while the other four bids are constants.
- **It says nothing about content.** That was settled separately and negatively in
  `bid_reduction_candidates_2026_08.md` and is not re-litigated here.
- **Silence figures are not comparable to live runs.** The replay reproduces the live
  WINNER but not the live silent fraction (0.003 live). No claim about workspace
  silence rests on this probe's absolute numbers; the mechanism above is read from
  the code, and the relative pattern across candidates is what is used.
- **One task.** DMTS only, where audio is structurally silent.
- **It does not re-score GWT-2.** The indicator's evidence is unchanged.

## Next, if this is picked up again

1. **The real blocker is that four of five bids are constants.** Making them vary is
   four separate module problems, not one bid formula. `memory_retrieval_repair_2026_08.md`
   already shows that repairing one moved it from one constant to another.
2. **The ignition rule deserves its own look.** `input_energy >= EMA(input_energy)`
   makes a stable workspace marginally conscious by construction, and that is a
   design question, not a bug report.
3. Do NOT wire the body bid. `ProprioceptiveProcessor` is never instantiated outside
   tests, its estimator is untrained, and DMTS supplies no proprioception.

## Reproduce

```
python -m scripts.analysis.probe_bid_counterfactual \
  --checkpoints runs/gate3_s42 runs/gate3_s43 runs/gate3_s44 \
  --episodes 8 --max-steps 200 --warmup 400 --out runs/_bidcf/stage0.json
```

About 9 minutes. The probe REFUSES to print a verdict if the harness fidelity check
fails, which it did three times during development.

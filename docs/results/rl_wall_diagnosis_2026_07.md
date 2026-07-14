# RL wall diagnosis (Track C1, 2026-07-06): the match is available but not expressible by the policy head

Read-only diagnosis, no training. After the perception chain was completed end to end
(identity now reaches tectum_content, `b1_continuous_latent_2026_07.md`), the agent still
does not learn DMTS (reward flat and negative in every run). C1 asks why. The cheapest
decisive question: is the correct DMTS choice decodable from what the policy actually reads?

## Method

Reused `scripts/analysis/probe_match_decode.py` + `scripts/analysis/decode_choice_records.py`
(built for the 2026-06-15 match-head arc). The obsmem-conv policy input is
`[current obs_map ; held sample memory]` (`ObsMapSampleMemory`). At the first choice frame
of each trial, one record per trial, decode the `target_position` label three ways. Probe
conditions (scripted policy, no training, no_grad), 280 records, DMTS seed 42. This env
config presents 2 choices, so chance = 0.500.

## Result

| decoder | test acc | chance |
|---|---:|---:|
| PCA-80 + MLP | 0.786 | 0.500 |
| PCA-80 + linear | 0.536 | 0.500 |
| conv MatchHead (the policy's head architecture, trained offline 300 epochs) | 0.655 (train 0.520) | 0.500 |

## Reading

1. **The match information IS present in the policy input.** A PCA+MLP decodes the correct
   choice at 0.786 vs 0.500 chance. The agent is not blind to the answer.
2. **The match is a NONLINEAR, non-local relation.** The linear probe gets 0.536 (near
   chance); only the MLP recovers it. Deciding "which choice equals the held sample" needs
   a comparison across spatial positions, not a linear readout.
3. **The conv policy head cannot express it.** The actual MatchHead architecture, trained
   offline with unlimited epochs and a held-out split, reaches only 0.655 test / 0.520
   train. It plateaus well below the PCA+MLP ceiling. This reproduces the 2026-06-15
   finding (the conv AdaptiveAvgPool destroys the position-specific comparison the match
   needs) on the current perception.

So the DMTS wall is NOT information availability and NOT the perception collapse (that is
now fixed upstream). It is two coupled problems: (a) the policy head architecture cannot
represent the non-local match comparison, and (b) the online RL credit assignment across
the sample-delay-choice gap (the 2026-06-14/15 in-loop-degradation result). Fixing
perception was necessary for the biologically faithful broadcast/signature path, but the
obsmem-conv policy already had the raw information and still could not learn, which places
this wall squarely on the policy-head + RL side.

## Honest scope

- Probe conditions use untrained components (obs_map is identity-rich even untrained), so
  0.786 is "the match is decodable in principle from the policy input", not a claim about
  a trained agent. The trained-agent in-loop degradation is the 2026-06-15 result.
- 2-choice env config (chance 0.500); a harder choice count would change the numbers, not
  the three-way reading.
- Single seed (42).

## Fork (owner decision, no fix built)

1. **Policy-head expressiveness:** replace the conv/AdaptiveAvgPool match path with a head
   that can do the non-local comparison (attention or an explicit sample-vs-choice
   cross-correlation). Read-only evidence says the ceiling is >= 0.786, so there is room.
2. **Credit assignment across the delay:** the aligned-resources map points at relational
   RL, temporal value transport, and synthetic returns for the sample-delay-choice gap.
3. **Accept and characterize:** the perception chain is complete and the wall is now
   precisely localized to the policy head + RL; report that as the honest state.

These are distinct bets; none is auto-started.

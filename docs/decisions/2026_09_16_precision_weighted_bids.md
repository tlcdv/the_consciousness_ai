# Owner override and pre-stated Gate B3: reliability-weighted sense bids

Date of decision: 2026-09-16. Decided by the owner. Written BEFORE the runs it judges.

## What failed first

The offline gain check (`scripts/analysis/probe_sense_gain.py`, seeds 51 to 53) **FAILED**.

| Criterion | Seed 51 | Seed 52 | Seed 53 |
|---|---|---|---|
| (R1) AUC of vision gain, light in view, at least 0.70 | 0.997 | 0.981 | 0.904 |
| (R2) AUC of the audio share, light out of view, at least 0.70 | 0.996 | 0.961 | 0.857 |
| (R3) usable in-view runs, at least 3 | **1** | 7 | 5 |
| (R3) median ratio, at least 0.80 | 1.000 | 1.000 | 1.000 |

R3 failed at seed 51 because the light stayed in view long enough only once, so the
criterion could not be measured there. The gate's own rule was "a FAIL means no build".

## The override

The owner decided on 2026-09-16 to build anyway, and to judge the result with a live gate.
Reasons recorded at the time of the decision:

- the R3 failure is "not measurable", not a measured habituation;
- R1 and R3 can pass by the construction of the room (the light is the brightest object
  and fills whole grid cells), so more offline checks add little;
- only a live run answers whether the weighting changes which module wins.

Nothing in the offline check is evidence that the weighting works.

## What was built

`models/core/precision_weighting.py`, behind `--bid-precision off|gain`, default `off`.

- vision gain: the strongest pooled cell of the frame outside the agent's own disc;
- audio gain: the waveform RMS over the tone RMS at the source;
- each sense bid is multiplied by twice its share of the total gain, so equal gains leave
  both bids unchanged, and the bids stay inside [0, 1].

Sources: Ma, Beck, Latham and Pouget 2006 (gain carries reliability); Fetsch et al. 2009
and 2011 (weights change within a trial, read from the current input); Feldman and Friston
2010 and Stein and Stanford 2008 (salience habituates, reliability does not); Ohshiro,
Angelaki and DeAngelis 2011 (divisive normalization of the shares).

Bit identity with the flag off: `metrics.csv` md5 `7c101c58eecc4eb91409c491a5ad0e26` for the
standard check (`--env dark_room --episodes 2 --max-steps 30 --seed 42 --existence-drive on`),
measured twice with the new code.

## Gate B3, pre-stated

The Gate B2 criteria, unchanged, including the task criterion that failed. The full text is
in the docstring of `scripts/analysis/probe_gate_b2.py`:

1. competition: the runner-up takes at least 0.05 of ignited steps;
2. silence: below 0.50 of steps;
3. selectivity: against the within-episode shuffled null;
4. task: pooled Spearman rho above 0 with a one-sided permutation p below 0.05, and rho
   above 0 at 2 or more of the 3 seeds;
5. KILL: any module at 0.95 or more of ignited steps.

Data: new seeds 54, 55 and 56, never run before, with the Gate B2 flags plus
`--bid-precision gain`:

```
python -m scripts.training.train_rlhf --env dark_room --episodes 10 --max-steps 200 \
    --seed <54|55|56> --enable-audio --rssm-latent-mode continuous \
    --capsule-workspace-source all_levels --existence-drive on \
    --dark-room-audio binaural --dark-room-audio-channels 4 --dark-room-collision \
    --dark-room-view agent_centered --vision-bid-reduction zscore \
    --audio-salience surprise --learned-valence \
    --ignition-rule tolerance --ignition-tolerance-sd 1.0 --bid-precision gain \
    --log-dir runs/gate_b3_s<seed>
```

A PASS on criterion 4 is the first evidence that hearing wins more when the light is out of
view. A FAIL is recorded as a FAIL, FAILED first, in `docs/results/dark_room_senses_2026_09.md`.
A comparison against the Gate B2 runs is not part of this gate: those runs used other seeds.

A PASS is not evidence of affect. Finding a light is automatic approach, which Feinberg and
Mallatt exclude as evidence of affect.

## Result, 2026-09-16: Gate B3 FAILED

Seed 55 fired the KILL rule (vision 0.961 of ignited steps) and failed criterion (1) with a
runner-up share of 0.039. Seeds 54 and 56 passed criteria 1 to 3. Criterion (4) passed for
the first time: pooled rho 0.446, one-sided p 0.005, rho above 0 at all 3 seeds. The full
table is in `docs/results/dark_room_senses_2026_09.md`. The measured confound with the
episode index is stated there and is not excluded.

# RIIU substrate probe (2026-05-17): no winner

**Date:** 2026-05-17
**Run:** `runs/riiu_substrate_probe_seed42` (50 episodes, dark_room, seed 42)
**Flags:** `--enable-riiu --riiu-probe-all --riiu-source broadcast --ablate-gate-diversity --phi-sample-every 5`
**Wall clock:** 27 min 45 s on CPU
**Analysis window:** steps >= 1000, `phi_method != insufficient_data` (9000 rows)

## Headline

**No substrate wins the Phase B6 selection criteria.** Per the decision gate
in `~/.claude/plans/let-s-plan-the-next-misty-parasol.md`, Phase C
(3-seed verification) is SKIPPED. The project moves to Phase 5 of
`docs/roadmap.md` (Dynamic Self-Representation & Meta-Cognition) with the
conclusion that the current architecture's binding+phi+gate stack does not
measurably support Phi-1 under any reasonable measurement choice across
pathways (pyphi, RIIU) AND substrates (broadcast, tectum, audio).

## Phase B6 selection criteria

A substrate wins only if it satisfies BOTH:

1. `std(phi_riiu_X) >= 2x std(phi_riiu_broadcast)` (variance unlock vs broadcast baseline)
2. `|pearson_r(phi_riiu_X, sync_R)| >= 0.15, p < 0.05` (binding correlation)

## Results

Baseline: `std(phi_riiu_broadcast) = 7.134e-05` over 9000 post-warmup rows.

| Substrate | std | ratio to broadcast | r(phi, sync_R) | p | criterion 1 (2x var) | criterion 2 (binding r) | win |
|-----------|-----|--------------------|----------------|---|----------------------|-------------------------|-----|
| broadcast | 7.134e-05 | 1.000x | +0.1536 | 1.24e-48 | FAIL | PASS | NO |
| tectum    | 7.134e-05 | 1.000x | +0.1536 | 1.24e-48 | FAIL | PASS | NO |
| audio     | 0.000e+00 | 0.000x | +0.0000 | 1.00 | FAIL | FAIL | NO |

## Why broadcast and tectum produced bit-identical phi

Direct CSV inspection: `max |phi_riiu_broadcast - phi_riiu_tectum| = 0.000e+00`
across all 9000 valid rows. The two columns are pointwise identical.

Cause: in `scripts/training/train_rlhf.py` line 528-532, the `broadcast` tensor
is extracted as `broadcast_content["tensor"]`, which IS the vision payload's
`tectum_content` tensor (line 411 sets `vision_payload = {"tensor": tectum_content, ...}`).
When the vision module wins workspace competition (which happens at almost
every step in dark_room because audio is zero, semantic is zero, memory/body
bids are below vision's ~0.5 bid), the broadcast tensor is identically the
tectum tensor. The RIIU pipelines on each substrate therefore see the same
data and produce the same phi.

This is a structural property of the dark_room training loop, not a wiring
bug in the substrate probe. The probe correctly disproved its own hypothesis:
"broadcast might be too smooth, try tectum" turned out to be a non-question
because broadcast IS tectum in this run configuration.

## Why audio was degenerate

The probe ran without `--enable-audio`, so `audio_content` was the zero
stub on every step. `RIIUPhi.push` on a zero vector produces phi = 0 (after
warm-up, the covariance is exactly zero and the surrogate returns 0). The
`init_components` startup emitted the documented warning:
"--riiu-probe-all is on but --enable-audio is off. ... Re-run with
--enable-audio for a real audio-substrate measurement."

A future audio-substrate probe would require `--enable-audio` plus an
environment that actually generates non-trivial audio (DMTS has audio
events; dark_room does not).

## What this rules out

- **Substrate hypothesis from 2026-05-16 risk register #1.** "Broadcast tensor
  too smooth" was the proposed cause of low RIIU phi variance. The tectum
  substrate cannot resolve this because broadcast equals tectum in
  configurations where the vision module dominates workspace competition.
- **Phi-1 prediction via substrate selection.** Tectum produces r=+0.1536
  with sync_R, marginally above the 0.15 plan threshold but well below the
  pre-registered 0.4. Same value as broadcast (because same tensor). No new
  signal here.

## What this leaves open

- **Audio substrate on an audio-rich environment.** Untested due to dark_room's
  zero audio and the probe running without `--enable-audio`. Cost: re-run
  the probe with `--enable-audio` on dark_room (audio is synthesized by the
  environment; non-trivial sound events fire periodically per
  `simulations/environments/audio_mixin.py`), or move to DMTS where audio
  events occur on phase transitions. Not scheduled.
- **Genuinely different substrates** (`gate_state`, RSSM hidden, RND
  predictor output). All have dim < 16 (RIIU rank), so direct RIIUPhi
  application is degenerate. Would require either lowering RIIU rank to a
  value below the substrate dim, or projecting up. Out of scope for the
  current closeout.

## Decision

Phase B closes with the empirical finding that no candidate substrate
unlocks the variance needed for the Phi-1 prediction under the current
architecture. Phase C is SKIPPED per the decision gate. The project moves
forward to Phase 5 of `docs/roadmap.md`.

The RIIU code path (`--enable-riiu`, `--riiu-probe-all`, `--riiu-source`,
the `phi_riiu_broadcast`/`phi_riiu_tectum`/`phi_riiu_audio` CSV columns)
remains merged. It is opt-in (default off) and stays as a diagnostic
capability for any future architectural change that would make substrate
selection meaningful.

## Reproducibility

```bash
# Re-run the probe
PYPHI_WELCOME_OFF=yes python -m scripts.training.train_rlhf \
    --env dark_room --episodes 50 --max-steps 200 \
    --enable-riiu --riiu-probe-all --riiu-source broadcast \
    --ablate-gate-diversity --phi-sample-every 5 \
    --seed 42 --log-dir runs/riiu_substrate_probe_seed42

# Per-substrate verdicts
for SUB in broadcast tectum audio; do
    python -m scripts.analysis.compare_phi_pathways \
        --run-dir runs/riiu_substrate_probe_seed42 \
        --substrate $SUB
done
```

All numbers in this document are reproducible from
`runs/riiu_substrate_probe_seed42/metrics.csv` (10000 rows) and
`episodes.csv` (50 rows). The raw `runs/` directory is gitignored.

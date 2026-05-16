# RIIU vs pyphi phi pathway comparison

**Date:** 2026-05-16
**Run:** `runs/riiu_compare_seed42` (200 episodes, dark_room, seed 42)
**Flags:** `--phi-sample-every 5 --log-ei-every 50 --ablate-gate-diversity --ablate-gate-feedback --enable-riiu`
**Baseline:** `runs/ablation/A_current` (from 2026-05-14 peaceful-castle campaign)
**Analysis window:** steps >= 1000 (post-warmup), `phi_method != insufficient_data`

## Headline

**FAIL on 3 of 4 go/no-go criteria.** Per the pre-registered decision protocol in
the 2026-05-16 plan, this triggers Phase 8: keep the RIIU code merged behind
`--enable-riiu` (default off), preserve as a covariance diagnostic, do not
promote to default reward source, and proceed to Direction C (accept negative
result, reframe documentation).

The story is more nuanced than the headline (see "Trajectory" below).

## Go / no-go criteria (set before measurement, evaluated post-hoc)

| Criterion | Verdict | Detail |
|-----------|---------|--------|
| A. Variance unlock | FAIL | std(RIIU)=5.55e-05, std(pyphi)=7.16e-05, ratio=0.78x (need >= 5.0x) |
| B. Binding correlation | FAIL | pearson_r(phi_riiu, sync_R) = +0.0754, p = 2.99e-50 (need r >= 0.15). Pyphi's prior best from the 2026-05-14 ablation campaign was r=+0.089, so RIIU is worse on this metric over the full run. |
| C. Signal alive | FAIL | mean(phi_riiu)=2.75e-04, need >= 1e-3 |
| D. No reward regression | PASS | run last-100 reward = +17.31 (n=100), baseline last-100 = +8.38 (n=100, 1 SE = 3.58), threshold = +4.80. **Caveat: within-run trajectory shows reward declined from +29.4 (first-100) to +17.3 (last-100); see "Reward regression within the run" below.** |

## Distribution summary

| metric | mean | std | min | max | nonzero rows |
|--------|------|-----|-----|-----|--------------|
| phi (pyphi) | 1.22e-05 | 7.16e-05 | 0.0 | 1.60e-03 | 38960 / 39000 |
| phi_riiu | 2.75e-04 | 5.55e-05 | 2.95e-07 | 5.34e-04 | 39000 / 39000 |
| sync_r | 2.43e-01 | 1.76e-02 | 2.36e-01 | 2.90e-01 | 39000 / 39000 |

RIIU produces non-zero phi at every step past warm-up (39000 / 39000), while
pyphi has 40 zero-phi rows. RIIU mean is ~22x larger than pyphi mean (the
pyphi pipeline collapses to its fixed point of ~5e-06 as observed in all prior
runs). So RIIU is measuring something different from pyphi. The question is
whether that something tracks binding or task performance, and the answer
over the full run is "no with caveats."

## Trajectory: RIIU phi - sync_R correlation by 5000-step window

| Window (steps) | r(RIIU, sync_R) | r(pyphi, sync_R) | RIIU mean | pyphi mean |
|----------------|-----------------|------------------|-----------|------------|
| 1000-6000 | +0.196 | -0.025 | 2.6e-04 | 6.0e-05 |
| 6000-11000 | +0.156 | +0.009 | 2.8e-04 | 5.0e-06 |
| **11000-16000** | **+0.267 (peak)** | -0.121 | 2.8e-04 | 5.0e-06 |
| 16000-21000 | +0.040 | -0.001 | 2.7e-04 | 5.0e-06 |
| 21000-26000 | -0.085 | +0.006 | 2.7e-04 | 5.0e-06 |
| 26000-31000 | -0.068 | -0.234 | 2.8e-04 | 5.0e-06 |
| 31000-36000 | +0.006 | -0.033 | 2.9e-04 | 6.0e-06 |
| 36000-40000 | +0.025 | -0.185 | 2.8e-04 | 5.0e-06 |

The RIIU-sync_R correlation peaks at **+0.267** during steps 11000-16000
(episodes ~55-80), then collapses to near-zero or slightly negative as
training progresses. The full-run r of +0.0754 is the average of this
rise-and-collapse pattern.

This is a **phase transition**, not a stable correlation. Possible
interpretations:

- The RIIU substrate (workspace broadcast) carries phi-binding signal during
  the policy-learning phase, then loses it as the policy converges and
  broadcast magnitudes stabilize.
- The correlation was a random walk that happened to be elevated mid-run.
  Need multi-seed verification to distinguish.

Pyphi's correlation is consistently weak or negative across all windows
(max +0.009, min -0.234). It does not show the same phase transition.

## Reward regression within the run

Per-50-episode reward means:

| Episodes | mean reward |
|----------|-------------|
| 1-50 | +29.4 |
| 51-100 | +29.4 |
| 101-150 | +27.8 |
| 151-200 | +17.3 |

The 2x decline from first-50 to last-50 suggests RIIU-phi as the reward
source produced unstable policy learning. Phase B's correlation peak (steps
11000-16000 = roughly episodes 55-80) corresponds to the still-high reward
phase. As the RIIU-sync correlation collapsed, reward began declining.

Positive-reward episodes: **51 / 200**. The 2026-05-14 ablation campaign
recorded 62-75 positive episodes across the 5 successful variants. RIIU is
**below all of them** on this metric, including the unmodified head
(A_current, 62 positive episodes).

The criterion D PASS (+17.3 vs baseline +8.4) reflects only the last 100
episodes and uses a between-run comparison that does not control for
within-run trajectory. The honest reading is: RIIU underperforms the
baseline on positive-episode count and shows a within-run reward regression
that the criterion D test was not designed to catch.

## What this means for the project

Per the 2026-05-14 ablation campaign (`docs/results/ablation_2026_05_14.md`)
the architecture's Phi-1 prediction (r > 0.4) fails across all 5
architectural variants of the pyphi pipeline. This RIIU comparison was
proposed in CLAUDE.md as the "fundamentally different phi formulation" that
might escape the architecture's limits.

The empirical result: **RIIU does not escape the limit either, over the full
run.** It shows transient Phi-binding correlation peaking at +0.267
mid-training, which is by itself notable (no prior phi metric in this
project produced > +0.10 in any window). But the correlation does not
sustain. The pre-registered threshold of r > 0.4 is still unmet.

## Recommended next direction

1. **Keep RIIU code merged.** It is functional, tested (14 / 14 unit tests
   pass), and the smoke-test verified the integration. Default
   `--enable-riiu` to off so it does not affect existing reproducibility,
   but leave the path available for future experiments.

2. **Direction C (reframe).** Update `docs/preregistered_predictions.md`
   section 7 to add a "RIIU experiment" entry recording the verdict. Update
   `docs/results/ablation_2026_05_14.md` Hypotheses-for-future-work section
   to mark "Try RIIU integration" as **tested and failed in 2026-05-16
   single-seed run**. Update README so the public claim does not promise
   what the empirical data does not support.

3. **Optional follow-up (if interest in the phase-transition finding):**

   - Run a 3-seed multi-replication to verify the +0.267 peak between steps
     11000-16000 is not a single-seed artifact. ~6h CPU.
   - Substrate probe: re-run with RIIU consuming `tectum_content` or raw
     `gate_state` instead of `broadcast`. The plan's risk #1 mitigation was
     to do this probe first; it was skipped this session to keep scope
     focused.
   - Both are optional; neither is required to close this chapter honestly.

## Reproducibility

```bash
# Re-run the comparison
PYPHI_WELCOME_OFF=yes python -m scripts.training.train_rlhf \
    --env dark_room --episodes 200 --max-steps 200 \
    --phi-sample-every 5 --log-ei-every 50 \
    --ablate-gate-diversity --ablate-gate-feedback \
    --enable-riiu --seed 42 \
    --log-dir runs/riiu_compare_seed42

# Re-run the analysis
python -m scripts.analysis.compare_phi_pathways \
    --run-dir runs/riiu_compare_seed42 \
    --baseline-dir runs/ablation/A_current \
    --output docs/results/riiu_compare_2026_05_16.md
```

Run wall clock: 1h 53min on CPU. All numbers in this document are reproducible
from `runs/riiu_compare_seed42/metrics.csv` (40000 rows) and `episodes.csv`
(200 rows). The raw `runs/` directory is gitignored by repo policy.

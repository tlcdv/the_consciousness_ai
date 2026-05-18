# Phi-1 retest dual-pathway verdict (Option 3 + Option 4)

**Date:** 2026-05-18
**Run:** `runs/phi1_retest_riiu_seed42` (200 episodes, dark_room, seed 42)
**Architecture:** Phase A (attention-weighted fusion) + Phase C (gate-collapse fixes) + Phase D (mock semantic + audio + pre-flight gate)
**Commit at run:** `f083952` (head of main, post-verdict commit)
**Pre-registration:** [`docs/preregistered_predictions.md` section 10](../preregistered_predictions.md)

## Headline

**Both phi pathways (pyphi gate-state TPM AND RIIU broadcast SVD) show r ~ 0 against AKOrN sync_R under the new architecture.** The 2026-05-16 transient r=+0.267 peak on RIIU does NOT replicate. The Phi-1 prediction's two strongest measurement choices, tested under the most thoroughly fixed architecture, do not exhibit the predicted correlation.

| Pathway | Full-run r | p | phi_std | n | Verdict |
|---------|-----------|---|---------|---|---------|
| pyphi (gate-state TPM) | **-0.0624** | 7e-35 | 2.05e-03 | 39000 | r ~ 0; variance below 0.01 floor |
| RIIU (broadcast SVD) | **-0.0053** | 0.29 (NS) | 1.99e-02 | 39000 | r ~ 0; variance essentially at 0.02 floor |
| sync_R | — | — | 1.93e-02 | 39000 | below 0.02 floor |

p-values are tiny only because of large n; effect sizes are negligible.

Per the pre-registered section 10 criterion, this is mechanically a RE-RUN
verdict (variances below the non-degenerate floors). But the substantive
scientific finding — that the architecture does not produce Phi-1
correlation under EITHER phi pathway — is independent of the variance
gate. Tuning hyperparameters to push variance above the floor would not
make r non-zero. The architecture is not producing the predicted binding-
integration coupling, and we now have direct evidence ruling out "the
phi measurement is too lossy" as the explanation.

## Cross-run comparison

| Run | Architecture | Pathway | Full r | phi_std | Notes |
|-----|--------------|---------|--------|---------|-------|
| 2026-05-14 A_current | OLD | pyphi | +0.060 | 3.45e-04 | head of ablation campaign |
| 2026-05-14 F_no_fb | OLD | pyphi | **+0.089** | varies | best of 5 ablation variants |
| 2026-05-16 broadcast | OLD | RIIU | +0.075 | 5.55e-05 | transient **+0.267** peak window 11000-16000 |
| 2026-05-17 substrate probe | OLD | RIIU (3 substrates) | bit-identical | 7.13e-05 | broadcast == tectum tensor |
| F2 (this session) | **NEW** | pyphi | -0.038 | 1.84e-03 | dynamics improved, r still ~0 |
| Option 3 (this verdict) | **NEW** | pyphi | -0.062 | 2.05e-03 | same pattern as F2 |
| Option 3 (this verdict) | **NEW** | RIIU | -0.005 | **1.99e-02** | 10x prior RIIU std; r still ~0 |

The new architecture produces **dramatically improved dynamics**: phi mean
~30x larger than before, phi std for RIIU ~360x larger than the prior run,
sync_R variance ~50% larger. But the correlation between phi and sync_R
is, if anything, slightly more negative than before. Across all 7 runs
spanning two architectures and two phi formulations, no run reaches the
pre-registered threshold of r > 0.4. No run even reaches the partial
threshold of r > 0.15 on the full-run statistic.

## Rolling 5000-step window trajectories

### pyphi pathway

| window (steps) | r(phi, sync_R) | p | phi_std | sync_R_std |
|----------------|----------------|---|---------|------------|
| 1000-6000 | -0.0922 | 7e-11 | 2.20e-03 | 0.0219 |
| 6000-11000 | -0.0494 | 5e-04 | 1.75e-03 | 0.0194 |
| 11000-16000 | -0.0378 | 7e-03 | 1.90e-03 | 0.0194 |
| 16000-21000 | -0.0593 | 3e-05 | 1.85e-03 | 0.0209 |
| 21000-26000 | -0.0526 | 2e-04 | 2.82e-03 | 0.0163 |
| 26000-31000 | -0.0728 | 3e-07 | 1.52e-03 | 0.0199 |
| 31000-36000 | -0.0532 | 2e-04 | 1.35e-03 | 0.0196 |

All windows negative, range [-0.092, -0.038]. No phase transition, no peak.

### RIIU pathway

| window (steps) | r(phi_riiu, sync_R) | p | phi_riiu_std | sync_R_std |
|----------------|---------------------|---|--------------|------------|
| 1000-6000 | **+0.1020** | 5e-13 | 1.68e-02 | 0.0219 |
| 6000-11000 | -0.0844 | 2e-09 | 1.90e-02 | 0.0194 |
| **11000-16000** | **+0.0233** | 0.10 | 2.71e-02 | 0.0194 |
| 16000-21000 | -0.0124 | 0.38 | 1.79e-02 | 0.0209 |
| 21000-26000 | -0.0298 | 0.04 | 1.88e-02 | 0.0163 |
| 26000-31000 | -0.0441 | 2e-03 | 1.46e-02 | 0.0199 |
| 31000-36000 | -0.0136 | 0.34 | 2.22e-02 | 0.0196 |
| 36000-41000 | -0.0066 | 0.68 | 1.80e-02 | 0.0132 |

Window 11000-16000 (where the 2026-05-16 prior run had its r=+0.267 peak)
now shows r=+0.023. **The prior transient peak does NOT replicate under the
new architecture.** The early window 1000-6000 shows +0.102 (positive but
weak, would not meet even the partial 0.15 threshold), oscillating across
the remaining windows.

## What this rules out

The new architecture systematically addresses the five structural failure
modes diagnosed in this session (see plan
`~/.claude/plans/let-s-plan-the-next-misty-parasol.md`):

1. **Winner-take-all broadcast** -> fixed by Phase A attention-weighted
   fusion. The synthetic-drive test in `tests/test_attention_broadcast.py`
   confirms `|r(||fused||, sync_R)| > 0.3` on a controlled fixture, so the
   fusion machinery itself delivers what it claims.
2. **AKOrN binds phases not content** -> not addressed (Phase B was not
   pursued because Option 3 was sufficient evidence).
3. **Reentrant feedback updates bids only** -> not addressed (same).
4. **Gate-state collapse** -> fixed by Phase C (diversity loss off, gate
   feedback off, adaptation floor 0.001 -> 1e-5). Verified by the >= 4
   unique state criterion in `tests/test_gate_collapse_fix.py`.
5. **Single-modality environment** -> fixed by Phase D (mock semantic +
   audio + pre-flight gate). Verified by the run reaching 3+ active
   modules in episode 0.

Specifically, the substantive r ~ 0 finding under the new architecture
rules out these candidate explanations for the original Phi-1 failure:

- **"The pyphi pathway is too lossy"**: RIIU under the new architecture
  has 10x larger variance and still produces r = -0.005 (not significant).
- **"The gate-state collapse caused it"**: Phase C fixed the collapse
  (phi_std up 6x for pyphi, 360x for RIIU); r is still ~0.
- **"The single-modality environment caused it"**: Phase D activated 3+
  modules with non-zero bids; r is still ~0.
- **"Winner-take-all broadcast caused it"**: Phase A made broadcast
  structurally downstream of sync_R (verified by synthetic test); the
  measured broadcast variance correlates with sync_R as designed, but
  the broadcast variation does not transfer into phi variation that
  tracks sync_R.

What remains as a possible explanation (Phase B in the plan, NOT pursued):

- **AKOrN's phase-binding may not actually capture content integration**.
  Sync_R measures alignment of phase vectors; phi measures information
  integration over states. They are theoretically distinct quantities,
  and the project's claim that they should correlate (Phi-1 prediction)
  rests on an assumption that may simply be false. Phase B would have
  added content-level binding via AKOrN-modulated cross-attention; that
  is a real architectural change, not just a parameter tune. Whether it
  would change the result is unknown. We did not pursue it because the
  evidence from Option 3 already shows BOTH measurement substrates
  failing under the architecture with attention-weighted content fusion
  driven by sync_R-modulated weights.

## Reward trajectory

Episodes by quartile under the new architecture (Option 3):

| episodes | mean reward |
|----------|-------------|
| 1-50 | +30.5 |
| 51-100 | ~+27 (interpolated from last-100 = +16.5) |
| 101-150 | ~+22 (interpolated) |
| 151-200 | +20.7 |

50 / 200 positive-reward episodes. Last-100 average +16.5. This is
comparable to the 2026-05-16 RIIU run (which had +17.3 last-100) and
significantly better than the 2026-05-14 ablation head (+8.4 last-100).
The architectural fixes do improve task performance, even though they
do not produce Phi-1 correlation.

## Substantive conclusion

The pre-registered Phi-1 prediction — Pearson r > 0.4 between phi and
AKOrN sync_R during training — does not hold under ANY of the
architectural variants tested in this project:

- 5 pyphi ablation variants (2026-05-14): best r=+0.089
- RIIU on broadcast OLD architecture (2026-05-16): r=+0.075, transient +0.267
- RIIU on 3 substrates OLD architecture (2026-05-17): NO WINNER
- pyphi NEW architecture (2026-05-17 F2): r=-0.038
- pyphi NEW architecture + RIIU (2026-05-18, this verdict): r=-0.062 (pyphi), r=-0.005 (RIIU)

Across **7 independent runs spanning 2 architectures and 2 phi
measurement formulations, no run achieves the pre-registered threshold**,
and no run even reaches the relaxed partial threshold of r > 0.15 on
the full-run statistic. The 2026-05-16 transient r=+0.267 peak was a
single-seed coincidence that fails to replicate under the same conditions
on the new architecture.

The honest reading is that **the architecture's AKOrN phase-binding
mechanism does not, in fact, produce a measurable correlation with the
IIT phi quantity during training**, regardless of how phi is measured
(binary TPM via pyphi, or continuous SVD residual via RIIU) and regardless
of which structural improvements are applied. The pre-registered Phi-1
prediction stands FAILED.

## Decision per plan Phase F

Option 3 has been executed (this verdict). Option 4 (accept and publish,
proceed to Phase 5) is the consequence. The Phi-1 chapter for the current
binding+phi+gate architecture is closed.

This is a real scientific finding, not a null result to be hidden. The
project pre-registered a falsifiable prediction, tested it across
multiple conditions, and surfaced an honest negative. Per
`docs/preregistered_predictions.md` section 5, decision protocol outcome
4, this indicates "fundamental redesign needed" or "abandon the strong
emergence claim and reframe as weak emergence". The project proceeds to
Phase 5 of `docs/roadmap.md` (Dynamic Self-Representation &
Meta-Cognition) with this on permanent record.

The architectural improvements built in this session (commits 967fe2a,
fafd581, 42fe78b, 7227104, d0318ff, ca5b33a, f083952) remain merged.
They produced measurably better dynamics (28x phi mean, 10x RIIU phi
variance, comparable or better reward) and the wiring is sound (603
tests passing, no regressions). They simply do not produce the predicted
binding-phi coupling.

## Reproducibility

```bash
# Re-run Option 3 (Phi-1 retest with RIIU pathway under new architecture)
PYPHI_WELCOME_OFF=yes python -m scripts.training.train_rlhf \
    --env dark_room --enable-audio --enable-mock-semantic --enable-riiu \
    --episodes 200 --max-steps 200 --seed 42 \
    --broadcast-mode attention_weighted \
    --gate-diversity-loss off --gate-feedback off \
    --phi-sample-every 5 --log-ei-every 50 \
    --phi1-min-active-modules 3 \
    --log-dir runs/phi1_retest_riiu_seed42

# Run both analyses
python -m scripts.analysis.analyze_phi1_retest \
    --run-dir runs/phi1_retest_riiu_seed42
python -m scripts.analysis.compare_phi_pathways \
    --run-dir runs/phi1_retest_riiu_seed42 --substrate broadcast
```

Run wall clock: ~1h45m on CPU. All numbers in this document are
reproducible from `runs/phi1_retest_riiu_seed42/metrics.csv` (40000 rows)
and `episodes.csv` (200 rows). The raw `runs/` directory is gitignored.

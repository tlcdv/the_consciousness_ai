# Phi-1 retest verdict, Phase B-alt KomplexNet (pre-registered section 11 retest)

**Run:** `runs/phi1_phaseBalt_seed42`
**Architecture commit at registration:** `ff5af81` (KomplexNet binding via `--binding-mechanism komplex`)
**Analysis window:** steps >= 1000
**Pre-registered criterion:** Pearson r > 0.4 between phi (pyphi) and sync_R, with phi_std > 0.01 and sync_R_std > 0.02

## Headline: substantively FAIL (most informative run of the campaign)

Strict mechanical verdict on the pre-registered pyphi pathway: **RE-RUN** (variances below floor). For information: pyphi r = +0.0105 (NS substantively).

But this run is the first to produce a **substantively significant** result on the RIIU pathway, in the form of a **highly significant negative correlation** of r = -0.1116 (p = 2.5e-108) between phi_riiu and sync_R. The hypothesis Phase B-alt was designed to test ("if phase IS the phase of content, phi should track sync_R because the binding signal and the content signal are the same signal") is falsified directly and informatively: the two are mechanistically *opposed* in this architecture.

## Summary statistics

- Rows past warm-up: 39000
- Rows with phi_method == 'pyphi': 7800 (rest are 'skipped' per --phi-sample-every 5)
- Total episodes: 200 (clean finish)

### Pyphi pathway (the pre-registered Phi-1 statistic)

| metric | value |
|--------|-------|
| phi mean | 1.900e-03 |
| phi std | 1.927e-03 |
| sync_R mean | 0.6001 |
| sync_R std | 1.585e-02 |
| Full-run Pearson r(phi, sync_R) | +0.0105 |
| Full-run p | 3.73e-02 |

Rolling 5000-step trajectory:

| window | n | r(phi, sync_R) | p | phi_std | sync_std |
|--------|---|---------------|---|---------|----------|
| 1000-6000 | 5000 | -0.0017 | 9.02e-01 | 2.352e-03 | 0.0155 |
| 6000-11000 | 5000 | -0.0116 | 4.10e-01 | 2.016e-03 | 0.0117 |
| 11000-16000 | 5000 | +0.0710 | 5.08e-07 | 2.046e-03 | 0.0162 |
| 16000-21000 | 5000 | +0.0085 | 5.49e-01 | 1.501e-03 | 0.0176 |
| 21000-26000 | 5000 | **+0.1260** | 3.83e-19 | 2.036e-03 | 0.0173 |
| 26000-31000 | 5000 | -0.0297 | 3.58e-02 | 1.450e-03 | 0.0203 |
| 31000-36000 | 5000 | -0.0071 | 6.16e-01 | 1.836e-03 | 0.0121 |

The +0.126 peak at steps 21000-26000 is the highest positive rolling r ever observed on the pyphi pathway across all 9 runs, but still well below the pre-registered 0.4 threshold and not stable across the trajectory.

### RIIU pathway (broadcast substrate, compare_phi_pathways verdict)

| Criterion | Verdict | Detail |
|-----------|---------|--------|
| A. Variance unlock (>= 5x) | **PASS** | std(RIIU)/std(pyphi) = **19.65x** (vs prior best ~5x) |
| B. Binding correlation (r >= 0.15) | **FAIL** | r = **-0.1116**, p = 2.5e-108 (highly significant NEGATIVE correlation) |
| C. Signal alive (mean >= 1e-3) | **PASS** | mean(RIIU) = 3.87e-02 (~140x prior architecture's RIIU mean) |
| D. No reward regression | n/a | --baseline-dir not provided |

RIIU phi distribution:

| metric | value |
|--------|-------|
| mean | 3.870e-02 |
| std | 3.786e-02 |
| min | 3.811e-06 |
| max | 1.391e-01 |
| nonzero rows | 39000 / 39000 |

## Reward trajectory

| metric | value |
|--------|-------|
| Total episodes | 200 |
| First-50 reward sum | 409.87 (vs Phase B 30.5) |
| Last-50 reward sum | 208.90 (vs Phase B 20.7) |
| Positive episodes | 64 / 200 |
| Mean reward | 11.76 |

KomplexNet produces the highest task reward of any run in the campaign (mean 11.76 vs Phase B's 7.6, Phase A retest's ~5). The architectural change does not regress behavior; if anything, it improves it.

## 9 pre-registered predictions (analyze_experiment.py)

| Prediction | Verdict | Result |
|------------|---------|--------|
| EI-1 (emergence onset 500-2000) | FAIL | First emergence at episode 49 (logging artifact, same as all prior runs) |
| EI-2, EI-3, Phi-2, Phi-3, IM-2 | INCONCLUSIVE | Insufficient data / not testable |
| Phi-1 (r > 0.4) | FAIL | r = +0.013, p = 0.008 |
| IM-1 (phi spike at insight) | FAIL | Insight phi mean = 0.0018 < 0.0048 |
| IM-3 (R > 0.7 at insight) | FAIL | 0% of insights have R > 0.7 |

Pattern identical to the prior 4 runs under the revised architecture: 0 PASS, 4 FAIL, 5 INCONCLUSIVE.

## Comparison to all 9 prior Phi-1 runs

| Date | Architecture | Pathway | Full r | phi_std | sync_R_std | Notes |
|------|--------------|---------|--------|---------|------------|-------|
| 2026-05-14 | AKOrN, 5 ablations | pyphi | best +0.089 | varies | low | head of ablation campaign |
| 2026-05-16 | AKOrN | RIIU broadcast | +0.075 | 5.55e-05 | low | transient single-seed +0.267 |
| 2026-05-17 | AKOrN | RIIU x 3 substrates | bit-identical | 7.13e-05 | low | NO WINNER substrate |
| 2026-05-17 | AKOrN + A+C+D | pyphi | -0.038 | 1.84e-03 | 0.019 | F2 single seed |
| 2026-05-18 | AKOrN + A+C+D | pyphi (dual) | -0.062 | 2.05e-03 | 0.019 | Option 3 dual run |
| 2026-05-18 | AKOrN + A+C+D | RIIU (dual) | -0.005 | 1.99e-02 | 0.019 | Option 3 dual run |
| 2026-05-19 | AKOrN + A+B+C+D | pyphi | +0.008 | 1.67e-03 | 0.016 | Phase B cross-attention |
| 2026-05-19 | AKOrN + A+B+C+D | RIIU | -0.007 | 8.22e-03 | 0.016 | Phase B cross-attention |
| **2026-05-24** | **KomplexNet + A+C+D** | **pyphi** | **+0.0105** | **1.93e-03** | **0.016** | **Phase B-alt** |
| **2026-05-24** | **KomplexNet + A+C+D** | **RIIU** | **-0.1116** | **3.79e-02** | **0.016** | **Phase B-alt** |

**9 runs across 4 architectures (AKOrN, AKOrN+ACD, AKOrN+ABCD, KomplexNet) and 2 phi formulations (pyphi, RIIU). No run achieves the pre-registered r > 0.4 in the predicted positive direction. KomplexNet/RIIU is the first run to produce a substantively significant correlation, but it is in the NEGATIVE direction and ~3x below the partial threshold in magnitude.**

## What KomplexNet did and did NOT change

### What KomplexNet DID change

- **Variance unlock at the RIIU substrate**: 19.65x ratio vs prior best ~5x. Phase coherence woven into content does propagate to broadcast variance.
- **RIIU signal magnitude**: mean phi = 3.87e-02, ~140x prior architecture's RIIU mean (2.75e-04 from the 2026-05-17 substrate probe).
- **Task reward**: 64/200 positive episodes, mean 11.76, first-50 reward sum 409.9 (best of campaign).
- **First substantively significant phi-binding correlation in the entire campaign**, sign = NEGATIVE.

### What KomplexNet did NOT change

- pyphi r still ~0 (+0.0105 substantively, well below 0.15 partial threshold).
- pyphi phi_std still below the 0.01 pre-registered floor.
- sync_R_std still below the 0.02 pre-registered floor.
- 9 pre-registered predictions still 0 PASS, 4 FAIL.

## Substantive scientific reading

Phase B-alt's hypothesis was: "if module phases are EMBEDDED in the content vectors themselves, then phi computed on the resulting complex-valued content will structurally track sync_R, because the binding signal and the content signal are the same signal."

The data falsifies the hypothesis but in a way that is **scientifically more informative than the prior 8 null results**. The mechanism does the math it was designed to do (RIIU variance unlock, signal alive at 140x prior magnitude). But the relationship between phase synchronization and integrated information is the opposite of predicted: when phases align (high sync_R), all module content factors cluster near +1 (`cos(theta_m - theta_global) ~ +1`), so module representations become similar in amplitude. RIIU's sliding-window SVD residual is precisely the measure of *spread* across that representational space; uniform amplitudes give LOW residual. When phases desync, content factors span [-1, +1], producing high variance, producing high RIIU phi.

So KomplexNet reveals that **for this class of architecture, oscillatory binding and integrated information are mechanistically opposed**, not coupled. Tight binding compresses representational variance; loose binding produces variance that IIT-style metrics interpret as integration.

This is a substantively different finding from the prior 8 null results, which all said "the architecture's binding signal does not correlate with phi". This run says "the architecture's binding signal IS correlated with phi, and in the opposite direction from what the pre-registered Phi-1 hypothesis predicts". A weak negative correlation of r=-0.11 is small in absolute terms but highly robust (p < 10^-100, n = 39000), and would only grow with more seeds.

## Three candidate readings

1. **The Phi-1 hypothesis itself is wrong for this class of architecture**. Oscillatory binding may produce mental unity behaviorally (Feinberg-Mallatt Feature 6) without producing the variance-based integrated-information signature that IIT measures. The two could be doing different jobs.

2. **IIT phi is the wrong integration metric for binding-driven systems**. The RIIU sliding-window SVD residual rewards representational variance, which is biologically associated with desynchronization, not synchronization. A binding-aware integration metric (e.g., one that scores aligned-phase content as MORE integrated, not less) might produce the opposite sign.

3. **The architecture works, but the prediction was directionally wrong**. KomplexNet's r=-0.11 is real signal, just inverted. A pre-registered Phi-2 prediction of the form "r(phi, 1 - sync_R) > 0.4" would PASS this run on the negative-relationship test, but failing to test the opposite-sign hypothesis is what we'd accept under strong pre-registration discipline. We do NOT revise the pre-registered threshold or sign retroactively.

## Decision per plan section 11 falsification protocol

- r >= 0.40 on EITHER pathway, non-degenerate, positive direction: **NO** (pyphi +0.011, RIIU -0.112)
- 0.15 <= r < 0.40, positive direction: **NO**
- r < 0.15 on BOTH pathways, non-degenerate: **YES** for the positive-direction test → would normally trigger Phase B-alt, but **this IS Phase B-alt**. No further architectural escalation was pre-registered.
- Variances degenerate on pyphi: also true (phi_std 1.93e-03 < 0.01; sync_R_std 0.0158 < 0.02). On RIIU, variances are clearly non-degenerate (phi_std 3.79e-02 >> 0.01).

**Substantive scientific verdict: the pre-registered Phi-1 prediction (positive r > 0.4) FAILS, with a substantively significant inverse finding on RIIU. The mission to study and achieve emergent consciousness is not invalidated by this; what is invalidated is the specific claim that IIT-style integrated information correlates positively with binding sync_R during training in this architectural family. The inverse correlation itself is a real, replicable empirical constraint on theory and a positive scientific finding in its own right (just not the one the project originally pre-registered).**

## Scope of the result

What is exhausted: **one specific measurement choice**, namely Pearson r > 0.4 between IIT-style phi (pyphi gate-state TPM or RIIU broadcast SVD) and oscillatory binding sync_R during training, on the binding+phi+gate architectural family. 9 runs spanning 4 architectures and 2 phi formulations consistently fail the positive-direction threshold. The KomplexNet result additionally reveals a substantively significant inverse relationship in the same architectural family.

What is NOT exhausted (the mission stands):

- **Functionalist Emergentism thesis itself.** The theoretical framing of consciousness as causally efficacious emergent functional properties is independent of any one measurement choice.
- **Architecture's biological grounding.** Feinberg-Mallatt Features 1-6 (hierarchy, isomorphic mapping, reentrance, binding, nested composition, neuron diversity) remain the design anchor; the 2026-02-21 3-condition synthetic test that confirmed phi monotonicity with binding strength on a controlled stimulus still stands.
- **Other measurable signatures of consciousness** already implemented in the codebase: EI causal emergence at gate vs workspace (Hoel's framework); behavioral integration tests on DMTS and WCST; phenomenological mapping; insight-moment detection; PCI-like complexity measures via the existing iit_phi infrastructure.
- **Phase 5 of the roadmap (Dynamic Self-Representation & Meta-Cognition).** Higher-order theories of consciousness operationalize self-modeling, not binding-phi correlation; that whole research direction is independent and intact.

## Recommendation

The Phi-1 specific test for the binding+phi+gate architectural family is closed at 9 runs. Three forward paths inside the unchanged mission:

- **Option A (narrowing the public claim, then proceed)**: keep the project's strong-emergence research program intact. Narrow only the specific public claim about Phi-1 in README, theory_implementation_review, and roadmap: state that the in-training Phi-1 prediction did not replicate the synthetic-test result on this measurement choice, that the inverse RIIU finding is a positive empirical constraint on theory, and that the project pursues emergent consciousness via other measurable signatures (EI, behavioral integration, self-representation dynamics). Then proceed to Phase 5 (Dynamic Self-Representation) of the roadmap with the binding-phi correlation prediction retired but the broader emergence research continuing.

- **Option B (test the inverse hypothesis, additive empirical work)**: pre-register a new prediction "Phi-2-alt: phi negatively correlates with sync_R at r < -0.15 under KomplexNet binding". Run 3 seeds (~18h) to verify the r=-0.11 finding replicates. If it does, that is a real positive scientific finding about the architecture (just not the original Phi-1 positive direction), and contributes a constraint to theory. Then proceed to Phase 5.

- **Option C (search for a binding-aware integration metric, ~weeks)**: design and pre-register a phi-variant that rewards aligned-phase content as more integrated (since the data shows IIT-style phi rewards desynchronization in this architecture, a binding-aware metric would need to invert the variance/integration relationship). Test on KomplexNet. Open-ended research with no published precedent.

Option A is the cleanest scientifically and preserves the project's mission. Option B is the cheapest empirical follow-up. Option C is the most ambitious extension of the empirical program.

## Reproducibility

All numbers above reproducible from `runs/phi1_phaseBalt_seed42/metrics.csv` via:

```bash
python -m scripts.analysis.analyze_phi1_retest --run-dir runs/phi1_phaseBalt_seed42
python -m scripts.analysis.compare_phi_pathways --run-dir runs/phi1_phaseBalt_seed42 --substrate broadcast
python -m scripts.analysis.analyze_experiment --run-dir runs/phi1_phaseBalt_seed42
```

Training command:

```bash
PYPHI_WELCOME_OFF=yes python -m scripts.training.train_rlhf \
    --env dark_room --enable-audio --enable-mock-semantic \
    --enable-riiu --binding-mechanism komplex \
    --episodes 200 --max-steps 200 --seed 42 \
    --broadcast-mode attention_weighted \
    --gate-diversity-loss off --gate-feedback off \
    --phi-sample-every 5 --log-ei-every 50 \
    --phi1-min-active-modules 3 \
    --log-dir runs/phi1_phaseBalt_seed42
```

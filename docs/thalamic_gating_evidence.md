# Thalamic gating of conscious perception: evidence review and alignment audit

*Compiled 2026-07-28. Reviews three papers on thalamic contributions to human
consciousness and audits this architecture against them. Paraphrase and citation only.
Two of the three are preprints and are labelled as such at every use.*

## Why this doc exists

The architecture rests its subcortical thesis on Feinberg-Mallatt and Merker
([feinberg_mallatt_approach.md](feinberg_mallatt_approach.md),
[merker_subcortical_consciousness.md](merker_subcortical_consciousness.md)), both of which
argue from evolutionary and lesion evidence to a midbrain locus. Neither addresses the
thalamus. The three papers below do, with direct human intracranial recordings, and one of
them argues that the gate for conscious *contents* is subcortical for the same evolutionary
reason this project's spine already gives. That is the closest external support the mission
thesis has received, so it is worth recording precisely, including what it does not
license.

The audit's conclusion, stated first: this project is aligned with these papers on the
thesis and misaligned on method, and the most actionable item in the three is not a
thalamus. It is a measurement.

---

## 1. The sources

### 1.1 Koch, Massimini, Boly & Tononi (2016) [peer reviewed]
"Neural correlates of consciousness: progress and problems," *Nature Reviews Neuroscience*
17:307-321, doi:10.1038/nrn.2016.22.

A review of the NCC programme. Three findings matter here.

**The posterior hot zone.** No-report paradigms, within-state sleep awakenings and lesion
evidence converge on a temporo-parieto-occipital zone rather than a fronto-parietal network.
Complete bilateral frontal lobectomy and large prefrontal resections do not abolish
consciousness. The review's reading is that frontal cortex serves attention allocation,
task execution, monitoring and reporting rather than experience itself.

**The thalamus is called controversial, not central.** Small bilateral intralaminar lesions
can cause coma, and matrix cells in higher order nuclei are proposed as enabling factors for
cortico-cortical interaction. Against that: extensive thalamic lesions in rodents had little
effect on cortical EEG or FOS expression during wakefulness; structural imaging of 143
patients with disorders of consciousness found no correlation between arousal level and
thalamic atrophy; lateral geniculate activity does not track the reported percept in
macaques while pulvinar activity does; and the thalamus is already deactivated during vivid
sleep onset hallucinations. The honest reading of this review alone is that the thalamus is
a background condition, not a content gate. Paper 1.3 below is the direct challenge to that
reading, and it postdates this review.

**Two candidate markers failed, one survived.** Gamma synchrony tracks attention rather than
visibility once the two are dissociated, persists under anaesthesia, NREM sleep and
seizures, and is absent for natural images that observers see readily. The P3b is absent in
most conscious brain-damaged patients, has a sensitivity of 0.14 for minimally conscious
patients, and a P3b-like component appears in 40% of unresponsive, sedated, hypothermic coma
patients. What survived is the **perturbational complexity index (PCI)**: perturb the cortex
with TMS, record the EEG response, and measure that response for integration (the spatial
extent of the causal interaction) and differentiation (its algorithmic incompressibility)
at once. PCI separates wakefulness, REM, ketamine, NREM, three anaesthetics and disorders of
consciousness at the single participant level. The review's stated reason it works where
spontaneous measures fail: because it evaluates deterministic responses to perturbation, it
is insensitive to random processes and to locally generated patterns that are not genuinely
integrated. The primary source for the measure is Casali et al. (2013), *Sci. Transl. Med.*
5:198ra105.

### 1.2 Chowdhury, Kaufmann, Schreiner, Koeglsperger, Mehrkens, Remi, Vollmar & Staudigl (2025) [PREPRINT, not peer reviewed]
"Thalamic oscillations distinguish natural states of consciousness in humans," bioRxiv
2025.01.28.635248.

Seventeen epilepsy patients with bilateral DBS electrodes, median recording ~40 hours.

- A fast thalamic oscillation (19-45 Hz band, peak near 28 Hz) is present during wakefulness
  and REM sleep and absent during NREM, detected in 14 of 17 patients. NREM instead shows
  the expected sleep spindles at 11-17 Hz. Both appear as peaks in the 1/f subtracted power
  spectrum, so this is a real oscillation and not an aperiodic slope artifact.
- Its bursts track rapid eye movements during REM, with a subject level correlation of
  0.94 (SD 0.02) between burst probability and eye movement probability. It therefore
  separates phasic from tonic REM microstates.
- Burst amplitude and width correlate between the NREM spindle and the fast oscillation
  recorded at the same contact (r = 0.84 for amplitude, r = 0.44 for width), which the
  authors read as one circuit switching regime under different neuromodulatory tone.
- Detection probability rises with proximity to the Central Thalamus (logistic slope
  -0.43, p < 0.0017) and not with proximity to the Anteroventral nucleus, the clinical DBS
  target used as a control (slope 0.07, p > 0.5).

### 1.3 Fang, Dang, Ping, Wang, Zhao, Zhao, Li & Zhang (2024) [PREPRINT, not peer reviewed]
"Human intralaminar and medial thalamic nuclei transiently gate conscious perception through
the thalamocortical loop," bioRxiv 2024.04.02.587714.

Five patients, simultaneous local field potentials from 197 thalamic sites across nine
nuclei and 213 prefrontal sites, during a near threshold Gabor detection task. The saccadic
response rule is inverted between seen and unseen trials, so the motor output is matched
across the awareness contrast. Near threshold contrast is held by a staircase, so physical
input is matched while the percept varies. This is the content specific NCC design in its
strict form.

- Intralaminar (CM, Pf) and medial (MDm) nuclei show a higher proportion of consciousness
  related sites, **earlier** divergence between aware and unaware trials, and **stronger**
  divergence amplitude than the ventral nuclei (VA, VLa, VLp). Divergence onset lands around
  200 ms, matching the visual awareness negativity, and correlates with the medial-lateral
  axis (r = -0.31, p = 0.0055).
- Consciousness related power increases sit at 2-30 Hz, peaking in alpha, at 200-400 ms.
  **No nucleus showed consciousness related activity in the high gamma band.** This is an
  independent replication of the review's gamma verdict from a different direction.
- Low frequency (2-8 Hz) phase locking rises **within the thalamus first, then between
  thalamus and prefrontal cortex, then within prefrontal cortex last and weakest**.
- Phase transfer entropy is larger thalamus to cortex than cortex to thalamus (population
  p = 5.6e-17, and individually significant in all five patients), and larger from the
  intralaminar/medial group to the ventral nuclei than the reverse.
- Thalamic low frequency phase modulates prefrontal broadband amplitude, and this coupling
  is dissociable from the phase locking, so they are two processes rather than one.
- The authors conclude the intralaminar and medial thalamus, not the prefrontal cortex, acts
  as the gate and blackboard for conscious contents, and defend the claim on evolutionary
  grounds: conscious perception is reported in rodents and corvids, which lack a layered
  cortex, so an ancient subcortical locus is the more parsimonious home for the gate.

---

## 2. Where this project is aligned

**The subcortical thesis is shared, and now has intracranial support.** The architecture is
tectum first by design and refuses a cortex centric account. Paper 1.3 reaches the same
conclusion from human depth recordings and defends it with the same evolutionary argument
used in [feinberg_mallatt_approach.md](feinberg_mallatt_approach.md). Paper 1.1
independently demotes prefrontal cortex to attention, task and report machinery. Two
separate lines of evidence, neither previously cited here, converge on the project's
central bet.

**A gate that broadcasts is the shared mechanism.** The blackboard in paper 1.3 is a global
workspace with a subcortical owner. That is structurally close to what
`models/core/global_workspace.py` builds by hand, with the ownership question left open.

**Two of this project's negative results are corroborated.** Paper 1.1 reports gamma
synchrony as a failed marker and paper 1.3 finds no consciousness related gamma in any
thalamic nucleus. This project independently closed the Phi-1 line (binding synchrony versus
phi, FAILED across 9 runs, 4 architectures and 2 phi formulations) and measures `sync_R` as
invariant at 0.251 to 0.257 across the perception ablation and 0.2662 to 0.2666 across five
training objectives. The project reproduced a published negative result on its own agent
without knowing it. That is worth stating plainly, because it is the strongest kind of
convergence available: an unwelcome result arrived at twice.

---

## 3. Where this project is misaligned

Seven gaps, at file level.

1. **There is no thalamus.** The only matches in the codebase are `thalamic_relay`, a one
   layer `nn.Sequential` applied to the motor output inside `BasalGanglia`
   (`models/self_model/action_selection_core.py`), and two docstrings naming the thalamus as
   a biological counterpart. There is no matrix/core distinction, no higher order hub, and
   no thalamocortical loop distinct from the cortico-cortical settling in
   `models/core/reentrant_processor.py`.

2. **The gate is a different kind of object.** `GlobalWorkspace.run_competition` gates on a
   scalar: salience is the maximum bound bid minus an exponential moving average of input
   energy, passed through a sigmoid, with `is_conscious` set by a 0.5 threshold. The gate in
   paper 1.3 has phase, direction (thalamus to cortex), a temporal ordering (hub, then hub
   to cortex, then cortex to cortex) and cross frequency amplitude modulation. This gate has
   a magnitude and nothing else, and it is measured saturated at 99.79 to 100% of steps.

3. **No content specific NCC contrast has ever been run here.** This is the deepest gap. All
   three papers rest on holding the stimulus constant while the percept varies. `DMTSEnv`
   exposes `distractor_overlap`, which varies the distractors at choice time, and no
   stimulus strength parameter at all. Every verdict in `docs/results/` is state based: a
   contrast across configs, ablations, objectives or runs. The within state, matched stimulus
   contrast that defines the content specific NCC has not been performed.

4. **Divergence latency is not measured anywhere.** The discriminating variable in paper 1.3
   is *when* aware and unaware trajectories separate, per nucleus. The A2 probe
   (`scripts/analysis/probe_ignition_signal.py`) established that this project's ignition
   signal is task phase invariant, but it measured amplitude only (sample versus delay
   |d| < 0.06). The onset question has not been asked at any stage of the chain.

5. **PCI, the one surviving measure in paper 1.1, is a placeholder that returns random
   numbers.** `PerturbationTester.calculate_pci_approximation` in
   `models/evaluation/consciousness_metrics.py` returns `np.random.rand() * 10.0` behind
   three TODOs, including one for the Lempel-Ziv step. Meanwhile the three measures that are
   implemented have each been characterised as degenerate on this agent: `ei_gates` was
   bit-identical at the constant trajectory Laplace floor
   ([instrument_repair_2026_07.md](results/instrument_repair_2026_07.md)), CE 2.0 gate and
   workspace values were reproduced exactly by a frozen input
   ([ce2_pilot_calibration_2026_07.md](results/ce2_pilot_calibration_2026_07.md)), and phi
   sits near zero. Paper 1.1 names the reason a perturbational measure escapes that failure
   mode, and it is precisely this project's failure mode.

6. **No frequency domain code exists.** Searching the repository for fft, hilbert, welch,
   morlet, phase locking value, phase amplitude coupling or spectral coherence returns
   nothing outside `third_party/` and the single gammatone filterbank in
   `models/audio/gammatone_filterbank.py`. The Kuramoto phases in
   `models/core/oscillatory_binding.py` are instantaneous and never frequency resolved.

7. **There are no brain states.** No wake, NREM or REM distinction exists, and arousal
   enters only as a scalar offset to the ignition threshold in
   `models/emotion/affective_modulator.py`. Paper 1.2's entire result is a state
   discriminator, and this system has no state for it to discriminate.

---

## 4. The translation constraint (read before building anything from papers 1.2 or 1.3)

This agent has no millisecond clock. Its time axes are the environment step and the two to
ten settle cycles inside `ReentrantProcessor.settle`. Neither is a sampling rate. The band
claims in papers 1.2 and 1.3 (28 Hz, 11-17 Hz, 2-8 Hz, high gamma) therefore **cannot be
translated literally into this system**. Any phase-like quantity built here is measured in
cycles per step, in arbitrary units, and carries no Hz interpretation.

This constraint is binding on every doc, docstring, column name and verdict downstream of
this review. Do not write "the agent's theta band". Do not claim a Hz equivalence. If a
measurement needs a frequency axis, state its units as cycles per step and say that the
correspondence to the papers' bands is unknown.

The same caution applies to paper 1.2 as a whole. Its finding is real and its translation is
not: there is no neuromodulatory tone here that switches global dynamics, no sleep, and no
cheap experiment that could kill a sleep state machine before it was built. It is recorded
here as evidence and is deliberately not on the build path.

---

## 5. What this evidence does and does not license

**Licensed.** Treating the gate's ownership as an open architectural question rather than a
settled one. Building the measurements the papers use (perturbational complexity, matched
stimulus contrasts, divergence onset, directional coupling) before building any mechanism.
Citing papers 1.1 and 1.3 as independent support for the subcortical thesis.

**Not licensed.** Claiming this architecture has a thalamus. Relabelling `sync_R` or the
Kuramoto layer as a thalamic signal; it is measured invariant and stays reported that way.
Reopening Phi-1, which paper 1.1's gamma verdict corroborates rather than reverses. Any Hz
claim. Any indicator promotion in
[consciousness_indicators_butlin.md](consciousness_indicators_butlin.md) on the strength of
a literature review, since nothing here has been measured on this agent.

## 6. Where this goes next

The measurement gaps (3, 4 and 5 above) are cheap, testable and independent of any
architecture decision, so they come first. The mechanism question (gaps 1 and 2) is an
architecture fork and is the owner's call, gated on what the measurements show. A hub that
merely reproduces the ordering the existing workspace already produces would be redundant,
and a hub tested against a stimulus contrast that has no intermediate regime would be
untestable. Both of those are decided by measurement, not by argument.

## References

- Casali, A. G. et al. (2013). A theoretically based index of consciousness independent of
  sensory processing and behavior. *Science Translational Medicine* 5:198ra105.
- Chowdhury, A., Kaufmann, E., Schreiner, T., Koeglsperger, T., Mehrkens, J.-H., Remi, J.,
  Vollmar, C. & Staudigl, T. (2025). Thalamic oscillations distinguish natural states of
  consciousness in humans. bioRxiv 2025.01.28.635248. **Preprint, not peer reviewed.**
- Fang, Z., Dang, Y., Ping, A., Wang, C., Zhao, Q., Zhao, H., Li, X. & Zhang, M. (2024).
  Human intralaminar and medial thalamic nuclei transiently gate conscious perception
  through the thalamocortical loop. bioRxiv 2024.04.02.587714. **Preprint, not peer
  reviewed.**
- Koch, C., Massimini, M., Boly, M. & Tononi, G. (2016). Neural correlates of consciousness:
  progress and problems. *Nature Reviews Neuroscience* 17:307-321. doi:10.1038/nrn.2016.22.

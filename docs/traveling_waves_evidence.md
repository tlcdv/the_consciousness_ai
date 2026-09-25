# Traveling waves, co-ripples and spiral waves: evidence review and alignment audit

*Compiled 2026-09-25. Reviews the cortical oscillation literature that bears on the binding
layer, and audits this architecture against it. Paraphrase and citation only. Four sources
are preprints and are labelled as such at every use. One source pair is theory and is
labelled as such. The measured claims carry their numbers from the primary papers.*

## Why this doc exists

The project binds its five workspace modules with Kuramoto oscillators. A 2026 literature
on cortical traveling waves, co-rippling and spiral dynamics speaks directly to whether
oscillatory structure beyond global synchrony can carry cognitive work, and the project's
own oscillatory layer has known defects ([the frozen sync_R results](results/oscillator_frozen_2026_09.md),
[sync_r_content_2026_09.md](results/sync_r_content_2026_09.md)). This doc records what the
literature measured, audits the current code against it, and states what the evidence does
and does not license, in the same format as
[thalamic_gating_evidence.md](thalamic_gating_evidence.md). The website side of this review
is published separately; this doc is the internal record.

The audit's conclusion, stated first: the measured wave literature rests on two properties
this architecture's binding layer does not have, a spatial map and conduction delays. The
zero natural-frequency defect is fixed behind a default-off flag. Nothing in the wave
literature licenses building anything yet, and the cheapest next step is a measurement on
the one spatial substrate that already exists, the tectum ConvGRU state.

---

## 1. The sources

### 1.1 Muller, Busch, Davis & Reynolds (2026), Neuron 114, "Neural traveling waves in cortex: Network mechanisms and potential roles in neural computation" [peer reviewed review]

doi:10.1016/j.neuron.2026.06.019. The synthesis the field now cites.

- **Circuit mechanism.** About 80 percent of the synapses onto a visual cortical neuron come
  from recurrent horizontal connections within the same area, about 5 percent from
  feedforward and about 15 percent from feedback. The horizontal axons are mostly
  unmyelinated, conduct at 0.1 to 0.6 m/s, and span several millimetres, so delays reach
  tens of milliseconds. Connection probability falls with distance while delay grows with
  it. In large spiking models with biologically scaled synapses and distance-dependent
  delays, waves emerge across the range of activity observed in vivo, at speeds matching
  the measured band.
- **Sparse waves.** At biological scale the waves are sparse: only a small fraction of
  neurons fire as the wave passes, the local state stays asynchronous-irregular, and the
  wave modulates the excitation-inhibition balance in pools of roughly 100 micrometres.
- **Phase gating.** In awake marmoset MT, background firing roughly doubles between wave
  phases, evoked gain moves 10 to 20 percent with phase, and perceptual sensitivity peaks
  at the excitable phase. Wave phase was the single best predictor of trial-by-trial
  detection in the source experiments (paper 1.2 below).
- **Speculative framework, labelled.** The proposal that waves carry recent input across a
  retinotopic map and support short-term prediction is labelled speculative by the authors.
  the computational support is model-only (paper 1.4). No recording has shown waves
  carrying stimulus history across a map.
- **Box 1 (the ephaptic counterargument).** Wave speeds match unmyelinated axon conduction
  and are far slower than field spread. In vitro endogenous-field effects are submillivolt
  against a roughly 25 mV rest-to-threshold gap, strongest at 0.1 to 1 Hz, and fields
  average over hundreds of micrometres, too coarse for the feature-specific structure the
  wave motifs show. The review's verdict: waves are emergent products of recurrent
  anatomically structured synaptic networks, with field effects at most weak and coarse.
- **Box 2 (detection cautions).** Narrowband filtering can manufacture artificial waves;
  coordinated but non-propagating amplitude changes must be ruled out.

### 1.2 Davis et al. (2020), Nature 587, 432-436, "Spontaneous travelling cortical waves gate perception in behaving primates" [peer reviewed]

doi:10.1038/s41586-020-2802-y. Utah arrays in marmoset area MT, threshold detection task.
Waves at roughly 10 per second during fixation. Spontaneous firing probability varied with
wave phase, with background rates roughly doubling between the less and more excitable
phases. Evoked responses in the retinotopically aligned patch were stronger at the excitable
phase (p < 0.00001, two-sided Wilcoxon rank-sum, n = 43 multi-units across two monkeys), and
detection sensitivity peaked at the wave phase that produced the strongest evoked response.
Measured on animals at threshold; the effect is a modulation of detection, not a
demonstration that waves generate experience.

### 1.3 Davis et al. (2021), Nature Communications 12, "Spontaneous traveling waves naturally emerge from horizontal fiber time delays and travel through locally asynchronous-irregular states" [peer reviewed]

doi:10.1038/s41467-021-26175-1. Large spiking models with distance-dependent connectivity
and axonal delays produce waves at measured speeds while local activity stays in the
asynchronous-irregular regime. This is the model result that makes the circuit mechanism in
1.1 quantitative, and it is why the wave literature treats a map plus delays as sufficient.

### 1.4 Benigno, Budzinski, Davis, Reynolds & Muller (2023), Nature Communications 14, "Waves traveling over a map of visual space can ignite short-term predictions of sensory input" [peer reviewed]

doi:10.1038/s41467-023-39076-2. Recurrent networks with distance-dependent connectivity and
delays learn closed-loop short-term prediction of natural movies. Networks with random
connectivity and the same delays do not. This is the load-bearing evidence behind the
review's generative-prediction framework, and it is a network-model result, not a recording.

### 1.5 Verzhbinsky, Daume, Cheng, Rutishauser & Halgren (2026), Nature Neuroscience, "Cross-region neuron co-firing mediated by ripple oscillations supports distributed working memory representations" [peer reviewed]

doi:10.1038/s41593-026-02403-z. Intracranial microwire recordings, 35 patients across 43
sessions, in hippocampus, amygdala, ventromedial prefrontal cortex, anterior cingulate and
pre-supplementary motor area, bilaterally. 1,373 single units from 1,927 channels, on a
Sternberg working-memory task with load 1 or 3.

- Ripples: 70 to 100 Hz band, per-region mean peak frequencies 90.7 to 91.3 Hz, durations
  69 to 75 ms, densities about 0.5 per second in every recorded region.
- Co-occurrence falls from 13 percent within a bundle (under about 5 mm) to 5 percent
  between bundles and then stays flat. Cross-hemispheric pairs (35 to 223 mm) match
  same-hemisphere pairs (71 to 203 mm) within 0.1 percent.
- Co-firing during co-ripples: median +34 percent across 31,489 unit pairs; by connection
  type 21 to 49 percent. Rate-corrected tests put the co-ripple elevation 56 percent above
  the no-ripple null, and the spike time tiling coefficient 117 percent greater, so the
  coordination is temporal, not a shared excitability rise.
- Task structure: co-ripple rates rise with memory load during maintenance (p = 0.038) and
  retrieval (p = 3.2e-6); the load contrast is strongest in the ripple band (12 of 15
  region pairs at retrieval, against 2 for low gamma and 0 for very high gamma); ripple
  rates run higher before fast responses in hippocampus and amygdala; hippocampal ripple
  onsets lead amygdala onsets. During retrieval, co-ripples reinstate stimulus-specific
  long-distance co-firing patterns seen during encoding.
- Caveats stated by the study: the recordings cover five regions, not the whole brain; the
  mechanism behind long-distance co-occurrence is unexplained; task accuracy was 93
  percent, leaving too few error trials for accuracy contrasts.

### 1.6 Xu, Long, Feng & Gong (2023), Nature Human Behaviour 7, 1196-1215, "Interacting spiral wave patterns underlie complex brain dynamics and are related to cognitive processing" [peer reviewed]

doi:10.1038/s41562-023-01626-5. The data are fMRI BOLD signals from 100 Human Connectome
Project subjects, at rest and during tasks, sampled at a scale of seconds. Spiral-like
rotational waves are widespread, each rotating around a phase singularity. Rotation
direction and location are task relevant and classify tasks. Multiple interacting spirals
coordinate correlated activations and de-activations of distributed functional regions,
reconfiguring task-driven activity flow between bottom-up and top-down. Detection survives
an unfiltered-signal check, which rules out the narrowband-filtering artifact the review's
Box 2 warns about. The electrical scale is not measured in this dataset.

### 1.7 Vishne, Gerber, Knight & Deouell (2023), Cell Reports 42, 112752, "Distinct ventral stream and prefrontal cortex representational dynamics during sustained conscious visual perception" [peer reviewed]

doi:10.1016/j.celrep.2023.112752. Ten patients, subdural recordings, images at 300, 900 and
1500 ms. Sensory regions hold a stable decodable representation of category and exemplar
content for the full presentation (the paper's "experience subspace"), while the prefrontal
cortex shows a transient onset burst without report and no offset response. Both the
registered IIT and GNWT predictions from the adversarial collaboration fit the result, the
sustained posterior representation for IIT and the transient prefrontal representation for
GNWT, with the registered GNWT offset component absent. Paradigm note: analyzed trials
required no report, but stimuli were task-relevant, so a fully passive version remains
untested at this resolution.

### 1.8 Das & Menon (2024), eLife 13, "Electrophysiological dynamics of salience, default mode, and frontoparietal networks during episodic memory formation and recall revealed through multi-experiment iEEG replication" [peer reviewed]

doi:10.7554/eLife.99018. 177 iEEG participants across four episodic memory experiments.
Phase transfer entropy shows consistently higher directed flow from the anterior insula to
default mode and frontoparietal network nodes, stronger during memory tasks than rest. This is the large-scale coordination layer the wave results sit inside.

### 1.9 Machine-learning wave networks [two preprints]

- Keller & Welling (2023), ICML, "Neural Wave Machines" [peer reviewed]. Locally coupled
  oscillatory RNNs; waves in the hidden state encode sequence structure.
- Liboni et al. (2025), PNAS 122, e2321319121 [peer reviewed]. Complex-valued RNN with
  distance-dependent coupling whose inputs set each node's natural frequency; object-specific
  waves segment images; exactly solvable. Code: github.com/mullerlab/liboniEA2025image.
- Jacobs, Budzinski, Muller, Ba & Keller (2025), arXiv:2502.06034. **Preprint, not peer
  reviewed.** Convolutional recurrent networks with waves; the waves widen the effective
  receptive field and help global segmentation.
- Karuvally, Sejnowski & Siegelmann (2024), arXiv:2402.10163. **Preprint, not peer
  reviewed.** Hidden traveling waves bind working-memory variables in RNNs.
- Shervani-Tabar, Brincat, Lundqvist & Miller (2026), bioRxiv 10.64898/2026.01.08.698281.
  **Preprint, not peer reviewed.** Waves emerge under biological constraints and keep
  working memory stable against distractors.

### 1.10 Miller, Brincat & Roy (2026), J. Neurosci. 46(33), and Pinotsis & Miller (2026), Cerebral Cortex 36(6) [THEORY]

The analog cognition framework: brain computation runs partly through continuous fields and
their feedback on spiking. The authors present it as theory. The measured ephaptic physics
it builds on is real (the Anastassiou, Perin, Markram and Koch 2011 Nature Neuroscience
slice results, recorded with the thalamic sources in section 8 of
[aligned_external_resources.md](aligned_external_resources.md)), and the counterargument is
1.1's Box 1. The project takes no side; see section 4.

---

## 2. Alignment audit against the current code

Audited 2026-09-25 against the current checkout, after the natural-frequency fix landed
behind a default-off flag. Line numbers refer to this checkout.

| Topic | Location | State |
| --- | --- | --- |
| AKOrN update | `models/core/oscillatory_binding.py:66-139` | 5 module oscillators via `WorkspaceBindingSystem` (D=2, all-to-all coupling, K=1.0, dt=0.1, 5 Euler iterations per call). No spatial map, no conduction delays. |
| Natural-frequency term | `models/core/oscillatory_binding.py:111-114` | FIXED behind a default-off flag. The legacy einsum `'ndd,bnd->bnd'` (zero diagonal of a skew-symmetric omega) remains the default so the baseline stays bit-identical; `--akorn-natural-frequency` (`scripts/training/train_rlhf.py:2573-2579`) switches to the `'nde,bne->bnd'` rotation. |
| Flag wiring | `scripts/training/train_rlhf.py:179,182` | `binding_mechanism` defaults to `akorn`; `akorn_natural_frequency` defaults to False. |
| Phases persist | `models/core/oscillatory_binding.py:141-165` | Phases persist across steps and episodes; `reset_state()` is not called in training (`scripts/training/train_rlhf.py` calls `reset_state` only on the tectum, line 479). |
| KomplexNet alternative | `models/core/complex_binding.py` | Scalar phases, no frequency term, no delays. Present, not the default. |
| sync_R consumers | `models/core/global_workspace.py:282-290`, `scripts/training/train_rlhf.py:1878-1886` | Bid boost for modules whose phase aligns with the mean field; a reward-weighted sync loss every 10 steps (`sync_loss = -reward_signal * sync_R`). |
| Brian2 validation | `models/validation/brian2_binding_validation.py:71-88` | Extracts a nonzero omega from the skew-symmetric matrix and runs its own parameterisation; it does not match the PyTorch layer's legacy default path (zero omega). Treat as a separate parameterisation, not a validation of the shipped default. |
| Spatial map | `models/core/sensory_tectum.py` (ConvGRU recurrence, around lines 183-236) | The only spatial recurrent substrate; a wave could propagate in its `h_t`. Nobody has measured whether one does. |
| Distance kernel | `models/core/topographic_loss.py:5-21` | An inverse-distance [HW, HW] matrix exists and is reusable for a spatial phase-gradient measure. |
| Phase instruments | `models/evaluation/coupling_measures.py` | PLV, phase transfer entropy and PAC on 1-D signals, in cycles per step. No spatial phase-gradient measure exists. |
| Results | [results/oscillator_frozen_2026_09.md](results/oscillator_frozen_2026_09.md), [results/sync_r_content_2026_09.md](results/sync_r_content_2026_09.md) | sync_R frozen on DMTS and contentless across 3 seeds. The zero omega was a probable part of the cause; the fix's effect is not yet measured here. |
| Rubric | `docs/consciousness_indicators_butlin.md:129-153` | RPT-2 PARTIAL, GWT-2 PARTIAL, PP-1 PARTIAL. Nothing here promotes them. |
| Ephaptic or field code | none in this repo | Implemented in the separate Neutral Core engine. The evidence is disputed (section 4), and no field mechanism is on this repo's build path. |

The audit's one actionable item: the binding layer has five oscillators with all-to-all
coupling and no map, so no traveling wave can exist in it by construction. The one spatial
substrate that could carry a wave is the tectum ConvGRU state, and the cheapest decisive
step is to measure whether propagating phase structure already exists there before any
mechanism is built. The inverse-distance kernel and the cycles-per-step instruments make
that measurement buildable without new machinery.

---

## 3. The translation constraint (read before building anything from these papers)

The same constraint as the thalamic review applies, unchanged. This agent has no
millisecond clock. Its time axes are the environment step and the two to ten settle cycles
inside `ReentrantProcessor.settle`. The band claims in these papers (90 Hz ripples, 0.1 to
0.6 m/s conduction, theta and alpha clocks) **cannot be translated literally into this
system**. Any phase-like quantity built here is measured in cycles per step, in arbitrary
units, and carries no Hz interpretation. A conduction delay is k steps. Do not write "the
agent's 90 Hz". If a measurement needs a frequency axis, state its units as cycles per step
and say that the correspondence to the papers' bands is unknown.

---

## 4. The ephaptic debate, stated as open

Miller and colleagues propose that wave fields feed back on spiking, with field coupling as
part of the computational substrate. Muller and colleagues hold that circuit mechanisms,
horizontal fibers with distance-dependent delays, explain the measured effects, that in
vitro field effects are submillivolt against a roughly 25 mV gap to threshold, and that
fields are too coarse to carry the feature-specific structure the wave motifs show. Both
sides cite real measurements. This project takes no side until data supports one. The
website presents the dispute as open, and nothing in this repo implements or assumes either
side.

---

## 5. What this evidence does and does not license

**Licensed.** Measuring before building: a spatial phase-gradient instrument on the tectum
ConvGRU state is licensed as the next step, since the substrate exists and the measure is
missing. Treating the AKOrN natural-frequency fix as an experimental variable behind its
default-off flag, with the frozen-sync baseline reproducible. Treating transient synchrony
(co-ripples) and sustained-versus-ignition content (the multiduration paradigm) as
candidate measurement designs for the workspace.

**Not licensed.** Building a wave lattice before the spatial detector has measured anything
on the existing ConvGRU state. Any claim that the agent's oscillators carry waves. Any Hz
claim. Any claim that co-ripples or spirals occur in this agent. Any indicator promotion in
[consciousness_indicators_butlin.md](consciousness_indicators_butlin.md) from this review.
Reopening Phi-1.

---

## 6. Where this goes next

The prototype order on the internal plan already matches this audit: the spatial wave
detector on the tectum ConvGRU state comes before any wave-lattice build, every prototype
carries the three-seed pre-registered gate, and the oscillation instrument results are
reported FAILED first. The re-scoring of RPT-2, GWT-2 and PP-1 stays owner-gated.
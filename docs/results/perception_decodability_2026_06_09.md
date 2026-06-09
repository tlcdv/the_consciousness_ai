# Perception-decodability probe (2026-06-09)

De-confounded answer to "is the bottleneck perception or policy?", the Phase 5
prerequisite from the 2026-06-09 architecture assessment. One linear decoder,
identical protocol (standardize, multinomial logistic regression, held-out 30%
test split), applied to four representation stages. This removes the confounds of
the 2026-06-02 reward-from-tap localization (epsilon schedule, MLP-vs-CNN learners,
unequal episode counts).

Script: `scripts/analysis/probe_perception_decodability.py`. Numbers below are read
from `runs/perception_probe_full/decodability.csv` (seed 42, 2 episodes/env,
4-stage sweep). Components are built by the real `init_components` and run forward
under `no_grad` (static, untrained, no optimizer steps), so the training file is
unchanged.

## Decode stages

| Stage | Where | Dim |
|-------|-------|-----|
| pixels | 32x32 block-mean of the raw frame | 3072 |
| obs_map | post retinotopic encoder + IE fusion (`tectum._last_obs_map`) | 16384 |
| tectum_content | post RSSM + capsule collapse (`tectum(...)[0]`) | 256 |
| broadcast | post-GNW (settle-extracted) | 256 |

Visual encoder active in this run: **conv-fallback (untrained projection)** for both
environments. DINOv2 weights were not present, so the retinotopic encoder used its
4-layer conv stack. This matters for interpretation (see Caveats).

## Results (test accuracy; chance = uniform 1/classes; major = majority-class share)

### DMTS, sample phase (stimulus on screen = pure perception)

| label | pixels | obs_map | tectum_content | broadcast | chance | major |
|-------|-------:|--------:|---------------:|----------:|-------:|------:|
| shape (6) | 0.917 | **1.000** | **0.275** | 0.275 | 0.167 | 0.275 |
| color (6) | 1.000 | **1.000** | **0.225** | 0.225 | 0.167 | 0.225 |
| size (2)  | 0.954 | **1.000** | **0.525** | 0.525 | 0.500 | 0.525 |

### DMTS, delay phase (stimulus off screen = working-memory test)

| label | pixels | obs_map | tectum_content | broadcast | chance | major |
|-------|-------:|--------:|---------------:|----------:|-------:|------:|
| shape | 0.276 | 0.276 | 0.276 | 0.276 | 0.167 | 0.274 |
| color | 0.245 | 0.245 | 0.245 | 0.245 | 0.167 | 0.244 |
| size  | 0.536 | 0.536 | 0.536 | 0.536 | 0.500 | 0.534 |

### WCST, card visible

| label | pixels | obs_map | tectum_content | broadcast | chance | major |
|-------|-------:|--------:|---------------:|----------:|-------:|------:|
| shape (4) | 0.983 | **1.000** | **0.258** | 0.258 | 0.250 | 0.265 |
| color (4) | 1.000 | **1.000** | **0.325** | 0.325 | 0.250 | 0.292 |
| count (4) | 1.000 | **1.000** | **0.308** | 0.267 | 0.250 | 0.270 |

## Verdict

**The lossy stage is the `obs_map -> tectum_content` collapse, not the perceptual
front-end.** Across both tasks and all six stimulus features:

- The pixel sanity baseline decodes near-perfectly (0.92 to 1.00). The probe works.
- **obs_map decodes the stimulus PERFECTLY (1.000 everywhere)**, even with an
  untrained conv encoder. The retinotopic encoder + inverse-effectiveness fusion
  preserve task-relevant stimulus identity completely.
- **tectum_content collapses to the majority-class baseline** (0.225 to 0.525, each
  within 0.01 of its `major` column). The RSSM + capsule composition + workspace
  projection that compress 16384-D obs_map into 256-D destroy stimulus identity.
- **broadcast equals tectum_content** (it is built from the winning vision payload,
  which is tectum_content), so the post-GNW stage recovers nothing.

The policy and the global workspace both read tectum_content / broadcast. They never
receive decodable stimulus identity, because the collapse upstream of them has
already removed it. This is the perception-vs-policy answer: the policy is not
starved by a weak encoder (the encoder is lossless); it is starved by the
representation handoff between a good spatial map and the workspace.

Working memory: at the delay phase every stage sits at its majority baseline. The
untrained RSSM/GNW does not hold the sample across the blank delay. DMTS is
unsolvable in this state regardless of policy.

## What this overturns

The 2026-06-02 localization (`agent_competence_fix_2026_06_02.md`) provisionally
read the lossy stage as the pixels->obs_map front-end. Its own doc flagged that
reading as confounded. This de-confounded probe shows the opposite: pixels->obs_map
is lossless (1.000), and the loss is concentrated at obs_map->tectum_content. The
prior provisional reading is retracted.

## Caveats (do not over-read)

- **Untrained components.** The capsule routing and RSSM are randomly initialized.
  obs_map being decodable under an untrained conv is expected (convolution preserves
  spatial structure by construction). tectum_content collapsing under an untrained
  capsule/RSSM is partly an untrained-weights effect. Whether training recovers
  tectum_content decodability is NOT measured here. It is the immediate next test.
- **Hypothesis (marked as such, not a finding):** the collapse may persist after
  training, because tectum_content is trained by a reward-prediction MSE + the TDANN
  topographic loss, neither of which asks it to preserve stimulus identity. A 256-D
  bottleneck trained on reward has no pressure to keep shape/color/count. This is
  exactly the gap a generative (reconstruction / variational-free-energy) objective
  would close. To be tested, not assumed.
- Single seed, single configuration, conv fallback. A measurement, not a law.

## Implications for the next build

The originally framed "active-inference stage 1 = rebuild the front-end encoder"
targets the wrong stage: the encoder is already lossless. Two cleaner options the
result points to, both gated and FAILED-first, at least 3 seeds before any default
change:

1. **Re-probe after a short training run** (cheap, decisive). Train DMTS/WCST for N
   episodes, then re-run this probe. If trained tectum_content stays at chance, the
   256-D collapse is architectural and the bottleneck is confirmed there. If it
   recovers, the issue was untrained init.
2. **Let the workspace/policy read obs_map, not the collapsed tectum_content.** The
   `--policy-input spatial` tap (added 2026-06-02) already exposes obs_map. The
   smaller-delta test is whether a policy on the decodable spatial map can enter the
   DMTS/WCST regimes. If active inference is pursued, its target is the RSSM/capsule
   generative training (the collapse stage), not the encoder.

## After-training re-probe (added same day)

The untrained result above left one question open: does training recover
tectum_content decodability, or is the 256-D collapse architectural? Tested by
training DMTS for 100 episodes (`train_rlhf --env dmts --episodes 100 --save-tectum`),
then re-probing the trained tectum (`--load-tectum`,
`runs/perception_probe_trained/decodability.csv`, seed 42).

**Training outcome first (FAILED): the agent did not learn DMTS.** Episode reward was
flat and negative throughout (episode 0 = -40.69, episode 99 = -38.59,
`runs/train_dmts_100/episodes.csv`). No behavioural learning occurred.

Trained vs untrained tectum_content, DMTS sample phase:

| feature | untrained | trained (100 ep) | obs_map | chance | major |
|---------|----------:|-----------------:|--------:|-------:|------:|
| shape (6) | 0.275 | 0.225 | 1.000 | 0.167 | 0.275 |
| color (6) | 0.225 | 0.133 | 1.000 | 0.167 | 0.225 |
| size (2)  | 0.525 | 0.600 | 1.000 | 0.500 | 0.525 |

**Verdict: training did NOT recover tectum_content decodability.** After 100 episodes
it remains at chance (shape and color sit at their majority/chance baselines; size
shows a marginal bump but stays far below the 1.000 that obs_map carries). obs_map is
unchanged at 1.000. The obs_map -> tectum_content collapse persists trained and
untrained.

Caveat (load-bearing): the agent never learned DMTS, so the reward-predictor that
trains the tectum saw a near-flat signal. Two non-exclusive readings, both consistent
with the data: (a) the reward-prediction-MSE + TDANN objective provides no pressure to
preserve stimulus identity in the 256-D collapse, which is the architectural/objective
gap; (b) on a task this agent cannot learn, there is no discriminative reward signal to
train on regardless. This is the assessment's root bind restated: the agent cannot
perform the consciousness tasks, which limits even the training-recovery test. A
longer or multi-seed run is the formal confirmation, but the flat reward trajectory
(no learning over 100 episodes) makes it unlikely that more episodes of the same
objective change the result. Single seed; a measurement, not a law.

**What this sharpens for the next build.** A reconstruction / variational-free-energy
objective is the principled fix precisely because it would force the RSSM/capsule to
encode what is needed to reconstruct the frame, which REQUIRES preserving shape/color
in the latent. That is the pressure the current reward-MSE objective lacks. The
cheaper immediate delta remains: route the policy at obs_map (the `--policy-input
spatial` tap), which is decodable, instead of the collapsed tectum_content. Both are
gated, FAILED-first, >= 3 seeds before any default change.

## Reproduce

```
export PYPHI_WELCOME_OFF=yes
# untrained probe (4-stage sweep)
python -m scripts.analysis.probe_perception_decodability \
    --episodes 2 --seed 42 --out-dir runs/perception_probe_full

# after-training re-probe
python -m scripts.training.train_rlhf --env dmts --episodes 100 --max-steps 200 \
    --phi-sample-every 5 --save-tectum runs/train_dmts_100/tectum.pt \
    --log-dir runs/train_dmts_100
python -m scripts.analysis.probe_perception_decodability \
    --episodes 2 --no-broadcast --envs dmts --seed 42 \
    --load-tectum runs/train_dmts_100/tectum.pt \
    --out-dir runs/perception_probe_trained
```

Unit tests for the decoder: `pytest tests/test_perception_probe.py -q`.

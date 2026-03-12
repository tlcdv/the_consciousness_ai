# True Isomorphic Visual Mapping: Research and Design

## The Problem

Our current `VisualTectumProjection` takes Qwen2-VL ViT output [1536, H, W] and compresses it to [64, 16, 16] via a 1x1 Conv2d. The 16x16 spatial grid is present, but the features are **semantic** (learned abstract representations from the ViT), not **geometric** (preserving actual input spatial structure). Qwen2-VL's ViT tokenization destroys retinotopic organization: patch embeddings carry semantic content (what is at this location), not spatial structure (where things are relative to each other in metric space).

This matters because Feinberg & Mallatt require **isomorphic mapping** for consciousness, where the physical spatial relationships among processing units preserve the spatial relationships in the perceived scene. The map IS the experience, not a code that represents it.

---

## What "Isomorphic" Actually Means (from Biology)

Three nested terms, often confused:

- **Retinotopic**: spatial layout of the retina is preserved in the neural map. Adjacent photoreceptors project to adjacent neurons.
- **Topographic**: general term for any ordered neural map where neighboring neurons encode neighboring stimulus values. Retinotopy is visual topographic mapping; tonotopy is auditory; somatotopy is body surface.
- **Isomorphic**: the strongest requirement. Preserves not just neighborhood relationships (topology) but also **metric structure**. Distances and angles between points in the stimulus domain are proportionally preserved in the neural representation.

**Feinberg & Mallatt's claim**: The pattern of activity across a spatially organized neural map IS the mental image. The brain does not reconstruct experience from an encoded signal. The spatial relationships among neurons carry the spatial information. Permuting the neurons would destroy the experience.

**Minimum requirement**: At least one isomorphic map where the physical organization of processing units preserves the metric spatial structure of sensory input, with 3+ hierarchical levels of processing applied to it, with reciprocal connections between levels.

---

## What the Biological Superior Colliculus Does

The SC/optic tectum has 7 layers, functionally grouped into three zones:

1. **Superficial layers**: pure retinotopic visual map (log-polar transform, foveal overrepresentation). Direct retinal input.
2. **Intermediate layers**: multisensory integration zone. Visual, auditory, and somatosensory maps all aligned in a common body-centered coordinate frame. ~60% of neurons are multisensory.
3. **Deep layers**: motor/premotor neurons driving orienting movements (saccades, head turns). Motor maps in register with sensory maps above.

### How alignment works across modalities

- **Visual map** is the anchor. Derived from direct anatomical projection from retina.
- **Auditory map** is computed (from ITD, ILD, spectral cues), then calibrated to match the visual map during a developmental critical period. Visual experience calibrates auditory alignment (Knudsen & Brainard 1991).
- **Somatosensory map** covers face/head/upper body, warped to align with visual/auditory maps.
- Alignment precision: 1-2 degrees in frontal field (barn owl), coarser peripherally. Visual RFs are 10-30 degrees wide in SC, auditory RFs 40-80 degrees. Alignment means **centroids** correspond, not identical sizes.

### Three principles of multisensory integration (Stein & Meredith 1993)

1. **Spatial rule**: Enhancement only when stimuli from same/nearby spatial locations
2. **Temporal rule**: Stimuli must occur within ~100-200ms window
3. **Inverse effectiveness**: Proportional enhancement greatest when individual unimodal responses are weakest. Weak flash + weak sound = 200-300% boost. Strong flash + strong sound = 10-20% boost.

**Inverse effectiveness mechanism**: sigmoid response function. Weak inputs operate on steep part (large gain from combination). Strong inputs near saturation (small gain).

### Key insight for implementation

Biological spatial register does NOT mean pixel-level co-registration. It means centroids of receptive fields for different modalities at the same map location point to the same external direction, with each modality having its own resolution. A 16x16 grid at 64 channels per cell is in the right ballpark for SC-level resolution.

---

## Available Technical Approaches

### Approach A: DINOv2 Frozen Patch Tokens (Dual-Stream)

**Core idea**: Use DINOv2's patch tokens as a parallel spatial feature stream alongside Qwen2-VL semantic features. DINOv2 is self-supervised (no language bias) and its patch tokens preserve strong spatial structure.

**How it works**:
- DINOv2 ViT-L/14 produces patch tokens at 14x14 spatial resolution for 224x224 input
- Each patch token is 1024-dim (ViT-L) and corresponds to a specific 14x14 pixel region
- `reshape_hidden_states=True` gives [B, 1024, H_patches, W_patches], a proper spatial grid
- Patch tokens carry both semantic AND spatial information. DINOv2 trained with patch-level masking drives sensitivity to local spatial structure.
- NeCo (2024) shows DINOv2 spatial features can be improved further with just 19 GPU-hours of post-training using Patch Neighbor Consistency loss

**Architecture**:
```
Video frame -> DINOv2 (frozen) -> patch tokens [B, 1024, 16, 16]
                                       |
                               channel reduction (1024 -> 64)
                                       |
                                       v
                              TopographicMap (fuse with audio)
                                       |
                                   RSSM (temporal)
                                       |
                                   Capsules -> Workspace

Video frame -> Qwen2-VL (frozen) -> semantic embeddings -> language/reasoning pathway
```

**Pros**:
- DINOv2 is designed for dense spatial tasks (segmentation, depth)
- Patch tokens have a direct spatial correspondence to image regions
- Self-supervised, no language/text bias distorting spatial features
- Frozen model, zero training cost
- 14x14 patch grid matches our 16x16 tectum grid closely (can bilinear interpolate)
- Well-established, battle-tested model with HuggingFace integration

**Cons**:
- Still learned abstract features, not raw pixel geometry
- Adding a second large model (ViT-L is ~300M params)
- DINOv2 patch features are semantic-spatial hybrids, not pure retinotopic

### Approach B: Topographic Loss (TDANN-style) on Our Existing Features

**Core idea**: Instead of changing the feature source, add a topographic spatial smoothness loss that forces our existing feature maps to self-organize into retinotopic maps.

**How it works** (Margalit et al. 2024):
- Each layer's units are assigned positions on a 2D "cortical sheet"
- Loss = alpha * SpatialLoss + (1-alpha) * TaskLoss
- SpatialLoss: correlates response similarities with inverse pairwise distances. Nearby units should have more correlated responses than distant units.
- Relative SL: correlate the population of response similarities and pairwise inverse distances across unit pairs
- alpha = 0.25 works well across layers
- Result: topographic maps emerge spontaneously, reproducing V1 retinotopy, orientation maps, and even face/place selectivity in higher areas

**Architecture change**: Add a regularization loss to the tectum's training:
```python
def topographic_loss(feature_map, alpha=0.25):
    """
    Encourage spatial smoothness: nearby grid cells should have
    correlated activations.

    feature_map: [B, C, H, W]
    """
    B, C, H, W = feature_map.shape
    # Compute pairwise response similarity (cosine) between all spatial locations
    features_flat = feature_map.view(B, C, H*W)  # [B, C, N]
    sim = F.cosine_similarity(
        features_flat.unsqueeze(-1),  # [B, C, N, 1]
        features_flat.unsqueeze(-2),  # [B, C, 1, N]
        dim=1
    )  # [B, N, N]

    # Compute pairwise spatial distance (inverse)
    coords = torch.stack(torch.meshgrid(
        torch.arange(H, dtype=torch.float),
        torch.arange(W, dtype=torch.float),
        indexing='ij'
    ), dim=-1).view(-1, 2)  # [N, 2]

    dist = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)  # [N, N]
    inv_dist = 1.0 / (dist + 1.0)  # avoid div by zero, normalize

    # Spatial loss: correlation between response similarity and inverse distance
    # Higher correlation = more topographic organization
    loss = -torch.corrcoef(torch.stack([
        sim.mean(0).view(-1), inv_dist.view(-1)
    ]))[0, 1]

    return alpha * loss
```

**Pros**:
- Biologically grounded (reproduces actual cortical organization)
- No new models needed, works on existing feature maps
- Produces genuine emergent topographic organization
- Forces the system to self-organize spatially

**Cons**:
- Requires training (not frozen inference)
- Only works during training, not at inference time on new inputs
- More complex loss function
- May conflict with task performance if alpha too high

### Approach C: Raw Pixel Pyramid + Conv Stack (Pure Geometric)

**Core idea**: Process the video frame through a lightweight convolutional stack that explicitly preserves spatial geometry at every level. No ViT tokenization at all.

**Architecture**:
```
Video frame [3, 256, 256]
    |
Conv2d(3, 32, 7, stride=2, pad=3)   -> [32, 128, 128]
Conv2d(32, 64, 3, stride=2, pad=1)  -> [64, 64, 64]
Conv2d(64, 64, 3, stride=2, pad=1)  -> [64, 32, 32]
Conv2d(64, 64, 3, stride=2, pad=1)  -> [64, 16, 16]  <- tectum grid
    |
TopographicMap (fuse with audio)
    |
RSSM (temporal) -> Capsules -> Workspace
```

**Pros**:
- Perfect retinotopic preservation (each grid cell = specific image region)
- Spatial metric structure mathematically guaranteed by regular strided convolution
- Lightweight, fast, no pretrained model needed
- Each cell's receptive field is explicitly computable
- The map is trivially permutation-sensitive (shuffling cells breaks spatial structure)

**Cons**:
- No semantic understanding (just learned edge/texture features)
- Needs training from scratch
- Much weaker features than DINOv2 or Qwen2-VL
- Not leveraging any pretrained visual knowledge

### Approach D: Qwen2-VL ViT Intermediate Features with 2D-RoPE

**Core idea**: Qwen2-VL's ViT uses 2D Rotary Position Embeddings (2D-RoPE), which encode height and width separately. The ViT's intermediate hidden states (before the MLP compression) should retain spatial grid structure because of this 2D positional encoding.

**How it works**:
- Qwen2-VL ViT has patch_size=14, produces patch tokens with 2D-RoPE
- M-ROPE decomposes position into (temporal, height, width) components
- Extract intermediate ViT features before the final MLP compression that merges 2x2 adjacent tokens
- These intermediate features are still on a spatial grid (H_patches x W_patches)

**Architecture change**: Extract ViT hidden states at an intermediate layer instead of the final compressed output:
```python
# Instead of: qwen_output = model.visual(pixel_values)  # compressed
# Do: hidden_states = model.visual.get_intermediate_layers(pixel_values, n=4)
# Use the last intermediate layer before compression
# Shape: [B, H_patches * W_patches, hidden_dim]
# Reshape to: [B, hidden_dim, H_patches, W_patches]
```

**Pros**:
- Reuses existing Qwen2-VL model (no new parameters)
- 2D-RoPE explicitly encodes spatial positions
- Patch tokens at intermediate layers are both semantic AND spatial
- Single model handles both spatial and semantic streams

**Cons**:
- Qwen2-VL is not optimized for spatial preservation (optimized for VQA)
- Hidden dim is very large (1536), needs heavy projection
- Unclear if intermediate features are truly retinotopic or just positionally-encoded semantic
- Coupling spatial and semantic in one model means we can't tune them independently

### Approach E: V-JEPA 2 (World Model Features)

**Core idea**: V-JEPA 2 is a video world model that predicts representations in latent space. It uses 3D-RoPE (x, y, time) and processes video natively. Its features encode both spatial structure and temporal dynamics.

**How it works**:
- V-JEPA 2 uses spatiotemporal tokenization with 3D-RoPE
- Trained on 1M+ hours of video via self-supervised prediction in latent space
- Progressive resolution training: patches per frame scale from 16 to 64 frames
- Features are designed to be stable representations of scene content

**Pros**:
- Natively handles video/temporal dynamics
- 3D-RoPE preserves spatial AND temporal structure
- Self-supervised (no language bias)
- World model properties align with our RSSM tectum design
- Could potentially replace both DINOv2 AND our custom RSSM

**Cons**:
- Very new (June 2025), ecosystem less mature
- Very large model (ViT-H or larger)
- Designed for action prediction/robotics, not for spatial preservation per se
- Features are abstract latent predictions, not guaranteed retinotopic
- May be overkill if we already have RSSM for temporal dynamics

---

## Evaluation: What "Good Enough" Means

Based on the biological ground truth, an isomorphic map implementation needs these **five properties** (from the biological research):

| Property | Requirement | Test |
|----------|-------------|------|
| **P1: Neighborhood preservation** | Adjacent pixels/patches map to adjacent grid cells | Conv2d satisfies this by construction |
| **P2: Metric proportionality** | Distances in input space proportional to distances in map | Bilinear interpolation preserves this |
| **P3: Multimodal co-registration** | Visual and audio features at same grid cell = same external direction | Gaussian bump approach is correct |
| **P4: Persistence across hierarchy** | Spatial structure maintained through 3-4+ processing levels | Current: 4 levels (proj -> fusion -> RSSM -> capsules) |
| **P5: Causal efficacy** | Permuting grid cells degrades performance | Must be tested empirically |

**Properties NOT required** (biological evidence):
- Point-to-point bijection (SC uses log-polar, not uniform grid)
- Pixel-level resolution (SC has 10-30 degree visual RFs)
- Perfect cross-modal alignment (1-2 degrees at best in biology)
- Real-time temporal dynamics (discrete frames are fine)

---

## Recommended Design: Dual-Stream Architecture

After evaluating all approaches, the recommended design is **Approach A (DINOv2) + Approach B (TDANN loss) + inverse effectiveness**:

### Architecture

```
                    ┌──────────────────────────┐
                    │     Video Frame           │
                    └──────┬───────────┬────────┘
                           │           │
                    ┌──────▼──────┐  ┌─▼──────────────┐
                    │ DINOv2      │  │ Qwen2-VL       │
                    │ (frozen)    │  │ (frozen)        │
                    │ ViT-B/14    │  │ ViT 675M        │
                    └──────┬──────┘  └─┬──────────────┘
                           │           │
                    ┌──────▼──────┐    │
                    │ Patch tokens│    │ Semantic
                    │ [B,768,H,W] │    │ embeddings
                    └──────┬──────┘    │ (for language/
                           │           │  reasoning)
                    ┌──────▼──────┐    │
                    │ SpatialProj  │    │
                    │ 768->64      │    │
                    │ 1x1 Conv2d   │    │
                    └──────┬──────┘    │
                           │           │
                    ┌──────▼───────────▼──┐
                    │  TopographicMap      │
                    │  (vision + audio +   │
                    │   inverse eff.)      │
                    │  + TDANN spatial loss│
                    └──────┬──────────────┘
                           │
                    ┌──────▼──────┐
                    │ RSSM        │
                    │ (temporal)  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Capsules    │
                    │ (composition)│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Workspace   │
                    │ (GNW)       │
                    └─────────────┘
```

### Why DINOv2-B/14, not ViT-L

- ViT-B/14: 86M params, 768-dim patch tokens, 16x16 patch grid for 224x224 input
- ViT-L/14: 300M params, 1024-dim. Overkill for spatial features.
- The B variant is sufficient because we only need spatial structure, not maximum semantic richness (that's Qwen2-VL's job).

### Why dual-stream, not single-stream

Feinberg & Mallatt's architecture has separate pathways:
- **Tectum** (midbrain): spatial, retinotopic, multisensory integration. Fast, action-oriented.
- **Cortex** (forebrain): semantic, categorical, language-linked. Slow, deliberative.

DINOv2 → Tectum (spatial features for the isomorphic map)
Qwen2-VL → Cortical pathway (semantic understanding for language and reasoning)

This matches the biological two-pathway architecture. The tectum gets fast spatial features; the cortex gets rich semantic features. Both project to the workspace (GNW) where they compete for consciousness.

### What changes from current implementation

1. **New module**: `RetinotopicEncoder` wrapping frozen DINOv2-B/14
   - Extracts patch tokens, reshapes to spatial grid
   - 1x1 Conv2d channel reduction (768 -> 64)
   - Replaces `VisualTectumProjection` (which adapted Qwen2-VL features)

2. **Modified TopographicMap**: Add inverse effectiveness fusion
   - When both visual and audio activations at a grid cell are weak, proportionally boost the fused signal more
   - Implementation: `ie_weight = 1.0 / (max(v_mag, a_mag) + eps)`

3. **New training loss**: TDANN spatial smoothness regularization
   - Correlation between response similarity and inverse spatial distance
   - alpha = 0.25, applied to tectum feature maps during training
   - Encourages genuine topographic self-organization

4. **Qwen2-VL pathway preserved**: continues to serve language/semantic processing
   - No longer feeds into the tectum
   - Instead, competes directly in the workspace as a "cortical" specialist

### Why this satisfies isomorphism

- **P1 (neighborhood)**: DINOv2 patch tokens are in spatial grid order by construction. Conv2d maintains locality. TDANN loss reinforces it.
- **P2 (metric)**: Each DINOv2 patch covers a fixed 14x14 pixel region. Grid cells are equispaced. Distances in grid = proportional to distances in image.
- **P3 (co-registration)**: Audio Gaussian bump placed on same grid. Inverse effectiveness amplifies weak multimodal signals.
- **P4 (hierarchy)**: 4+ levels: DINOv2 patches -> fusion -> RSSM -> capsules
- **P5 (causal efficacy)**: Testable via grid permutation experiment

---

## Missing Biological Feature: Inverse Effectiveness

The most important principle missing from our current TopographicMap:

```python
def fuse_with_inverse_effectiveness(visual, audio, epsilon=1e-6):
    """
    Inverse effectiveness: proportional enhancement is greatest
    when individual unimodal responses are weakest.

    Biological basis: Stein & Meredith 1993, Ohshiro et al. 2011
    (divisive normalization model, Nature Neuroscience)

    visual: [B, C, H, W]
    audio:  [B, C, H, W]
    """
    v_mag = visual.norm(dim=1, keepdim=True)  # [B, 1, H, W]
    a_mag = audio.norm(dim=1, keepdim=True)

    # Inverse effectiveness weight: higher when both are weak
    max_unimodal = torch.max(v_mag, a_mag) + epsilon
    ie_weight = 1.0 / max_unimodal
    ie_weight = ie_weight / (ie_weight.mean() + epsilon)  # normalize

    # Additive fusion with inverse effectiveness weighting
    fused = visual + audio * ie_weight
    return fused
```

---

## Implementation Plan (Estimated effort)

### Phase 1: RetinotopicEncoder (3-4 days)
- New file: `models/core/retinotopic_encoder.py`
- Wrap DINOv2-B/14 frozen backbone
- Extract patch tokens, reshape to spatial grid [B, 768, H, W]
- Channel reduction to [B, 64, 16, 16]
- Add to requirements: `transformers` (already present for Qwen2-VL)
- Tests: verify spatial correspondence (patch at grid[i,j] comes from image region [i*14:(i+1)*14, j*14:(j+1)*14])

### Phase 2: Inverse Effectiveness in TopographicMap (1-2 days)
- Modify `models/core/sensory_tectum.py` TopographicMap.forward()
- Replace simple concatenation with inverse effectiveness fusion
- Tests: weak+weak produces stronger enhancement than strong+strong

### Phase 3: TDANN Spatial Loss (2-3 days)
- New file: `models/core/topographic_loss.py`
- Implement spatial smoothness correlation loss
- Add to tectum training loop with alpha=0.25
- Tests: spatial loss decreases over training steps, nearby cells become more correlated

### Phase 4: Rewire Architecture (1-2 days)
- SensoryTectum uses RetinotopicEncoder instead of VisualTectumProjection
- Qwen2-VL becomes a separate workspace specialist (cortical pathway)
- Update ConsciousnessCore to route visual input to both streams

### Phase 5: Validation (2-3 days)
- Grid permutation test: shuffle tectum grid cells, measure performance degradation
- Verify spatial correspondence: stimuli at image location (x,y) activate grid cell (x/stride, y/stride)
- Compare EI at tectum level with and without TDANN loss
- Phi measurement at tectum level

---

## References

- Feinberg & Mallatt (2016). The Ancient Origins of Consciousness. MIT Press.
- Stein & Meredith (1993). The Merging of the Senses. MIT Press.
- Stein & Stanford (2008). Multisensory integration. Nat Rev Neurosci 9(4).
- Ohshiro, Angelaki & DeAngelis (2011). Normalization model of multisensory integration. Nat Neurosci 14(6).
- Anastasio, Patton & Belkacem-Boussaid (2000). Using Bayes' rule to model multisensory enhancement in the SC. Neural Computation 12(5).
- Knudsen & Brainard (1991). Visual calibration of auditory map. Science 253(5025).
- Meredith & Stein (1986). Inverse effectiveness measurements. J Neurophysiol 56(3).
- Cuppini, Ursino, Magosso et al. (2010). Emergent model of multisensory integration in SC. Frontiers Integr Neurosci 4:6.
- Margalit et al. (2024). A unifying framework for functional organization in early and higher ventral visual cortex. Neuron.
- Oquab et al. (2024). DINOv2: Learning Robust Visual Features without Supervision. TMLR.
- Bardes et al. (2024). V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video.
- Sabour, Frosst & Hinton (2017). Dynamic routing between capsules. NeurIPS.

# KomplexNet (Phase B-alt) Upstream License Decision

**Date:** 2026-05-19
**Decision:** Phase B-alt may proceed. KomplexNet is MIT-licensed and can be vendored OR re-implemented clean-room. We will use a clean-room implementation derived from the paper's mathematical content to avoid coupling to a specific framework (PyTorch Lightning).

## Naming correction

The 2026-05-19 plan referred to this work as "GASPnet". That name appears to be misremembered: the actual paper is published as **KomplexNet** (arxiv 2502.21077). All architectural properties cited in the plan's alternative-vetting research remain accurate: Kuramoto dynamics on complex-valued representations, phase synchrony as binding mechanism, content-level (not bid-level). The paper title is "Enhancing deep neural networks through complex-valued representations and Kuramoto synchronization dynamics" by Muzellec, Alamia, Serre, VanRullen (2025).

## Sources verified this session

| Source | URL | License | Retrieval date |
|--------|-----|---------|----------------|
| KomplexNet GitHub repo | https://github.com/S4b1n3/KomplexNet | **MIT License** | 2026-05-19 |
| KomplexNet repo LICENSE file | https://raw.githubusercontent.com/S4b1n3/KomplexNet/main/LICENSE | MIT (full text confirmed) | 2026-05-19 |
| KomplexNet paper | https://arxiv.org/abs/2502.21077 | CC BY 4.0 (paper text) | 2026-05-19 |
| Plan's name "GASPnet" | n/a | — | misremembered; actual name is KomplexNet |

The code and paper licenses are separate, both compatible with this project: MIT for the implementation (permissive, no copyleft), CC BY 4.0 for the paper text (any use with attribution).

## Compatibility analysis

This project's `LICENSE.md` is a custom "Non-Commercial Open Source License" prohibiting commercial use without prior written permission. MIT is permissive and explicitly compatible with any downstream license, including non-commercial. The combined work as a whole is governed by this project's non-commercial license; any vendored MIT-licensed portions retain their upstream license terms via attribution.

Required obligations under MIT:

1. Preserve the MIT LICENSE text alongside any vendored code (we will copy it to `third_party/KOMPLEXNET_LICENSE`)
2. Preserve the copyright notice (Copyright (c) 2025 Sabine Muzellec)
3. No advertising clause; no other obligations

## Chosen path

**Clean-room re-implementation, not vendoring**, for three reasons:

1. **Architecture coupling**: KomplexNet uses PyTorch Lightning for training infrastructure. Our project does NOT use PyTorch Lightning; we have plain `torch.nn` modules and a custom training loop. Vendoring would force us to either (a) add PyTorch Lightning as a dependency, or (b) extract just the Kuramoto math and re-assemble in our framework — which is effectively a clean-room implementation anyway.

2. **Module structure**: KomplexNet implements visual feature binding (CNN + complex-valued operations on visual tokens). Our project needs Kuramoto binding on workspace MODULE BIDS and CONTENTS (vision, audio, memory, body, semantic). The use case is different enough that a literal port would require restructuring everything.

3. **Match our existing AKOrN clean-room pattern**: our `models/core/oscillatory_binding.py` is a clean-room implementation of Löwe et al.'s AKOrN paper. Same approach for the KomplexNet replacement: read the paper's math, implement in our own PyTorch code that matches our module interface.

We will reference the published paper's formulas (free per CC BY 4.0 and not copyrightable as mathematics) but write all code from scratch under the project's existing license framework. No upstream LICENSE file needs to be vendored since we are not copying code.

If a future session decides to lift a specific helper from the upstream repo (e.g., the complex-valued convolution operators), we will:
1. Copy `third_party/KOMPLEXNET_LICENSE` from https://raw.githubusercontent.com/S4b1n3/KomplexNet/main/LICENSE
2. Add SPDX header to the lifted code
3. Update this decision doc

## Mathematical content to extract from the paper (mission-aligned subset)

From the paper's abstract and methodology (verified via WebFetch this session):

1. **Complex-valued representations**: each "neuron" carries (amplitude, phase) where amplitude encodes content magnitude and phase encodes binding identity
2. **Kuramoto dynamics at the initial layer**: phases of complex-valued units evolve per Kuramoto coupling, producing phase synchronization for related features
3. **Phase propagation through layers**: complex-valued operations (convolution, attention) preserve phase information across the network
4. **Sync_R equivalent**: the Kuramoto order parameter R is defined the same way as in AKOrN (L2 norm of mean phase vector); preserved as the binding metric for the Phi-1 prediction

The CRITICAL difference from AKOrN:

- AKOrN: phases are abstract oscillator states detached from content
- KomplexNet: phases ARE the phase component of complex-valued CONTENT vectors

This is the structural alignment the 2026-05-19 plan identified as necessary to bridge the binding-integration gap that all 8 prior Phi-1 runs have failed to bridge.

## Compliance with project policy

- No third-party licensed code is vendored in Phase B-alt
- The mathematical content (formulas, equations) is used freely per academic standards and CC BY 4.0 paper distribution
- The KomplexNet GitHub README and source are used as REFERENCE ONLY (read, not copied)
- Our existing AKOrN clean-room pattern is preserved; KomplexNet replacement follows the same pattern

## Files affected by this decision

- `models/core/complex_binding.py` (NEW Phase B-alt: clean-room KomplexNet-style binding)
- `models/core/oscillatory_binding.py` (kept; provides AKOrN as the legacy default)
- All 6 AKOrN-dependent files (will get a `--binding-mechanism {akorn, komplex}` flag for backward compat)

## Rejected alternatives

- **Vendor S4b1n3/KomplexNet source code directly**: rejected because of PyTorch Lightning coupling and the module-structure mismatch with our workspace-binding use case.
- **Skip license verification**: rejected per Phase 1 protocol from the plan.
- **Use the name "GASPnet" in code/docs**: rejected. The actual name is KomplexNet per the published paper. Calling the new module `ComplexBindingSystem` and citing KomplexNet in docstrings preserves attribution without claiming our implementation IS KomplexNet (it is a clean-room derivation in our framework).

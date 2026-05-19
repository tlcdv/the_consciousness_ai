# AKOrN Upstream License Decision

**Date:** 2026-05-19
**Decision:** Phase B may proceed. NO upstream code is vendored. Phase B extends our existing clean-room implementation and adds a new module written from scratch.

## Sources verified this session

| Source | URL | Status | Retrieval date |
|--------|-----|--------|----------------|
| AKOrN GitHub repo (correct URL) | https://github.com/autonomousvision/akorn | **NO LICENSE file** | 2026-05-19 |
| AKOrN paper | https://arxiv.org/abs/2410.13821 | arxiv nonexclusive-distrib/1.0 (paper text) | 2026-05-19 |
| AKOrN project page | https://takerum.github.io/akorn_project_page/ | not inspected | 2026-05-19 |
| Prior CLAUDE.md claim | "MIT license" | **INCORRECT** | 2026-02-21 entry |
| Prior CLAUDE.md URL guess | github.com/loeweX/AKOrN | **404 NOT FOUND** | 2026-05-19 |

Both CLAUDE.md misclaims (the license type and the repo owner) are now corrected by this decision doc.

## Legal analysis

**GitHub repo without a LICENSE file**: under US copyright law (and most jurisdictions), absence of an explicit license means the code is "all rights reserved" by the copyright holder. We may NOT copy, modify, or redistribute the code. We may only view it for personal study.

**arXiv paper under nonexclusive-distrib/1.0**: this is arXiv's standard distribution agreement. The paper TEXT may be redistributed. Mathematical formulas described in published academic papers are not copyrightable subject matter (Feist v. Rural, Baker v. Selden). Scientific ideas, methods, and equations are free for any party to implement.

**Our existing `models/core/oscillatory_binding.py`**: written 2026-02-21 onward (per the CLAUDE.md session log). Implements the standard Kuramoto-on-N-spheres dynamics from the published paper. The Explore agent this session verified the code: it uses standard PyTorch operations (einsum, torch.norm) on standard mathematical formulas (mean-field order parameter, tangent-plane projection). This is a clean-room implementation derived from the published paper's mathematical content, not a port of the GitHub code. We are free to use it.

## Chosen path

**Phase B proceeds without any upstream code vendoring.** Specifically:

1. Phase B1 extends `models/core/oscillatory_binding.py` (our own code) to expose the pairwise phase-coherence matrix `dot_ij` that is already computed inside `KuramotoLayer.forward()`. We are modifying our own clean-room implementation, not copying anything from upstream.

2. Phase B2 creates a NEW file `models/core/binding_attention.py` implementing AKOrN-modulated cross-attention. The cross-attention formula `attn = softmax(QK^T / sqrt(d)); out = attn @ V` is from Vaswani 2017, freely implementable. The coherence-modulated variant `K_ij = W_k(payload_j) * coherence[i,j]` is our own design.

3. No upstream AKOrN code is read, copied, or referenced beyond the publicly-available paper's mathematical descriptions.

## Compliance with project policy

- The project's own LICENSE.md is a custom "Non-Commercial Open Source License"
- We add no third-party licensed code in Phase B (zero new entries in `third_party/`)
- The decision honors the CLAUDE.md NEVER LIE protocol: the prior MIT claim and the wrong URL are both corrected here

## Notes for future sessions

If a future session ever wants to vendor specific functions from the upstream AKOrN repo (e.g., a more efficient `KuramotoLayer` implementation), the absence of a LICENSE is a hard blocker. The mitigation would be:

1. Open an issue on github.com/autonomousvision/akorn asking the authors to add a permissive license
2. Contact the corresponding author directly (their email is on the arxiv paper)
3. In the meantime, continue with our clean-room implementation

For Phase B and Phase B-alt (GASPnet contingency), neither vendoring path is needed because both can be implemented from scratch using publicly available paper formulas.

## Rejected alternatives

- **Vendor from autonomousvision/akorn without an explicit license**: rejected. This would be a copyright violation. Even if the authors clearly intend the code to be permissive, the lack of an explicit grant means redistribution is not authorized.
- **Use loeweX/AKOrN per prior CLAUDE.md**: rejected. URL is 404. CLAUDE.md was wrong about both the owner and the license.
- **Skip license verification**: rejected per plan Phase 1 protocol. The 30-minute cost prevented a much larger error.

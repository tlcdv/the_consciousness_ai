# RIIU Upstream License Decision

**Date:** 2026-05-16
**Decision:** Vendor `AutoPhiSurrogate` from upstream RIIU with Apache-2.0 attribution.

## Sources verified this session

| Source | URL | License | Retrieval date |
|--------|-----|---------|----------------|
| RIIU GitHub repo | https://github.com/ReFractals/RIIU | Apache-2.0 | 2026-05-16 |
| RIIU repo LICENSE file | https://raw.githubusercontent.com/ReFractals/RIIU/main/LICENSE | Apache License 2.0 (full text confirmed) | 2026-05-16 |
| RIIU paper (arxiv) | https://arxiv.org/abs/2506.13825 | CC BY-NC-SA 4.0 | 2026-05-16 |

The code and the paper text have separate licenses, which is standard. The code is permissive (Apache-2.0). The paper text is non-commercial share-alike (CC BY-NC-SA 4.0).

## Compatibility analysis

This project's `LICENSE.md` is a custom "Non-Commercial Open Source License" prohibiting commercial use without prior written permission. Apache-2.0 is permissive and explicitly compatible with any downstream license, including non-commercial ones. The combined work as a whole is governed by this project's non-commercial license; the vendored Apache-2.0 portions retain their upstream license terms.

Required obligations under Apache-2.0:

1. Preserve the Apache-2.0 LICENSE text alongside the vendored code.
2. Preserve copyright notices from the upstream file.
3. Mark modifications as such.
4. Include a NOTICE file if upstream provides one (none observed).

## Corrected CLAUDE.md claim

CLAUDE.md (2026-04-06 entry and 2026-05-14 entry) describes RIIU as "MIT license". This is incorrect. The actual license is Apache-2.0. Both are permissive and compatible with this project, but the specific identifier in CLAUDE.md should be updated when the next session log entry is written.

## Chosen path

**Vendor** the `AutoPhiSurrogate` class verbatim (~30 lines of substantive code) into `models/evaluation/phi_riiu.py`. Add a header comment block citing:

- Source: https://github.com/ReFractals/RIIU/blob/main/riiu.py
- Paper: N'guessan and Karambal, "The Reflexive Integrated Information Unit: A Differentiable Primitive for Artificial Consciousness", arxiv:2506.13825
- License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)
- Modifications: wrapped in a `RIIUPhi` adapter class with `push`, `compute`, `compute_value`, `reset`, and `is_warm` API to match the project's sliding-window integration pattern.

Copy the upstream Apache-2.0 LICENSE text to `third_party/RIIU_LICENSE` so the obligation is met outside the source file header.

## Files affected by this decision

- `models/evaluation/phi_riiu.py` (new): vendored `AutoPhiSurrogate` plus `RIIUPhi` adapter.
- `third_party/RIIU_LICENSE` (new): full Apache-2.0 text from upstream.
- `models/evaluation/__init__.py`: add `from .phi_riiu import RIIUPhi`.

## Rejected alternatives

- **Clean-room reimplementation from the paper formula alone:** rejected because the upstream license is permissive and vendoring is cheaper and reduces risk of subtle implementation drift.
- **No license consideration before coding:** rejected explicitly. CLAUDE.md misidentified the license; verifying first prevented a CC BY-NC-SA mistake.

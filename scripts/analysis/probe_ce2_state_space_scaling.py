"""
Does CE 2.0 depend on state-space size when the macro structure is held fixed?

The training loop divides the workspace CE 2.0 (8 states) by the gate CE 2.0
(243 states) and logs it as `ce2_ratio`, which carries the whole of the CE2-1
onset prediction. That division is only meaningful if CE 2.0 is comparable
across state-space sizes. This probe tests that directly.

METHOD. Hold the macro structure fixed (k equivalency classes, the thing CE 2.0
claims to detect) and vary only the number of microstates n. A metric that is
comparable across cardinality should return the same value for the same macro
structure at any n. Block sizes may be uneven; each uniform block
(1/m) * J_m contributes exactly one singular value of 1 regardless of m, so the
family is well defined for any n >= k.

PRE-STATED GATE (written before any number from this probe was read, per the
convention in docs/results/gate_binning_2026_07.md):

    PASS, comparable:  at fixed k, |CE(n=243) - CE(n=8)| <= 0.05, so `ce2_ratio`
                       is reading macro structure and not cardinality.
    FAIL, confounded:  at fixed k, |CE(n=243) - CE(n=8)| > 0.05, so a `ce2_ratio`
                       away from 1.0 can be produced by state-count alone and
                       CE2-1 cannot be read from it as written.

The 0.05 tolerance is set against the effect CE2-1 needs to detect: the pilot
recorded `ce2_ratio` at 0.7551, i.e. a claimed 24 percent departure from parity.
A cardinality artifact of comparable size would be indistinguishable from it.

CONTROL. The uniform-block family has a closed form, so every measured value is
checked against it rather than trusted:

    CE(n, k) = 1 - (k-1)/(n-1)

Already pinned independently in tests/test_causal_emergence_svd.py, which asserts
CE = 6/7 at (n=8, k=2) and CE = 0.75 at (n=9, k=3). Both agree with the closed
form. A leaky variant is also scanned, because a knife-edge result that holds only
for exactly block-diagonal matrices would not justify a verdict.

Run:
    python -m scripts.analysis.probe_ce2_state_space_scaling
"""
from __future__ import annotations

import numpy as np

from models.evaluation.causal_emergence_svd import compute_ce2_from_tpm

# The two cardinalities the training loop actually divides.
WORKSPACE_STATES = 8
GATE_STATES = 243
GATE_TOLERANCE = 0.05


def uniform_block_tpm(n_states: int, n_blocks: int) -> np.ndarray:
    """k uniform blocks over n states, row-stochastic. Blocks may be uneven."""
    if n_blocks < 1 or n_states < n_blocks:
        raise ValueError(f"need 1 <= n_blocks <= n_states, got {n_blocks}, {n_states}")
    tpm = np.zeros((n_states, n_states), dtype=np.float64)
    for block in np.array_split(np.arange(n_states), n_blocks):
        tpm[np.ix_(block, block)] = 1.0 / len(block)
    return tpm


def leaky_block_tpm(n_states: int, n_blocks: int, leak: float) -> np.ndarray:
    """Block structure softened toward all-to-all uniform by `leak` in [0, 1]."""
    if not 0.0 <= leak <= 1.0:
        raise ValueError(f"leak must be in [0, 1], got {leak}")
    uniform = np.full((n_states, n_states), 1.0 / n_states, dtype=np.float64)
    return (1.0 - leak) * uniform_block_tpm(n_states, n_blocks) + leak * uniform


def closed_form_block_ce2(n_states: int, n_blocks: int) -> float:
    """
    Analytic CE 2.0 for the uniform-block family.

    A block-diagonal TPM of k uniform blocks has k singular values equal to 1 and
    n-k equal to 0. Discarding the trivial sigma_1 leaves gamma_star = (k-1)/(n-1)
    and sigma_2 = 1, so CE = 1 - (k-1)/(n-1). Independent of block sizes.
    """
    if n_states < 3 or n_blocks < 2:
        return 0.0
    return 1.0 - (n_blocks - 1) / (n_states - 1)


def scan_cardinality(n_blocks: int, state_counts: list[int],
                     leak: float = 0.0) -> list[dict]:
    """CE 2.0 at each state count with the macro structure held at n_blocks."""
    readings = []
    for n_states in state_counts:
        tpm = leaky_block_tpm(n_states, n_blocks, leak) if leak else \
            uniform_block_tpm(n_states, n_blocks)
        ce2 = compute_ce2_from_tpm(tpm)
        readings.append({
            "n_states": n_states,
            "ce2": ce2.causal_emergence,
            "closed_form": closed_form_block_ce2(n_states, n_blocks),
            "complexity": ce2.emergent_complexity,
        })
    return readings


def verify_against_closed_form(readings: list[dict], tol: float = 1e-9) -> None:
    """Raise if any measured value departs from the analytic control."""
    for row in readings:
        gap = abs(row["ce2"] - row["closed_form"])
        if gap > tol:
            raise AssertionError(
                f"measured CE 2.0 {row['ce2']:.9f} at n={row['n_states']} departs "
                f"from closed form {row['closed_form']:.9f} by {gap:.2e}"
            )


def print_scan(title: str, readings: list[dict]) -> None:
    print(f"\n{title}")
    print(f"{'n_states':>9} | {'CE 2.0':>10} | {'closed form':>12} | {'complexity':>10}")
    print("-" * 51)
    for row in readings:
        print(f"{row['n_states']:>9} | {row['ce2']:>10.6f} | "
              f"{row['closed_form']:>12.6f} | {row['complexity']:>10}")


def report_verdict(n_blocks: int, readings: list[dict]) -> bool:
    """Apply the pre-stated gate. Returns True on PASS (comparable)."""
    by_n = {row["n_states"]: row["ce2"] for row in readings}
    small, large = by_n[WORKSPACE_STATES], by_n[GATE_STATES]
    spread = abs(large - small)
    print(f"\nAt fixed macro structure k={n_blocks}:")
    print(f"  CE 2.0 at n={WORKSPACE_STATES:<4} (workspace cardinality) = {small:.6f}")
    print(f"  CE 2.0 at n={GATE_STATES:<4} (gate cardinality)      = {large:.6f}")
    print(f"  spread                                     = {spread:.6f}")
    print(f"  implied ce2_ratio for IDENTICAL structure   = {small / large:.6f}")
    passed = spread <= GATE_TOLERANCE
    print(f"\n  gate: spread <= {GATE_TOLERANCE} -> {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> None:
    print(__doc__.split("Run:")[0].strip())
    state_counts = [WORKSPACE_STATES, 16, 32, 64, 128, GATE_STATES]

    exact = scan_cardinality(2, state_counts)
    verify_against_closed_form(exact)
    print_scan("Uniform blocks, k=2 (control: matches closed form exactly)", exact)
    passed = report_verdict(2, exact)

    for n_blocks in (3, 4):
        scan = scan_cardinality(n_blocks, state_counts)
        verify_against_closed_form(scan)
        print_scan(f"Uniform blocks, k={n_blocks}", scan)

    for leak in (0.1, 0.5):
        scan = scan_cardinality(2, state_counts, leak=leak)
        print_scan(f"Leaky blocks, k=2, leak={leak} (closed form not applicable)", scan)

    print(f"\nVERDICT: ce2_ratio is "
          f"{'COMPARABLE' if passed else 'CONFOUNDED'} across state-space size.")


if __name__ == "__main__":
    main()

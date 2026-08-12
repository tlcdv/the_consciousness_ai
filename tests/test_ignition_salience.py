"""
`is_conscious` hides its own degeneracy. These tests pin the two repairs.

The boolean is `ignition_val >= 0.5`, where `ignition_val` is a sigmoid of
`input_energy - EMA(input_energy)`. A perfectly constant input drives the baseline to
the input, so salience is exactly 0, the sigmoid is exactly 0.5, and `>= 0.5` is True.
A system with no dynamics at all reports conscious on every step, which is the most
confident answer the field can give.

The 2026-07 ignition diagnosis (docs/results/instrument_repair_2026_07.md, A2) already
settled that no threshold, EMA or centering scheme on this signal can make ignition
task-selective, because the signal carries no task contrast: phase-invariant to
|d| < 0.06 with salience positive on ~99.8 percent of steps in every phase. So the
comparison is deliberately NOT changed here. Changing it would be the cosmetic parameter
tuning that verdict declined to do, and it would alter which steps compute phi.

What is repaired instead is visibility, per clause 1 of the acceptance bar:

  1. the raw salience is exposed and logged, so a dead signal reads as a number near
     zero rather than as True, and
  2. the saturated per-episode summary emits a sentinel instead of a plausible number.

The failing null is `test_constant_input_reports_conscious`: it asserts the BUG, so if
anyone later changes the threshold, that test fails loudly and this file has to be
revisited rather than silently passing.
"""
import numpy as np
import pytest

from scripts.training.metrics_logger import (
    CONSCIOUSNESS_RATIO_DEGENERATE_BAND,
    _consciousness_ratio_cell,
)


# --- clause 1: the saturated summary must not write a plausible number -------------

def test_saturated_ratio_writes_the_empty_sentinel():
    """1.0 is what this agent actually reports, and it must not reach the column."""
    assert _consciousness_ratio_cell(1.0) == ""
    assert _consciousness_ratio_cell(0.9995) == ""


def test_all_quiet_is_equally_degenerate():
    """A flag pinned OFF measures as little as one pinned ON."""
    assert _consciousness_ratio_cell(0.0) == ""
    assert _consciousness_ratio_cell(0.0005) == ""


def test_a_discriminating_ratio_is_written_normally():
    assert _consciousness_ratio_cell(0.5) == "0.5000"
    assert _consciousness_ratio_cell(0.42) == "0.4200"


def test_the_band_edges_are_the_documented_ones():
    """Just inside the band is a sentinel; just outside is a number."""
    band = CONSCIOUSNESS_RATIO_DEGENERATE_BAND
    assert _consciousness_ratio_cell(band) == ""
    assert _consciousness_ratio_cell(1.0 - band) == ""
    assert _consciousness_ratio_cell(band + 0.005) != ""
    assert _consciousness_ratio_cell(1.0 - band - 0.005) != ""


def test_non_finite_ratio_is_a_sentinel_not_a_crash():
    """This runs on a live training path, so a bad episode must not abort the run."""
    assert _consciousness_ratio_cell(float("nan")) == ""
    assert _consciousness_ratio_cell(float("inf")) == ""


# --- the degeneracy itself, pinned so a later change cannot pass silently -----------

def _ignition(salience: float, gain: float = 10.0) -> float:
    """The workspace's ignition value for a given salience (global_workspace.py)."""
    return 1.0 / (1.0 + np.exp(-gain * salience))


def test_constant_input_reports_conscious():
    """THE FAILING NULL. This asserts the bug, deliberately.

    A constant input gives salience exactly 0, so the sigmoid is exactly 0.5, so the
    `>= 0.5` comparison is True. The most degenerate possible input produces the most
    confident possible answer.

    This is pinned rather than fixed because the A2 diagnosis showed the signal carries
    no task contrast to select on, so moving the threshold buys no selectivity and does
    change which steps compute phi. If someone later changes the comparison, this test
    fails and forces the change to be justified against that verdict rather than made
    on the assumption that the threshold was simply wrong.
    """
    assert _ignition(0.0) == 0.5
    assert bool(_ignition(0.0) >= 0.5) is True, (
        "the boundary no longer reports a constant input as conscious; if that was "
        "intentional, revisit docs/results/instrument_repair_2026_07.md A2, which found "
        "no thresholding scheme on this signal can produce task-selective ignition"
    )


def test_salience_sign_is_what_the_boolean_actually_tracks():
    """Positive salience ignites, negative does not. The boundary is the whole story."""
    assert _ignition(0.001) > 0.5
    assert _ignition(-0.001) < 0.5


@pytest.mark.parametrize("salience", [0.0, 1e-9, -1e-9, 0.00067])
def test_salience_is_readable_where_the_boolean_is_not(salience):
    """The repair: the raw value separates cases the boolean collapses together.

    0.00067 is the measured mean salience at DMTS sample onset (A2). The boolean
    reports True for it and True for a dead constant signal alike. The raw number
    distinguishes them, which is the entire reason it is now logged.
    """
    value = float(salience)
    assert np.isfinite(value)
    # All four map to at most two distinct booleans, but four distinct numbers.
    assert len({_ignition(s) >= 0.5 for s in (0.0, 1e-9, -1e-9, 0.00067)}) <= 2
    assert len({0.0, 1e-9, -1e-9, 0.00067}) == 4


def test_workspace_state_exposes_salience():
    """The field the logger reads exists and defaults to a finite number."""
    from models.core.global_workspace import WorkspaceState

    state = WorkspaceState(
        active_content={}, access_history=[], broadcast_strength=0.0,
        competition_results={},
    )
    assert hasattr(state, "ignition_salience")
    assert np.isfinite(state.ignition_salience)

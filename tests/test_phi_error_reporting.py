"""
How IITMetrics reports a phi it could not compute.

1. A pyphi exception used to become phi 0.0 labelled "pyphi", the same label as a
   computed value, with the error logged at DEBUG through `self.logger`, which the
   consciousness monitor sets to a MetricsLogger. Analysis that keeps
   `phi_method == "pyphi"` rows (scripts/analysis/probe_scalar_content.py) counted
   those zeros as measurements. The row is now labelled "pyphi_error"; the value
   stays 0.0 so no column changes type.
2. The binarization floor warning said a dimension whose median sits below its floor
   "may be saturated to 1". The threshold is then the floor, and every value below it
   binarizes to 0, so the bit is 0 on at least half of the steps.

pyphi 1.2.0 does not import on Python 3.10, so these tests install a stand-in module.
"""
from __future__ import annotations

import logging
import os
import sys
import types
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core.consciousness_gating import GatingState
from models.evaluation import iit_phi
from models.evaluation.iit_phi import IITMetrics


def _stand_in_pyphi(sia):
    return types.SimpleNamespace(
        Network=lambda tpm, cm=None: ("network", tpm, cm),
        Subsystem=lambda network, state: ("subsystem", network, state),
        compute=types.SimpleNamespace(sia=sia),
        config=types.SimpleNamespace(PROGRESS_BARS=True, PARALLEL_CUT_EVALUATION=True),
    )


def _gate_states(n, adaptation=0.01):
    rng = np.random.default_rng(0)
    for _ in range(n):
        attention, stability, coherence, confidence = rng.uniform(0.0, 1.0, 4)
        yield GatingState(attention_level=attention, stability_score=stability,
                          adaptation_rate=adaptation, meta_memory_coherence=coherence,
                          narrator_confidence=confidence)


def _last_result(metrics, n=8):
    result = None
    for state in _gate_states(n):
        result = metrics.compute_phi_from_gate_state(state)
    return result


def test_a_pyphi_exception_is_labelled_pyphi_error(monkeypatch, caplog):
    def failing_sia(subsystem):
        raise RuntimeError("TPM is not conditionally independent")

    monkeypatch.setattr(iit_phi, "pyphi", _stand_in_pyphi(failing_sia))
    metrics = IITMetrics()
    with caplog.at_level(logging.WARNING, logger=iit_phi.__name__):
        result = _last_result(metrics)

    assert result.method == "pyphi_error"
    assert result.phi == 0.0
    assert metrics.phi_error_count >= 1
    assert "conditionally independent" in metrics.last_phi_error
    warnings_logged = [r for r in caplog.records
                       if r.levelno == logging.WARNING and r.name == iit_phi.__name__]
    assert len(warnings_logged) == 1, "the first failure warns; later ones are counted"


def test_a_computed_phi_keeps_the_pyphi_label(monkeypatch):
    monkeypatch.setattr(iit_phi, "pyphi",
                        _stand_in_pyphi(lambda subsystem: types.SimpleNamespace(phi=0.25)))
    metrics = IITMetrics()
    result = _last_result(metrics)

    assert result.method == "pyphi"
    assert result.phi == 0.25
    assert metrics.last_phi_error is None
    assert metrics.phi_error_count == 0


def test_a_success_after_a_failure_is_labelled_pyphi(monkeypatch):
    calls = {"n": 0}

    def flaky_sia(subsystem):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first call fails")
        return types.SimpleNamespace(phi=0.5)

    monkeypatch.setattr(iit_phi, "pyphi", _stand_in_pyphi(flaky_sia))
    metrics = IITMetrics()
    methods = [metrics.compute_phi_from_gate_state(s).method for s in _gate_states(8)]
    sampled = [m for m in methods if m != "insufficient_data"]
    assert sampled[0] == "pyphi_error"
    assert set(sampled[1:]) == {"pyphi"}


def test_floor_warning_says_the_dimension_is_pinned_to_zero():
    metrics = IITMetrics()
    below_floor = 1e-6  # the adaptation floor is 1e-5
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bits = [metrics.update_from_gate_state(s)[2]
                for s in _gate_states(60, adaptation=below_floor)]

    messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    adaptation = [m for m in messages if "'adaptation'" in m]
    assert len(adaptation) == 1
    assert "binarizes to 0" in adaptation[0]
    assert "saturated to 1" not in adaptation[0]
    assert set(bits) == {0}

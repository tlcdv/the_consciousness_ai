"""
Regression tests for the placeholder metrics retired 2026-07-29.

WHY THIS FILE EXISTS. An audit found 14 functions that returned a number they never
computed: a phi "approximation" that was a module count plus `np.random.rand()`, a
GWT ignition detector that fired on a 20% coin flip, three self-awareness scores that
were scaled random floats, a dashboard serving `random.uniform(0, 1)` as a
consciousness score, and several methods returning a hardcoded 0.0, 0.5 or 1.0 while
their names promised a measurement.

None was wired into the training loop, so no committed verdict came from them. They
were retired because they are landmines: they sit in `models/evaluation/` beside the
real instruments, and one of them (`PerturbationTester`) was very nearly used before
it was noticed to return `np.random.rand() * 10.0`.

These tests are the durable protection. A skill file saying "do not fabricate
numbers" cannot stop someone restoring a `return 0.0`; a failing test can. If you are
implementing one of these for real, delete its test here in the SAME commit that adds
the implementation and its own tests. Do not weaken an assertion to make it pass.

See the write-code skill, rule 0.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestEvaluationPlaceholdersRaise:
    """models/evaluation/: the directory that holds the real instruments."""

    def test_phi_approximation_raises_instead_of_returning_noise(self):
        from models.evaluation.consciousness_metrics import (
            IntegratedInformationCalculator,
        )

        calculator = IntegratedInformationCalculator({})
        with pytest.raises(NotImplementedError, match="iit_phi"):
            calculator.calculate_phi_approximation({"perception": 1}, [])

    def test_gwt_ignition_raises_instead_of_flipping_a_coin(self):
        from models.evaluation.consciousness_metrics import GlobalWorkspaceTracker

        tracker = GlobalWorkspaceTracker({})
        with pytest.raises(NotImplementedError, match="run_competition"):
            tracker.detect_ignition({"attention_level": 0.99}, [])

    def test_self_awareness_raises_instead_of_returning_random_scores(self):
        from models.evaluation.consciousness_metrics import SelfAwarenessMonitor

        monitor = SelfAwarenessMonitor({}, None, None)
        with pytest.raises(NotImplementedError):
            monitor.evaluate_self_awareness({"self_model": {}, "agent_status": "ok"})

    def test_pci_placeholder_stays_retired(self):
        # Retired 2026-07-28, one day before the rest. Pinned here so the whole
        # family is covered by one file.
        from models.evaluation.consciousness_metrics import PerturbationTester

        tester = PerturbationTester({}, None)
        with pytest.raises(NotImplementedError, match="perturbational_complexity"):
            tester.calculate_pci_approximation({})

    def test_dashboard_does_not_serve_random_numbers_as_metrics(self):
        # flask is an optional dependency and is not installed in the default env.
        pytest.importorskip("flask")
        from models.evaluation.consciousness_dashboard import fetch_metrics

        with pytest.raises(NotImplementedError, match="metrics.csv"):
            fetch_metrics()


class TestMemoryAndFusionPlaceholdersRaise:
    def test_retrieval_accuracy_raises(self):
        from models.memory.memory_core import MemoryCore

        core = MemoryCore.__new__(MemoryCore)  # no __init__: this is a pure stub check
        with pytest.raises(NotImplementedError):
            core._calculate_retrieval_accuracy()

    def test_temporal_consistency_raises(self):
        from models.memory.memory_core import MemoryCore

        core = MemoryCore.__new__(MemoryCore)
        with pytest.raises(NotImplementedError):
            core._calculate_temporal_consistency()

    def test_narrative_alignment_raises(self):
        from models.memory.memory_core import MemoryCore

        core = MemoryCore.__new__(MemoryCore)
        with pytest.raises(NotImplementedError):
            core._calculate_narrative_alignment()

    def test_semantic_and_retrieval_quality_raise(self):
        from models.memory.memory_integration import MemoryIntegrationCore

        integration = MemoryIntegrationCore.__new__(MemoryIntegrationCore)
        with pytest.raises(NotImplementedError):
            integration._evaluate_semantic_quality()
        with pytest.raises(NotImplementedError):
            integration._evaluate_retrieval_quality()

    def test_cross_modal_alignment_no_longer_reports_perfect(self):
        from models.fusion.emotional_memory_fusion import EmotionalMemoryFusion

        fusion = EmotionalMemoryFusion.__new__(EmotionalMemoryFusion)
        with pytest.raises(NotImplementedError, match="cosine"):
            fusion._calculate_alignment([])

    def test_generative_coherence_no_longer_reports_perfect(self):
        from models.generative.generative_emotional_core import GenerativeEmotionalCore

        core = GenerativeEmotionalCore.__new__(GenerativeEmotionalCore)
        with pytest.raises(NotImplementedError):
            core._evaluate_coherence("any response", None)


class TestEthicsFilterFailsClosed:
    """
    The ethics fallback is the one case where the fix is a value, not a raise.

    Before 2026-07-29 a failed AsimovComplianceFilter initialization installed a dummy
    whose is_compliant returned True unconditionally, so a safety check that could not
    run approved every action. A safety check that cannot run must deny.
    """

    def test_fallback_filter_denies_rather_than_approves(self):
        """
        Source-level check, and it says so. The fallback class is defined inside
        ConsciousnessCore.__init__'s exception handler, so reaching it behaviourally
        would mean forcing AsimovComplianceFilter to fail during a full core
        construction. That is a heavy and brittle setup for a one-line invariant, so
        this reads the source instead and asserts the shape directly.
        """
        import inspect

        from models.core.consciousness_core import ConsciousnessCore

        source = inspect.getsource(ConsciousnessCore.__init__)

        assert "FailClosedEthicsFilter" in source, (
            "the ethics fallback class was renamed or removed"
        )
        assert "def is_compliant(self, action, state): return True" not in source, (
            "the fail-OPEN ethics dummy is back: a compliance filter that could not "
            "initialize would approve every action"
        )

        fallback_body = source.split("class FailClosedEthicsFilter", 1)[1]
        fallback_body = fallback_body.split("self.ethics_filter =", 1)[0]
        assert "return False" in fallback_body, (
            "the ethics fallback no longer denies"
        )
        assert "return True" not in fallback_body


def test_no_new_fabricated_numbers_in_evaluation():
    """
    Tripwire matching check 1 of the audit-project skill.

    Scans models/evaluation/ for random-number generation in EXECUTABLE CODE. It
    tokenizes rather than grepping text, because the retirement messages quote the
    old fabricating expressions verbatim and a text grep flags those quotes. Comments
    and string literals are therefore excluded by construction.

    The known-benign hits are `if __name__ == '__main__'` demo sections in
    gnw_metrics.py and subjective_testing_suite.py, which build explicitly named Mock
    objects. Anything else is a fabricated measurement.
    """
    import io
    import tokenize
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "models" / "evaluation"
    fabricators = {"rand", "uniform", "normal", "random", "randint"}

    offenders = []
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        demo_starts_at = next(
            (n for n, line in enumerate(text.splitlines(), 1)
             if line.startswith("if __name__")),
            None,
        )
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        except (tokenize.TokenError, IndentationError):
            continue

        names = [t for t in tokens if t.type == tokenize.NAME]
        for index, token in enumerate(names[:-1]):
            if token.string != "random":
                continue
            following = names[index + 1].string
            if following not in fabricators:
                continue
            line_number = token.start[0]
            if demo_starts_at is not None and line_number >= demo_starts_at:
                continue
            offenders.append(
                f"{path.name}:{line_number}: "
                f"{text.splitlines()[line_number - 1].strip()}"
            )

    assert offenders == [], (
        "Fabricated numbers found in executable code under models/evaluation/. "
        "See the write-code skill, rule 0:\n  " + "\n  ".join(offenders)
    )

"""models/predictive/emotional_predictor.py imported a ConsciousnessState class that
models/core/consciousness_core.py does not define, so the module could not be
imported at all. The class inside is still unusable (see the audit notes); this pins
only that the module imports."""
from __future__ import annotations

import importlib


def test_the_module_imports():
    module = importlib.import_module("models.predictive.emotional_predictor")
    assert hasattr(module, "EmotionalPredictor")

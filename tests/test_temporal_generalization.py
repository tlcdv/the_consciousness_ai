"""
Tests for models/evaluation/temporal_generalization.py.

Constructed data only. A code that is present at every step with the same
geometry must generalize across the whole block. A code present only in a short
window must decode inside that window and fall to chance outside it. Labels
shuffled across trials must decode at chance, which is what the null measures.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.evaluation.temporal_generalization import (  # noqa: E402
    classify_dynamics,
    label_shuffle_threshold,
    temporal_generalization,
)

TRIALS, T, D, CLASSES = 90, 12, 10, 3


def _data(active_steps, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(CLASSES), TRIALS // CLASSES)
    means = 3.0 * rng.standard_normal((CLASSES, D))
    X = rng.standard_normal((TRIALS, T, D))
    for t in active_steps:
        X[:, t, :] += means[y]
    return X, y


def test_stable_code_generalizes_across_the_whole_block():
    # Arrange
    X, y = _data(range(T))

    # Act
    acc = temporal_generalization(X, y, n_splits=3, seed=0)

    # Assert
    assert acc.shape == (T, T)
    assert acc.min() > 0.9


def test_transient_code_decodes_only_inside_its_window():
    # Arrange: the class signal exists at steps 2, 3 and 4 only.
    X, y = _data([2, 3, 4])

    # Act
    acc = temporal_generalization(X, y, n_splits=3, seed=0)

    # Assert
    assert acc[2:5, 2:5].min() > 0.9
    assert acc[8:, 8:].max() < 0.6          # chance is 1/3
    assert acc[3, 9] < 0.6                  # a decoder from step 3 fails at step 9


def test_shuffled_labels_sit_at_chance():
    # Arrange
    X, y = _data(range(T))

    # Act
    threshold = label_shuffle_threshold(X, y, n_splits=3, n_perm=20, seed=0)

    # Assert: the p95 of a label-shuffled decoder stays near 1/3.
    assert threshold < 0.55


def test_classify_dynamics_names_the_two_patterns():
    # Arrange
    stable = temporal_generalization(*_data(range(T)), n_splits=3, seed=0)
    brief = temporal_generalization(*_data([0, 1, 2]), n_splits=3, seed=0)

    # Act and Assert: the window of interest is steps 0..11.
    assert classify_dynamics(stable, threshold=0.55, onset=0, length=T) == "SUSTAINED"
    assert classify_dynamics(brief, threshold=0.55, onset=0, length=T) == "TRANSIENT"

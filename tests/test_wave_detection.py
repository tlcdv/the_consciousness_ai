"""
Tests for models/evaluation/wave_detection.py.

Every expectation follows from the definition of the measure on a constructed
field, never from a run.

  PGD            a plane wave has one gradient direction everywhere, so
                 |mean gradient| / mean |gradient| = 1 exactly. Independent
                 noise has random directions, so PGD falls toward 0.
  singularities  the phase winding around a closed loop is 2 pi times the
                 number of phase singularities inside it. One spiral has
                 exactly one. A plane wave has none.
  shuffle null   moving whole cell time series to random positions keeps every
                 cell's signal and destroys the spatial layout, so a plane wave
                 must sit above that null and noise must not.

Units are cells per step. No Hz and no metres per second.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.evaluation.wave_detection import (  # noqa: E402
    WaveResult,
    count_phase_singularities,
    measure_waves,
    phase_field,
    phase_gradient_directionality,
    project_channels,
)

T, H, W = 256, 16, 16
TEMPORAL = 0.05   # cycles per step
SPATIAL = 0.1     # cycles per cell


def plane_wave(direction=(1.0, 0.0)) -> np.ndarray:
    t = np.arange(T)[:, None, None]
    y, x = np.mgrid[0:H, 0:W]
    k = 2 * np.pi * SPATIAL
    return np.cos(2 * np.pi * TEMPORAL * t - k * (direction[0] * x + direction[1] * y))


def spiral_wave() -> np.ndarray:
    t = np.arange(T)[:, None, None]
    y, x = np.mgrid[0:H, 0:W]
    angle = np.arctan2(y - 7.5, x - 7.5)
    return np.cos(2 * np.pi * TEMPORAL * t - angle)


def noise(seed=0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((T, H, W))


def test_plane_wave_pgd_is_one():
    # Arrange
    phases = phase_field(plane_wave())

    # Act
    pgd = phase_gradient_directionality(phases)

    # Assert: edge samples of the Hilbert transform are excluded by the caller,
    # so check the interior of the record.
    assert np.all(pgd[32:-32] > 0.99)


def test_diagonal_plane_wave_pgd_is_one():
    # Arrange
    phases = phase_field(plane_wave(direction=(0.6, 0.8)))

    # Act
    pgd = phase_gradient_directionality(phases)

    # Assert
    assert np.median(pgd[32:-32]) > 0.99


def test_independent_noise_pgd_is_low():
    # Arrange: 14 x 14 interior gradients with random directions. The expected
    # |mean unit vector| of n random directions is about sqrt(pi / (4 n)).
    phases = phase_field(noise())

    # Act
    pgd = phase_gradient_directionality(phases)

    # Assert
    assert np.median(pgd) < 0.15


def test_spiral_has_one_singularity_and_plane_has_none():
    # Arrange
    spiral = phase_field(spiral_wave())[T // 2]
    plane = phase_field(plane_wave())[T // 2]

    # Act and Assert
    assert count_phase_singularities(spiral) == 1
    assert count_phase_singularities(plane) == 0


def test_spiral_pgd_is_low_which_is_why_singularities_are_reported():
    # Arrange: rotating waves have gradient directions that cancel.
    phases = phase_field(spiral_wave())

    # Act
    pgd = phase_gradient_directionality(phases)

    # Assert
    assert np.median(pgd[32:-32]) < 0.1


def test_measure_waves_separates_plane_wave_from_its_shuffle_null():
    # Act
    wave = measure_waves(plane_wave(), n_surrogates=20, seed=0)
    flat = measure_waves(noise(1), n_surrogates=20, seed=0)

    # Assert
    assert isinstance(wave, WaveResult)
    assert wave.pgd_mean > wave.null_p95
    assert flat.pgd_mean <= flat.null_p95 + 0.02


def test_measure_waves_counts_the_spiral():
    # Act
    spiral = measure_waves(spiral_wave(), n_surrogates=5, seed=0)

    # Assert: one singularity at every interior step.
    assert spiral.singularities_median == 1


def test_measure_waves_refuses_a_constant_field():
    # Arrange: no phase can be defined for a signal that never changes.
    constant = np.ones((T, H, W))

    # Act and Assert
    with pytest.raises(ValueError):
        measure_waves(constant, n_surrogates=2)


def test_project_channels_recovers_a_wave_carried_on_one_direction():
    # Arrange: a plane wave written onto a fixed 8-D direction plus small noise.
    rng = np.random.default_rng(3)
    direction = rng.standard_normal(8)
    h = plane_wave()[:, None, :, :] * direction[None, :, None, None]
    h = h + 0.01 * rng.standard_normal(h.shape)

    # Act
    field = project_channels(h)

    # Assert: the projection is the wave up to sign and scale.
    r = np.corrcoef(field.ravel(), plane_wave().ravel())[0, 1]
    assert abs(r) > 0.99

"""
Tests for the DMTS near-threshold manipulation (sample_contrast / sample_noise).

The first test is the load-bearing one: at the defaults, DMTSEnv must render
byte-for-byte what it rendered before this parameter existed, so that every DMTS
result already in docs/results/ stays comparable. The rest pin the properties the
manipulation has to have to be usable as a matched-stimulus contrast:

  - lowering contrast reduces the sample's deviation from the background, and only
    during the sample phase (fixation, delay and choice are untouched),
  - raising noise does NOT change which trials are generated, so the same stimulus
    sequence can be replayed at several visibility levels,
  - the whole thing is reproducible under a fixed seed.

Expected values are derived from the blend definition, never from a run.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulations.environments._stimulus_renderer import BACKGROUND_GRAY  # noqa: E402
from simulations.environments.dmts_env import DMTSEnv  # noqa: E402


def _sample_frame(env: DMTSEnv, seed: int = 42) -> np.ndarray:
    """Step to the first sample-phase frame and return it."""
    obs, info = env.reset(seed=seed)
    for _ in range(env.max_steps_per_trial):
        if info["phase"] == "sample":
            return obs
        obs, _, _, _, info = env.step(0)
    raise AssertionError("never reached the sample phase")


def _trial_signature(env: DMTSEnv, seed: int = 42, n_steps: int = 300) -> list[tuple]:
    """The stimulus sequence, independent of how visible it was rendered."""
    _, info = env.reset(seed=seed)
    signature = []
    for _ in range(n_steps):
        _, _, terminated, _, info = env.step(0)
        signature.append(
            (
                info["phase"],
                info["sample_shape"],
                info["sample_color"],
                info["sample_size"],
                info["target_position"],
                info["delay_length"],
            )
        )
        if terminated:
            break
    return signature


class TestDefaultsAreBitIdentical:
    """The guarantee that keeps prior DMTS results comparable."""

    def test_default_construction_is_the_identity_transform(self):
        env = DMTSEnv()
        assert env.sample_contrast == 1.0
        assert env.sample_noise == 0.0

    def test_explicit_defaults_match_implicit_defaults_exactly(self):
        implicit = _sample_frame(DMTSEnv())
        explicit = _sample_frame(DMTSEnv(sample_contrast=1.0, sample_noise=0.0))
        assert np.array_equal(implicit, explicit)

    def test_full_episode_is_bit_identical_at_defaults(self):
        # Not just the sample frame: every frame of a whole episode.
        a, b = DMTSEnv(), DMTSEnv(sample_contrast=1.0, sample_noise=0.0)
        obs_a, _ = a.reset(seed=7)
        obs_b, _ = b.reset(seed=7)
        assert np.array_equal(obs_a, obs_b)
        for _ in range(200):
            obs_a, r_a, t_a, _, _ = a.step(0)
            obs_b, r_b, t_b, _, _ = b.step(0)
            assert np.array_equal(obs_a, obs_b)
            assert r_a == r_b
            if t_a or t_b:
                break

    def test_identity_path_returns_the_canvas_unmodified(self):
        env = DMTSEnv()
        canvas = np.full((8, 8, 3), 200, dtype=np.uint8)
        out = env._apply_sample_visibility(canvas)
        assert out is canvas


class TestContrast:
    def test_zero_contrast_erases_the_sample(self):
        frame = _sample_frame(DMTSEnv(sample_contrast=0.0))
        assert np.all(frame == BACKGROUND_GRAY)

    def test_lower_contrast_moves_the_sample_toward_background(self):
        deviations = []
        for contrast in (1.0, 0.75, 0.5, 0.25):
            frame = _sample_frame(DMTSEnv(sample_contrast=contrast)).astype(np.float64)
            deviations.append(np.abs(frame - BACKGROUND_GRAY).mean())
        assert deviations == sorted(deviations, reverse=True)

    def test_half_contrast_halves_the_deviation(self):
        full = _sample_frame(DMTSEnv()).astype(np.float64) - BACKGROUND_GRAY
        half = _sample_frame(DMTSEnv(sample_contrast=0.5)).astype(np.float64) - BACKGROUND_GRAY
        # Exact up to uint8 rounding of the blend.
        assert np.allclose(half, full * 0.5, atol=1.0)

    def test_contrast_is_clamped_to_unit_interval(self):
        assert DMTSEnv(sample_contrast=5.0).sample_contrast == 1.0
        assert DMTSEnv(sample_contrast=-2.0).sample_contrast == 0.0

    def test_only_the_sample_phase_is_degraded(self):
        # The choice array must stay fully visible: difficulty comes from what the
        # agent encoded, not from what it can currently see.
        faint = DMTSEnv(sample_contrast=0.1)
        crisp = DMTSEnv()
        obs_f, info_f = faint.reset(seed=13)
        obs_c, info_c = crisp.reset(seed=13)
        seen_sample = False
        for _ in range(200):
            if info_f["phase"] == "sample":
                seen_sample = True
                assert not np.array_equal(obs_f, obs_c)
            else:
                assert np.array_equal(obs_f, obs_c)
            obs_f, _, term, _, info_f = faint.step(0)
            obs_c, _, _, _, info_c = crisp.step(0)
            if term:
                break
        assert seen_sample


class TestNoise:
    def test_noise_perturbs_the_sample_frame(self):
        quiet = _sample_frame(DMTSEnv())
        noisy = _sample_frame(DMTSEnv(sample_noise=20.0))
        assert not np.array_equal(quiet, noisy)

    def test_noise_does_not_change_which_trials_are_generated(self):
        # This is what makes the manipulation a MATCHED-stimulus contrast: the same
        # trial sequence can be replayed at several visibility levels.
        baseline = _trial_signature(DMTSEnv(), seed=99)
        for noise in (5.0, 20.0, 60.0):
            assert _trial_signature(DMTSEnv(sample_noise=noise), seed=99) == baseline

    def test_contrast_does_not_change_which_trials_are_generated(self):
        baseline = _trial_signature(DMTSEnv(), seed=99)
        for contrast in (0.8, 0.4, 0.05):
            assert _trial_signature(DMTSEnv(sample_contrast=contrast), seed=99) == baseline

    def test_noise_is_reproducible_under_a_fixed_seed(self):
        first = _sample_frame(DMTSEnv(sample_noise=25.0), seed=5)
        second = _sample_frame(DMTSEnv(sample_noise=25.0), seed=5)
        assert np.array_equal(first, second)

    def test_negative_noise_is_floored_at_zero(self):
        assert DMTSEnv(sample_noise=-3.0).sample_noise == 0.0

    def test_output_stays_in_uint8_range(self):
        frame = _sample_frame(DMTSEnv(sample_contrast=0.5, sample_noise=200.0))
        assert frame.dtype == np.uint8
        assert frame.min() >= 0
        assert frame.max() <= 255


class TestInfoExposesTheManipulation:
    def test_info_reports_contrast_and_noise(self):
        env = DMTSEnv(sample_contrast=0.3, sample_noise=12.0)
        _, info = env.reset(seed=1)
        assert info["sample_contrast"] == pytest.approx(0.3)
        assert info["sample_noise"] == pytest.approx(12.0)

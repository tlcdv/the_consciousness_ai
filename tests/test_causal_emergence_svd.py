"""
Tests for the Causal Emergence 2.0 SVD heuristic
(models/evaluation/causal_emergence_svd.py).

All expected values are derived analytically from the singular-value structure of
the constructed TPMs, not copied from any external run:

  * A permutation matrix has all singular values equal to 1, so after discarding
    the trivial sigma_1 every remaining sigma equals gamma_star -> CE = 0 and no
    scale exceeds the mean -> complexity 0.
  * An all-to-all uniform matrix has rank 1: sigma = [1, 0, ..., 0] -> CE = 0,
    complexity 0.
  * k equal uniform blocks (k equivalency classes over k*m states) give exactly k
    singular values equal to 1 and the rest 0. Discarding sigma_1 leaves (k-1)
    ones above gamma_star, so emergent_complexity == k - 1. This is the paper's
    "more contributing scales = more complex" behaviour.
"""
import numpy as np
import pytest
import torch

from models.evaluation.causal_emergence_svd import (
    CE2Result,
    compute_ce2_from_tpm,
    compute_ce2_from_trajectories,
    frozen_trajectory_ce2_value,
    trajectory_degeneracy,
    latent_class_indices,
    build_latent_tpm,
    new_transition_counts,
    update_transition_counts,
    counts_to_tpm,
)
from models.evaluation.effective_information import _build_tpm


def _k_block_uniform_tpm(k: int, m: int) -> np.ndarray:
    """k uniform blocks of size m on the diagonal -> row-stochastic, rank k."""
    n = k * m
    tpm = np.zeros((n, n))
    for b in range(k):
        lo, hi = b * m, (b + 1) * m
        tpm[lo:hi, lo:hi] = 1.0 / m
    return tpm


def _cyclic_permutation_tpm(n: int) -> np.ndarray:
    tpm = np.zeros((n, n))
    for i in range(n):
        tpm[i, (i + 1) % n] = 1.0
    return tpm


class TestSVDHeuristicMath:
    def test_permutation_matrix_has_zero_emergence(self):
        # Identity (self-loops) and a cyclic shift are both permutations:
        # all singular values are 1, so CE = 0 and complexity = 0.
        for tpm in (np.eye(5), _cyclic_permutation_tpm(6)):
            res = compute_ce2_from_tpm(tpm)
            assert res.causal_emergence == pytest.approx(0.0, abs=1e-9)
            assert res.emergent_complexity == 0
            assert res.sigma[0] == pytest.approx(1.0, abs=1e-9)

    def test_uniform_all_to_all_has_zero_emergence(self):
        n = 5
        tpm = np.full((n, n), 1.0 / n)  # rank 1: sigma = [1, 0, 0, 0, 0]
        res = compute_ce2_from_tpm(tpm)
        assert res.causal_emergence == pytest.approx(0.0, abs=1e-9)
        assert res.gamma_star == pytest.approx(0.0, abs=1e-9)
        assert res.emergent_complexity == 0

    def test_two_block_system_is_top_heavy(self):
        # 2 uniform blocks of size 4 (n = 8): sigma = [1, 1, 0, 0, 0, 0, 0, 0].
        # Discard sigma_1 -> [1, 0, 0, 0, 0, 0, 0], gamma_star = 1/7,
        # CE = 1 - 1/7 = 6/7, exactly one sigma above the mean -> complexity 1.
        res = compute_ce2_from_tpm(_k_block_uniform_tpm(2, 4))
        assert res.n_states == 8
        assert res.gamma_star == pytest.approx(1.0 / 7.0, abs=1e-9)
        assert res.causal_emergence == pytest.approx(6.0 / 7.0, abs=1e-9)
        assert res.emergent_complexity == 1
        assert len(res.contributions) == 1

    def test_three_block_system_has_more_complexity(self):
        # 3 uniform blocks of size 3 (n = 9): sigma = [1, 1, 1, 0, ...].
        # Discard sigma_1 -> [1, 1, 0, 0, 0, 0, 0, 0], gamma_star = 2/8 = 0.25,
        # CE = 1 - 0.25 = 0.75, two sigma above the mean -> complexity 2.
        res = compute_ce2_from_tpm(_k_block_uniform_tpm(3, 3))
        assert res.gamma_star == pytest.approx(0.25, abs=1e-9)
        assert res.causal_emergence == pytest.approx(0.75, abs=1e-9)
        assert res.emergent_complexity == 2

    def test_complexity_grows_with_number_of_equivalency_classes(self):
        c2 = compute_ce2_from_tpm(_k_block_uniform_tpm(2, 4)).emergent_complexity
        c3 = compute_ce2_from_tpm(_k_block_uniform_tpm(3, 3)).emergent_complexity
        c4 = compute_ce2_from_tpm(_k_block_uniform_tpm(4, 3)).emergent_complexity
        assert c2 < c3 < c4  # 1 < 2 < 3

    def test_small_n_guard_returns_zeros(self):
        for tpm in (np.eye(1), np.eye(2), _k_block_uniform_tpm(1, 2)):
            res = compute_ce2_from_tpm(tpm)
            assert res.causal_emergence == 0.0
            assert res.emergent_complexity == 0
            assert res.n_states < 3

    def test_trivial_sigma1_at_least_one_for_row_stochastic(self):
        # sigma_1 >= spectral radius = 1 for any row-stochastic matrix.
        rng = np.random.default_rng(0)
        for _ in range(20):
            n = int(rng.integers(3, 12))
            m = rng.random((n, n))
            tpm = m / m.sum(axis=1, keepdims=True)
            res = compute_ce2_from_tpm(tpm)
            assert res.sigma[0] >= 1.0 - 1e-9
            assert res.causal_emergence >= 0.0  # sigma_2 >= gamma_star always

    def test_accepts_torch_and_numpy_equally(self):
        tpm = _k_block_uniform_tpm(2, 4)
        a = compute_ce2_from_tpm(tpm)
        b = compute_ce2_from_tpm(torch.as_tensor(tpm))
        assert a.causal_emergence == pytest.approx(b.causal_emergence, abs=1e-9)
        assert a.emergent_complexity == b.emergent_complexity

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            compute_ce2_from_tpm(np.ones((3, 4)))

    def test_from_trajectories_matches_from_tpm(self):
        trajs = [np.array([0, 1, 2, 0, 1, 2, 0]), np.array([2, 2, 1, 0, 0, 1, 2])]
        res = compute_ce2_from_trajectories(trajs, 3)
        direct = compute_ce2_from_tpm(_build_tpm(trajs, 3))
        assert res.causal_emergence == pytest.approx(direct.causal_emergence, abs=1e-12)


class TestLatentExtraction:
    def test_latent_class_indices_discrete_argmax(self):
        # [B=1, C=2, K=3, H=2, W=2] -> 1*2*2*2 = 8 variables, indices in [0, 3).
        z = torch.randn(1, 2, 3, 2, 2)
        idx = latent_class_indices(z, latent_mode="discrete")
        expected = z.argmax(dim=2).reshape(-1).numpy()
        assert idx.shape == (8,)
        assert np.array_equal(idx, expected)
        assert idx.min() >= 0 and idx.max() < 3

    def test_latent_class_indices_continuous_bins(self):
        z = torch.rand(1, 2, 4, 2, 2)  # in [0,1], safe for the clip-based binning
        idx = latent_class_indices(z, latent_mode="continuous", num_bins=4)
        assert idx.shape == (8,)
        assert idx.min() >= 0 and idx.max() < 4

    def test_latent_class_indices_rejects_wrong_rank(self):
        with pytest.raises(ValueError):
            latent_class_indices(torch.randn(2, 3, 4), latent_mode="discrete")

    def test_build_latent_tpm_matches_build_tpm(self):
        fields = [np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]
        tpm = build_latent_tpm(fields, 2)
        # variable 0 trajectory: 0,1,1 ; variable 1: 1,0,1
        ref = _build_tpm([np.array([0, 1, 1]), np.array([1, 0, 1])], 2)
        assert np.allclose(tpm, ref)


class TestIncrementalCounting:
    def test_update_and_normalize(self):
        counts = new_transition_counts(3)
        update_transition_counts(counts, np.array([0, 1, 2]), np.array([1, 1, 0]))
        assert counts[0, 1] == 1.0
        assert counts[1, 1] == 1.0
        assert counts[2, 0] == 1.0
        tpm = counts_to_tpm(counts, laplace=1.0)
        # row 0 raw counts [0,1,0] + 1 = [1,2,1], sum 4 -> [0.25, 0.5, 0.25]
        assert tpm[0, 1] == pytest.approx(0.5)
        assert np.allclose(tpm.sum(axis=1), 1.0)

    def test_out_of_range_indices_ignored(self):
        counts = new_transition_counts(3)
        update_transition_counts(counts, np.array([0, 5, -1]), np.array([1, 2, 0]))
        assert counts.sum() == 1.0  # only the 0->1 pair is in range

    def test_incremental_matches_pooled_build(self):
        # Two steps of a 2-variable field: incremental counts + Laplace should
        # match build_latent_tpm on the same fields.
        f0 = np.array([0, 1, 1])
        f1 = np.array([1, 1, 0])
        f2 = np.array([2, 0, 1])
        counts = new_transition_counts(3)
        update_transition_counts(counts, f0, f1)
        update_transition_counts(counts, f1, f2)
        tpm_inc = counts_to_tpm(counts, laplace=1.0)
        tpm_pooled = build_latent_tpm([f0, f1, f2], 3)
        assert np.allclose(tpm_inc, tpm_pooled)


class TestDegeneracyConfound:
    """CE 2.0 rises as the input trajectory degenerates, because sigma_2 minus
    gamma_star measures the coarse-graining gain still AVAILABLE. These tests lock
    that behaviour in so a high value is never misread as strong emergence."""

    def test_ce2_decreases_as_trajectory_gets_richer(self):
        L, N = 2000, 243
        rng = np.random.default_rng(0)
        frozen = np.zeros(L, dtype=np.int64)
        two = np.tile([0, 1], L // 2).astype(np.int64)
        cyc = (np.arange(L) % 10).astype(np.int64)
        rich = rng.integers(0, N, L).astype(np.int64)
        vals = [compute_ce2_from_trajectories([t], N).causal_emergence
                for t in (frozen, two, cyc, rich)]
        assert vals[0] > vals[1] > vals[2] > vals[3], vals

    def test_frozen_reference_matches_the_2026_07_pilot(self):
        # The dark_room pilot's gate and workspace values were constant across all
        # 50 windows and equal to the frozen-input reference at the 2000-step
        # window, proving both trajectories were frozen rather than emergent.
        assert frozen_trajectory_ce2_value(243, 2000) == pytest.approx(0.877884, abs=1e-6)
        assert frozen_trajectory_ce2_value(8, 2000) == pytest.approx(0.662874, abs=1e-6)

    def test_frozen_reference_is_length_dependent(self):
        a = frozen_trajectory_ce2_value(243, 100)
        b = frozen_trajectory_ce2_value(243, 2000)
        c = frozen_trajectory_ce2_value(243, 10000)
        assert a < b < c

    def test_degeneracy_flags_a_frozen_trajectory(self):
        d = trajectory_degeneracy([np.zeros(500, dtype=np.int64)], 243)
        assert d["distinct_states"] == 1
        assert d["degenerate"] is True
        assert d["n_transitions"] == 499

    def test_degeneracy_passes_a_varied_trajectory(self):
        rng = np.random.default_rng(1)
        d = trajectory_degeneracy([rng.integers(0, 243, 2000).astype(np.int64)], 243)
        assert d["distinct_states"] > 100
        assert d["degenerate"] is False
        assert 0.0 < d["coverage"] <= 1.0

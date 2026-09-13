"""
Tests for models/evaluation/perturbational_complexity.py.

Every expected value here is derived analytically or by hand-parsing, never copied
from a run. The LZ76 values in TestLempelZiv are hand-parsed in the module
docstring of lempel_ziv_complexity; the three PCI regimes are constructed so that
the ordering between them follows from the definition rather than from a
measurement.

The three regimes under test are the ones the 2016 review says PCI separates:
  1. local / dying response      -> low PCI
  2. global but stereotyped      -> low PCI
  3. global and non-repeating    -> high PCI
plus the property that makes the measure usable at all here:
  4. a response no larger than baseline fluctuation scores ~0 (this comes from the
     significance binarization, NOT from the entropy normalization).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.evaluation.perturbational_complexity import (  # noqa: E402
    DEFAULT_NOISE_EPS,
    DEFAULT_VAR_FLOOR,
    PCIResult,
    binarize_response,
    binary_entropy,
    compute_pci,
    lempel_ziv_complexity,
    resolve_var_floor,
)


class TestLempelZiv:
    """LZ76 exhaustive-history parsing, hand-verified."""

    def test_empty_and_singleton(self):
        assert lempel_ziv_complexity([]) == 0
        assert lempel_ziv_complexity([1]) == 1

    def test_constant_sequence_parses_into_two_components(self):
        # "0" | "000"
        assert lempel_ziv_complexity([0, 0, 0, 0]) == 2
        assert lempel_ziv_complexity([1] * 50) == 2

    def test_hand_parsed_short_strings(self):
        assert lempel_ziv_complexity([0, 1]) == 2           # "0" | "1"
        assert lempel_ziv_complexity([0, 0, 1, 0]) == 3     # "0" | "01" | "0"
        assert lempel_ziv_complexity([0, 1, 0, 1]) == 3     # "0" | "1" | "01"

    def test_periodic_is_less_complex_than_random_at_equal_length(self):
        rng = np.random.default_rng(0)
        n = 2000
        periodic = np.tile([0, 1], n // 2)
        random_seq = rng.integers(0, 2, size=n)
        assert lempel_ziv_complexity(periodic) < lempel_ziv_complexity(random_seq)

    def test_accepts_2d_input_by_raveling(self):
        assert lempel_ziv_complexity(np.zeros((4, 4), dtype=int)) == 2

    def test_is_deterministic(self):
        rng = np.random.default_rng(7)
        seq = rng.integers(0, 2, size=500)
        assert lempel_ziv_complexity(seq) == lempel_ziv_complexity(seq)


class TestBinaryEntropy:
    def test_endpoints_are_zero(self):
        assert binary_entropy(0.0) == 0.0
        assert binary_entropy(1.0) == 0.0

    def test_maximum_at_one_half(self):
        assert binary_entropy(0.5) == pytest.approx(1.0)

    def test_symmetric(self):
        assert binary_entropy(0.25) == pytest.approx(binary_entropy(0.75))


class TestBinarizeResponse:
    def test_subthreshold_response_marks_nothing(self):
        rng = np.random.default_rng(1)
        baseline = rng.normal(0.0, 1.0, size=(5, 200))
        response = rng.normal(0.0, 1.0, size=(5, 50))  # same scale as baseline
        binary = binarize_response(response, baseline, threshold_sigma=3.0)
        # A 3-sigma threshold on same-scale noise marks well under 1% of entries.
        assert binary.mean() < 0.02

    def test_suprathreshold_response_marks_everything(self):
        baseline = np.tile(np.array([-1.0, 1.0]), (3, 100))
        response = np.full((3, 20), 100.0)
        binary = binarize_response(response, baseline, threshold_sigma=3.0)
        assert binary.all()

    def test_dead_channel_never_marks_significant(self):
        # Channel 1 has variability below the floor. Even a huge response on it must
        # not count, because its threshold would otherwise be ~0. This is the
        # `adaptation` gate node case (std 6.08e-06).
        baseline = np.zeros((2, 100))
        baseline[0] = np.tile([-1.0, 1.0], 50)
        baseline[1] = np.tile([-1e-7, 1e-7], 50)
        response = np.full((2, 10), 50.0)
        binary = binarize_response(response, baseline, var_floor=DEFAULT_VAR_FLOOR)
        assert binary[0].all()
        assert not binary[1].any()

    def test_channel_mismatch_raises(self):
        with pytest.raises(ValueError, match="channel mismatch"):
            binarize_response(np.zeros((3, 10)), np.zeros((4, 10)))

    def test_too_short_baseline_raises(self):
        with pytest.raises(ValueError, match="at least 2 timesteps"):
            binarize_response(np.zeros((3, 10)), np.zeros((3, 1)))


def _baseline(n_channels: int, seed: int = 0) -> np.ndarray:
    """Unit-variance baseline fluctuation, so a threshold_sigma of 3 sits at 3.0."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(n_channels, 400))


class TestPCIRegimes:
    """The three regimes the measure exists to separate."""

    N_CHANNELS = 32
    N_STEPS = 64

    def test_no_response_scores_zero(self):
        baseline = _baseline(self.N_CHANNELS)
        response = np.zeros((self.N_CHANNELS, self.N_STEPS))
        result = compute_pci(response, baseline)
        assert result.pci == 0.0
        assert result.active_fraction == 0.0

    def test_local_dying_response_scores_low(self):
        # One channel responds, and only for the first few steps.
        baseline = _baseline(self.N_CHANNELS)
        response = np.zeros((self.N_CHANNELS, self.N_STEPS))
        response[0, :4] = 100.0
        local = compute_pci(response, baseline)

        rng = np.random.default_rng(3)
        spread = np.zeros((self.N_CHANNELS, self.N_STEPS))
        mask = rng.random((self.N_CHANNELS, self.N_STEPS)) < 0.5
        spread[mask] = 100.0
        global_varied = compute_pci(spread, baseline)

        assert local.pci < global_varied.pci
        assert local.pci < 0.1

    def test_casali_normalization_diverges_on_sparse_responses(self):
        # Regression pin for the deviation documented in the module docstring. The
        # published normalizer L*H/log2(L) collapses toward zero as the response
        # gets sparse, so it ranks a 4-entry dying response ABOVE a fully
        # differentiated one. This is why `pci` does not use it. If this assertion
        # ever flips, the deviation is no longer needed and should be revisited.
        baseline = _baseline(self.N_CHANNELS)

        dying = np.zeros((self.N_CHANNELS, self.N_STEPS))
        dying[0, :4] = 100.0

        rng = np.random.default_rng(3)
        differentiated = np.where(
            rng.random((self.N_CHANNELS, self.N_STEPS)) < 0.5, 100.0, 0.0
        )

        dying_result = compute_pci(dying, baseline)
        diff_result = compute_pci(differentiated, baseline)

        # The published normalization inverts the correct ordering here.
        assert dying_result.pci_casali > diff_result.pci_casali
        # The normalization actually used does not.
        assert dying_result.pci < diff_result.pci

    def test_global_stereotyped_response_scores_low(self):
        # Every channel does the same thing at the same time, forever: maximally
        # integrated in the naive sense, but carrying no differentiation.
        baseline = _baseline(self.N_CHANNELS)
        pattern = np.tile([100.0, 0.0], self.N_STEPS // 2)
        stereotyped = np.tile(pattern, (self.N_CHANNELS, 1))
        stereo = compute_pci(stereotyped, baseline)

        rng = np.random.default_rng(4)
        varied = np.where(
            rng.random((self.N_CHANNELS, self.N_STEPS)) < 0.5, 100.0, 0.0
        )
        varied_result = compute_pci(varied, baseline)

        assert stereo.pci < varied_result.pci
        # Both have the same active fraction, so the gap is differentiation alone.
        assert stereo.active_fraction == pytest.approx(0.5, abs=0.05)
        assert varied_result.active_fraction == pytest.approx(0.5, abs=0.05)

    def test_noise_response_scores_near_zero(self):
        # A "response" statistically identical to the baseline. This is the property
        # that makes PCI insensitive to random processes, and it comes from the
        # significance binarization: nothing crosses threshold.
        rng = np.random.default_rng(5)
        baseline = _baseline(self.N_CHANNELS, seed=5)
        response = rng.normal(0.0, 1.0, size=(self.N_CHANNELS, self.N_STEPS))
        result = compute_pci(response, baseline, threshold_sigma=3.0)
        assert result.pci < 0.05
        assert result.active_fraction < 0.02

    def test_differentiated_global_response_approaches_one(self):
        # A response as incompressible as a random binary matrix of the same size
        # is the ceiling of this normalization, so it should land near 1.0.
        baseline = _baseline(self.N_CHANNELS)
        rng = np.random.default_rng(21)
        response = np.where(
            rng.random((self.N_CHANNELS, self.N_STEPS)) < 0.5, 100.0, 0.0
        )
        result = compute_pci(response, baseline)
        assert 0.8 < result.pci < 1.3

    def test_saturated_response_scores_zero_not_high(self):
        # Every entry significant means zero source entropy: no differentiation.
        baseline = _baseline(self.N_CHANNELS)
        response = np.full((self.N_CHANNELS, self.N_STEPS), 500.0)
        result = compute_pci(response, baseline)
        assert result.active_fraction == 1.0
        assert result.source_entropy == 0.0
        assert result.pci == 0.0
        assert result.pci_casali == 0.0

    def test_regime_ordering_holds_end_to_end(self):
        baseline = _baseline(self.N_CHANNELS)

        dying = np.zeros((self.N_CHANNELS, self.N_STEPS))
        dying[0, :4] = 100.0

        stereotyped = np.tile(
            np.tile([100.0, 0.0], self.N_STEPS // 2), (self.N_CHANNELS, 1)
        )

        rng = np.random.default_rng(6)
        differentiated = np.where(
            rng.random((self.N_CHANNELS, self.N_STEPS)) < 0.5, 100.0, 0.0
        )

        pci_dying = compute_pci(dying, baseline).pci
        pci_stereo = compute_pci(stereotyped, baseline).pci
        pci_diff = compute_pci(differentiated, baseline).pci

        assert pci_dying < pci_diff
        assert pci_stereo < pci_diff


class TestPCIProperties:
    def test_returns_result_dataclass_with_shape_metadata(self):
        baseline = _baseline(8)
        response = np.zeros((8, 16))
        result = compute_pci(response, baseline)
        assert isinstance(result, PCIResult)
        assert result.n_channels == 8
        assert result.n_timesteps == 16

    def test_is_deterministic_for_fixed_input(self):
        baseline = _baseline(16, seed=11)
        rng = np.random.default_rng(12)
        response = np.where(rng.random((16, 32)) < 0.5, 100.0, 0.0)
        first = compute_pci(response, baseline)
        second = compute_pci(response, baseline)
        assert first.pci == second.pci
        assert first.lz_complexity == second.lz_complexity

    def test_pci_is_non_negative(self):
        baseline = _baseline(8, seed=13)
        rng = np.random.default_rng(14)
        for _ in range(5):
            response = rng.normal(0.0, 20.0, size=(8, 32))
            assert compute_pci(response, baseline).pci >= 0.0

    def test_higher_threshold_marks_no_more_than_lower(self):
        baseline = _baseline(8, seed=15)
        rng = np.random.default_rng(16)
        response = rng.normal(0.0, 6.0, size=(8, 64))
        lenient = compute_pci(response, baseline, threshold_sigma=2.0)
        strict = compute_pci(response, baseline, threshold_sigma=5.0)
        assert strict.active_fraction <= lenient.active_fraction


class TestVarianceFloorMode:
    """
    The dead-channel floor must fit the site it is applied to.

    `DEFAULT_VAR_FLOOR` is 1e-4, chosen for the rssm whose PCI-window baseline sd is
    2.4e-02 to 1.8e-01. The gate's is 1.2e-05 to 1.2e-04, entirely below it, so at
    the default the gate cannot register a response AT ANY SIZE. Measured
    2026-09-13: lowering the floor turns `gate3_s42` from 0.0000 into 0.0439 while
    `gate3_s43` and `gate3_s44` stay at exactly 0.0000, raw response 1.192e-07.

    These tests pin both halves. A floor mode that wakes a real response is only
    useful if it does NOT wake noise, so the noise case is pinned too.
    """

    # Four live gate nodes plus gate_adaptation, roughly 100x quieter.
    GATE_SIGMAS = np.array([5e-5, 6e-5, 4e-5, 6.9e-5, 1.04e-7])

    def _gate_site(self, seed=7, response_sigmas=8.0):
        rng = np.random.default_rng(seed)
        base = np.stack([rng.normal(0, s, 40) for s in self.GATE_SIGMAS])
        resp = np.stack(
            [rng.normal(0, s * response_sigmas, 60) for s in self.GATE_SIGMAS])
        return base, resp

    def test_absolute_floor_disqualifies_the_whole_gate_substrate(self):
        """Every channel responds at 8 sigma and the default floor reports nothing.

        This is the defect, pinned. It is not a statement about the agent.
        """
        base, resp = self._gate_site()
        floor = resolve_var_floor(base, var_floor_mode="absolute")
        assert floor == DEFAULT_VAR_FLOOR
        binary = binarize_response(resp, base, var_floor=floor)
        assert not binary.any(), "the absolute floor should silence the whole site"

    def test_relative_floor_admits_the_live_nodes_and_excludes_the_quiet_one(self):
        """The behaviour the floor was written for: dead means quiet RELATIVE to peers."""
        base, resp = self._gate_site()
        floor = resolve_var_floor(base, var_floor_mode="relative")
        binary = binarize_response(resp, base, var_floor=floor)
        registering = [i for i in range(len(self.GATE_SIGMAS)) if binary[i].any()]
        assert registering == [0, 1, 2, 3], (
            "expected the four live nodes and not gate_adaptation, got %r" % registering)

    def test_relative_mode_DOES_manufacture_signal_from_noise(self):
        """The failure mode of `relative`, pinned so it is never forgotten.

        This test asserts a DEFECT, not a property. A site whose response is the
        same size as its own fluctuation should score zero. Under `relative` it does
        not: measured over 40 seeds, 29 of 40 score non-zero with a mean of 0.083 and
        a maximum of 0.192.

        That range matters. The real gate reading this mode produced on
        `gate3_s42` was 0.0603, which sits INSIDE it. So `relative` cannot currently
        separate a gate response from gate noise, and no gate PCI obtained under it
        may be cited as a finding.

        `noise_eps` does not help: it guards float noise near 1e-9, while the gate
        channels sit at 1e-5 to 1e-7, far above it.

        The honest reading is that neither floor works at the gate. `absolute`
        silences a real substrate, `relative` admits its noise, and the gap between
        the gate's fluctuation and numerical noise is too small for a fixed
        threshold of either kind. Separating them needs a NULL: score the probe with
        no impulse and use that distribution as the floor, which is the control this
        project already uses for content.
        """
        noisy = 0
        for seed in range(40):
            base, resp = self._gate_site(seed=seed, response_sigmas=1.0)
            floor = resolve_var_floor(base, var_floor_mode="relative")
            if compute_pci(resp, base, var_floor=floor).pci > 0.0:
                noisy += 1
        assert noisy > 20, (
            "relative mode no longer admits noise at this rate (%d/40). If this is a "
            "deliberate fix, replace this test with the property it now has." % noisy)

        # absolute does not have this defect; it has the opposite one.
        base, resp = self._gate_site(seed=11, response_sigmas=1.0)
        floor = resolve_var_floor(base, var_floor_mode="absolute")
        assert compute_pci(resp, base, var_floor=floor).pci == 0.0

    def test_noise_eps_stops_a_dead_site_scaling_its_own_floor_down(self):
        """A site that is entirely float noise must not admit that noise."""
        rng = np.random.default_rng(3)
        base = rng.normal(0, 1e-12, size=(5, 40))
        floor = resolve_var_floor(base, var_floor_mode="relative")
        assert floor == DEFAULT_NOISE_EPS

    def test_default_mode_is_absolute_so_published_numbers_are_unchanged(self):
        """Backward compatibility, pinned rather than assumed."""
        base, resp = self._gate_site()
        assert compute_pci(resp, base).resolved_var_floor == DEFAULT_VAR_FLOOR

    def test_compute_pci_reports_the_floor_it_actually_used(self):
        """A zero PCI is unreadable without the floor, so the result must carry it."""
        base, resp = self._gate_site()
        result = compute_pci(resp, base, var_floor_mode="relative")
        assert result.resolved_var_floor < DEFAULT_VAR_FLOOR
        assert result.resolved_var_floor == resolve_var_floor(
            base, var_floor_mode="relative")

    def test_an_unknown_mode_raises_rather_than_defaulting(self):
        base, _ = self._gate_site()
        with pytest.raises(ValueError, match="var_floor_mode"):
            resolve_var_floor(base, var_floor_mode="whatever")

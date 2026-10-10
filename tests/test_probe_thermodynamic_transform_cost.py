"""Tests for the transform-cost probe. Pure arithmetic and small linear algebra, CPU only.

Closed forms used.
    A dense 256 x 256 matrix needs 65536 multiply-adds. A diagonal needs 256. So the extra of an equalised arm
    over the diagonal z score of plain is 65280 for a raw 256-D stream.
    Energy of one decision, in channel events, is R T + rho M for M multiply-adds. The break-even rho* solves
    R T_plain + rho M_plain = R T_arm + rho M_arm, so rho* = R (T_plain - T_arm) / (M_arm - M_plain).
    A signed permutation matrix has max |entry| = 1 in every row.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_thermodynamic_transform_cost as tc  # noqa: E402
from scripts.analysis.probe_thermodynamic_encoding import encode_arm  # noqa: E402


class TestCounts:
    def test_dense_rotation_costs_r_squared_minus_the_diagonal(self):
        assert tc.extra_macs("full", 256) == 65536 - 256 == 65280
        assert tc.extra_macs("half", 256) == tc.extra_macs("pca0", 256) == 65280
        assert tc.extra_macs("plain", 256) == 0

    def test_large_streams_fold_the_scaling_into_the_pca_matrix(self):
        for dim in (1024, 16384):
            assert all(tc.extra_macs(a, dim) == 0 for a in tc.ARM_NAMES)

    def test_common_cost_uses_the_pca_projection_for_large_streams(self):
        assert tc.common_macs(256) == 256 + 256 * 6
        assert tc.common_macs(1024) == 1024 * 256 + 256 * 6

    def test_online_cost_adds_the_extra(self):
        assert tc.online_macs("full", 256) - tc.online_macs("plain", 256) == 65280


class TestEnergy:
    def test_energy_is_events_plus_rho_times_macs(self):
        assert tc.energy("plain", 256, 16, rho=0) == 256 * 16
        assert tc.energy("plain", 256, 16, rho=2) == 256 * 16 + 2 * tc.online_macs("plain", 256)

    def test_break_even_solves_the_equal_energy_condition(self):
        b = tc.break_even(1024, 16, 256, "full")
        rho = b["rho_star"]
        assert tc.energy("plain", 256, 1024, rho) == pytest.approx(tc.energy("full", 256, 16, rho))
        assert rho == pytest.approx(256 * (1024 - 16) / 65280)

    def test_net_saving_is_one_at_break_even_and_tends_to_the_event_ratio_at_zero_rho(self):
        rho = tc.break_even(1024, 16, 256, "full")["rho_star"]
        assert tc.net_saving(1024, 16, 256, "full", rho) == pytest.approx(1.0)
        assert tc.net_saving(1024, 16, 256, "full", 0.0) == pytest.approx(1024 / 16)
        assert tc.net_saving(1024, 16, 256, "full", 1e9) < 1.0

    def test_free_transform_has_no_break_even_and_a_pure_event_saving(self):
        assert tc.break_even(1024, 16, 1024, "full")["status"] == "free"
        assert tc.net_saving(1024, 16, 1024, "full", 1e6) == pytest.approx(
            (256 * 1024 + 1e6 * tc.online_macs("plain", 1024)) / (256 * 16 + 1e6 * tc.online_macs("full", 1024)))

    def test_plain_never_reaching_the_target_gives_a_lower_bound(self):
        b = tc.break_even(None, 16, 256, "full")
        assert b["lower_bound"] is True and b["rho_star"] == pytest.approx(256 * (tc.T_CAP - 16) / 65280)
        assert tc.break_even(1024, None, 256, "full")["status"] == "not reached"
        assert tc.net_saving(1024, None, 256, "full", 10) is None

    def test_reference_ratio_is_the_stated_division(self):
        assert tc.REFERENCE_RHO == pytest.approx(2300.0)


class TestChecks:
    def test_signed_permutation_has_diagonality_one(self):
        gen = torch.Generator().manual_seed(0)
        scores = torch.randn(500, 8, generator=gen, dtype=torch.float64) * torch.tensor([5, 4, 3, 2, 1.5, 1.2, 1.1, 1.0], dtype=torch.float64)
        assert tc.diagonality(scores) > 0.98

    def test_rotated_data_has_diagonality_below_one(self):
        gen = torch.Generator().manual_seed(1)
        scores = torch.randn(500, 8, generator=gen, dtype=torch.float64) * torch.tensor([5, 4, 3, 2, 1.5, 1.2, 1.1, 1.0], dtype=torch.float64)
        q, _ = torch.linalg.qr(torch.randn(8, 8, generator=gen, dtype=torch.float64))
        assert tc.diagonality(scores @ q) < 0.9

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_the_equalised_encoding_is_one_matrix(self, alpha):
        gen = torch.Generator().manual_seed(2)
        mix = torch.randn(10, 10, generator=gen, dtype=torch.float64)
        z = torch.randn(300, 10, generator=gen, dtype=torch.float64) @ mix
        z = z - z.mean(0)
        a = tc.composed_matrix(z, alpha)
        utr, ute = encode_arm(z, z[:7], alpha)
        assert torch.allclose(z @ a.T, utr, atol=1e-9) and torch.allclose(z[:7] @ a.T, ute, atol=1e-9)


class TestReport:
    def test_build_entry_has_all_arms_and_the_sweep(self):
        t95 = {"plain": 1024, "pca0": 1024, "half": 64, "full": 16}
        e = tc.build_entry(t95, "tectum_content")
        assert set(e) == {"pca0", "half", "full"}
        assert set(e["full"]["net_saving"]) == {str(r) for r in tc.RHO_SWEEP} | {"reference"}
        assert e["full"]["break_even"]["status"] == "finite"
        assert tc.build_entry(t95, "obs_map")["full"]["break_even"]["status"] == "free"


class TestDiagonalOnly:
    def _scores(self, seed=0):
        gen = torch.Generator().manual_seed(seed)
        scales = torch.tensor([6.0, 4.0, 2.5, 1.5, 1.0, 0.7, 0.4, 0.2], dtype=torch.float64)
        return torch.randn(600, 8, generator=gen, dtype=torch.float64) * scales

    def test_diag_encoding_has_no_rotation(self):
        z = self._scores()
        utr, _ = tc.encode_diag(z, z[:5], 1.0)
        # a pure per-column scaling leaves the correlation matrix of the columns unchanged
        assert torch.allclose(torch.corrcoef(utr.T), torch.corrcoef(z.T), atol=1e-9)

    def test_full_diag_equalises_component_variances(self, monkeypatch):
        monkeypatch.setattr(tc, "EPS_FRACTION", 1e-12)
        z = self._scores()
        utr, _ = tc.encode_diag(z, z[:5], 1.0)
        assert torch.allclose(utr.var(0), torch.ones(8, dtype=torch.float64), rtol=1e-6)

    def test_mean_training_variance_is_one_for_both_exponents(self):
        z = self._scores()
        for alpha in (0.5, 1.0):
            assert tc.encode_diag(z, z[:5], alpha)[0].var(0).mean().item() == pytest.approx(1.0, rel=1e-9)

    def test_diag_matches_the_rotated_arm_when_the_scores_are_already_principal(self):
        z = self._scores()
        z = z - z.mean(0)
        _, _, vh = torch.linalg.svd(z, full_matrices=False)
        principal = z @ vh.T
        diag, _ = tc.encode_diag(principal, principal[:3], 0.5)
        rotated, _ = encode_arm(principal, principal[:3], 0.5)
        assert torch.allclose(torch.sort(diag.var(0)).values, torch.sort(rotated.var(0)).values, rtol=1e-6)

    def test_diag_stream_evaluation_scores_every_arm_regime_and_dose(self):
        from tests.test_probe_thermodynamic_encoding import _low_variance_signal
        rec = _low_variance_signal(trials=60)
        out = tc.evaluate_diag_stream(rec.streams["synthetic"], rec.labels, rec.trials, seed=3)["trial_accuracy"]
        expected = {f"{a}|clean" for a in tc.DIAG_ARMS} | {f"{a}|{r}|pbit|{d}" for a in tc.DIAG_ARMS
                                                            for r in ("clean-trained", "matched") for d in tc.LADDER}
        assert set(out) == expected and all(len(v) == 60 for v in out.values())

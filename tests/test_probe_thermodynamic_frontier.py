"""Tests for the frontier probe, on synthetic tensors, CPU only.

Closed forms used.
    Events per decision are k x T. Multiply-adds follow the folding rules of the docstring: a raw 256-D stream
    with k axes needs 256 x k + 6 k, the plain 256-axis arm needs 256 + 1536; a larger stream needs d x k + 6 k.
    E_plain / E_arm = (k_p T_p + rho M_p) / (k_a T_a + rho M_a).
    With alpha = 1 and eps -> 0 the retained axes have equal variance; with alpha = 0 they keep their variances.
"""
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_thermodynamic_encoding as enc  # noqa: E402
from scripts.analysis import probe_thermodynamic_fidelity as fid  # noqa: E402
from scripts.analysis import probe_thermodynamic_frontier as fr  # noqa: E402
from tests.test_probe_thermodynamic_encoding import _low_variance_signal  # noqa: E402


def _z(rows=500, dims=20, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, dims, generator=gen, dtype=torch.float64) * torch.logspace(1, -1, dims, dtype=torch.float64)
    return x - x.mean(0)


class TestEncodings:
    def test_full_rank_matches_the_encoding_probe(self):
        z = _z()
        a, b = fr.encode_rank(z, z[:5], 1.0, 20)
        c, d = enc.encode_arm(z, z[:5], 1.0)
        assert torch.allclose(a.var(0), c.var(0), rtol=1e-8) and a.shape == c.shape

    @pytest.mark.parametrize("k", [4, 8, 16])
    def test_keeps_k_axes_with_unit_mean_variance(self, k):
        z = _z()
        utr, ute = fr.encode_rank(z, z[:5], 1.0, k)
        assert utr.shape == (500, k) and ute.shape == (5, k)
        assert utr.var(0).mean().item() == pytest.approx(1.0, rel=1e-9)

    def test_equalised_axes_have_equal_variance_and_unscaled_axes_keep_their_ratio(self, monkeypatch):
        monkeypatch.setattr(fr, "EPS_FRACTION", 1e-12)
        z = _z()
        full, _ = fr.encode_rank(z, z[:5], 1.0, 8)
        pca, _ = fr.encode_rank(z, z[:5], 0.0, 8)
        assert torch.allclose(full.var(0), torch.ones(8, dtype=torch.float64), rtol=1e-6)
        assert (pca.var(0).max() / pca.var(0).min()).item() > 5

    def test_the_arm_table_has_the_planned_arms(self):
        assert set(fr.ARMS) == {"plain256"} | {f"full{k}" for k in (8, 16, 32, 64, 128, 256)} | {f"pca{k}" for k in (16, 64, 256)}


class TestCosts:
    def test_raw_stream_counts(self):
        assert fr.arm_macs("plain256", 256) == 256 + 1536
        assert fr.arm_macs("full256", 256) == 256 * 256 + 1536
        assert fr.arm_macs("full16", 256) == 256 * 16 + 6 * 16

    def test_larger_stream_counts_follow_the_projection_to_k_axes(self):
        assert fr.arm_macs("plain256", 1024) == 1024 * 256 + 1536
        assert fr.arm_macs("full16", 1024) == 1024 * 16 + 6 * 16
        assert fr.arm_macs("full16", 16384) < fr.arm_macs("plain256", 16384)

    def test_events_are_k_times_dose(self):
        assert fr.events("full16", 64) == 16 * 64 and fr.events("plain256", 4) == 1024

    def test_energy_ratio_matches_the_formula_and_marks_missing_targets(self):
        r = fr.energy_ratio(256, "full16", 64, 1024, 10.0)
        expected = (256 * 1024 + 10 * fr.arm_macs("plain256", 256)) / (16 * 64 + 10 * fr.arm_macs("full16", 256))
        assert r == pytest.approx(expected)
        assert fr.energy_ratio(256, "full16", None, 1024, 10.0) is None
        assert fr.energy_ratio(256, "full16", 64, None, 10.0) == fr.energy_ratio(256, "full16", 64, fr.T_CAP, 10.0)

    def test_fewer_axes_save_total_energy_on_a_larger_stream_even_at_a_high_ratio(self):
        # the projection to 16 axes costs 16 / 256 of the projection to 256 axes, so the ratio tends to 256 / 16 = 16
        assert fr.energy_ratio(16384, "full16", 64, 1024, 2300.0) == pytest.approx(16.0, rel=0.01)


class TestEvaluation:
    def test_every_arm_and_dose_is_scored_per_trial(self):
        rec = _low_variance_signal(trials=60)
        out = fr.evaluate_stream(rec.streams["synthetic"], rec.labels, rec.trials, seed=3)["trial_accuracy"]
        expected = {f"{a}|clean" for a in fr.ARMS} | {f"{a}|matched|{d}" for a in fr.ARMS for d in fid.LADDER}
        assert set(out) == expected and all(len(v) == 60 for v in out.values())

    def test_small_k_can_keep_a_class_that_lives_in_few_axes(self):
        rec = _low_variance_signal(trials=90)
        acc = {k: float(np.mean(v)) for k, v in fr.evaluate_stream(rec.streams["synthetic"], rec.labels, rec.trials, 3)["trial_accuracy"].items()}
        assert acc["full8|clean"] > 0.5

    def test_summary_has_events_energy_ratios_and_the_plain_yardstick(self):
        names = [f"{a}|clean" for a in fr.ARMS] + [f"{a}|matched|{d}" for a in fr.ARMS for d in fid.LADDER]
        rng = np.random.default_rng(0)
        per_trial = {n: np.clip(rng.normal(0.7, 0.1, 120), 0, 1) for n in names}
        entry = fr.summarise_stream(per_trial, "obs_map", rng)
        assert entry["estimable"] is True and set(entry["arms"]) == set(fr.ARMS)
        e = entry["arms"]["full16"]
        assert {"events95", "energy_ratio", "T95", "macs", "k"} <= set(e) and set(e["energy_ratio"]) == {str(r) for r in fr.RHOS}

    def test_chance_stream_is_not_estimable(self):
        names = [f"{a}|clean" for a in fr.ARMS] + [f"{a}|matched|{d}" for a in fr.ARMS for d in fid.LADDER]
        assert fr.summarise_stream({n: np.full(120, fid.CHANCE) for n in names}, "obs_map", np.random.default_rng(0))["estimable"] is False


class TestCli:
    def test_round_trip(self, tmp_path):
        rec = _low_variance_signal(trials=60)
        torch.save({"streams": rec.streams, "labels": rec.labels, "trials": rec.trials}, fid.recording_path(tmp_path, 42, 300))
        fr.main(["evaluate", "--out-dir", str(tmp_path)])
        assert fr.frontier_path(tmp_path, 42, 300).exists()
        out = tmp_path / "s.json"
        fr.main(["summarise", "--out-dir", str(tmp_path), "--output-json", str(out)])
        assert "synthetic" in json.loads(out.read_text())["models"]["42"]

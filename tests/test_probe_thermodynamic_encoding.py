"""Tests for the variance-equalised encoding probe, on synthetic tensors, CPU only.

Closed forms used.
    Binomial counts. A count over T bins with probability p has mean T p and variance T p (1 - p).
    pbit decoder: count / T, mean x, variance x (1 - x) / T. poisson decoder: count / (T F_DT), mean x.
    Whitening. With eps -> 0 and alpha = 1 the training covariance of u is the identity up to one gain,
    so every component variance equals 1. For alpha = 0 the encoding is a rotation, so distances are
    preserved up to the one gain.
    Every arm is rescaled so the mean training variance per dimension is 1.
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
from scripts.analysis.probe_thermodynamic_cost import Recording  # noqa: E402


def _anisotropic(rows: int = 600, dims: int = 12, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    scales = torch.logspace(1, -1, dims, dtype=torch.float64)
    x = torch.randn(rows, dims, generator=gen, dtype=torch.float64) * scales
    return x - x.mean(0)


class TestEncodings:
    def test_plain_returns_the_inputs_unchanged(self):
        z = _anisotropic()
        a, b = enc.encode_arm(z, z[:5], None)
        assert a is z and torch.equal(b, z[:5])

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_mean_training_variance_per_dimension_is_one(self, alpha):
        z = _anisotropic()
        utr, _ = enc.encode_arm(z, z[:5], alpha)
        assert utr.var(0).mean().item() == pytest.approx(1.0, rel=1e-9)

    def test_full_equalisation_gives_equal_component_variances(self, monkeypatch):
        monkeypatch.setattr(enc, "EPS_FRACTION", 1e-12)
        z = _anisotropic()
        utr, _ = enc.encode_arm(z, z[:5], 1.0)
        assert torch.allclose(utr.var(0), torch.ones(12, dtype=torch.float64), rtol=1e-6)

    def test_rotation_arm_preserves_distances_up_to_one_gain(self):
        z = _anisotropic()
        utr, _ = enc.encode_arm(z, z[:5], 0.0)
        d_before = torch.cdist(z[:20], z[:20])
        d_after = torch.cdist(utr[:20], utr[:20])
        ratio = (d_after[d_before > 0] / d_before[d_before > 0])
        assert ratio.std().item() < 1e-9

    def test_partial_equalisation_lies_between_plain_rotation_and_full(self):
        z = _anisotropic()
        spread = {a: enc.encode_arm(z, z[:5], a)[0].var(0) for a in (0.0, 0.5, 1.0)}
        ratio = {a: (v.max() / v.min()).item() for a, v in spread.items()}
        assert ratio[0.0] > ratio[0.5] > ratio[1.0]

    def test_test_rows_use_the_training_axes(self):
        z = _anisotropic()
        utr, ute = enc.encode_arm(z[:400], z[400:], 1.0)
        assert utr.shape[1] == ute.shape[1] == 12 and torch.isfinite(ute).all()


class TestChannels:
    @pytest.mark.parametrize("value", [0.2, 0.5, 0.8])
    def test_pbit_decoder_matches_the_closed_forms(self, value):
        x = torch.full((4000, 3), value, dtype=torch.float64)
        est = enc.binomial_estimates(x, (64,), torch.Generator().manual_seed(1), "pbit")[0]
        assert est.mean().item() == pytest.approx(value, abs=0.01)
        assert est.var().item() == pytest.approx(value * (1 - value) / 64, rel=0.1)

    def test_poisson_decoder_matches_the_closed_forms(self):
        x = torch.full((4000, 3), 0.5, dtype=torch.float64)
        est = enc.binomial_estimates(x, (64,), torch.Generator().manual_seed(2), "poisson")[0]
        assert est.mean().item() == pytest.approx(0.5, abs=0.01)
        assert est.var().item() == pytest.approx(0.5 * (1 - fid.F_DT * 0.5) / (fid.F_DT * 64), rel=0.1)

    def test_binomial_path_agrees_with_the_per_bin_path(self):
        x = torch.full((3000, 3), 0.3, dtype=torch.float64)
        per_bin, _ = fid.pbit_estimates(x, (16, 256), torch.Generator().manual_seed(3))
        exact = enc.binomial_estimates(x, (16, 256), torch.Generator().manual_seed(4), "pbit")
        for a, b in zip(per_bin, exact):
            assert a.mean().item() == pytest.approx(b.mean().item(), abs=0.01)
            assert a.var().item() == pytest.approx(b.var().item(), rel=0.15)

    def test_doses_are_nested_so_the_variance_falls_with_dose(self):
        x = torch.full((3000, 3), 0.5, dtype=torch.float64)
        est = enc.binomial_estimates(x, (16, 256), torch.Generator().manual_seed(5), "pbit")
        assert est[0].var().item() / est[1].var().item() == pytest.approx(16.0, rel=0.25)

    def test_extreme_inputs_stay_finite(self):
        x = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        est = enc.binomial_estimates(x, (16,), torch.Generator().manual_seed(6), "pbit")[0]
        assert torch.isfinite(est).all()


def _low_variance_signal(seed: int = 0, trials: int = 90, rows_per_trial: int = 5, dims: int = 48, k: int = 6) -> Recording:
    """Six classes. The class lives in k minor principal axes. k shared factors with large variance
    mix into every dimension, so per-dimension z scoring cannot remove them, and they carry only nuisance."""
    gen = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(dims, 2 * k, generator=gen, dtype=torch.float64))
    major, minor = q[:, :k], q[:, k:]
    offsets = torch.randn(6, k, generator=gen, dtype=torch.float64)
    rows, labels, tids = [], [], []
    for t in range(trials):
        c = t % 6
        shift = 0.1 * torch.randn(dims, generator=gen, dtype=torch.float64)
        for _ in range(rows_per_trial):
            factors = 5.0 * torch.randn(k, generator=gen, dtype=torch.float64)
            noise = 0.3 * torch.randn(dims, generator=gen, dtype=torch.float64)
            rows.append(major @ factors + minor @ offsets[c] + shift + noise)
            labels.append(c)
            tids.append(t)
    return Recording({"synthetic": torch.stack(rows).float()}, torch.tensor(labels), torch.tensor(tids))


class TestEvaluation:
    def test_every_arm_channel_and_dose_is_scored_per_trial(self):
        rec = _low_variance_signal()
        out = enc.evaluate_recording(rec, seed=3)["synthetic"]
        names = set(out["trial_accuracy"])
        expected = {f"{a}|clean" for a in enc.ARMS} | {f"{a}|{c}:{d}" for a in enc.ARMS for c in fid.CHANNELS for d in fid.LADDER}
        assert names == expected and all(len(v) == 90 for v in out["trial_accuracy"].values())
        assert all(len(p) == 5 for p in out["penalties"].values())

    def test_equalisation_raises_accuracy_at_equal_events_when_the_class_is_in_low_variance_axes(self):
        rec = _low_variance_signal()
        acc = {k: float(np.mean(v)) for k, v in enc.evaluate_recording(rec, seed=3)["synthetic"]["trial_accuracy"].items()}
        dose = fid.LADDER[1]
        assert acc[f"half|pbit:{dose}"] > acc[f"plain|pbit:{dose}"] + 0.1
        assert acc[f"full|pbit:{dose}"] > acc[f"plain|pbit:{dose}"] + 0.1

    def test_the_rotation_control_does_not_explain_the_gain(self):
        rec = _low_variance_signal()
        acc = {k: float(np.mean(v)) for k, v in enc.evaluate_recording(rec, seed=3)["synthetic"]["trial_accuracy"].items()}
        dose = fid.LADDER[1]
        assert acc[f"half|pbit:{dose}"] > acc[f"pca0|pbit:{dose}"] + 0.1

    def test_every_arm_reaches_its_clean_accuracy_at_the_top_dose(self):
        rec = _low_variance_signal()
        acc = {k: float(np.mean(v)) for k, v in enc.evaluate_recording(rec, seed=3)["synthetic"]["trial_accuracy"].items()}
        top = fid.LADDER[-1]
        for arm in enc.ARMS:
            assert acc[f"{arm}|pbit:{top}"] >= acc[f"{arm}|clean"] - 0.08

    def test_paired_difference_and_rho_star_use_the_plain_clean_yardstick(self):
        rng = np.random.default_rng(0)
        plain_clean = np.full(100, 0.7)
        index = enc.resample_index(rng, 100)
        assert enc.rho_star(np.full(100, 0.7), plain_clean, index)["rho"] == pytest.approx(1.0)
        assert enc.rho_star(np.full(100, fid.CHANCE), plain_clean, index)["rho"] == pytest.approx(0.0)
        diff = enc.paired_difference(np.full(100, 0.5), np.full(100, 0.4), index)
        assert diff["difference"] == pytest.approx(0.1) and diff["low"] == pytest.approx(0.1)

    def test_summary_marks_a_chance_stream_not_estimable(self):
        names = [f"{a}|clean" for a in enc.ARMS] + [f"{a}|{c}:{d}" for a in enc.ARMS for c in fid.CHANNELS for d in fid.LADDER]
        chance = {k: np.full(120, fid.CHANCE) for k in names}
        assert enc.summarise_stream(chance, np.random.default_rng(0))["estimable"] is False


class TestCli:
    def test_evaluate_and_summarise_round_trip(self, tmp_path):
        rec = _low_variance_signal(trials=60)
        torch.save({"streams": rec.streams, "labels": rec.labels, "trials": rec.trials}, fid.recording_path(tmp_path, 42, 300))
        enc.main(["evaluate", "--out-dir", str(tmp_path)])
        assert enc.encoding_path(tmp_path, 42, 300).exists()
        out = tmp_path / "s.json"
        enc.main(["summarise", "--out-dir", str(tmp_path), "--output-json", str(out)])
        entry = json.loads(out.read_text())["models"]["42"]["synthetic"]
        assert entry["estimable"] is True and set(entry["arms"]) == set(enc.ARMS)
        assert len(entry["arms"]["full"]["pbit"]["curve"]) == len(fid.LADDER)

    def test_models_filter_skips_other_models(self, tmp_path):
        rec = _low_variance_signal(trials=60)
        for model in (42, 43):
            torch.save({"streams": rec.streams, "labels": rec.labels, "trials": rec.trials}, fid.recording_path(tmp_path, model, 300))
        enc.main(["evaluate", "--out-dir", str(tmp_path), "--models", "43"])
        assert enc.encoding_path(tmp_path, 43, 300).exists() and not enc.encoding_path(tmp_path, 42, 300).exists()

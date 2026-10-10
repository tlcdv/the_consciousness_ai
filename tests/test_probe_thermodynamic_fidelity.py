"""Tests for the fidelity dose-response probe, on synthetic tensors, CPU only.

Closed forms used.
    Poisson code. Bins are Bernoulli(F_DT * x), so a count over T bins is Binomial(T, F_DT * x).
    The decoder count / (T * F_DT) has mean x and variance x * (1 - F_DT * x) / (F_DT * T).
    P-bit code. P(+1) = (1 + tanh(h)) / 2 = x when h = atanh(2x - 1). The mean of S samples of
    +-1 is 2x - 1 with variance 4 x (1 - x) / S, so the decoder (1 + mean) / 2 has mean x and
    variance x (1 - x) / S.
    Retained information rho = (A_T - 1/6) / (A_clean - 1/6) is 1 when A_T = A_clean and 0 at chance.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_thermodynamic_fidelity as fid  # noqa: E402
from scripts.analysis.probe_thermodynamic_cost import Recording, synthetic_recording  # noqa: E402
from models.thermodynamic.interfaces.snn_bridge import rate_decode, rate_encode  # noqa: E402
from models.thermodynamic.p_bit_emulator import BlockGibbsSampler  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _x(value: float, rows: int = 2000, dims: int = 4) -> torch.Tensor:
    return torch.full((rows, dims), value, dtype=torch.float64)


class TestChannels:
    @pytest.mark.parametrize("value", [0.2, 0.5, 0.8])
    def test_poisson_decoder_is_unbiased_with_the_binomial_variance(self, value):
        est, _ = fid.poisson_estimates(_x(value), (64,), torch.Generator().manual_seed(1))
        expected_var = value * (1 - fid.F_DT * value) / (fid.F_DT * 64)
        assert est[0].mean().item() == pytest.approx(value, abs=0.01)
        assert est[0].var().item() == pytest.approx(expected_var, rel=0.1)

    @pytest.mark.parametrize("value", [0.2, 0.5, 0.8])
    def test_pbit_decoder_is_unbiased_with_the_bernoulli_variance(self, value):
        est, _ = fid.pbit_estimates(_x(value), (64,), torch.Generator().manual_seed(2))
        assert est[0].mean().item() == pytest.approx(value, abs=0.01)
        assert est[0].var().item() == pytest.approx(value * (1 - value) / 64, rel=0.1)

    def test_variance_falls_as_one_over_the_dose(self):
        est, _ = fid.pbit_estimates(_x(0.5), (16, 256), torch.Generator().manual_seed(3))
        assert est[0].var().item() / est[1].var().item() == pytest.approx(16.0, rel=0.25)

    def test_doses_are_nested_prefixes_of_one_stream(self):
        gen_a, gen_b = torch.Generator().manual_seed(4), torch.Generator().manual_seed(4)
        nested, _ = fid.poisson_estimates(_x(0.5, 50), (8, 600), gen_a)
        direct = torch.zeros(50, 4, dtype=torch.float64)
        left = 600
        while left > 0:
            step = min(left, fid.CHUNK)
            direct += rate_encode(_x(0.5, 50), step, fid.F_MAX_HZ, fid.DT_S, gen_b).sum(0)
            left -= step
        assert torch.allclose(nested[1], direct / (600 * fid.F_DT))

    def test_poisson_decoder_matches_the_bridge_rate_decode(self):
        x = _x(0.4, 30)
        spikes = rate_encode(x, 100, fid.F_MAX_HZ, fid.DT_S, torch.Generator().manual_seed(5))
        est, _ = fid.poisson_estimates(x, (100,), torch.Generator().manual_seed(5))
        assert torch.allclose(est[0], rate_decode(spikes, fid.F_MAX_HZ, fid.DT_S))

    def test_pbit_rule_matches_the_block_gibbs_sampler_on_a_shared_bias(self):
        value, n = 0.3, 6
        bias = torch.full((n,), float(np.arctanh(2 * value - 1)), dtype=torch.float64)
        sampler = BlockGibbsSampler(torch.zeros(n, n, dtype=torch.float64), bias, beta=1.0, seed=6)
        spins = sampler.random_spins(4000)
        mean_up = (sampler.step(spins) > 0).double().mean().item()
        est, _ = fid.pbit_estimates(_x(value, 4000, n), (1,), torch.Generator().manual_seed(7))
        assert mean_up == pytest.approx(value, abs=0.02)
        assert est[0].mean().item() == pytest.approx(mean_up, abs=0.03)

    def test_extreme_inputs_do_not_break_the_pbit_channel(self):
        est, _ = fid.pbit_estimates(torch.tensor([[0.0, 1.0]], dtype=torch.float64), (16,), torch.Generator().manual_seed(8))
        assert torch.isfinite(est[0]).all() and est[0][0, 0] < 0.2 and est[0][0, 1] > 0.8

    def test_measured_spike_count_follows_the_rate(self):
        _, spikes = fid.poisson_estimates(_x(0.5), (256,), torch.Generator().manual_seed(9))
        assert spikes[0] == pytest.approx(256 * fid.F_DT * 0.5, rel=0.02)


class TestReadout:
    def _data(self, noise: float, seed: int = 0):
        rec = synthetic_recording(seed, separation=1.0, nuisance=noise, dim=32, n_trials=90)
        return torch.sigmoid(rec.streams["synthetic"].double()), rec.labels, rec.trials

    def test_readout_learns_separable_classes(self):
        x, y, _ = self._data(0.0)
        assert (fid.Readout(x, y, 1.0).predict(x) == y).float().mean().item() > 0.95

    def test_selection_never_does_worse_than_the_legacy_penalty_and_moves_off_it_on_low_signal(self):
        rec = synthetic_recording(0, separation=0.15, nuisance=0.0, dim=200, n_trials=120)
        x, y, t = torch.sigmoid(rec.streams["synthetic"].double()), rec.labels, rec.trials
        chosen = fid.select_penalty(x, y, t, seed=0)
        folds = fid.make_folds(t, 0)

        def error(penalty):
            return sum(int((fid.Readout(x[~h], y[~h], penalty).predict(x[h]) != y[h]).sum()) for h in folds)

        assert chosen < 100.0 and error(chosen) <= error(100.0)

    def test_ties_go_to_the_larger_penalty(self):
        x = torch.zeros(40, 3, dtype=torch.float64)
        y, t = torch.arange(40) % 6, torch.arange(40)
        assert fid.select_penalty(x, y, t, seed=0) == fid.PENALTIES[-1]


class TestStatistics:
    def test_rho_is_one_at_clean_and_zero_at_chance(self):
        rng = np.random.default_rng(0)
        clean = np.full(200, 0.6)
        assert fid.bootstrap_rho(clean, clean, rng)["rho"] == pytest.approx(1.0)
        assert fid.bootstrap_rho(clean, np.full(200, fid.CHANCE), rng)["rho"] == pytest.approx(0.0)

    def test_interval_is_wider_for_fewer_trials_and_contains_the_estimate(self):
        rng = np.random.default_rng(1)
        gen = np.random.default_rng(2)
        wide = fid.bootstrap_rho(gen.binomial(1, 0.6, 40).astype(float), gen.binomial(1, 0.5, 40).astype(float), rng)
        narrow = fid.bootstrap_rho(gen.binomial(1, 0.6, 2000).astype(float), gen.binomial(1, 0.5, 2000).astype(float), rng)
        assert wide["rho_high"] - wide["rho_low"] > narrow["rho_high"] - narrow["rho_low"]
        assert narrow["rho_low"] <= narrow["rho"] <= narrow["rho_high"]

    def test_estimable_needs_five_points_above_chance_at_the_lower_bound(self):
        assert not fid.estimable(fid.CHANCE + 0.05) and fid.estimable(fid.CHANCE + 0.06)

    def test_first_dose_returns_the_smallest_dose_or_none(self):
        rows = [{"rho": r} for r in (0.1, 0.5, 0.91, 0.96, 0.99, 1.0)]
        assert fid.first_dose(rows, 0.90, "rho") == fid.LADDER[2]
        assert fid.first_dose(rows, 0.95, "rho") == fid.LADDER[3]
        assert fid.first_dose([{"rho": 0.2}] * 6, 0.9, "rho") is None
        assert fid.first_dose([{"rho": float("nan")}] * 6, 0.9, "rho") is None


class TestEvaluation:
    def _recording(self) -> Recording:
        return synthetic_recording(0, separation=1.0, nuisance=0.0, dim=48, n_trials=60)

    def test_curve_rises_with_dose_and_reaches_clean_accuracy(self):
        out = fid.evaluate_recording(self._recording(), seed=3)["synthetic"]
        acc = {k: float(np.mean(v)) for k, v in out["trial_accuracy"].items()}
        for channel in fid.CHANNELS:
            curve = [acc[f"{channel}:{d}"] for d in fid.LADDER]
            assert curve[-1] >= curve[0] and curve[-1] >= acc["clean"] - 0.05
        assert acc["clean"] > 0.9

    def test_every_trial_is_scored_once_per_condition(self):
        out = fid.evaluate_recording(self._recording(), seed=3)["synthetic"]
        assert all(len(v) == 60 for v in out["trial_accuracy"].values())
        assert len(out["penalties"]) == 5 and set(out["mean_spikes_per_dim"]) == set(fid.CHANNELS)

    def test_clean_beats_legacy_penalty_on_a_low_signal_stream(self):
        rec = synthetic_recording(1, separation=0.15, nuisance=0.0, dim=200, n_trials=120)
        out = fid.evaluate_recording(rec, seed=4)["synthetic"]["trial_accuracy"]
        assert np.mean(out["clean"]) >= np.mean(out["legacy100"]) - 0.02

    def test_summary_marks_a_chance_stream_not_estimable(self):
        rng = np.random.default_rng(0)
        chance = {k: np.full(120, fid.CHANCE) for k in ["clean", "legacy100"] + [f"{c}:{d}" for c in fid.CHANNELS for d in fid.LADDER]}
        assert fid.summarise_model_stream(chance, rng)["estimable"] is False


class TestCli:
    def test_help_runs_standalone(self):
        done = subprocess.run([sys.executable, "scripts/analysis/probe_thermodynamic_fidelity.py", "--help"],
                              cwd=ROOT, capture_output=True, text=True)
        assert done.returncode == 0 and "record" in done.stdout

    def test_evaluate_and_summarise_round_trip(self, tmp_path):
        rec = synthetic_recording(0, separation=1.0, dim=48, n_trials=60)
        torch.save({"streams": rec.streams, "labels": rec.labels, "trials": rec.trials}, fid.recording_path(tmp_path, 42, 300))
        fid.main(["evaluate", "--out-dir", str(tmp_path)])
        assert fid.result_path(tmp_path, 42, 300).exists()
        out = tmp_path / "s.json"
        fid.main(["summarise", "--out-dir", str(tmp_path), "--output-json", str(out)])
        summary = json.loads(out.read_text())
        entry = summary["models"]["42"]["synthetic"]
        assert summary["status"] == "UNPROVEN" and entry["estimable"] is True
        assert set(entry["poisson"]) >= {"curve", "T90", "T95"} and len(entry["pbit"]["curve"]) == len(fid.LADDER)

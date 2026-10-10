"""Tests for the offline thermodynamic profiler, on synthetic tensors, CPU only.

Closed forms used.
    Entropy production per sweep is beta * (E_t - E_{t+1}), so it sums to beta * (E_0 - E_T).
    One stored pattern xi gives J = (xi xi^T - I) / n and H(xi) = -(n - 1) / 2, the minimum.
    The overlap of a pattern with itself is 1, and of its negation -1.
    With P_o = I and P_s = I, the posterior mean is (W^T W + I)^-1 W^T o.
    Under a trial-level label shuffle, accuracy has mean 1 / K for K balanced classes.
"""
import json
import os
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_thermodynamic_cost as probe  # noqa: E402
from models.thermodynamic.energy_minimization import relax_free_energy  # noqa: E402
from models.thermodynamic.p_bit_emulator import ising_energy  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSUMED = {"gpu_flops_per_joule": 1e10, "pbit_power_w": 1.0, "pbit_rate_hz": 5e7}


def _pattern(n: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.where(torch.rand(1, n, generator=gen, dtype=torch.float64) > 0.5, 1.0, -1.0).to(torch.float64)


class TestNormalisation:
    def test_small_vectors_get_zero_mean_unit_variance_from_training_rows(self):
        x = torch.randn(80, 20, generator=torch.Generator().manual_seed(0)) * 3 + 5
        ztr, _ = probe.normalise(x[:60], x[60:])
        assert torch.allclose(ztr.mean(0), torch.zeros(20, dtype=torch.float64), atol=1e-9)
        assert torch.allclose(ztr.std(0), torch.ones(20, dtype=torch.float64), atol=1e-9)

    def test_large_vectors_are_reduced_with_training_fold_statistics(self):
        x = torch.randn(40, 600, generator=torch.Generator().manual_seed(1))
        ztr, zte = probe.normalise(x[:30], x[30:])
        assert ztr.shape == (30, 30) and zte.shape == (10, 30)
        assert ztr.std().item() == pytest.approx(1.0, abs=1e-9)

    def test_quantize_maps_zero_to_plus_one(self):
        assert probe.quantize(torch.tensor([-0.5, 0.0, 2.0])).tolist() == [-1.0, 1.0, 1.0]

    def test_spins_are_plus_minus_one_and_deterministic_per_seed(self):
        z = torch.randn(5, 30, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
        a, b = probe.to_spins(z, 3), probe.to_spins(z, 3)
        assert torch.equal(a, b) and set(a.unique().tolist()) <= {-1.0, 1.0}

    def test_strong_positive_input_transduces_to_plus_one(self):
        assert torch.all(probe.to_spins(torch.full((4, 16), 6.0, dtype=torch.float64), 0) == 1.0)


class TestMemory:
    def test_stored_pattern_is_the_energy_minimum(self):
        xi = _pattern(32, 0)
        energy = ising_energy(xi[0], probe.hebbian_couplings(xi), torch.zeros(32, dtype=torch.float64))
        assert energy.item() == pytest.approx(-(32 - 1) / 2)

    def test_overlap_of_pattern_with_itself_is_one(self):
        xi = _pattern(40, 1)
        assert probe.overlap(xi, xi).item() == pytest.approx(1.0)
        assert probe.overlap(-xi, xi).item() == pytest.approx(-1.0)

    def test_missing_class_raises(self):
        with pytest.raises(ValueError):
            probe.class_means(torch.randn(4, 3), torch.tensor([0, 0, 1, 1]), 3)

    def test_corrupted_prototype_settles_back_to_its_class(self):
        protos = torch.cat([_pattern(128, s) for s in range(6)])
        flips = torch.rand(6, 128, generator=torch.Generator().manual_seed(5)) < 0.25
        trace = probe.settle(protos, torch.where(flips, -protos, protos), beta=4.0, sweeps=20, seed=1)
        assert torch.equal(probe.predict_by_overlap(trace["spins"], protos), torch.arange(6))
        assert trace["energy"][-1].mean() < trace["energy"][0].mean()

    def test_settle_does_not_modify_its_input(self):
        protos = _pattern(32, 3)
        start = -protos.clone()
        probe.settle(protos, start, 4.0, 5, 0)
        assert torch.equal(start, -protos)

    def test_entropy_production_telescopes(self):
        energy = torch.tensor([[0.0], [-1.0], [-3.0], [-2.0]])
        assert probe.entropy_production(energy, beta=2.0).sum().item() == pytest.approx(4.0)


class TestFolds:
    def test_no_trial_is_split_across_folds(self):
        trials = torch.arange(30).repeat_interleave(4)
        folds = probe.make_folds(trials, seed=0)
        assert len(folds) == probe.N_FOLDS
        assert torch.stack(folds).sum(0).eq(1).all()
        for mask in folds:
            for t in trials[mask].unique():
                assert mask[trials == t].all()

    def test_label_permutation_keeps_counts_and_trial_consistency(self):
        trials = torch.arange(24).repeat_interleave(3)
        labels = (torch.arange(24) % 6).repeat_interleave(3)
        out = probe.permute_labels(labels, trials, torch.Generator().manual_seed(0))
        assert torch.equal(torch.bincount(out), torch.bincount(labels))
        for t in range(24):
            assert out[trials == t].unique().numel() == 1


class TestPipeline:
    def _stream(self, separation: float):
        rec = probe.synthetic_recording(0, separation=separation)
        return rec.streams["synthetic"], rec.labels, rec.trials

    def test_separated_classes_survive_transduction(self):
        z, y, t = self._stream(1.0)
        folds = probe.prepare_folds(z, t, 0)
        acc = probe.run_arms(folds, y, 6, 4.0, 20, 0, ("reference", "ridge", "quantizer", "settled", "control"))
        assert acc["ridge"] > 0.9
        assert acc["reference"] > 0.9 and acc["settled"] >= acc["reference"] - probe.MARGIN
        assert acc["control"] < 0.5

    def test_shuffled_labels_fall_to_chance(self):
        z, y, t = self._stream(1.0)
        null = probe.permutation_null(probe.prepare_folds(z, t, 0), y, t, 6, 4.0, 10, 0, n_perm=12)
        assert sum(null["settled"]) / 12 == pytest.approx(1 / 6, abs=0.06)

    def test_stream_without_class_information_is_untestable(self):
        z, y, t = self._stream(0.0)
        profile = probe.profile_stream(z, y, t, 0, 4.0, 10, 12, ASSUMED)
        assert not probe.gate(profile)["G1"]
        assert probe.stream_verdict([profile] * 3).startswith("UNTESTABLE")

    def test_fewer_than_three_seeds_is_a_hypothesis(self):
        fake = {"accuracy": {"reference": 1, "settled": 1, "control": 0}, "null_p95": {"reference": .3, "settled": .3}}
        assert probe.stream_verdict([fake]).startswith("HYPOTHESIS")
        assert probe.stream_verdict([fake] * 3) == "PASSED"

    def test_high_control_accuracy_fails_the_gate(self):
        bad = {"accuracy": {"reference": 1, "settled": 1, "control": 0.9}, "null_p95": {"reference": .3, "settled": .3}}
        assert not probe.gate(bad)["G4"]
        assert probe.stream_verdict([bad] * 3) == "FAILED"

    def test_settled_accuracy_below_margin_fails_g2(self):
        low = {"accuracy": {"reference": 1, "settled": 0.7, "control": 0}, "null_p95": {"reference": .3, "settled": .3}}
        assert not probe.gate(low)["G2"]


class TestFreeEnergy:
    def test_relaxation_reaches_closed_form_posterior(self):
        batch = torch.randn(20, 10, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
        model = probe.pca_model(batch, n_latent=3, obs_weight=1.0)
        W = model.weights
        exact = torch.linalg.solve(W.T @ W + torch.eye(3, dtype=torch.float64), W.T @ batch[0])
        step = 0.5 * probe.stable_step_bound(model)
        traj = relax_free_energy(model.prior_mean.clone(), batch[0], model, step, 200, "heun")
        assert torch.allclose(traj.mu[-1], exact, atol=1e-6)

    def test_trace_delta_is_negative(self):
        batch = torch.randn(20, 10, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
        assert probe.fep_relaxation_trace(probe.pca_model(batch, 3, 1.0), batch[0], 50)["delta_free_energy"] < 0


class TestEfficiency:
    def test_ratio_is_linear_in_gpu_efficiency(self):
        kw = dict(n_spins=64, n_chains=8, n_sweeps=10, n_colors=3, pbit_power_w=1.0, pbit_rate_hz=5e7)
        a = probe.energy_estimate(gpu_flops_per_joule=1e10, **kw)["ratio_gpu_over_pbit"]
        b = probe.energy_estimate(gpu_flops_per_joule=2e10, **kw)["ratio_gpu_over_pbit"]
        assert a == pytest.approx(2 * b)

    def test_pbit_time_is_sweeps_times_colors_over_rate(self):
        est = probe.energy_estimate(10, 1, 100, 4, 1e10, 1.0, 5e7)
        assert est["pbit_seconds_per_chain"] == pytest.approx(100 * 4 / 5e7)


class TestCli:
    ARGS = ["--synthetic", "--sweeps", "8", "--permutations", "6"]

    def test_fewer_than_three_seeds_is_a_hypothesis(self):
        report = probe.run(probe.build_parser().parse_args(self.ARGS + ["--seed", "0"]))
        assert report["verdicts"]["synthetic"].startswith("HYPOTHESIS")

    def test_json_report_has_required_fields(self, tmp_path):
        out = tmp_path / "report.json"
        probe.main(self.ARGS + ["--seed", "0", "1", "2", "--output-json", str(out)])
        report = json.loads(out.read_text())
        assert report["status"] == "UNPROVEN"
        first = report["streams"]["synthetic"][0]
        assert {"pbit_energy", "thermodynamic_entropy", "fep_free_energy", "accuracy", "null_p95"} <= set(first)

    def test_missing_source_exits(self):
        with pytest.raises(SystemExit):
            probe.run(probe.build_parser().parse_args([]))

    def test_checkpoint_count_must_match_seeds(self):
        with pytest.raises(SystemExit):
            probe.run(probe.build_parser().parse_args(["--checkpoint", "a", "b", "--seed", "0", "1", "2"]))

    def test_help_runs_standalone(self):
        done = subprocess.run([sys.executable, "scripts/analysis/probe_thermodynamic_cost.py", "--help"],
                              cwd=ROOT, capture_output=True, text=True)
        assert done.returncode == 0 and "--output-json" in done.stdout


class TestGateV2:
    """Gate v2 reads a discriminative subspace. Nuisance directions without class content are the
    failure that gate v1 showed on real checkpoints, so the synthetic case includes them."""

    def _fold_inputs(self, nuisance: float):
        rec = probe.synthetic_recording(0, separation=0.5, nuisance=nuisance)
        z = rec.streams["synthetic"]
        return probe.prepare_folds(z, rec.trials, 0), rec.labels, rec

    def test_basis_is_orthonormal_and_spans_the_ridge_weights(self):
        rec = probe.synthetic_recording(1, separation=0.5)
        z = rec.streams["synthetic"].double()
        ztr, _ = probe.normalise(z, z[:1])
        basis = probe.discriminative_basis(ztr, rec.labels, 6)
        weights = probe.ridge_weights(ztr, rec.labels, 6)
        assert torch.allclose(basis.T @ basis, torch.eye(6, dtype=torch.float64), atol=1e-9)
        assert torch.allclose(basis @ (basis.T @ weights), weights, atol=1e-9)

    def test_ridge_weights_solve_the_normal_equations(self):
        gen = torch.Generator().manual_seed(4)
        z = torch.randn(50, 8, generator=gen, dtype=torch.float64)
        y = torch.arange(50) % 3
        w = probe.ridge_weights(z, y, 3)
        onehot = torch.nn.functional.one_hot(y, 3).double()
        residual = (z.T @ z + probe.RIDGE_LAMBDA * torch.eye(8, dtype=torch.float64)) @ w - z.T @ onehot
        assert residual.abs().max() < 1e-9

    def test_subspace_memory_survives_nuisance_that_defeats_the_centroid_memory(self):
        folds, labels, _ = self._fold_inputs(nuisance=1.0)
        acc = probe.run_arms(folds, labels, 6, 4.0, 20, 0,
                             ("reference", "ridge", "settled", "disc_settled", "disc_control"))
        assert acc["ridge"] > 0.9
        assert acc["reference"] < 0.5 and acc["settled"] < 0.5
        assert acc["disc_settled"] > acc["settled"] + 0.3
        assert acc["disc_control"] < 0.3

    def test_v2_null_is_refitted_and_sits_at_chance(self):
        folds, labels, rec = self._fold_inputs(nuisance=1.0)
        null = probe.permutation_null(folds, labels, rec.trials, 6, 4.0, 10, 0, 10, probe.NULL_ARMS["v2"])
        assert set(null) == {"ridge", "disc_settled"}
        assert sum(null["disc_settled"]) / 10 == pytest.approx(1 / 6, abs=0.07)

    def test_v2_profile_passes_on_separable_synthetic_data(self):
        rec = probe.synthetic_recording(0, separation=0.5)
        profile = probe.profile_stream(rec.streams["synthetic"], rec.labels, rec.trials, 0, 4.0, 15, 10,
                                       ASSUMED, "v2")
        assert profile["gate_version"] == "v2" and set(probe.gate(profile)) == {"H1", "H2", "H3", "H4"}
        assert all(probe.gate(profile).values())

    def test_v2_gate_logic(self):
        ok = {"gate_version": "v2", "accuracy": {"ridge": 0.8, "disc_settled": 0.75, "disc_control": 0.15},
              "null_p95": {"ridge": 0.25, "disc_settled": 0.25}}
        assert probe.stream_verdict([ok] * 3) == "PASSED"
        weak = {**ok, "accuracy": {"ridge": 0.2, "disc_settled": 0.75, "disc_control": 0.15}}
        assert probe.stream_verdict([weak] * 3).startswith("UNTESTABLE")
        lost = {**ok, "accuracy": {"ridge": 0.8, "disc_settled": 0.5, "disc_control": 0.15}}
        assert not probe.gate(lost)["H2"] and probe.stream_verdict([lost] * 3) == "FAILED"
        leaky = {**ok, "accuracy": {"ridge": 0.8, "disc_settled": 0.75, "disc_control": 0.6}}
        assert not probe.gate(leaky)["H4"]

    def test_cli_defaults_to_gate_v2(self):
        args = probe.build_parser().parse_args(["--synthetic"])
        assert args.gate == "v2" and args.episodes == 120


class TestProjectionRule:
    def _correlated(self):
        gen = torch.Generator().manual_seed(7)
        base = torch.where(torch.rand(1, 64, generator=gen, dtype=torch.float64) > 0.5, 1.0, -1.0)
        flips = torch.rand(4, 64, generator=gen, dtype=torch.float64) < 0.25
        return torch.where(flips, -base, base)

    def test_projection_couplings_are_valid_for_the_sampler(self):
        J = probe.projection_couplings(self._correlated())
        assert torch.equal(J, J.T) and torch.count_nonzero(torch.diagonal(J)) == 0

    def test_projector_maps_each_stored_pattern_to_itself(self):
        P = self._correlated()
        projector = P.T @ torch.linalg.solve(P @ P.T, P)
        assert torch.allclose(P @ projector, P, atol=1e-9)

    def test_projection_stores_correlated_patterns_as_fixed_points(self):
        P = self._correlated()
        J = probe.projection_couplings(P)
        assert torch.equal(torch.where(P @ J >= 0, 1.0, -1.0), P)

    def test_unknown_rule_raises(self):
        with pytest.raises(KeyError):
            probe.settle(self._correlated(), self._correlated(), 4.0, 1, 0, rule="nope")

    def test_duplicate_prototypes_do_not_raise(self):
        P = self._correlated()
        P[1] = P[0]
        J = probe.projection_couplings(P)
        assert torch.equal(J, J.T) and torch.isfinite(J).all()

    def test_partial_report_is_written_after_each_seed(self, tmp_path):
        out = tmp_path / "r.json"
        probe.main(["--synthetic", "--sweeps", "4", "--permutations", "3", "--seed", "0", "1", "--output-json", str(out)])
        assert json.loads(out.read_text())["complete"] is True
        assert json.loads(out.with_suffix(".partial.json").read_text())["complete"] is False

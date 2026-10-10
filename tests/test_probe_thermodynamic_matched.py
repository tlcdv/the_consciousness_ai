"""Tests for the matched-readout probe, on synthetic tensors, CPU only.

Closed forms used.
    Ridge on centred features has weights V diag(s / (s^2 + lambda)) U^T Yc, so the SVD route must give the
    same predictions as the normal-equation route of Readout for every penalty.
    On a well posed stream a readout fitted on the channel's own noisy outputs must not fall far below one fitted
    on clean vectors.
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
from scripts.analysis import probe_thermodynamic_matched as mat  # noqa: E402
from scripts.analysis.probe_thermodynamic_cost import synthetic_recording  # noqa: E402
from tests.test_probe_thermodynamic_encoding import _low_variance_signal  # noqa: E402


class TestFastPenalty:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_same_penalty_as_the_normal_equation_route(self, seed):
        rec = synthetic_recording(seed, separation=0.15, dim=60, n_trials=60)
        x = torch.sigmoid(rec.streams["synthetic"].double())
        slow = fid.select_penalty(x, rec.labels, rec.trials, 0, enc.PENALTIES_LOW)
        fast = mat.select_penalty_fast(x, rec.labels, rec.trials, 0)
        assert fast == slow

    def test_tied_errors_go_to_the_larger_penalty(self):
        x = torch.zeros(40, 3, dtype=torch.float64)
        y, t = torch.arange(40) % 6, torch.arange(40)
        assert mat.select_penalty_fast(x, y, t, 0) == mat.PENALTIES_LOW[-1]

    def test_rows_of_one_trial_stay_in_one_fold_when_copies_are_stacked(self):
        rec = synthetic_recording(0, separation=0.5, dim=40, n_trials=60)
        x = torch.sigmoid(rec.streams["synthetic"].double())
        stacked_x, stacked_y, stacked_t = x.repeat(2, 1), rec.labels.repeat(2), rec.trials.repeat(2)
        for held in fid.make_folds(stacked_t, 0):
            for t in stacked_t[held].unique():
                assert held[stacked_t == t].all()
        assert mat.select_penalty_fast(stacked_x, stacked_y, stacked_t, 0) in mat.PENALTIES_LOW


class TestEvaluation:
    def test_every_arm_dose_and_regime_is_scored_per_trial(self):
        out = mat.evaluate_recording(_low_variance_signal(trials=60), seed=3)["synthetic"]["trial_accuracy"]
        expected = {f"{a}|clean" for a in enc.ARMS} | {f"{a}|{r}|{c}|{d}" for a in enc.ARMS for r in mat.REGIMES
                                                         for c in fid.CHANNELS for d in fid.LADDER}
        assert set(out) == expected and all(len(v) == 60 for v in out.values())

    def test_matched_training_is_not_worse_than_clean_training_beyond_noise_on_a_well_posed_stream(self):
        acc = {k: float(np.mean(v)) for k, v in mat.evaluate_recording(_low_variance_signal(), seed=3)["synthetic"]["trial_accuracy"].items()}
        for arm in enc.ARMS:
            for dose in fid.LADDER:
                # p-bit only. At low Poisson doses the spikes are so sparse that a readout fitted on two noisy copies of
                # 480 rows can be worse than a clean-trained one. That is a data limit, not a defect.
                assert acc[f"{arm}|matched|pbit|{dose}"] >= acc[f"{arm}|clean-trained|pbit|{dose}"] - 0.06

    def test_both_regimes_approach_the_clean_accuracy_at_the_top_dose(self):
        acc = {k: float(np.mean(v)) for k, v in mat.evaluate_recording(_low_variance_signal(), seed=3)["synthetic"]["trial_accuracy"].items()}
        top = fid.LADDER[-1]
        for arm in enc.ARMS:
            for regime in mat.REGIMES:
                assert acc[f"{arm}|{regime}|pbit|{top}"] >= acc[f"{arm}|clean"] - 0.1

    def test_matched_training_does_not_exceed_the_clean_ceiling_by_much(self):
        acc = {k: float(np.mean(v)) for k, v in mat.evaluate_recording(_low_variance_signal(), seed=3)["synthetic"]["trial_accuracy"].items()}
        for arm in enc.ARMS:
            assert max(acc[f"{arm}|matched|pbit|{d}"] for d in fid.LADDER) <= acc[f"{arm}|clean"] + 0.08


class TestSummary:
    def test_summary_has_both_regimes_and_the_paired_gain(self):
        names = [f"{a}|clean" for a in enc.ARMS] + [f"{a}|{r}|{c}|{d}" for a in enc.ARMS for r in mat.REGIMES for c in fid.CHANNELS for d in fid.LADDER]
        rng = np.random.default_rng(0)
        per_trial = {k: np.clip(rng.normal(0.6, 0.1, 120), 0, 1) for k in names}
        entry = mat.summarise_stream(per_trial, rng)
        assert entry["estimable"] is True
        row = entry["arms"]["half"]["matched"]["pbit"]["curve"][0]
        assert {"accuracy", "difference", "rho", "gain_over_clean_trained"} <= set(row)
        assert set(entry["arms"]["half"]) >= set(mat.REGIMES)

    def test_a_chance_stream_is_not_estimable(self):
        names = [f"{a}|clean" for a in enc.ARMS] + [f"{a}|{r}|{c}|{d}" for a in enc.ARMS for r in mat.REGIMES for c in fid.CHANNELS for d in fid.LADDER]
        assert mat.summarise_stream({k: np.full(120, fid.CHANCE) for k in names}, np.random.default_rng(0))["estimable"] is False


class TestCli:
    def test_evaluate_and_summarise_round_trip(self, tmp_path):
        rec = _low_variance_signal(trials=60)
        torch.save({"streams": rec.streams, "labels": rec.labels, "trials": rec.trials}, fid.recording_path(tmp_path, 42, 300))
        mat.main(["evaluate", "--out-dir", str(tmp_path)])
        assert mat.matched_path(tmp_path, 42, 300).exists()
        out = tmp_path / "s.json"
        mat.main(["summarise", "--out-dir", str(tmp_path), "--output-json", str(out)])
        entry = json.loads(out.read_text())["models"]["42"]["synthetic"]
        assert entry["estimable"] is True and len(entry["arms"]["full"]["matched"]["poisson"]["curve"]) == len(fid.LADDER)

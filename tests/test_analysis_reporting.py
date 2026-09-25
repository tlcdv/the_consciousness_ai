"""
Analysis scripts must report what they computed and nothing else.

1. compare_experiments wrote a fixed "Findings" section, with numbers, into
   docs/results/experiment_comparison.md on every run, whatever the run folders held.
2. probe_gate_b2's task criterion compared the null against a NaN rank correlation.
   Nothing is >= NaN, so the permutation p came out as 1/2001, the smallest possible.
3. report_signatures counted every phi row except "skipped" and "insufficient_data"
   as computed, so rows from a pyphi error or with no gate entered the statistics.
4. aggregate_seeds reported a peak r of 0.0 when no window could be scored, a value
   that enters the cross seed mean as if measured.
"""
from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# --- 1. compare_experiments ------------------------------------------------------

def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def test_comparison_report_holds_computed_tables_only(tmp_path, monkeypatch):
    from scripts.analysis import compare_experiments

    run = tmp_path / "runs" / "agent"
    _write_csv(run / "episodes.csv", ["episode", "total_reward", "avg_phi"],
               [[0, 1.0, 0.1], [1, 3.0, 0.3]])
    baseline = tmp_path / "baseline.csv"
    _write_csv(baseline, ["episode", "reward", "steps", "epsilon"],
               [[0, 5.0, 10, 1.0], [1, 7.0, 10, 0.9]])
    monkeypatch.setattr(compare_experiments, "ENVIRONMENTS", {
        "dark_room": {"consciousness_dir": str(run), "dqn_csv": str(baseline),
                      "label": "Dark Room", "note": ""},
    })

    rows = compare_experiments.compare_all(str(tmp_path / "out"))
    report = (tmp_path / "out" / "experiment_comparison.md").read_text()

    assert rows[0]["c_first100"] == 2.0 and rows[0]["d_first100"] == 6.0
    assert "| Dark Room | 2 | 2.00 | 2.00 | 2 | 6.00 | 6.00 |" in report
    for fixed_text in ("Findings", "DQN outperforms", "sync_R range", "200ms"):
        assert fixed_text not in report


# --- 2. probe_gate_b2 task criterion ------------------------------------------------

def _tables(audio_share):
    """Three seeds of six qualifying episodes, each row (episode, f, a, ignited)."""
    return [[(ep, ep / 10.0, audio_share(seed, ep), 40) for ep in range(6)]
            for seed in range(3)]


def test_task_with_an_undefined_rank_correlation_is_not_measurable():
    from scripts.analysis.probe_gate_b2 import task, task_failures

    result = task(_tables(lambda seed, ep: 0.0))  # hearing never wins
    assert result["measurable"] is False
    assert "p_one_sided" not in result
    assert task_failures(result) == ["(4) task not measurable"]


def test_task_with_a_defined_rank_correlation_still_gets_a_p_value():
    from scripts.analysis.probe_gate_b2 import task

    result = task(_tables(lambda seed, ep: ep / 10.0 + seed / 100.0))
    assert result["measurable"] is True
    assert result["rho_pooled"] > 0
    assert 0.0 < result["p_one_sided"] <= 1.0


# --- 3. report_signatures phi filter ------------------------------------------------

def test_signature_report_counts_only_computed_phi(tmp_path):
    from scripts.analysis.report_signatures import analyze_run

    rows = [(0.5, "pyphi"), (0.25, "proxy"), (0.0, "pyphi_error"), (0.0, "no_gate"),
            (0.5, "skipped"), (0.0, "insufficient_data"), (0.0, "")]
    _write_csv(tmp_path / "metrics.csv", ["phi", "phi_method"], rows)
    _write_csv(tmp_path / "episodes.csv", ["episode"], [])

    out = analyze_run(tmp_path)
    assert out["phi_computed"]["n"] == 2
    assert out["phi_computed"]["mean"] == pytest.approx(0.375)


# --- 4. aggregate_seeds peak r ------------------------------------------------------

def _steps(n, phi, sync):
    return pd.DataFrame({"global_step": np.arange(1000, 1000 + n),
                         "phi_riiu": phi, "sync_r": sync})


def test_peak_r_is_nan_when_no_window_can_be_scored():
    from scripts.analysis.aggregate_seeds import rolling_peak_r

    too_short = _steps(30, np.arange(30.0), np.arange(30.0))
    constant = _steps(400, np.ones(400), np.arange(400.0))
    for frame in (too_short, constant, _steps(0, [], [])):
        peak, _, _ = rolling_peak_r(frame, window=100)
        assert math.isnan(peak)


def test_peak_r_is_measured_when_a_window_qualifies():
    from scripts.analysis.aggregate_seeds import rolling_peak_r

    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    peak, start, end = rolling_peak_r(_steps(400, x, x + 0.1 * rng.normal(size=400)), 100)
    assert 0.9 < peak <= 1.0
    assert end - start == 100


def test_a_seed_without_variance_reports_nan_not_zero(tmp_path):
    from scripts.analysis.aggregate_seeds import evaluate_seed

    n = 200
    _write_csv(tmp_path / "metrics.csv",
               ["global_step", "phi", "phi_riiu", "sync_r", "phi_method"],
               [[1000 + i, 0.1, 0.2, 0.3, "pyphi"] for i in range(n)])
    result = evaluate_seed(str(tmp_path), substrate="broadcast", peak_window_size=100)
    assert math.isnan(result.full_run_r)
    assert math.isnan(result.peak_r)
    assert result.n_rows == n

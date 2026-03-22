"""
Analyze training experiment results against pre-registered predictions.

Loads CSV output from train_rlhf.py and tests each prediction from
docs/preregistered_predictions.md.

Usage:
    python -m scripts.analysis.analyze_experiment --run-dir runs/dark_room_5k
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats


def load_data(run_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load step-level and episode-level CSV data."""
    metrics_path = os.path.join(run_dir, "metrics.csv")
    episodes_path = os.path.join(run_dir, "episodes.csv")

    if not os.path.exists(metrics_path):
        print(f"ERROR: {metrics_path} not found")
        sys.exit(1)
    if not os.path.exists(episodes_path):
        print(f"ERROR: {episodes_path} not found")
        sys.exit(1)

    steps = pd.read_csv(metrics_path)
    episodes = pd.read_csv(episodes_path)
    return steps, episodes


def test_ei_1(episodes: pd.DataFrame) -> dict:
    """EI-1: EI emergence onset between episode 500 and 2000."""
    ei_rows = episodes[episodes["ei_workspace"] > 0]
    if ei_rows.empty:
        return {"prediction": "EI-1", "result": "NO DATA", "verdict": "INCONCLUSIVE",
                "detail": "No EI measurements found (log-ei-every may be too high or run too short)"}

    emergent = ei_rows[ei_rows["ei_workspace"] > ei_rows["ei_gates"]]
    if emergent.empty:
        return {"prediction": "EI-1", "result": "NEVER EMERGED", "verdict": "FAIL",
                "detail": "EI(workspace) never exceeded EI(gates) across all episodes"}

    first_ep = int(emergent["episode"].iloc[0])
    in_range = 500 <= first_ep <= 2000
    return {
        "prediction": "EI-1",
        "result": f"First emergence at episode {first_ep}",
        "verdict": "PASS" if in_range else "FAIL",
        "detail": f"Expected onset in [500, 2000], got {first_ep}"
    }


def test_ei_2(episodes: pd.DataFrame) -> dict:
    """EI-2: EI ratio stabilizes between 1.1 and 3.0 in episodes 3000-5000."""
    late = episodes[(episodes["episode"] >= 3000) & (episodes["ei_ratio"] > 0)]
    if late.empty or len(late) < 3:
        return {"prediction": "EI-2", "result": "INSUFFICIENT DATA", "verdict": "INCONCLUSIVE",
                "detail": "Not enough EI measurements after episode 3000"}

    mean_ratio = float(late["ei_ratio"].mean())
    std_ratio = float(late["ei_ratio"].std())
    in_range = 1.1 <= mean_ratio <= 3.0
    stable = std_ratio < 1.0

    return {
        "prediction": "EI-2",
        "result": f"Mean ratio={mean_ratio:.3f}, std={std_ratio:.3f}",
        "verdict": "PASS" if (in_range and stable) else "FAIL",
        "detail": f"Expected stable ratio in [1.1, 3.0]. Got mean={mean_ratio:.3f}, std={std_ratio:.3f}"
    }


def test_ei_3(episodes: pd.DataFrame) -> dict:
    """EI-3: EI emergence correlates with reward (Pearson r > 0.3)."""
    ei_rows = episodes[episodes["ei_ratio"] > 0].copy()
    if len(ei_rows) < 10:
        return {"prediction": "EI-3", "result": "INSUFFICIENT DATA", "verdict": "INCONCLUSIVE",
                "detail": f"Only {len(ei_rows)} EI measurements, need at least 10"}

    emergent_flag = (ei_rows["ei_workspace"] > ei_rows["ei_gates"]).astype(float)
    reward_median = ei_rows["total_reward"].median()
    above_median = (ei_rows["total_reward"] > reward_median).astype(float)

    r, p = stats.pearsonr(emergent_flag, above_median)
    return {
        "prediction": "EI-3",
        "result": f"r={r:.3f}, p={p:.4f}",
        "verdict": "PASS" if r > 0.3 else "FAIL",
        "detail": f"Expected Pearson r > 0.3 between emergence and above-median reward. Got r={r:.3f}"
    }


def test_phi_1(steps: pd.DataFrame) -> dict:
    """Phi-1: Phi correlates with AKOrN sync R (Pearson r > 0.4)."""
    valid = steps.dropna(subset=["phi", "sync_r"])
    if len(valid) < 50:
        return {"prediction": "Phi-1", "result": "INSUFFICIENT DATA", "verdict": "INCONCLUSIVE",
                "detail": f"Only {len(valid)} valid step measurements"}

    phi = valid["phi"].values
    sync_r = valid["sync_r"].values

    if np.std(phi) < 1e-8 or np.std(sync_r) < 1e-8:
        return {"prediction": "Phi-1", "result": "CONSTANT VALUES", "verdict": "FAIL",
                "detail": "Phi or sync_R is constant across all steps"}

    r, p = stats.pearsonr(phi, sync_r)
    return {
        "prediction": "Phi-1",
        "result": f"r={r:.3f}, p={p:.6f}",
        "verdict": "PASS" if r > 0.4 else "FAIL",
        "detail": f"Expected Pearson r > 0.4 between phi and sync_R. Got r={r:.3f}"
    }


def test_phi_2(steps: pd.DataFrame) -> dict:
    """Phi-2: Phi drops >40% when sync_R < 0.3 (zombie mode proxy)."""
    normal = steps[steps["sync_r"] >= 0.3]
    zombie = steps[steps["sync_r"] < 0.3]

    if len(normal) < 20 or len(zombie) < 20:
        return {"prediction": "Phi-2", "result": "INSUFFICIENT DATA", "verdict": "INCONCLUSIVE",
                "detail": f"Need 20+ steps in each mode. Normal: {len(normal)}, Zombie: {len(zombie)}"}

    mean_normal = float(normal["phi"].mean())
    mean_zombie = float(zombie["phi"].mean())

    if mean_normal < 1e-8:
        return {"prediction": "Phi-2", "result": "ZERO NORMAL PHI", "verdict": "FAIL",
                "detail": "Mean phi in normal mode is near zero"}

    drop_pct = (mean_normal - mean_zombie) / mean_normal * 100
    return {
        "prediction": "Phi-2",
        "result": f"Normal phi={mean_normal:.4f}, Zombie phi={mean_zombie:.4f}, Drop={drop_pct:.1f}%",
        "verdict": "PASS" if drop_pct > 40 else "FAIL",
        "detail": f"Expected >40% drop. Got {drop_pct:.1f}%"
    }


def test_phi_3(steps: pd.DataFrame) -> dict:
    """Phi-3: Phi increases with reentrant cycles. Not testable from current logs."""
    return {
        "prediction": "Phi-3",
        "result": "NOT TESTABLE",
        "verdict": "INCONCLUSIVE",
        "detail": "Per-cycle phi logging not yet implemented. Requires instrumentation of ReentrantProcessor.settle()."
    }


def test_im_1(steps: pd.DataFrame) -> dict:
    """IM-1: Phi at insight moments > mean + 1.5 SD of episode phi."""
    # Insight moments are not directly flagged in the CSV.
    # We approximate using the 4-criterion definition:
    # high reward + high broadcast + novel (can't check novelty from CSV alone).
    # For now, flag steps where reward > 1.5x rolling mean AND broadcast_mag > 75th percentile.

    if len(steps) < 100:
        return {"prediction": "IM-1", "result": "INSUFFICIENT DATA", "verdict": "INCONCLUSIVE",
                "detail": "Need at least 100 steps"}

    reward_rolling = steps["reward"].rolling(50, min_periods=1).mean()
    reward_jump = steps["reward"] > 1.5 * reward_rolling
    broadcast_75 = steps["broadcast_mag"].quantile(0.75)
    high_broadcast = steps["broadcast_mag"] >= broadcast_75

    insight_mask = reward_jump & high_broadcast
    insight_steps = steps[insight_mask]

    if len(insight_steps) < 5:
        return {"prediction": "IM-1", "result": f"Only {len(insight_steps)} candidate insights", "verdict": "INCONCLUSIVE",
                "detail": "Need at least 5 insight moments for statistical test"}

    phi_mean = float(steps["phi"].mean())
    phi_std = float(steps["phi"].std())
    insight_phi_mean = float(insight_steps["phi"].mean())

    threshold = phi_mean + 1.5 * phi_std
    above = insight_phi_mean >= threshold

    return {
        "prediction": "IM-1",
        "result": f"Insight phi mean={insight_phi_mean:.4f}, threshold={threshold:.4f}",
        "verdict": "PASS" if above else "FAIL",
        "detail": f"Expected insight phi > mean + 1.5*SD ({threshold:.4f}). Got {insight_phi_mean:.4f}"
    }


def test_im_2(episodes: pd.DataFrame) -> dict:
    """IM-2: Insight frequency 2x higher in emergent episodes."""
    # We don't have insight counts per episode in the CSV.
    # This would require additional logging.
    return {
        "prediction": "IM-2",
        "result": "NOT TESTABLE",
        "verdict": "INCONCLUSIVE",
        "detail": "Per-episode insight counts not in CSV. Add insight_count column to episodes.csv to enable."
    }


def test_im_3(steps: pd.DataFrame) -> dict:
    """IM-3: sync_R > 0.7 at 80%+ of insight moments."""
    if len(steps) < 100:
        return {"prediction": "IM-3", "result": "INSUFFICIENT DATA", "verdict": "INCONCLUSIVE",
                "detail": "Need at least 100 steps"}

    reward_rolling = steps["reward"].rolling(50, min_periods=1).mean()
    reward_jump = steps["reward"] > 1.5 * reward_rolling
    broadcast_75 = steps["broadcast_mag"].quantile(0.75)
    high_broadcast = steps["broadcast_mag"] >= broadcast_75

    insight_mask = reward_jump & high_broadcast
    insight_steps = steps[insight_mask]

    if len(insight_steps) < 5:
        return {"prediction": "IM-3", "result": f"Only {len(insight_steps)} candidate insights", "verdict": "INCONCLUSIVE",
                "detail": "Need at least 5 insight moments"}

    high_r_pct = float((insight_steps["sync_r"] > 0.7).mean()) * 100
    return {
        "prediction": "IM-3",
        "result": f"{high_r_pct:.1f}% of insights have R > 0.7",
        "verdict": "PASS" if high_r_pct >= 80 else "FAIL",
        "detail": f"Expected >=80% with R > 0.7. Got {high_r_pct:.1f}%"
    }


def run_analysis(run_dir: str):
    """Run all prediction tests and output results."""
    steps, episodes = load_data(run_dir)

    print(f"\nAnalyzing experiment: {run_dir}")
    print(f"Steps: {len(steps)}, Episodes: {len(episodes)}")
    print(f"Total episodes: {int(episodes['episode'].max()) + 1 if len(episodes) > 0 else 0}")
    print("=" * 80)

    results = [
        test_ei_1(episodes),
        test_ei_2(episodes),
        test_ei_3(episodes),
        test_phi_1(steps),
        test_phi_2(steps),
        test_phi_3(steps),
        test_im_1(steps),
        test_im_2(episodes),
        test_im_3(steps),
    ]

    # Print table
    print(f"\n{'Prediction':<12} {'Verdict':<14} {'Result':<50}")
    print("-" * 76)
    for r in results:
        verdict_color = {"PASS": "PASS", "FAIL": "FAIL", "INCONCLUSIVE": "INCONC."}
        v = verdict_color.get(r["verdict"], r["verdict"])
        print(f"{r['prediction']:<12} {v:<14} {r['result']:<50}")

    print("\n" + "=" * 80)

    # Summary counts
    passes = sum(1 for r in results if r["verdict"] == "PASS")
    fails = sum(1 for r in results if r["verdict"] == "FAIL")
    inconc = sum(1 for r in results if r["verdict"] == "INCONCLUSIVE")
    print(f"Summary: {passes} PASS, {fails} FAIL, {inconc} INCONCLUSIVE out of 9 predictions")

    # Decision protocol
    ei_pass = all(r["verdict"] == "PASS" for r in results[:3])
    phi_pass = all(r["verdict"] == "PASS" for r in results[3:6])
    im_pass = all(r["verdict"] == "PASS" for r in results[6:9])

    print("\nDecision protocol:")
    if ei_pass and phi_pass:
        print("  -> All EI and Phi confirmed. Architecture consistent with strong emergence.")
    elif ei_pass and not phi_pass:
        print("  -> EI confirmed but Phi fails. Gate subsystem may not be the right IIT substrate.")
    elif phi_pass and not ei_pass:
        print("  -> Phi confirmed but EI fails. Macro causal power not emerging despite integration.")
    else:
        print("  -> Both EI and Phi fail. Fundamental architecture revision may be needed.")

    if not im_pass and (ei_pass or phi_pass):
        print("  -> Insight predictions fail while EI/Phi confirm. Emergence present but not functionally useful.")

    # Save markdown report
    output_path = os.path.join(run_dir, "analysis_results.md")
    with open(output_path, "w") as f:
        f.write(f"# Experiment Analysis: {os.path.basename(run_dir)}\n\n")
        f.write(f"Steps analyzed: {len(steps)}\n")
        f.write(f"Episodes analyzed: {len(episodes)}\n\n")
        f.write("## Results\n\n")
        f.write(f"| Prediction | Verdict | Result |\n")
        f.write(f"|---|---|---|\n")
        for r in results:
            f.write(f"| {r['prediction']} | {r['verdict']} | {r['result']} |\n")
        f.write(f"\n## Details\n\n")
        for r in results:
            f.write(f"### {r['prediction']}\n")
            f.write(f"- Verdict: **{r['verdict']}**\n")
            f.write(f"- {r['detail']}\n\n")

    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze training experiment against pre-registered predictions")
    parser.add_argument("--run-dir", type=str, required=True, help="Directory containing metrics.csv and episodes.csv")
    args = parser.parse_args()
    run_analysis(args.run_dir)


if __name__ == "__main__":
    main()

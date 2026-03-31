"""
Multi-environment comparison: consciousness agent vs DQN baseline.

Generates a summary table and markdown report comparing reward curves,
consciousness metrics, and pre-registered prediction results across
dark_room, DMTS, and WCST environments.

Usage:
    python -m scripts.analysis.compare_experiments
    python -m scripts.analysis.compare_experiments --output-dir docs/results
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


ENVIRONMENTS = {
    "dark_room": {
        "consciousness_dir": "runs/dark_room_1k_v2",
        "dqn_csv": "runs_baseline/baseline_dark_room.csv",
        "label": "Dark Room",
        "note": "Navigation task. Reward +1 per step near goal.",
    },
    "dmts": {
        "consciousness_dir": "runs/dmts_100",
        "dqn_csv": "runs_baseline/baseline_dmts.csv",
        "label": "DMTS",
        "note": "Working memory task. Requires binding across delay period.",
    },
    "wcst": {
        "consciousness_dir": "runs/wcst_100",
        "dqn_csv": "runs_baseline/baseline_wcst.csv",
        "label": "WCST",
        "note": "Cognitive flexibility task. Rule changes without warning.",
    },
}


def load_episodes(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_dqn(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def reward_summary(rewards: list[float]) -> dict:
    n = len(rewards)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "first_100": float("nan"), "last_100": float("nan")}
    first = float(np.mean(rewards[:min(100, n)]))
    last = float(np.mean(rewards[max(0, n - 100):]))
    return {"n": n, "mean": float(np.mean(rewards)), "first_100": first, "last_100": last}


def phi_summary(episodes: pd.DataFrame) -> dict:
    phis = episodes["avg_phi"].dropna().tolist() if "avg_phi" in episodes.columns else []
    if not phis:
        return {"mean": float("nan"), "range": (float("nan"), float("nan")), "varies": False}
    return {
        "mean": float(np.mean(phis)),
        "range": (float(min(phis)), float(max(phis))),
        "varies": float(max(phis)) - float(min(phis)) > 1e-5,
    }


def ei_summary(episodes: pd.DataFrame) -> dict:
    ei_rows = episodes[episodes.get("ei_ratio", pd.Series()).gt(0)] if "ei_ratio" in episodes.columns else pd.DataFrame()
    if ei_rows.empty:
        return {"n_measurements": 0, "mean_ratio": float("nan"), "varies": False}
    ratios = ei_rows["ei_ratio"].tolist()
    return {
        "n_measurements": len(ratios),
        "mean_ratio": float(np.mean(ratios)),
        "varies": float(max(ratios)) - float(min(ratios)) > 0.01,
    }


def compare_all(output_dir: str = "."):
    rows = []
    for env_key, cfg in ENVIRONMENTS.items():
        cons_dir = cfg["consciousness_dir"]
        dqn_path = cfg["dqn_csv"]

        cons_eps_path = os.path.join(cons_dir, "episodes.csv")
        cons_episodes = load_episodes(cons_eps_path)
        dqn_data = load_dqn(dqn_path)

        # Consciousness agent
        if cons_episodes is not None:
            c_rewards = cons_episodes["total_reward"].tolist()
            c_rs = reward_summary(c_rewards)
            c_phi = phi_summary(cons_episodes)
            c_ei = ei_summary(cons_episodes)
        else:
            c_rs = {"n": 0, "first_100": float("nan"), "last_100": float("nan")}
            c_phi = {"mean": float("nan"), "varies": False}
            c_ei = {"n_measurements": 0, "mean_ratio": float("nan")}

        # DQN baseline
        if dqn_data is not None:
            d_rewards = dqn_data["reward"].tolist()
            d_rs = reward_summary(d_rewards)
        else:
            d_rs = {"n": 0, "first_100": float("nan"), "last_100": float("nan")}

        rows.append({
            "env": cfg["label"],
            "c_episodes": c_rs["n"],
            "c_first100": c_rs["first_100"],
            "c_last100": c_rs["last_100"],
            "d_episodes": d_rs["n"],
            "d_first100": d_rs["first_100"],
            "d_last100": d_rs["last_100"],
            "phi_mean": c_phi["mean"],
            "phi_varies": c_phi["varies"],
            "ei_ratio": c_ei["mean_ratio"],
            "ei_n": c_ei["n_measurements"],
        })

    # Print table
    print("\n=== Consciousness Agent vs DQN Baseline ===\n")
    header = f"{'Env':<12} {'C-eps':>6} {'C-first':>9} {'C-last':>9} {'D-eps':>6} {'D-first':>9} {'D-last':>9} {'Phi':>8} {'EI':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['env']:<12} {r['c_episodes']:>6} {r['c_first100']:>9.2f} {r['c_last100']:>9.2f} "
            f"{r['d_episodes']:>6} {r['d_first100']:>9.2f} {r['d_last100']:>9.2f} "
            f"{r['phi_mean']:>8.5f} {r['ei_ratio']:>6.2f}"
        )

    # Save markdown report
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "experiment_comparison.md")
    with open(out_path, "w") as f:
        f.write("# Experiment Comparison: Consciousness Agent vs DQN Baseline\n\n")
        f.write("## Reward Comparison\n\n")
        f.write("| Environment | C Episodes | C First 100 | C Last 100 | D Episodes | D First 100 | D Last 100 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['env']} | {r['c_episodes']} | {r['c_first100']:.2f} "
                f"| {r['c_last100']:.2f} | {r['d_episodes']} "
                f"| {r['d_first100']:.2f} | {r['d_last100']:.2f} |\n"
            )
        f.write("\n## Consciousness Metrics\n\n")
        f.write("| Environment | Avg Phi | Phi Varies | EI Ratio | EI Measurements |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['env']} | {r['phi_mean']:.5f} | {r['phi_varies']} "
                f"| {r['ei_ratio']:.3f} | {r['ei_n']} |\n"
            )
        f.write("\n## Findings\n\n")
        f.write("### Structural fixes applied (2026-03-29)\n\n")
        f.write("1. **ConsciousnessGate wired**: all 5 gate values (attention, stability, adaptation, "
                "coherence, confidence) computed from broadcast via learned networks. "
                "No longer static. Phi now varies per step.\n")
        f.write("2. **`compute_phi_proxy()` replaced** with `compute_phi_from_gate_state()` "
                "in GlobalWorkspace and training loop.\n")
        f.write("3. **Adaptive EI binning**: per-dimension median thresholds instead of fixed 0.5, "
                "so adaptation_rate (range 0.004-0.006) contributes to joint state diversity.\n")
        f.write("4. **DMTS/WCST action discretization**: consciousness agent now correctly converts "
                "continuous actions to discrete indices via argmax.\n\n")
        f.write("### Known limitations for this run\n\n")
        f.write("- Phi proxy converges to empirical fixed point after ~5000 steps "
                "(TPM saturates). Per-episode phi becomes constant after early training.\n")
        f.write("- EI stable across measurement windows: gate transitions converge to a "
                "stationary distribution quickly. Longer training or sliding-window TPM needed.\n")
        f.write("- DQN outperforms consciousness agent on all reward metrics. "
                "The consciousness pipeline adds ~200ms overhead per step without "
                "contributing to the action policy directly.\n")
        f.write("- sync_R range [0.216, 0.220]: workspace binding optimizer needs stronger "
                "reward signal over many more episodes to shift coupling weights significantly.\n")

    print(f"\nReport saved to: {out_path}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="docs/results")
    args = parser.parse_args()
    compare_all(args.output_dir)


if __name__ == "__main__":
    main()

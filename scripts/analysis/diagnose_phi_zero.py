"""Diagnose why phi=0 across all ablation runs.

Loads a completed ablation run's metrics.csv and inspects the gate state
trajectory and the empirical TPM that pyphi sees. Tests whether pyphi
returns 0 for a legitimate reason (reducible TPM) or whether the gate
state distribution is degenerate.

Output: prints one diagnostic block per run dir provided on the CLI.

Usage:
    python -m scripts.analysis.diagnose_phi_zero runs/ablation/A_current

The diagnostic does NOT load any saved gate states (none exist), so it
re-runs the iit_phi pipeline on a synthetic state distribution drawn
from the metrics.csv broadcast_mag, valence, arousal, dominance columns
to approximate what gate inputs looked like.

A second mode reproduces the exact computation by re-running the gate
network on a few hundred synthetic broadcast vectors with statistics
matching the run, then computing phi on the resulting state history.
This is the only way to know whether the empirical TPM that pyphi saw
in training was degenerate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.core.consciousness_gating import ConsciousnessGate
from models.evaluation.iit_phi import IITMetrics, GATE_CM, GATE_NODE_LABELS

import torch


def diagnose_run(run_dir: str) -> dict:
    """Load metrics.csv, replay gate network on synthetic broadcasts that
    mirror the broadcast_mag distribution, and report what pyphi sees."""
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.csv"
    if not metrics_path.exists():
        return {"error": f"missing {metrics_path}"}

    df = pd.read_csv(metrics_path)
    n = len(df)
    bm = df["broadcast_mag"].to_numpy()
    bm_mean, bm_std = float(bm.mean()), float(bm.std())

    print(f"\n=== {run_path.name} ===")
    print(f"  rows={n}  pyphi_calls={(df['phi_method']=='pyphi').sum()}")
    print(f"  broadcast_mag mean={bm_mean:.3f} std={bm_std:.3f} "
          f"min={bm.min():.3f} max={bm.max():.3f}")
    print(f"  phi mean={df['phi'].mean():.6f} std={df['phi'].std():.6f}")
    print(f"  sync_r mean={df['sync_r'].mean():.4f} std={df['sync_r'].std():.4f}")

    # --- Replay: build synthetic broadcast vectors with matching norm
    # statistics, run them through a fresh gate network, observe gate state
    # distribution and resulting empirical TPM.
    config = {"hidden_size": 256, "workspace_dim": 256,
              "gating": {"attention_threshold": 0.5,
                         "stability_threshold": 0.6,
                         "base_adaptation_rate": 0.01}}
    gate = ConsciousnessGate(config)
    gate.eval()
    iit = IITMetrics(history_len=200, tpm_window=200, tpm_decay=0.995)

    torch.manual_seed(0)
    np.random.seed(0)

    # Sample 300 broadcast magnitudes from the actual run (with replacement)
    n_steps = 300
    sampled_mags = np.random.choice(bm, size=n_steps, replace=True)

    raw_values = []  # [n_steps, 5]
    bin_states = []  # [n_steps] tuples
    phi_values = []
    phi_methods = []
    for step in range(n_steps):
        # Build a unit-norm random direction, scale to the sampled magnitude.
        # This matches the distribution of broadcast vectors that the gate
        # saw in training (we don't have the exact tensors but the norm
        # distribution is the dominant signal here).
        direction = torch.randn(256)
        direction = direction / (direction.norm() + 1e-8)
        broadcast = direction * float(sampled_mags[step])
        broadcast = broadcast.unsqueeze(0)

        with torch.no_grad():
            _, state = gate(broadcast)

        raw = np.array([state.attention_level, state.stability_score,
                        state.adaptation_rate, state.meta_memory_coherence,
                        state.narrator_confidence])
        raw_values.append(raw)
        result = iit.compute_phi_from_gate_state(state)
        phi_values.append(result.phi)
        phi_methods.append(result.method)
        bin_states.append(result.current_state)

    raw_arr = np.array(raw_values)
    print("  --- Replay (300 steps, same broadcast_mag distribution) ---")
    for i, label in enumerate(GATE_NODE_LABELS):
        col = raw_arr[:, i]
        print(f"  raw {label:12s}: mean={col.mean():.4f} std={col.std():.4f} "
              f"min={col.min():.4f} max={col.max():.4f}")

    unique_states = set(bin_states)
    print(f"  unique binarized states: {len(unique_states)}/32")
    print(f"  most common state: {pd.Series(bin_states).value_counts().head(3).to_dict()}")

    phi_arr = np.array(phi_values)
    method_counts = pd.Series(phi_methods).value_counts().to_dict()
    print(f"  phi (replay): mean={phi_arr.mean():.6f} std={phi_arr.std():.6f} "
          f"max={phi_arr.max():.6f}")
    print(f"  phi methods: {method_counts}")

    # TPM stats from the replayed history
    tpm_stats = iit.get_tpm_stats()
    print(f"  TPM stats: {tpm_stats}")

    # Final TPM diagnostics: how reducible is it?
    if len(iit.state_history) > 5:
        tpm = iit.build_empirical_tpm(5)
        print(f"  TPM shape: {tpm.shape}")
        print(f"  TPM rows that are pure 0.5 (no info): "
              f"{(np.abs(tpm - 0.5).sum(axis=1) < 0.01).sum()}/{tpm.shape[0]}")
        # Check whether any node's column is entirely uniform (which would
        # make that node trivially independent and pyphi return 0)
        col_var = tpm.var(axis=0)
        print(f"  TPM per-node column variance: "
              f"{dict(zip(GATE_NODE_LABELS, [f'{v:.4f}' for v in col_var]))}")
        # Check the connectivity matrix the run used
        print(f"  GATE_CM:")
        for i, label in enumerate(GATE_NODE_LABELS):
            print(f"    {label:12s}: {GATE_CM[i].tolist()}")

    return {
        "run": run_path.name,
        "broadcast_mag_mean": bm_mean,
        "phi_replay_mean": float(phi_arr.mean()),
        "phi_replay_max": float(phi_arr.max()),
        "unique_states": len(unique_states),
        "method_counts": method_counts,
    }


def main():
    if len(sys.argv) < 2:
        runs = ["runs/ablation/A_current",
                "runs/ablation/C_no_replay",
                "runs/ablation/G_no_rnd_zero",
                "runs/ablation/E_no_div"]
    else:
        runs = sys.argv[1:]

    summaries = []
    for r in runs:
        try:
            s = diagnose_run(r)
            summaries.append(s)
        except Exception as e:
            import traceback
            print(f"ERROR processing {r}: {e}")
            traceback.print_exc()

    print("\n=== Summary ===")
    for s in summaries:
        if "error" in s:
            print(f"  {s}")
            continue
        print(f"  {s['run']:18s}: phi_replay_mean={s['phi_replay_mean']:.6f} "
              f"max={s['phi_replay_max']:.6f}  unique_states={s['unique_states']}/32")


if __name__ == "__main__":
    main()

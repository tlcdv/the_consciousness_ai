"""Stage 4 diagnostic: phi as a function of TPM transition count.

train_rlhf.py calls iit_metrics.reset_tpm() at every episode start. So phi
is computed on a TPM that grows from 5 transitions (first valid call) up
to 200 (full window). With Laplace smoothing alpha=0.1, a sparse TPM is
near-uniform and pyphi finds the system reducible (returns phi=0).

This script populates state_history incrementally and reports phi at each
step, mimicking the per-episode growth pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.core.consciousness_gating import ConsciousnessGate
from models.evaluation.iit_phi import IITMetrics

import pyphi


def main():
    config = {"hidden_size": 256, "workspace_dim": 256,
              "gating": {"attention_threshold": 0.5,
                         "stability_threshold": 0.6,
                         "base_adaptation_rate": 0.01}}
    torch.manual_seed(0)
    np.random.seed(0)

    gate = ConsciousnessGate(config)
    iit = IITMetrics(history_len=200, tpm_window=200, tpm_decay=0.995)
    optimizer = torch.optim.Adam(gate.parameters(), lr=3e-4)

    df = pd.read_csv("runs/ablation/A_current/metrics.csv")
    bm = df["broadcast_mag"].to_numpy()

    # Pre-train gate to mimic actual training-time gate dynamics
    print("Pre-training gate with diversity loss for 1000 steps...")
    train_mags = np.random.choice(bm, size=1000, replace=True)
    for step, mag in enumerate(train_mags):
        d = torch.randn(256)
        d = d / (d.norm() + 1e-8)
        broadcast = (d * float(mag)).unsqueeze(0)
        _, _ = gate(broadcast)
        if step % 5 == 0:
            gv = gate.last_gate_values_tensor
            if gv is not None:
                loss = -torch.log(torch.abs(gv - 0.5).clamp(min=0.01)).mean() * 0.05
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
                optimizer.step()

    # Simulate one episode start: reset_tpm, then incrementally compute phi
    iit.reset_tpm()
    print("\nEpisode 1 simulation (200 steps), reset_tpm called at start:")
    print(f"{'step':>5} {'transitions':>12} {'unique':>8} "
          f"{'phi':>10} {'method':>20}")

    test_mags = np.random.choice(bm, size=200, replace=True)
    for step, mag in enumerate(test_mags):
        d = torch.randn(256)
        d = d / (d.norm() + 1e-8)
        broadcast = (d * float(mag)).unsqueeze(0)
        with torch.no_grad():
            _, state = gate(broadcast)
        result = iit.compute_phi_from_gate_state(state)
        unique = len(set(iit.state_history))
        if step < 20 or step % 20 == 0 or step >= 195:
            print(f"{step:>5} {result.num_transitions:>12} {unique:>8} "
                  f"{result.phi:>10.6f} {result.method:>20}")


if __name__ == "__main__":
    main()

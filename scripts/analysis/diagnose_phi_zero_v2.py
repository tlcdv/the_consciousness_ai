"""Stage 2 diagnostic: simulate the *trained* gate dynamics.

The v1 diagnostic ran a fresh untrained gate and got phi up to 0.005. The
actual training shows phi=0.0 exactly. This version reproduces the
training loop's effect on the gate by applying the same gate diversity
loss for N steps, then probes phi.

If after diversity training the gate state distribution collapses such
that every state binarizes to the same tuple (or nearly so), pyphi
correctly returns 0 because the empirical TPM has no transition
structure.

Usage:
    python -m scripts.analysis.diagnose_phi_zero_v2
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.core.consciousness_gating import ConsciousnessGate
from models.evaluation.iit_phi import IITMetrics, GATE_NODE_LABELS


def run_phase(label, gate, iit, broadcasts, apply_diversity, optimizer,
              compute_phi=True):
    """Run gate forward for each broadcast, optionally with diversity loss.

    When compute_phi=False, skips pyphi calls (faster, avoids memory pressure
    while the ablation campaign is also running). Still records binarized
    states via update_from_gate_state.
    """
    iit.state_history.clear()
    iit._raw_history.clear()
    raw_traj = []
    bin_traj = []
    phi_traj = []
    for step, b in enumerate(broadcasts):
        b = b.unsqueeze(0)
        gated, state = gate(b)
        raw_traj.append([state.attention_level, state.stability_score,
                         state.adaptation_rate, state.meta_memory_coherence,
                         state.narrator_confidence])
        if compute_phi:
            result = iit.compute_phi_from_gate_state(state)
            phi_traj.append(result.phi)
            bin_traj.append(result.current_state)
        else:
            # Just record the state without pyphi call
            bin_state = iit.update_from_gate_state(state)
            bin_traj.append(bin_state)

        if apply_diversity and step % 5 == 0:
            gate_values_tensor = gate.last_gate_values_tensor
            if gate_values_tensor is not None:
                diversity_loss = -torch.log(
                    torch.abs(gate_values_tensor - 0.5).clamp(min=0.01)
                ).mean() * 0.05
                optimizer.zero_grad()
                diversity_loss.backward()
                torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
                optimizer.step()

    raw = np.array(raw_traj)
    print(f"\n--- {label} ---")
    for i, n in enumerate(GATE_NODE_LABELS):
        c = raw[:, i]
        print(f"  raw {n:12s}: mean={c.mean():.4f} std={c.std():.4f} "
              f"min={c.min():.4f} max={c.max():.4f}")
    bin_counts = pd.Series(bin_traj).value_counts()
    print(f"  unique bin states: {len(bin_counts)}/32")
    print(f"  top-3 bin states (count): {bin_counts.head(3).to_dict()}")
    nonzero = sum(1 for p in phi_traj if p > 0)
    if phi_traj:
        print(f"  phi: mean={np.mean(phi_traj):.6f} max={max(phi_traj):.6f} "
              f"nonzero/total={nonzero}/{len(phi_traj)}")
    return bin_counts


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

    # Load real broadcast magnitudes from A_current to mirror training input
    df = pd.read_csv("runs/ablation/A_current/metrics.csv")
    bm = df["broadcast_mag"].to_numpy()
    n_steps = 400
    sampled_mags = np.random.choice(bm, size=n_steps, replace=True)

    def broadcast_iter(mags):
        for m in mags:
            d = torch.randn(256)
            d = d / (d.norm() + 1e-8)
            yield d * float(m)

    print("=== Phase A: untrained gate, no diversity loss (baseline) ===")
    broadcasts = list(broadcast_iter(sampled_mags[:200]))
    run_phase("untrained, no loss", gate, iit, broadcasts, False, optimizer,
              compute_phi=False)

    print("\n=== Phase B: apply diversity loss for 200 steps (no phi calls) ===")
    broadcasts = list(broadcast_iter(sampled_mags[200:]))
    run_phase("with diversity loss", gate, iit, broadcasts, True, optimizer,
              compute_phi=False)

    print("\n=== Phase C: fresh inference after diversity training ===")
    iit2 = IITMetrics(history_len=200, tpm_window=200, tpm_decay=0.995)
    fresh_mags = np.random.choice(bm, size=300, replace=True)
    broadcasts = list(broadcast_iter(fresh_mags))
    run_phase("post-diversity, no loss", gate, iit2, broadcasts,
              False, optimizer, compute_phi=True)

    if iit2.state_history:
        tpm = iit2.build_empirical_tpm(5)
        print(f"\n  Final TPM stats: {iit2.get_tpm_stats()}")
        print(f"  TPM rows that are uniform 0.5: "
              f"{(np.abs(tpm - 0.5).sum(axis=1) < 0.01).sum()}/{tpm.shape[0]}")


if __name__ == "__main__":
    main()

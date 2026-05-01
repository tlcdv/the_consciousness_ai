"""Stage 3 diagnostic: instrument calculate_phi and see what pyphi actually does.

Monkey-patches calculate_phi to capture the first 30 calls' inputs, the
result, and any exception. Then runs the v2 simulation pipeline once to
generate calls. The point is to know whether pyphi is:
  (a) running cleanly and returning 0 because the TPM is reducible,
  (b) raising an exception that is swallowed and returning 0 in the except branch.

If (b), we'll know the exception type. If (a), we'll see what TPMs and
states cause phi=0 vs phi>0.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.core.consciousness_gating import ConsciousnessGate
from models.evaluation.iit_phi import IITMetrics, GATE_CM, GATE_NODE_LABELS

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

    # First, train the gate with diversity loss for 200 steps to mimic actual
    # training-time gate dynamics
    np.random.seed(0)
    train_mags = np.random.choice(bm, size=200, replace=True)
    print("Training gate with diversity loss for 200 steps...")
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

    # Now build state_history with 200 binarized states from gate
    print("\nGenerating 200 binarized states post-training...")
    test_mags = np.random.choice(bm, size=200, replace=True)
    for mag in test_mags:
        d = torch.randn(256)
        d = d / (d.norm() + 1e-8)
        broadcast = (d * float(mag)).unsqueeze(0)
        with torch.no_grad():
            _, state = gate(broadcast)
        iit.update_from_gate_state(state)

    # Now we have a fully populated state_history with realistic gate dynamics
    print(f"\nstate_history len: {len(iit.state_history)}")
    bin_counts = pd.Series(list(iit.state_history)).value_counts()
    print(f"unique states: {len(bin_counts)}/32")
    print(f"top-5 states (count): {bin_counts.head(5).to_dict()}")

    # Build TPM
    tpm = iit.build_empirical_tpm(5)
    print(f"\nTPM shape: {tpm.shape}")
    print(f"TPM rows uniform 0.5: {(np.abs(tpm - 0.5).sum(axis=1) < 0.01).sum()}/32")
    print(f"TPM column variance: {tpm.var(axis=0)}")

    # Now manually call pyphi on this TPM with the most common state
    most_common_state = bin_counts.index[0]
    print(f"\nMost common state: {most_common_state}")

    print("\n--- Manual pyphi call with verbose error reporting ---")
    pyphi.config.PROGRESS_BARS = False
    pyphi.config.PARALLEL_CUT_EVALUATION = False
    try:
        network = pyphi.Network(tpm, cm=GATE_CM)
        subsystem = pyphi.Subsystem(network, most_common_state)
        print(f"Subsystem created. Computing SIA...")
        sia = pyphi.compute.sia(subsystem)
        print(f"SIA result: phi={sia.phi}, "
              f"partition={getattr(sia, 'partition', 'N/A')}")
    except Exception as e:
        print(f"EXCEPTION in pyphi: {type(e).__name__}: {e}")
        traceback.print_exc()

    # Now try with several different states
    print("\n--- Testing multiple states ---")
    for state in list(set(iit.state_history))[:5]:
        try:
            network = pyphi.Network(tpm, cm=GATE_CM)
            subsystem = pyphi.Subsystem(network, state)
            sia = pyphi.compute.sia(subsystem)
            print(f"  state={state}: phi={sia.phi:.6f}")
        except Exception as e:
            print(f"  state={state}: EXCEPTION {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

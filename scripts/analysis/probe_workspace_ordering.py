"""
Workspace ordering and directionality probe (read-only).

Asks Fang et al.'s question of the workspace this project ALREADY has, before any
thalamic hub is built. Their three measurable claims about a subcortical gate
(bioRxiv 2024.04.02.587714, PREPRINT, not peer reviewed):

  ordering     phase locking rises within the hub first, then hub to cortex, then
               within cortex last and weakest
  direction    directed phase flow runs hub to cortex more than cortex to hub
  cross-scale  the hub's slow phase modulates the amplitude of faster activity
               elsewhere, dissociably from the phase locking

THIS PROBE IS A KILL GATE, NOT A DEMONSTRATION. The decision it informs:

  If the current workspace ALREADY shows the ordering and the directional
  asymmetry, a thalamic hub would be redundant and should not be built.

  If it shows neither, that is the gap a hub would fill, and it is the evidence
  that would justify opening the architecture fork.

  If the signals do not vary enough to support any coupling estimate at all, then
  NEITHER conclusion is available and the honest output is that the question cannot
  be asked on this agent yet. Given that sync_R is measured invariant at 0.251 to
  0.257 across the whole perception ablation, this third outcome is a live
  possibility and the probe checks for it FIRST, before printing any coupling
  number.

UNITS. Every band here is in CYCLES PER STEP and carries no Hz interpretation. See
the units warning in models/evaluation/coupling_measures.py. Do not map the slow
band onto anything in the source papers.

HONESTY. Single seed is a hypothesis. Coupling estimators on short records from a
substrate they were not designed for are diagnostics, not results. Nothing here may
be reported without at least 3-seed replication.

Usage:
    python -m scripts.analysis.probe_workspace_ordering --env dmts --seed 42
    python -m scripts.analysis.probe_workspace_ordering --steps 1024 \\
        --load-tectum runs/x/tectum.pt --latent-mode continuous \\
        --capsule-workspace-source all_levels
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from models.evaluation.coupling_measures import (
    phase_amplitude_coupling,
    phase_locking_value,
    phase_transfer_entropy,
)
from scripts.analysis.probe_perception_decodability import (
    _build_components,
    _compute_broadcast,
)
from scripts.analysis.probe_pci import _gate_vector, _make_env
from scripts.training.train_rlhf import frame_to_tensor

# Cycles per step. The slow band must stay strictly below the fast band for the
# modulation index to mean anything; both are arbitrary and carry no Hz reading.
SLOW_BAND = (0.01, 0.05)
FAST_BAND = (0.15, 0.35)

# A signal flatter than this cannot support a phase estimate: its "phase" would be
# the phase of float noise. Matches the var_floor convention used elsewhere.
VARIANCE_FLOOR = 1e-4


def collect_signals(args) -> dict:
    """One rollout, returning a named per-step scalar signal for each node."""
    action_dim = 5 if args.env == "dmts" else 2
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config, tectum, workspace, reentrant, self_model, memory, mock_sem = _build_components(
        args.env,
        action_dim=action_dim,
        seed=args.seed,
        mock_semantic=False,
        load_tectum=args.load_tectum,
        latent_mode=args.latent_mode,
        capsule_workspace_source=args.capsule_workspace_source,
    )
    gate_module = _build_gate(config)
    env = _make_env(args.env, args.seed)
    obs, _ = env.reset(seed=args.seed)

    device = config["device"]
    ws_dim = config["workspace_dim"]
    rng = np.random.default_rng(args.seed)

    signals: dict[str, list[float]] = {
        "vision": [], "broadcast": [], "sync_R": [],
        "gate_attention": [], "gate_coherence": [],
    }

    with torch.no_grad():
        for _ in range(args.steps):
            frame = frame_to_tensor(obs, device)
            audio = torch.zeros(1, config["tectum_feature_dim"], 2, device=device)
            tectum_content, vision_bid = tectum(frame, audio)

            broadcast = _compute_broadcast(
                config, tectum, workspace, reentrant, self_model, memory,
                mock_sem, tectum_content, vision_bid, obs,
            )
            if broadcast is None:
                broadcast = np.zeros(ws_dim, dtype=np.float64)

            gate_in = torch.tensor(
                broadcast[:ws_dim], dtype=torch.float32, device=device
            )
            if gate_in.numel() < gate_module.hidden_size:
                gate_in = torch.nn.functional.pad(
                    gate_in, (0, gate_module.hidden_size - gate_in.numel())
                )
            gate_module(gate_in[: gate_module.hidden_size].unsqueeze(0))
            gate_vec = _gate_vector(gate_module)

            signals["vision"].append(float(tectum_content.detach().norm()))
            signals["broadcast"].append(float(np.linalg.norm(broadcast)))
            signals["sync_R"].append(float(getattr(workspace, "last_sync_R", 0.0) or 0.0))
            signals["gate_attention"].append(float(gate_vec[0]))
            signals["gate_coherence"].append(float(gate_vec[3]))

            action = int(rng.integers(0, action_dim)) if args.action_rate > 0 else 0
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=args.seed + 1)

    return {k: np.asarray(v, dtype=np.float64) for k, v in signals.items()}


def _build_gate(config):
    from models.core.consciousness_gating import ConsciousnessGate

    gate = ConsciousnessGate(
        {
            "hidden_size": config["workspace_dim"],
            "ablate_feedback": config.get("ablate_gate_feedback", False),
            "use_self_vector": config.get("enable_self_vector_gating", False),
            "self_vector_dim": config.get("self_vector_dim", 64),
        }
    )
    gate.eval()
    return gate


def report_variance(signals: dict) -> list[str]:
    """
    Gate on signal variation BEFORE any coupling number is printed.

    Returns the names of signals flat enough that a phase estimate would be
    meaningless. A coupling value computed on those is an artifact.
    """
    print("Signal variation (the precondition for every measure below)")
    print(f"  {'signal':<16} {'std':>12} {'range':>12}   usable")
    flat = []
    for name, sig in signals.items():
        sd = float(sig.std())
        rng_ = float(sig.max() - sig.min())
        usable = sd > VARIANCE_FLOOR
        if not usable:
            flat.append(name)
        print(f"  {name:<16} {sd:>12.3e} {rng_:>12.3e}   {'yes' if usable else 'NO'}")
    return flat


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env", default="dmts", choices=["dmts", "dark_room"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1024,
                        help="Rollout length. Coupling estimates need a long record; "
                             "below ~512 the surrogate null dominates.")
    parser.add_argument("--action-rate", type=float, default=0.0)
    parser.add_argument("--load-tectum", default=None)
    parser.add_argument("--latent-mode", default="discrete",
                        choices=["discrete", "continuous"])
    parser.add_argument("--capsule-workspace-source", default="final",
                        choices=["final", "all_levels"])
    parser.add_argument("--out", default=None, help="Optional CSV path")
    args = parser.parse_args()

    print(f"Workspace ordering probe: env={args.env} seed={args.seed} "
          f"steps={args.steps}")
    print(f"  latent={args.latent_mode} capsule={args.capsule_workspace_source}")
    print(f"  bands (CYCLES PER STEP, no Hz reading): slow={SLOW_BAND} "
          f"fast={FAST_BAND}\n")

    signals = collect_signals(args)
    flat = report_variance(signals)
    print()

    if len(flat) == len(signals):
        print("VERDICT: every signal is flat below the variance floor. No coupling "
              "estimate is available, so the ordering question CANNOT be asked on "
              "this agent in this configuration. This is neither support for nor "
              "against a thalamic hub. Report it as such.")
        return

    usable = [n for n in signals if n not in flat]
    if len(usable) < 2:
        print(f"VERDICT: only {usable} varies. Coupling needs at least two varying "
              f"signals, so no pairwise measure is available.")
        return

    print(f"Pairwise coupling over {len(usable)} usable signals: {usable}\n")
    rows = []
    print(f"  {'pair':<34} {'PLV':>8} {'PTE a->b':>10} {'PTE b->a':>10} "
          f"{'asym':>9} {'null sd':>9}")
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            plv = phase_locking_value(signals[a], signals[b], SLOW_BAND)
            fwd = phase_transfer_entropy(signals[a], signals[b], SLOW_BAND,
                                         seed=args.seed)
            bwd = phase_transfer_entropy(signals[b], signals[a], SLOW_BAND,
                                         seed=args.seed)
            asym = fwd.corrected - bwd.corrected
            null_sd = max(fwd.surrogate_std, bwd.surrogate_std)
            rows.append({
                "signal_a": a, "signal_b": b,
                "plv": round(plv, 6),
                "pte_a_to_b": round(fwd.corrected, 6),
                "pte_b_to_a": round(bwd.corrected, 6),
                "asymmetry": round(asym, 6),
                "null_sd": round(null_sd, 6),
                "significant": bool(abs(asym) > 2.0 * null_sd) if null_sd > 0 else False,
            })
            flag = "*" if rows[-1]["significant"] else " "
            print(f"  {a + ' <-> ' + b:<34} {plv:>8.4f} {fwd.corrected:>10.4f} "
                  f"{bwd.corrected:>10.4f} {asym:>8.4f}{flag} {null_sd:>9.4f}")

    print("\n  * asymmetry exceeds twice the circular-shift null spread.")

    # The cross-scale measure, only where both bands can be estimated.
    print("\nCross-scale coupling (slow phase of a driving fast envelope of b)")
    pac_rows = []
    for a in usable:
        for b in usable:
            if a == b:
                continue
            try:
                mi = phase_amplitude_coupling(
                    signals[a], signals[b], SLOW_BAND, FAST_BAND
                )
            except ValueError:
                continue
            pac_rows.append({"phase_signal": a, "amplitude_signal": b,
                             "modulation_index": round(mi, 6)})
    for row in sorted(pac_rows, key=lambda r: -r["modulation_index"])[:6]:
        print(f"  {row['phase_signal']:<16} -> {row['amplitude_signal']:<16} "
              f"MI={row['modulation_index']:.4f}")

    significant = [r for r in rows if r["significant"]]
    print()
    if significant:
        print(f"VERDICT: {len(significant)} of {len(rows)} pairs show a directional "
              f"asymmetry beyond the null. The existing workspace already produces "
              f"some of the structure a hub would supply; identify which pairs "
              f"before opening the architecture fork.")
    else:
        print(f"VERDICT: no pair shows a directional asymmetry beyond the null "
              f"({len(rows)} pairs tested). The current workspace produces no "
              f"measurable hub-like ordering. This is the gap a thalamic hub would "
              f"fill, and it is the evidence that would justify the fork.")

    print("\nSingle seed is a hypothesis. Replicate across >= 3 seeds before "
          "reporting any of these numbers.")

    if args.out and rows:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

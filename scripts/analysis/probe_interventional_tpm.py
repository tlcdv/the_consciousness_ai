"""
Is the gate's degeneracy in the ESTIMATOR or in the system?

Every EI and CE 2.0 number this project has produced came from a TPM built by counting
OBSERVED transitions (`effective_information._build_tpm`, "from observed state
trajectories"). The theory specifies an INTERVENTIONAL matrix under a maximum-entropy
`do()` distribution, which is a specification of full state coverage. With the gate
visiting one joint state, 242 of 243 rows are pure Laplace smoothing, so the uniform
average returns the prior's entropy rather than the system's.

This probe builds the matrix both ways from the SAME trained checkpoint and scores them
side by side. The discretization is held fixed (tertile, exactly as
`metrics_logger._discretize_gate_window` applies it) so the estimator is the only thing
that varies.

FOUR-WAY, not two-way. A second defect was found while designing this
(tests/test_ei_degeneracy_term.py): the local EI formula uses `log2(n)` where Hoel's
uses `H(<TPM>)`, so it omits the degeneracy term and returns the MAXIMUM for a totally
degenerate system. That defect is invisible on a smoothed observational matrix and
appears immediately on a sharp interventional one, so scoring only the legacy formula
here would have reported a spectacular success for a dead gate. Both formulas are
reported on both matrices.

TWO STATE SETS. The gate nodes live in a band about 0.01 wide around 0.49
(docs/results/gate_binning_2026_07.md), so forcing joint state 0 writes values near 0.17
that `gate_feedback` has never seen.

  (a) FULL: all 243 joint states. This is the theory's max-entropy intervention, and it
      evaluates the model far outside its own input distribution.
  (b) VISITED: only joint states the rollout actually entered. Stays on-distribution.

With (a) alone, a degenerate result is ambiguous between "the estimator was the defect"
and "the intervention produced extrapolation noise". Running both separates them.

PRE-STATED GATE, written before any number from this probe was read:

    ESTIMATOR: the interventional EI (corrected formula) on the FULL set differs from
        the observational EI by more than 0.10 bits AND the interventional matrix is
        non-degenerate by the clause-1 check below. The degeneracy lived in the
        estimator.
    ARCHITECTURAL: the interventional matrix is degenerate under full forced coverage,
        i.e. the forced states collapse onto few effects. The root is the system, not
        the instrument, and the EI family is retired on evidence.
    UNINTERPRETABLE: (a) and (b) disagree in direction. The full-set result is
        extrapolation and neither branch is supported.

CLAUSE 1. A degenerate interventional matrix must emit a sentinel, not a number. Here
that is an explicit DEGENERATE verdict printed instead of a bare EI value, triggered
when the forced transitions land on fewer than 3 distinct effects.

CLAUSE 2. The failing null is the frozen trajectory. Interventional EI must NOT
reproduce `constant_trajectory_floor`, or the intervention is not doing what it claims.

Run:
    python -m scripts.analysis.probe_interventional_tpm \\
        --load-tectum runs/gate_ckpt_s42/tectum.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.core.consciousness_gating import ConsciousnessGate, gate_checkpoint_path
from models.evaluation.causal_emergence_svd import (
    compute_ce2_from_tpm, frozen_trajectory_ce2_value,
)
from models.evaluation.effective_information import (
    _build_tpm, constant_trajectory_floor, effective_information_from_tpm,
)
from scripts.analysis.probe_perception_decodability import (
    _build_components, _compute_broadcast,
)
from scripts.training.train_rlhf import frame_to_tensor
from simulations.environments.dmts_env import DMTSEnv

N_NODES = 5
N_STATES = 3 ** N_NODES          # 243, the gate joint space the logger scores
TERTILE_MIDPOINTS = (1.0 / 6.0, 0.5, 5.0 / 6.0)
MIN_DISTINCT_EFFECTS = 3         # clause 1 threshold
ESTIMATOR_GATE_BITS = 0.10       # pre-stated


def joint_index(values) -> int:
    """Tertile joint index. Same arithmetic as metrics_logger._discretize_gate_window."""
    idx = 0
    for i, val in enumerate(values):
        trit = 0 if val < 1 / 3 else (1 if val < 2 / 3 else 2)
        idx += trit * (3 ** i)
    return idx


def state_to_values(index: int) -> list[float]:
    """Inverse: a representative 5-tuple whose joint index is `index`.

    Bin midpoints, so a forced state sits in the middle of its cell rather than on a
    boundary. joint_index(state_to_values(k)) == k by construction, asserted below.
    """
    values, rest = [], index
    for _ in range(N_NODES):
        values.append(TERTILE_MIDPOINTS[rest % 3])
        rest //= 3
    return values


def collect_observational(config, tectum, workspace, reentrant, self_model, memory,
                          mock_sem, gate, episodes: int, seed: int):
    """Roll out normally. Return the observed joint-state trajectory and gate inputs."""
    env = DMTSEnv(num_trials=20)
    trajectory, gate_inputs = [], []
    with torch.no_grad():
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            if hasattr(tectum, "reset_state"):
                tectum.reset_state(1)
            gate.reset_episode()
            done, steps = False, 0
            while not done and steps < 400:
                frame = frame_to_tensor(obs, config["device"])
                audio = torch.zeros(1, config["tectum_feature_dim"], 2,
                                    device=config["device"])
                tectum_content, vision_bid = tectum(frame, audio)
                broadcast = _compute_broadcast(config, tectum, workspace, reentrant,
                                               self_model, memory, mock_sem,
                                               tectum_content, vision_bid, obs)
                if broadcast is None:
                    break
                gate_input = _as_gate_input(broadcast, config)
                gate(gate_input)
                vals = gate.prev_gate_values.detach().cpu().numpy().reshape(-1)[:N_NODES]
                trajectory.append(joint_index(vals))
                gate_inputs.append(gate_input.detach().clone())
                obs, _, term, trunc, info = env.step(0)
                done = term or trunc
                steps += 1
    return np.asarray(trajectory, dtype=np.int64), gate_inputs


def _as_gate_input(broadcast, config) -> torch.Tensor:
    """Build the gate's input exactly as train_rlhf does (line ~1337)."""
    tensor = torch.as_tensor(np.asarray(broadcast).reshape(-1), dtype=torch.float32,
                             device=config["device"])
    dim = config["workspace_dim"]
    tensor = tensor[:dim]
    if tensor.shape[0] < dim:
        tensor = torch.nn.functional.pad(tensor, (0, dim - tensor.shape[0]))
    return tensor.unsqueeze(0)


def quantile_joint_indices(raw: np.ndarray, var_floor: float = 1e-4) -> list[int]:
    """Per-dimension terciles from the data's own distribution.

    Mirrors `metrics_logger._discretize_gate_window` in 'quantile' mode, including the
    var_floor that pins a near-constant dimension to bin 0 so float noise cannot
    manufacture states.
    """
    mat = np.asarray(raw, dtype=float)
    edges = [None if mat[:, i].std() <= var_floor
             else np.quantile(mat[:, i], [1 / 3, 2 / 3])
             for i in range(mat.shape[1])]
    out = []
    for row in mat:
        idx = 0
        for i, val in enumerate(row):
            if edges[i] is None:
                trit = 0
            else:
                trit = 0 if val < edges[i][0] else (1 if val < edges[i][1] else 2)
            idx += trit * (3 ** i)
        out.append(idx)
    return out


def subtertile_control(gate, gate_input, states) -> dict:
    """MANDATORY control: does forcing move the CONTINUOUS output at all?

    Without this, a probe that only reads binned states cannot tell "the gate has no
    causal structure" from "the gate has causal structure the binning cannot see".
    Those support opposite verdicts, and the first run of this probe reported
    ARCHITECTURAL when the second was true.
    """
    raw = []
    with torch.no_grad():
        for cause in states:
            gate.prev_gate_values = torch.tensor(state_to_values(cause),
                                                 dtype=torch.float32).reshape(-1)
            gate(gate_input)
            raw.append(gate.prev_gate_values.detach().cpu().numpy().reshape(-1)[:N_NODES].copy())
    raw = np.asarray(raw)
    return {
        "raw": raw,
        "distinct_continuous": len({tuple(r) for r in raw}),
        "per_node_range": raw.max(axis=0) - raw.min(axis=0),
        "per_node_std": raw.std(axis=0),
    }


def build_interventional_tpm(gate, gate_input, states) -> tuple[np.ndarray, list[int]]:
    """Force each state in `states`, step once, record the effect.

    Rows for states NOT in `states` are left uniform, which is the honest encoding of
    "not intervened on" and matches the Laplace convention elsewhere.
    """
    counts = np.ones((N_STATES, N_STATES), dtype=np.float64)
    effects = []
    with torch.no_grad():
        for cause in states:
            forced = torch.tensor(state_to_values(cause), dtype=torch.float32).reshape(-1)
            gate.prev_gate_values = forced
            gate(gate_input)
            vals = gate.prev_gate_values.detach().cpu().numpy().reshape(-1)[:N_NODES]
            effect = joint_index(vals)
            counts[cause, effect] += 1.0
            effects.append(effect)
    return counts / counts.sum(axis=1, keepdims=True), effects


def score(tpm, label: str) -> dict:
    return {
        "label": label,
        "ei_legacy": effective_information_from_tpm(tpm, degeneracy_corrected=False),
        "ei_corrected": effective_information_from_tpm(tpm, degeneracy_corrected=True),
        "ce2": compute_ce2_from_tpm(tpm).causal_emergence,
        "ce2_complexity": compute_ce2_from_tpm(tpm).emergent_complexity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-tectum", default="runs/gate_ckpt_s42/tectum.pt")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    # These MUST match the flags the checkpoint was trained under, or the state_dict
    # will not load: --capsule-workspace-source changes workspace_proj's input width
    # (64 for "final", 384 for "all_levels") and --rssm-latent-mode changes the RSSM.
    # Defaults match the perception-fixed configuration, which is what gets checkpointed.
    parser.add_argument("--rssm-latent-mode", default="continuous",
                        choices=["discrete", "continuous"])
    parser.add_argument("--capsule-workspace-source", default="all_levels",
                        choices=["final", "all_levels"])
    args = parser.parse_args()

    # Sanity: the state encoding must round-trip before anything is measured.
    for k in (0, 1, 121, 242):
        assert joint_index(state_to_values(k)) == k, "state encoding is not invertible"

    config, tectum, workspace, reentrant, self_model, memory, mock_sem = \
        _build_components("dmts", action_dim=5, seed=args.seed, mock_semantic=False,
                          load_tectum=args.load_tectum,
                          latent_mode=args.rssm_latent_mode,
                          capsule_workspace_source=args.capsule_workspace_source)

    gate = ConsciousnessGate({
        "hidden_size": config["workspace_dim"], "ablate_feedback": False,
        "use_self_vector": False, "self_vector_dim": 64,
    })
    gate_path = gate_checkpoint_path(args.load_tectum)
    if not Path(gate_path).exists():
        raise SystemExit(
            f"FAILED: no gate checkpoint at {gate_path}. The gate trains inside "
            f"tectum_optimizer but was not persisted before 2026-08-11, so an "
            f"interventional TPM here would measure a RANDOM gate_feedback. Re-run "
            f"training with --save-tectum to write one."
        )
    gate.load_state_dict(torch.load(gate_path, map_location="cpu"))
    gate.eval()
    print(f"loaded trained gate: {gate_path}\n")

    observed, gate_inputs = collect_observational(
        config, tectum, workspace, reentrant, self_model, memory, mock_sem,
        gate, args.episodes, args.seed)
    visited = sorted(set(observed.tolist()))
    print(f"OBSERVATIONAL: {len(observed)} steps, {len(visited)} of {N_STATES} joint "
          f"states visited -> {visited[:8]}{'...' if len(visited) > 8 else ''}")

    obs_tpm = _build_tpm([observed], N_STATES)
    rows = [score(obs_tpm, "observational")]

    # Representative input: the mean observed gate input, so the intervention isolates
    # the gate's recurrence rather than confounding it with input variation.
    mean_input = torch.stack(gate_inputs).mean(dim=0)

    for label, states in (("interventional FULL (243)", list(range(N_STATES))),
                          ("interventional VISITED", visited)):
        gate.reset_episode()
        tpm, effects = build_interventional_tpm(gate, mean_input, states)
        distinct = len(set(effects))
        row = score(tpm, label)
        row["distinct_effects"] = distinct
        row["degenerate"] = distinct < MIN_DISTINCT_EFFECTS
        rows.append(row)
        print(f"{label}: forced {len(states)} states -> {distinct} distinct effects")

    print("\n" + "=" * 92)
    print(f"{'matrix':<28} {'EI legacy':>10} {'EI corrected':>13} {'CE 2.0':>9} {'cx':>4}  note")
    print("=" * 92)
    for r in rows:
        note = ""
        if r.get("degenerate"):
            note = "DEGENERATE -> values are SENTINEL, not measurements"
        print(f"{r['label']:<28} {r['ei_legacy']:>10.6f} {r['ei_corrected']:>13.6f} "
              f"{r['ce2']:>9.6f} {r['ce2_complexity']:>4}  {note}")

    # --- the control that decides which verdict the numbers above support ---------
    gate.reset_episode()
    control = subtertile_control(gate, mean_input, list(range(N_STATES)))
    print("\n" + "=" * 92)
    print("SUB-TERTILE CONTROL: does forcing move the CONTINUOUS output?")
    print("=" * 92)
    print(f"  distinct continuous effects from 243 distinct causes: "
          f"{control['distinct_continuous']}/243")
    names = ["attention", "stability", "adaptation", "coherence", "confidence"]
    for i, name in enumerate(names):
        print(f"    {name:<11} range={control['per_node_range'][i]:.3e}  "
              f"std={control['per_node_std'][i]:.3e}")

    # Re-bin the SAME forced effects by their own quantiles, which is what the gate
    # binning fix does for the observational path.
    q_effects = quantile_joint_indices(control["raw"])
    q_counts = np.ones((N_STATES, N_STATES), dtype=np.float64)
    for cause, effect in zip(range(N_STATES), q_effects):
        q_counts[cause, effect] += 1.0
    q_tpm = q_counts / q_counts.sum(axis=1, keepdims=True)
    q_row = score(q_tpm, "interventional QUANTILE")
    q_row["distinct_effects"] = len(set(q_effects))
    rows.append(q_row)
    print(f"\n  Re-binned by quantile: {len(set(q_effects))} distinct effect states "
          f"(tertile gave 1)")
    print(f"  {'matrix':<28} {'EI legacy':>10} {'EI corrected':>13} {'CE 2.0':>9} {'cx':>4}")
    print(f"  {q_row['label']:<28} {q_row['ei_legacy']:>10.6f} "
          f"{q_row['ei_corrected']:>13.6f} {q_row['ce2']:>9.6f} {q_row['ce2_complexity']:>4}")

    floor = constant_trajectory_floor(N_STATES, len(observed))
    frozen_ce2 = frozen_trajectory_ce2_value(N_STATES, len(observed))
    print("\nCLAUSE 2, the failing null (frozen trajectory reference):")
    print(f"  EI floor      = {floor:.6f}")
    print(f"  CE 2.0 frozen = {frozen_ce2:.6f}")

    print("\n" + "=" * 92)
    print("PRE-STATED GATE")
    print("=" * 92)
    full = next(r for r in rows if "FULL" in r["label"])
    vis = next(r for r in rows if "VISITED" in r["label"])
    obs = rows[0]

    # The control gates the ARCHITECTURAL branch. Binned collapse alone does not
    # license it: the first run of this probe reported ARCHITECTURAL while the gate
    # was in fact producing 243 distinct effects that the tertile bins could not see.
    alive = control["distinct_continuous"] > MIN_DISTINCT_EFFECTS

    if full["degenerate"] and alive:
        print("  DISCRETIZATION, not estimator and not architecture.")
        print(f"  Forcing 243 distinct causes produces {control['distinct_continuous']} "
              f"distinct CONTINUOUS effects, so the gate has real causal structure")
        print("  under intervention. Every one of them falls in a single tertile bin,")
        print("  so the 1/3 boundaries destroy it before any metric sees it.")
        print(f"  Re-binned by quantile the same effects give "
              f"{q_row['distinct_effects']} states and EI {q_row['ei_corrected']:.6f}.")
        print("  This confirms the 2026-07-26 gate-binning result under intervention,")
        print("  and it is a FOURTH outcome the pre-stated gate did not enumerate.")
    elif full["degenerate"]:
        print("  ARCHITECTURAL. The forced states collapse onto fewer than "
              f"{MIN_DISTINCT_EFFECTS} distinct effects under FULL coverage, AND the")
        print("  continuous outputs are equally collapsed, so this is not a binning")
        print("  artifact. The root is the system. EI values above are a SENTINEL.")
    else:
        delta = abs(full["ei_corrected"] - obs["ei_corrected"])
        same_dir = (np.sign(full["ei_corrected"] - obs["ei_corrected"])
                    == np.sign(vis["ei_corrected"] - obs["ei_corrected"]))
        if not same_dir:
            print("  UNINTERPRETABLE. FULL and VISITED disagree in direction, so the")
            print("  full-set result is extrapolation. Neither branch is supported.")
        elif delta > ESTIMATOR_GATE_BITS:
            print(f"  ESTIMATOR. Interventional EI differs by {delta:.6f} bits "
                  f"(> {ESTIMATOR_GATE_BITS}), and FULL agrees with VISITED in")
            print("  direction. The degeneracy lived in the estimator.")
        else:
            print(f"  NO DIFFERENCE. {delta:.6f} bits <= {ESTIMATOR_GATE_BITS}. The "
                  "estimator is not the defect.")

    print("\n  Legacy-vs-corrected gap on the interventional matrix "
          f"= {full['ei_legacy'] - full['ei_corrected']:.6f} bits.")
    print("  A large gap here is the degeneracy the legacy formula cannot see.")


if __name__ == "__main__":
    main()

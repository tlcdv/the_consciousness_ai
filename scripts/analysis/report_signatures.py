"""Aggregate consciousness-signature distributions from existing run csvs.

Read-only pilot for the signature assessment: no training, no model loading.
Reads runs/<run>/metrics.csv and runs/<run>/episodes.csv and prints, per run
and side by side across runs:
  - distribution stats (mean/std/CV/min/max) for each signature column,
  - phi stats restricted to steps where phi was actually computed (phi_method),
  - ignition stats (per-step is_conscious, per-episode consciousness_ratio),
  - the EI windows from episodes.csv.

Two targeted diagnoses:
  (a) ei_gates froze at the same value in every window of every run. Test the
      constant-trajectory hypothesis: with Laplace smoothing, a gate trajectory
      stuck in ONE joint state yields an EI that depends only on window length
      and state count, hence bit-identical across runs. Reproduce numerically.
  (b) is_conscious saturation: characterize when the selective-ignition gate
      (salience above an EMA baseline, models/core/global_workspace.py) ever
      goes quiet.

Usage:
    python -m scripts.analysis.report_signatures [--log-dir runs] [--runs a b c]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.evaluation.effective_information import (
    compute_effective_information,
    constant_trajectory_floor,
)

DEFAULT_RUNS = [
    "collapse_trained",
    "wmrecon_trained",
    "wmobs_trained",
    "wmpredict_trained",
    "wmpredict_trained2",
]

SIGNAL_COLUMNS = [
    "phi", "sync_r", "broadcast_mag", "valence", "arousal", "dominance",
    "phi_riiu", "levin_bioelectric_complexity", "levin_morphological_adaptation",
    "levin_collective_intelligence", "levin_goal_directed", "levin_basal_cognition",
    "self_pred_mse", "self_pred_skill",
]


def load_csv(path: Path) -> dict[str, list[str]]:
    cols: dict[str, list[str]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
    return cols


def floats(values: list[str]) -> np.ndarray:
    out = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            pass
    return np.asarray(out)


def stats(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"n": 0}
    mean = float(np.mean(x))
    std = float(np.std(x))
    cv = std / abs(mean) if abs(mean) > 1e-12 else float("nan")
    return {"n": int(x.size), "mean": mean, "std": std, "cv": cv,
            "min": float(np.min(x)), "max": float(np.max(x)),
            "all_zero": bool(np.all(x == 0.0))}


def fmt(s: dict) -> str:
    if s.get("n", 0) == 0:
        return "no data"
    if s.get("all_zero"):
        return f"all zero (n={s['n']})"
    return (f"mean={s['mean']:.6g} std={s['std']:.3g} cv={s['cv']:.3g} "
            f"min={s['min']:.6g} max={s['max']:.6g} n={s['n']}")


def analyze_run(run_dir: Path) -> dict:
    metrics = load_csv(run_dir / "metrics.csv")
    episodes = load_csv(run_dir / "episodes.csv")
    out: dict = {"run": run_dir.name}

    for col in SIGNAL_COLUMNS:
        if col in metrics:
            out[col] = stats(floats(metrics[col]))

    # phi restricted to steps where it was actually computed
    if "phi" in metrics and "phi_method" in metrics:
        methods = metrics["phi_method"]
        counts: dict[str, int] = {}
        for m in methods:
            counts[m] = counts.get(m, 0) + 1
        out["phi_method_counts"] = counts
        computed = np.asarray([
            float(p) for p, m in zip(metrics["phi"], methods)
            if m not in ("skipped", "insufficient_data")
        ])
        out["phi_computed"] = stats(computed)

    # ignition: per-step is_conscious
    if "is_conscious" in metrics:
        ic = floats(metrics["is_conscious"]).astype(int)
        quiet = np.where(ic == 0)[0]
        out["ignition"] = {
            "steps": int(ic.size),
            "fraction_ignited": float(np.mean(ic)) if ic.size else float("nan"),
            "quiet_steps": int(quiet.size),
            "quiet_first_idx": int(quiet[0]) if quiet.size else None,
            "quiet_last_idx": int(quiet[-1]) if quiet.size else None,
            "quiet_in_first_1000": int(np.sum(quiet < 1000)) if quiet.size else 0,
        }

    # episodes.csv: consciousness_ratio and EI windows.
    #
    # From 2026-08-12 `consciousness_ratio` is EMPTY whenever it saturates, because a
    # binary summary pinned at an extreme measures nothing and must not be written as a
    # plausible number (clause 1 of the instrument acceptance bar). floats() drops the
    # empties, so an all-degenerate run reports n=0 rather than a confident 1.0000.
    # The unfiltered value is kept in `consciousness_ratio_raw` and reported separately
    # below, so the diagnostic survives without being usable as a measurement.
    if "consciousness_ratio" in episodes:
        cr = floats(episodes["consciousness_ratio"])
        out["consciousness_ratio"] = stats(cr)
        out["eps_below_1"] = int(np.sum(cr < 1.0))
    if "consciousness_ratio_raw" in episodes:
        raw = floats(episodes["consciousness_ratio_raw"])
        out["consciousness_ratio_raw"] = stats(raw)
        n_written = out.get("consciousness_ratio", {}).get("n", 0)
        out["consciousness_ratio_degenerate_eps"] = int(raw.size) - int(n_written)
    ei_rows = []
    if "ei_gates" in episodes:
        for ep, g, w, r in zip(episodes["episode"], episodes["ei_gates"],
                               episodes["ei_workspace"], episodes["ei_ratio"]):
            if float(g) != 0.0 or float(w) != 0.0:
                ei_rows.append((int(ep), float(g), float(w), float(r)))
    out["ei_windows"] = ei_rows
    return out


def diagnose_constant_ei(observed: float, num_states: int, label: str) -> None:
    print(f"\n== Diagnosis: can a CONSTANT trajectory reproduce "
          f"{label} = {observed:.6f} ({num_states} states)?")
    print("   (Laplace-smoothed TPM; EI of a single-state trajectory depends only "
          "on window length and state count, so it is bit-identical across runs.)")
    for n in (200, 2000, 5000, 10000, 20000):
        traj = np.zeros(n, dtype=int)  # one joint state for the whole window
        ei = compute_effective_information([traj], num_states)
        marker = "  <-- MATCH" if abs(ei - observed) < 5e-7 else ""
        print(f"   constant trajectory, window={n:>6} steps: EI={ei:.6f}{marker}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", default="runs")
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--ei-window-len", type=int, default=10000,
                    help="Trajectory length per logged EI window, for the post-hoc "
                         "floor correction (log_ei_every * max_steps; 50 * 200 = "
                         "10000 for the 2026-06/07 DMTS runs).")
    args = ap.parse_args()

    results = []
    for name in args.runs:
        run_dir = Path(args.log_dir) / name
        if not (run_dir / "metrics.csv").exists():
            print(f"SKIP {name}: no metrics.csv")
            continue
        results.append(analyze_run(run_dir))

    for r in results:
        print(f"\n===== {r['run']} =====")
        for col in SIGNAL_COLUMNS:
            if col in r:
                print(f"  {col:32s} {fmt(r[col])}")
        if "phi_computed" in r:
            print(f"  {'phi (computed steps only)':32s} {fmt(r['phi_computed'])}")
            print(f"  {'phi_method counts':32s} {r['phi_method_counts']}")
        if "ignition" in r:
            ig = r["ignition"]
            print(f"  {'is_conscious':32s} fraction={ig['fraction_ignited']:.4f} "
                  f"quiet_steps={ig['quiet_steps']}/{ig['steps']} "
                  f"(first@{ig['quiet_first_idx']}, last@{ig['quiet_last_idx']}, "
                  f"{ig['quiet_in_first_1000']} in first 1000 steps)")
        if "consciousness_ratio" in r:
            print(f"  {'consciousness_ratio (per ep)':32s} {fmt(r['consciousness_ratio'])} "
                  f"episodes<1.0: {r['eps_below_1']}")
        if "consciousness_ratio_raw" in r:
            degenerate = r.get("consciousness_ratio_degenerate_eps", 0)
            note = (f"  [{degenerate} episodes SATURATED, withheld from the column above]"
                    if degenerate else "")
            print(f"  {'consciousness_ratio (raw)':32s} "
                  f"{fmt(r['consciousness_ratio_raw'])}{note}")
        for ep, g, w, ratio in r["ei_windows"]:
            print(f"  EI window @ep{ep:<3d} ei_gates={g:.6f} ei_workspace={w:.6f} "
                  f"ratio={ratio:.4f}")

    # cross-run side-by-side for the headline signals
    print("\n== Cross-run comparison (mean [cv]) ==")
    keys = ["phi_computed", "sync_r", "broadcast_mag", "valence", "arousal",
            "dominance"]
    header = f"{'signal':28s}" + "".join(f"{r['run'][:18]:>22s}" for r in results)
    print(header)
    for k in keys:
        cells = []
        for r in results:
            s = r.get(k, {})
            if s.get("n"):
                cells.append(f"{s['mean']:.4g} [{s['cv']:.2g}]")
            else:
                cells.append("-")
        print(f"{k:28s}" + "".join(f"{c:>22s}" for c in cells))
    ig_cells = "".join(
        f"{r.get('ignition', {}).get('fraction_ignited', float('nan')):>22.4f}"
        for r in results)
    print(f"{'ignition fraction':28s}{ig_cells}")

    gate_values = {g for r in results for _, g, _, _ in r["ei_windows"]}
    ws_values = sorted({w for r in results for _, _, w, _ in r["ei_windows"]})
    print(f"\nDistinct ei_gates values across ALL windows of ALL runs: "
          f"{sorted(gate_values)}")
    print(f"Distinct ei_workspace values across ALL windows of ALL runs: {ws_values}")
    if len(gate_values) == 1:
        diagnose_constant_ei(next(iter(gate_values)), 243, "ei_gates")
    if ws_values:
        # the LOWEST recurring workspace value is the candidate frozen floor
        diagnose_constant_ei(ws_values[0], 8, "ei_workspace (lowest observed)")

    # Post-hoc floor-corrected EI (constant-trajectory Laplace baseline subtracted;
    # models/evaluation/effective_information.py). Runs from 2026-07 onward log the
    # corrected values directly (ei_*_corr columns); for older runs this recomputes
    # them from the raw logged values, valid because the floor depends only on
    # (state count, window length).
    L = args.ei_window_len
    floor_g = constant_trajectory_floor(243, L)
    floor_w = constant_trajectory_floor(8, L)
    print(f"\n== Floor-corrected EI (window={L}: floor_gates={floor_g:.6f}, "
          f"floor_workspace={floor_w:.6f}) ==")
    eps = 1e-6  # episodes.csv logs 6 decimals; sub-precision residues are frozen
    for r in results:
        for ep, g, w, _ in r["ei_windows"]:
            cg = max(0.0, g - floor_g)
            cw = max(0.0, w - floor_w)
            cg = cg if cg >= eps else 0.0
            cw = cw if cw >= eps else 0.0
            if cg > 0:
                cr = f"{cw / cg:.4f}"
            elif cw > 0:
                cr = "inf (micro frozen, macro not)"
            else:
                cr = "0 (both levels frozen)"
            print(f"  {r['run']:22s} @ep{ep:<3d} corr_gates={cg:.6f} "
                  f"corr_workspace={cw:.6f} corr_ratio={cr}")


if __name__ == "__main__":
    main()

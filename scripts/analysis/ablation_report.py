"""
Ablation report (Phase 4 empirical validation).

Loads every episodes.csv + metrics.csv under runs/ablation/ (and the two
historical pre-fix references) into one pandas table, prints the comparison,
and applies the strict acceptance thresholds from the plan.

Honesty rule: every cited number is loaded from disk in this script's own
invocation. No copy-paste from prior runs or CLAUDE.md.

Usage:
    python -m scripts.analysis.ablation_report
    python -m scripts.analysis.ablation_report --campaign-dir runs/ablation
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# Strict acceptance thresholds (peaceful-castle Task 2.1, scaled to 200 eps).
# A single-flag ablation is a "fix candidate" iff it meets ALL four.
THRESHOLDS = {
    "phi_std_min": 0.0015,
    "last100_reward_min": 5.0,
    "positive_eps_min": 60,         # 30 % rate over 200 eps
    "phi_sync_r_correlation_min": 0.4,
    "phi_std_min_for_correlation": 0.0005,  # blocks the r=1.000 flat-trajectory pathology
}


def load_run(run_dir: Path) -> dict | None:
    """Load one ablation run. Returns None if episodes.csv is missing or
    has zero data rows; returns a partial row when the run crashed mid-way."""
    ep_path = run_dir / "episodes.csv"
    metrics_path = run_dir / "metrics.csv"

    if not ep_path.exists():
        return {"run": run_dir.name, "status": "NO_EPISODES_CSV"}

    try:
        edf = pd.read_csv(ep_path)
    except Exception as exc:
        return {"run": run_dir.name, "status": f"EP_READ_ERROR:{exc}"}

    if edf.empty:
        return {"run": run_dir.name, "status": "EMPTY_EPISODES_CSV", "n_episodes": 0}

    # Step-level metrics for phi_method breakdown and phi-vs-sync_R correlation.
    mdf = None
    if metrics_path.exists():
        try:
            mdf = pd.read_csv(metrics_path)
        except Exception:
            mdf = None

    n = len(edf)
    last100 = edf["total_reward"].iloc[-min(100, n):].mean()
    pos_eps = int((edf["total_reward"] > 0).sum())
    phi_mean = float(edf["avg_phi"].mean()) if "avg_phi" in edf.columns else float("nan")
    phi_std = float(edf["avg_phi"].std()) if "avg_phi" in edf.columns else float("nan")

    # phi-method breakdown from step-level CSV
    phi_methods = ""
    phi_corr = float("nan")
    phi_corr_p = float("nan")
    if mdf is not None and len(mdf) > 0:
        if "phi_method" in mdf.columns:
            counts = mdf["phi_method"].value_counts().to_dict()
            phi_methods = " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        if "phi" in mdf.columns and "sync_r" in mdf.columns and len(mdf) >= 10:
            try:
                # Drop rows where either is exactly zero (no signal); then correlate.
                mask = ~(mdf["phi"].isna() | mdf["sync_r"].isna())
                sub = mdf.loc[mask, ["phi", "sync_r"]]
                if len(sub) >= 10 and sub["phi"].std() > 0 and sub["sync_r"].std() > 0:
                    r, p = stats.pearsonr(sub["phi"], sub["sync_r"])
                    phi_corr = float(r)
                    phi_corr_p = float(p)
            except Exception:
                pass

    return {
        "run": run_dir.name,
        "status": "OK_FULL" if n >= 200 else f"OK_PARTIAL({n}eps)",
        "n_episodes": n,
        "phi_mean": phi_mean,
        "phi_std": phi_std,
        "last100_reward": float(last100),
        "positive_eps": pos_eps,
        "phi_sync_r_r": phi_corr,
        "phi_sync_r_p": phi_corr_p,
        "phi_methods": phi_methods,
    }


def evaluate_thresholds(row: dict) -> tuple[bool, list[str]]:
    """Apply the four-criterion strict acceptance test from the plan.
    Returns (passes_all, list_of_failed_thresholds)."""
    if row.get("status", "").startswith("NO_") or row.get("n_episodes", 0) < 50:
        return False, ["INSUFFICIENT_DATA"]

    failed = []
    if not (row["phi_std"] >= THRESHOLDS["phi_std_min"]):
        failed.append(f"phi_std<{THRESHOLDS['phi_std_min']}")
    if not (row["last100_reward"] >= THRESHOLDS["last100_reward_min"]):
        failed.append(f"last100<{THRESHOLDS['last100_reward_min']}")
    if not (row["positive_eps"] >= THRESHOLDS["positive_eps_min"]):
        failed.append(f"pos_eps<{THRESHOLDS['positive_eps_min']}")
    # phi-sync_R correlation gate: r must be >= 0.4 AND phi_std must clear
    # the pathology floor, otherwise a flat trajectory can hit r=1.000.
    if not (
        not np.isnan(row.get("phi_sync_r_r", np.nan))
        and row["phi_sync_r_r"] >= THRESHOLDS["phi_sync_r_correlation_min"]
        and row["phi_std"] >= THRESHOLDS["phi_std_min_for_correlation"]
    ):
        failed.append(
            f"phi-sync_r r<{THRESHOLDS['phi_sync_r_correlation_min']} "
            f"or phi_std<{THRESHOLDS['phi_std_min_for_correlation']}"
        )

    return (len(failed) == 0), failed


def print_table(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    cols = [
        "run", "status", "n_episodes", "phi_mean", "phi_std",
        "last100_reward", "positive_eps", "phi_sync_r_r", "phi_methods",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    # Prettier formatting for numeric columns.
    if "phi_mean" in df.columns:
        df["phi_mean"] = df["phi_mean"].apply(lambda x: f"{x:.5f}" if pd.notna(x) else "NaN")
    if "phi_std" in df.columns:
        df["phi_std"] = df["phi_std"].apply(lambda x: f"{x:.5f}" if pd.notna(x) else "NaN")
    if "last100_reward" in df.columns:
        df["last100_reward"] = df["last100_reward"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "NaN")
    if "phi_sync_r_r" in df.columns:
        df["phi_sync_r_r"] = df["phi_sync_r_r"].apply(
            lambda x: f"{x:+.3f}" if pd.notna(x) else "NaN"
        )

    with pd.option_context("display.max_colwidth", 60, "display.width", 220):
        print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Ablation campaign report")
    parser.add_argument(
        "--campaign-dir", default="runs/ablation",
        help="Directory holding A_current/, C_no_replay/, etc.",
    )
    parser.add_argument(
        "--include-historical", action="store_true",
        help="Also load runs/dark_room_improved_1k and runs/dark_room_phase3_1k "
             "as descriptive references. These are pre-fix; do NOT compare phi "
             "values directly to ablation runs.",
    )
    args = parser.parse_args()

    campaign = Path(args.campaign_dir)
    if not campaign.exists():
        print(f"ERROR: campaign dir {campaign} does not exist")
        return 1

    # Load every subdirectory that has an episodes.csv. Sort so output is stable.
    rows: list[dict] = []
    for sub in sorted(campaign.iterdir()):
        if sub.is_dir() and not sub.name.startswith("_"):
            row = load_run(sub)
            if row is not None:
                rows.append(row)

    if args.include_historical:
        for hist in ("dark_room_improved_1k", "dark_room_phase3_1k"):
            hpath = Path("runs") / hist
            if hpath.exists():
                row = load_run(hpath)
                if row is not None:
                    row["run"] = f"[hist] {row['run']}"
                    rows.append(row)

    if not rows:
        print(f"No ablation runs found under {campaign}/")
        return 1

    print("=" * 90)
    print("Ablation campaign report")
    print(f"Campaign dir: {campaign}")
    print(f"Loaded {len(rows)} runs")
    print("=" * 90)
    print()
    print_table(rows)
    print()

    # Strict acceptance evaluation. State FAILED before any interpretation.
    print("=" * 90)
    print("Acceptance test (4 criteria, all required):")
    for k, v in THRESHOLDS.items():
        print(f"  {k}: {v}")
    print("=" * 90)
    print()
    candidates = []
    for row in rows:
        if row["run"].startswith("[hist]"):
            continue
        passes, failed = evaluate_thresholds(row)
        marker = "PASS" if passes else "FAILED"
        detail = "all four met" if passes else f"FAILED on: {', '.join(failed)}"
        print(f"  {row['run']:20s}  {marker}  ({detail})")
        if passes:
            candidates.append(row["run"])

    print()
    print("=" * 90)
    if not candidates:
        print("No fix candidate. State: NONE_RESTORED.")
        print("Action: stop, report to user, do NOT paper over.")
    elif len(candidates) == 1:
        print(f"Single fix candidate identified: {candidates[0]}")
        print("Action: run verify protocol (200 eps, runs/ablation/fix_verify),")
        print("        then apply preference order (remove > relax > retune).")
    else:
        print(f"Multiple partial candidates: {candidates}")
        print("Action: compose them, run a verify run before declaring success.")
    print("=" * 90)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

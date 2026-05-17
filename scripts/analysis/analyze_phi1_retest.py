"""Evaluate a Phi-1 retest run against the pre-registered falsification criterion.

Reads `metrics.csv` from a run created under the 2026-05-17 revised
architecture (Phase A attention-weighted fusion + Phase C gate fixes +
Phase D mock semantic + audio) and reports the Phi-1 verdict per
`docs/preregistered_predictions.md` section 10.

Pre-registered criterion (section 10, 2026-05-17, untouchable post-hoc):

  Pearson r > 0.4 between pyphi-phi (column 'phi') and AKOrN sync_R
  (column 'sync_r'), with non-degenerate variances:
    phi_std > 0.01
    sync_R_std > 0.02

  Degenerate variance triggers a re-run, not threshold revision.

Decision-gate outcomes mapped to the plan Phase F2 table:
  r >= 0.40, non-degenerate     -> PASS  (proceed to F3 3-seed)
  0.15 <= r < 0.40              -> PARTIAL (Phase B contingent or accept)
  r < 0.15, non-degenerate      -> FAIL  (stop or run Phase B)
  any variance degenerate       -> RE-RUN

Usage:
    python -m scripts.analysis.analyze_phi1_retest \\
        --run-dir runs/phi1_retest_v1_seed42

Output:
    - Stdout: per-criterion verdict + rolling-window trajectory
    - Optional --output writes a markdown verdict doc
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


MIN_STEP = 1000              # plan: post-warmup window
PHI_THRESHOLD = 0.4          # pre-registered Phi-1 threshold (not revisable)
PARTIAL_THRESHOLD = 0.15     # plan decision gate for F3 / Phase B branch
PHI_STD_FLOOR = 0.01         # pre-registered non-degenerate criterion
SYNC_STD_FLOOR = 0.02        # pre-registered non-degenerate criterion
ROLLING_WINDOW = 5000        # step window for rolling-r trajectory


@dataclass
class RetestResult:
    n_valid_rows: int
    n_pyphi_rows: int
    phi_mean: float
    phi_std: float
    sync_r_mean: float
    sync_r_std: float
    full_run_r: float
    full_run_p: float
    rolling: list[dict]
    verdict: str  # PASS / PARTIAL / FAIL / RE-RUN / NO DATA
    detail: str


def load_metrics(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        sys.exit(1)
    return pd.read_csv(path)


def filter_valid(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows past warm-up. Do NOT drop pyphi-skipped rows from the
    correlation calc: phi values carry forward on skipped steps by design,
    and removing them would oversample the high-variance pyphi calls."""
    return df[df["global_step"] >= MIN_STEP].copy()


def rolling_window_r(valid: pd.DataFrame, window: int) -> list[dict]:
    """Compute (start, end, r, n) for non-overlapping windows of size `window`."""
    out: list[dict] = []
    if len(valid) == 0:
        return out
    min_step = int(valid["global_step"].min())
    max_step = int(valid["global_step"].max())
    start = min_step
    while start + window <= max_step + 1:
        end = start + window
        chunk = valid[(valid["global_step"] >= start) & (valid["global_step"] < end)]
        if len(chunk) >= 50 and chunk["phi"].std() > 0 and chunk["sync_r"].std() > 0:
            r, p = stats.pearsonr(chunk["phi"], chunk["sync_r"])
            out.append({
                "start": start, "end": end,
                "r": float(r), "p": float(p), "n": len(chunk),
                "phi_mean": float(chunk["phi"].mean()),
                "phi_std": float(chunk["phi"].std()),
                "sync_mean": float(chunk["sync_r"].mean()),
                "sync_std": float(chunk["sync_r"].std()),
            })
        start += window
    return out


def evaluate(df: pd.DataFrame) -> RetestResult:
    valid = filter_valid(df)
    if len(valid) == 0:
        return RetestResult(0, 0, 0, 0, 0, 0, 0, 1, [],
                            "NO DATA", "no rows past warm-up")
    n_pyphi = int((valid["phi_method"] == "pyphi").sum()) if "phi_method" in valid else len(valid)
    phi_mean, phi_std = float(valid["phi"].mean()), float(valid["phi"].std())
    sync_mean, sync_std = float(valid["sync_r"].mean()), float(valid["sync_r"].std())

    if phi_std == 0 or sync_std == 0:
        return RetestResult(
            len(valid), n_pyphi, phi_mean, phi_std, sync_mean, sync_std, 0.0, 1.0,
            rolling_window_r(valid, ROLLING_WINDOW),
            "RE-RUN",
            f"degenerate variance: phi_std={phi_std:.3e}, sync_R_std={sync_std:.3e}",
        )

    r, p = stats.pearsonr(valid["phi"], valid["sync_r"])
    r_full, p_full = float(r), float(p)
    rolling = rolling_window_r(valid, ROLLING_WINDOW)

    # Variance check (pre-registered non-degenerate criterion)
    variance_ok = (phi_std > PHI_STD_FLOOR) and (sync_std > SYNC_STD_FLOOR)

    if not variance_ok:
        verdict = "RE-RUN"
        detail = (
            f"variance below pre-registered floor: phi_std={phi_std:.3e} "
            f"(floor {PHI_STD_FLOOR}), sync_R_std={sync_std:.3e} "
            f"(floor {SYNC_STD_FLOOR}). Plan section 10 says: degenerate "
            f"variance triggers a re-run, not threshold revision. "
            f"For information: r = {r_full:+.4f}, p = {p_full:.2e}."
        )
    elif r_full >= PHI_THRESHOLD:
        verdict = "PASS"
        detail = (
            f"r = {r_full:+.4f} >= pre-registered {PHI_THRESHOLD} "
            f"with p = {p_full:.2e}. Variances non-degenerate "
            f"(phi_std={phi_std:.3e}, sync_R_std={sync_std:.3e}). "
            f"Proceed to F3 3-seed confirmation."
        )
    elif r_full >= PARTIAL_THRESHOLD:
        verdict = "PARTIAL"
        detail = (
            f"r = {r_full:+.4f} in [{PARTIAL_THRESHOLD}, {PHI_THRESHOLD}). "
            f"Pre-registered threshold not met but partial signal present. "
            f"Plan decision gate: if Phase B not yet done, do Phase B and "
            f"re-run F2. Otherwise run F3 anyway and report best honestly. "
            f"p = {p_full:.2e}, phi_std={phi_std:.3e}, sync_R_std={sync_std:.3e}."
        )
    else:
        verdict = "FAIL"
        detail = (
            f"r = {r_full:+.4f} < partial threshold {PARTIAL_THRESHOLD}. "
            f"Pre-registered threshold not met. Plan: if Phase B not yet "
            f"done, do Phase B and re-run F2. Otherwise pre-registration "
            f"FAILED, write verdict and stop. "
            f"p = {p_full:.2e}, phi_std={phi_std:.3e}, sync_R_std={sync_std:.3e}."
        )

    return RetestResult(
        len(valid), n_pyphi, phi_mean, phi_std, sync_mean, sync_std,
        r_full, p_full, rolling, verdict, detail,
    )


def write_report(output_path: str, run_dir: str, res: RetestResult) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    lines = [
        "# Phi-1 retest verdict (pre-registered section 10)",
        "",
        f"**Run:** `{run_dir}`",
        f"**Analysis window:** steps >= {MIN_STEP}",
        f"**Pre-registered criterion:** Pearson r > {PHI_THRESHOLD} between "
        f"phi (pyphi) and sync_R, with phi_std > {PHI_STD_FLOOR} and "
        f"sync_R_std > {SYNC_STD_FLOOR}",
        "",
        f"## Headline: {res.verdict}",
        "",
        res.detail,
        "",
        "## Summary statistics",
        "",
        f"- Rows past warm-up: {res.n_valid_rows}",
        f"- Rows with phi_method == 'pyphi': {res.n_pyphi_rows} "
        f"(rest are 'skipped' or 'insufficient_data')",
        f"- phi: mean = {res.phi_mean:.4e}, std = {res.phi_std:.4e}",
        f"- sync_R: mean = {res.sync_r_mean:.4f}, std = {res.sync_r_std:.4e}",
        f"- Full-run Pearson r(phi, sync_R) = {res.full_run_r:+.4f}, p = {res.full_run_p:.2e}",
        "",
        f"## Rolling {ROLLING_WINDOW}-step window trajectory",
        "",
        "| window (steps) | n | r(phi, sync_R) | p | phi_mean | phi_std | sync_mean | sync_std |",
        "|----------------|---|----------------|---|----------|---------|-----------|----------|",
    ]
    for w in res.rolling:
        lines.append(
            f"| {w['start']}-{w['end']} | {w['n']} | {w['r']:+.4f} | "
            f"{w['p']:.2e} | {w['phi_mean']:.3e} | {w['phi_std']:.3e} | "
            f"{w['sync_mean']:.4f} | {w['sync_std']:.4f} |"
        )

    lines.extend([
        "",
        "## Decision-gate mapping (plan Phase F2)",
        "",
        f"| outcome | what to do |",
        f"|---------|------------|",
        f"| PASS (r >= {PHI_THRESHOLD}, non-degenerate) | Run F3 (3-seed confirmation) |",
        f"| PARTIAL ({PARTIAL_THRESHOLD} <= r < {PHI_THRESHOLD}) | Run Phase B (content-level binding) and re-run F2; OR run F3 anyway and report |",
        f"| FAIL (r < {PARTIAL_THRESHOLD}, non-degenerate) | If Phase B not yet done, do Phase B and re-run F2. Otherwise: pre-reg FAILED, stop. |",
        f"| RE-RUN (variance degenerate) | Tune attention_temperature, audio modulation, mock_semantic salience; do NOT revise threshold |",
        "",
        "## Reproducibility",
        "",
        f"All numbers reproducible from `{run_dir}/metrics.csv` via:",
        "",
        "```bash",
        f"python -m scripts.analysis.analyze_phi1_retest --run-dir {run_dir} --output {output_path}",
        "```",
        "",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True,
                        help="Directory with metrics.csv from a Phi-1 retest run")
    parser.add_argument("--output", default=None,
                        help="Path to write markdown report. If omitted, prints to stdout.")
    args = parser.parse_args()

    df = load_metrics(args.run_dir)
    res = evaluate(df)

    print("=" * 72)
    print(f"Phi-1 retest verdict for {args.run_dir}")
    print("=" * 72)
    print(f"Verdict: {res.verdict}")
    print(f"  {res.detail}")
    print()
    print(f"Stats:")
    print(f"  rows past warm-up = {res.n_valid_rows} (pyphi = {res.n_pyphi_rows})")
    print(f"  phi mean / std    = {res.phi_mean:.4e} / {res.phi_std:.4e}")
    print(f"  sync_R mean / std = {res.sync_r_mean:.4f} / {res.sync_r_std:.4e}")
    print(f"  Full-run r        = {res.full_run_r:+.4f}, p = {res.full_run_p:.2e}")
    print()
    if res.rolling:
        print(f"Rolling {ROLLING_WINDOW}-step windows:")
        for w in res.rolling:
            print(
                f"  steps {w['start']:6d}-{w['end']:6d}: "
                f"r = {w['r']:+.4f} (p = {w['p']:.2e}, n = {w['n']}, "
                f"phi_std = {w['phi_std']:.3e}, sync_std = {w['sync_std']:.4f})"
            )

    if args.output:
        write_report(args.output, args.run_dir, res)
        print(f"\nWrote markdown report to {args.output}")


if __name__ == "__main__":
    main()

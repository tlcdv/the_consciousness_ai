"""Aggregate multiple RIIU-substrate runs across seeds and report replication.

Reads `metrics.csv` from N run directories (one per seed), selects the same
substrate column from each, and reports:

  - Per-seed full-run pearson_r(phi_riiu, sync_R) and phi_std
  - Per-seed peak r in any rolling --peak-window-size window (default 5000)
  - The window position of each peak
  - Cross-seed mean and SE of peak_r
  - Whether peaks overlap the 2026-05-16 reference window (steps 11000-16000)

Replication verdict (per plan Phase C):
  REPLICATED     if >= 2 of N seeds have peak_r >= 0.20 AND windows overlap
  WEAK           if 1 of N seeds meets that bar
  NOT REPLICATED otherwise

Usage:
    python -m scripts.analysis.aggregate_seeds \\
        --run-dirs runs/riiu_tectum_seed43 runs/riiu_tectum_seed44 runs/riiu_tectum_seed45 \\
        --substrate tectum \\
        --peak-window-size 5000 \\
        --output docs/results/riiu_multiseed_2026_05_17.md
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# Reuse the substrate-aware loader and filter from the single-run analyzer.
from scripts.analysis.compare_phi_pathways import (
    MIN_STEP,
    filter_valid,
    load_metrics,
)


REF_PEAK_WINDOW = (11000, 16000)  # 2026-05-16 single-seed broadcast peak


@dataclass
class SeedResult:
    run_dir: str
    seed: int | None
    full_run_r: float
    full_run_p: float
    phi_mean: float
    phi_std: float
    peak_r: float
    peak_window_start: int
    peak_window_end: int
    overlaps_reference: bool
    n_rows: int


def extract_seed(run_dir: str) -> int | None:
    """Parse a seed int from a path like 'runs/riiu_tectum_seed43'."""
    tail = os.path.basename(run_dir.rstrip("/\\"))
    if "seed" in tail:
        try:
            return int(tail.rsplit("seed", 1)[-1])
        except ValueError:
            return None
    return None


def rolling_peak_r(valid: pd.DataFrame, window: int) -> tuple[float, int, int]:
    """Return (peak_r, window_start, window_end) over rolling fixed windows.

    Steps are step-indexed by `global_step`. We slide a window of `window`
    steps and compute r(phi_riiu, sync_R) inside each, skipping windows with
    < 50 rows or zero variance. Returns the maximum r encountered.
    """
    if "global_step" not in valid.columns:
        return 0.0, 0, 0
    if len(valid) == 0:
        return 0.0, 0, 0
    min_step = int(valid["global_step"].min())
    max_step = int(valid["global_step"].max())
    best_r = -2.0
    best_start = min_step
    best_end = min_step + window
    stride = max(1, window // 4)
    start = min_step
    while start + window <= max_step + 1:
        end = start + window
        chunk = valid[(valid["global_step"] >= start) & (valid["global_step"] < end)]
        if len(chunk) >= 50 and chunk["phi_riiu"].std() > 0 and chunk["sync_r"].std() > 0:
            r, _ = stats.pearsonr(chunk["phi_riiu"], chunk["sync_r"])
            if r > best_r:
                best_r = float(r)
                best_start = start
                best_end = end
        start += stride
    if best_r == -2.0:
        return 0.0, min_step, min(min_step + window, max_step)
    return best_r, best_start, best_end


def windows_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def evaluate_seed(run_dir: str, substrate: str, peak_window_size: int) -> SeedResult:
    df = load_metrics(run_dir, substrate=substrate)
    valid = filter_valid(df)
    if len(valid) == 0 or valid["phi_riiu"].std() == 0 or valid["sync_r"].std() == 0:
        return SeedResult(
            run_dir=run_dir,
            seed=extract_seed(run_dir),
            full_run_r=0.0,
            full_run_p=1.0,
            phi_mean=float(valid["phi_riiu"].mean()) if len(valid) else 0.0,
            phi_std=float(valid["phi_riiu"].std()) if len(valid) else 0.0,
            peak_r=0.0,
            peak_window_start=0,
            peak_window_end=0,
            overlaps_reference=False,
            n_rows=len(valid),
        )
    full_r, full_p = stats.pearsonr(valid["phi_riiu"], valid["sync_r"])
    peak_r, peak_s, peak_e = rolling_peak_r(valid, peak_window_size)
    return SeedResult(
        run_dir=run_dir,
        seed=extract_seed(run_dir),
        full_run_r=float(full_r),
        full_run_p=float(full_p),
        phi_mean=float(valid["phi_riiu"].mean()),
        phi_std=float(valid["phi_riiu"].std()),
        peak_r=peak_r,
        peak_window_start=peak_s,
        peak_window_end=peak_e,
        overlaps_reference=windows_overlap((peak_s, peak_e), REF_PEAK_WINDOW),
        n_rows=len(valid),
    )


def replication_verdict(results: list[SeedResult]) -> tuple[str, str]:
    """Return (verdict_token, one-line explanation) per plan Phase C."""
    qualifying = [r for r in results if r.peak_r >= 0.20 and r.overlaps_reference]
    n_qual = len(qualifying)
    n_total = len(results)
    if n_qual >= 2:
        token = "REPLICATED"
    elif n_qual == 1:
        token = "WEAK"
    else:
        token = "NOT REPLICATED"
    detail = (
        f"{n_qual} of {n_total} seeds met both bars: peak_r >= 0.20 AND "
        f"peak window overlapped the 2026-05-16 reference window "
        f"{REF_PEAK_WINDOW}."
    )
    return token, detail


def write_report(
    output_path: str,
    substrate: str,
    peak_window_size: int,
    results: list[SeedResult],
    verdict_token: str,
    verdict_detail: str,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    peak_rs = [r.peak_r for r in results]
    mean_peak = float(np.mean(peak_rs)) if peak_rs else 0.0
    se_peak = float(np.std(peak_rs, ddof=1) / max(1, np.sqrt(len(peak_rs)))) if len(peak_rs) > 1 else 0.0

    lines = [
        "# RIIU multi-seed verification",
        "",
        f"**Substrate:** {substrate}",
        f"**Rolling window size:** {peak_window_size} steps",
        f"**Reference peak window (from 2026-05-16):** "
        f"steps {REF_PEAK_WINDOW[0]} to {REF_PEAK_WINDOW[1]}",
        f"**Number of seeds:** {len(results)}",
        f"**Replication verdict:** {verdict_token}",
        f"**Verdict detail:** {verdict_detail}",
        "",
        "## Per-seed results",
        "",
        "| Seed | n_rows | phi_mean | phi_std | full_run_r | full_p | peak_r | peak_window | overlaps_ref |",
        "|------|--------|----------|---------|------------|--------|--------|-------------|--------------|",
    ]
    for r in results:
        seed_str = str(r.seed) if r.seed is not None else os.path.basename(r.run_dir)
        lines.append(
            f"| {seed_str} | {r.n_rows} | {r.phi_mean:.3e} | {r.phi_std:.3e} | "
            f"{r.full_run_r:+.4f} | {r.full_run_p:.2e} | {r.peak_r:+.4f} | "
            f"{r.peak_window_start}-{r.peak_window_end} | "
            f"{'yes' if r.overlaps_reference else 'no'} |"
        )

    lines.extend([
        "",
        "## Cross-seed summary",
        "",
        f"- Mean peak_r across seeds: **{mean_peak:+.4f}**",
        f"- SE of peak_r: {se_peak:.4f}",
        "",
        "## Decision",
        "",
    ])

    if verdict_token == "REPLICATED":
        lines.append(
            "REPLICATED. The peak phi-sync_R correlation observed in the "
            "2026-05-16 single-seed broadcast run reproduces in this multi-seed "
            "verification. This revives the Phi-1 thread as a phase-transition "
            "phenomenon and warrants a follow-up plan to investigate the "
            "mechanism (when does the peak appear, why does it collapse, can "
            "it be stabilized). The pre-registered r > 0.4 threshold is NOT "
            "revised; it is still not met by any individual run. What is "
            "replicated is the transient mid-training peak."
        )
    elif verdict_token == "WEAK":
        lines.append(
            "WEAK. Only one of the seeds reproduced the peak. Recommendation: "
            "expand to 5 seeds before any architectural change or claim "
            "revision. Do not promote RIIU-as-reward to default; keep "
            "--enable-riiu off by default."
        )
    else:
        lines.append(
            "NOT REPLICATED. The 2026-05-16 transient peak of r=+0.267 did not "
            "reproduce under multi-seed verification. The Phi-1 prediction "
            "stands FAILED across pathways (pyphi, RIIU) and substrates. The "
            "project closes the Phi-1 chapter for the current architecture "
            "and proceeds to Phase 5 (Dynamic Self-Representation & "
            "Meta-Cognition) per docs/roadmap.md."
        )

    lines.extend([
        "",
        "## Reproducibility",
        "",
        "Every number is reproducible from the per-seed `metrics.csv` files via:",
        "",
        "```bash",
        "python -m scripts.analysis.aggregate_seeds \\",
        f"  --run-dirs {' '.join(r.run_dir for r in results)} \\",
        f"  --substrate {substrate} --peak-window-size {peak_window_size} \\",
        f"  --output {output_path}",
        "```",
        "",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True,
                        help="Two or more run directories (one per seed)")
    parser.add_argument("--substrate", default="broadcast",
                        choices=["broadcast", "tectum", "audio"],
                        help="RIIU substrate column to evaluate")
    parser.add_argument("--peak-window-size", type=int, default=5000,
                        help="Rolling window size in steps for peak detection")
    parser.add_argument("--output", default=None,
                        help="Path to write markdown report. If omitted, prints to stdout.")
    args = parser.parse_args()

    if len(args.run_dirs) < 2:
        print("WARNING: aggregate_seeds is designed for >= 2 runs; got 1. Continuing.")

    results = [
        evaluate_seed(run_dir, args.substrate, args.peak_window_size)
        for run_dir in args.run_dirs
    ]
    verdict_token, verdict_detail = replication_verdict(results)

    print("=" * 70)
    print(f"Multi-seed RIIU aggregation (substrate: {args.substrate})")
    print("=" * 70)
    for r in results:
        seed_str = str(r.seed) if r.seed is not None else os.path.basename(r.run_dir)
        overlap_str = "yes" if r.overlaps_reference else "no"
        print(
            f"  seed={seed_str}  n={r.n_rows}  full_r={r.full_run_r:+.4f}  "
            f"phi_std={r.phi_std:.3e}  peak_r={r.peak_r:+.4f}  "
            f"window={r.peak_window_start}-{r.peak_window_end}  "
            f"overlaps_ref={overlap_str}"
        )
    print()
    print(f"Verdict: {verdict_token}")
    print(f"  {verdict_detail}")

    if args.output:
        write_report(
            args.output, args.substrate, args.peak_window_size,
            results, verdict_token, verdict_detail,
        )
        print(f"\nWrote markdown report to {args.output}")


if __name__ == "__main__":
    main()

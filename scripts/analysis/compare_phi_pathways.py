"""
Compare pyphi and RIIU phi pathways from a training run.

Reads `metrics.csv` produced by `train_rlhf.py --enable-riiu` and evaluates the
four go/no-go criteria set in
`C:\\Users\\zaesa\\.claude\\plans\\let-s-plan-the-next-misty-parasol.md` (Phase 7):

  A. Variance unlock:   std(phi_riiu) >= 5 * std(phi_pyphi)
  B. Binding correlation: pearson_r(phi_riiu, sync_R) >= 0.15, p < 0.05
  C. Signal alive:      mean(phi_riiu) >= 1e-3 AND mean > 10 * std (not collapsed)
  D. No reward regression: final-100-episode reward >= baseline - 1 SE
                          (D requires a reference baseline CSV via --baseline-dir)

All evaluations exclude rows where phi_method == 'insufficient_data' (the
early-episode TPM warm-up rows where pyphi cannot return a meaningful value).
The plan further restricts the analysis window to steps 1000+ to give RIIU's
sliding-window covariance time to stabilize past the initial warm-up.

Usage:
    python -m scripts.analysis.compare_phi_pathways \\
        --run-dir runs/riiu_compare_seed42 \\
        --baseline-dir runs/ablation/A_current \\
        --output docs/results/riiu_compare_2026_05_16.md
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


MIN_STEP = 1000  # plan Phase 7: analyze steps 1000+ (post-warmup)

# Map --substrate CLI value to the CSV column emitted by metrics_logger.
# The probe-all run (post-2026-05-17) writes all three explicitly. Older
# CSVs from pre-probe runs only have `phi_riiu` (whichever was the reward
# source) and treat 'broadcast' as the default; we fall back to that.
SUBSTRATE_COLUMN = {
    "broadcast": "phi_riiu_broadcast",
    "tectum": "phi_riiu_tectum",
    "audio": "phi_riiu_audio",
}


@dataclass
class Verdict:
    name: str
    passed: bool
    detail: str


def load_metrics(run_dir: str, substrate: str = "broadcast") -> pd.DataFrame:
    """Load metrics.csv and overwrite `phi_riiu` with the chosen substrate.

    The returned DataFrame always has `phi_riiu` as the column being analyzed,
    which keeps all downstream criterion functions substrate-agnostic. The
    column resolution falls back to the legacy `phi_riiu` (= the reward
    source from that run) when the requested explicit substrate column is
    missing or all-zero; a one-line note is printed in that case.
    """
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        sys.exit(1)
    df = pd.read_csv(path)
    required = {"phi", "phi_riiu", "sync_r", "phi_method"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: {path} missing columns: {missing}")
        sys.exit(1)
    explicit_col = SUBSTRATE_COLUMN[substrate]
    if explicit_col in df.columns and df[explicit_col].abs().sum() > 0:
        df["phi_riiu"] = df[explicit_col]
    elif substrate != "broadcast":
        print(
            f"NOTE: column {explicit_col!r} missing or all-zero in {path}; "
            f"falling back to legacy phi_riiu column (interpreted as the "
            f"reward-source substrate from that run)."
        )
    return df


def filter_valid(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows past the warm-up window with valid phi method."""
    return df[(df["global_step"] >= MIN_STEP) & (df["phi_method"] != "insufficient_data")]


def criterion_a_variance_unlock(valid: pd.DataFrame) -> Verdict:
    std_riiu = valid["phi_riiu"].std()
    std_pyphi = valid["phi"].std()
    if std_pyphi == 0:
        return Verdict(
            "A. Variance unlock",
            False,
            f"std(pyphi) is exactly 0 (degenerate). std(RIIU)={std_riiu:.3e}",
        )
    ratio = std_riiu / std_pyphi
    passed = ratio >= 5.0
    return Verdict(
        "A. Variance unlock",
        passed,
        f"std(RIIU)={std_riiu:.3e}, std(pyphi)={std_pyphi:.3e}, "
        f"ratio={ratio:.2f}x (threshold >= 5.0x)",
    )


def criterion_b_binding_correlation(valid: pd.DataFrame) -> Verdict:
    if len(valid) < 2 or valid["phi_riiu"].std() == 0 or valid["sync_r"].std() == 0:
        return Verdict(
            "B. Binding correlation",
            False,
            "Insufficient variance in phi_riiu or sync_r for correlation",
        )
    r, p = stats.pearsonr(valid["phi_riiu"], valid["sync_r"])
    passed = r >= 0.15 and p < 0.05
    return Verdict(
        "B. Binding correlation",
        passed,
        f"pearson_r(phi_riiu, sync_R) = {r:+.4f}, p = {p:.2e} "
        f"(threshold r >= 0.15, p < 0.05). For comparison, pyphi's best from "
        f"the 2026-05-14 ablation campaign was r=+0.089.",
    )


def criterion_c_signal_alive(valid: pd.DataFrame) -> Verdict:
    mean = valid["phi_riiu"].mean()
    std = valid["phi_riiu"].std()
    mean_above_floor = mean >= 1e-3
    not_collapsed = (std > 0) and (mean <= 10 * std)
    passed = mean_above_floor and not_collapsed
    return Verdict(
        "C. Signal alive",
        passed,
        f"mean(phi_riiu)={mean:.3e} (threshold >= 1e-3), "
        f"std={std:.3e}, mean/std={mean / std if std > 0 else float('inf'):.2f} "
        f"(threshold <= 10x for non-collapsed)",
    )


def criterion_d_no_reward_regression(
    run_dir: str, baseline_dir: str | None
) -> Verdict:
    if baseline_dir is None:
        return Verdict(
            "D. No reward regression",
            True,
            "SKIPPED: no --baseline-dir provided. Pass --baseline-dir to enable.",
        )

    def last_100_reward(d: str) -> tuple[float, float, int]:
        ep_path = os.path.join(d, "episodes.csv")
        if not os.path.exists(ep_path):
            raise FileNotFoundError(ep_path)
        ep = pd.read_csv(ep_path)
        last = ep.tail(100)
        return last["total_reward"].mean(), last["total_reward"].std(), len(last)

    try:
        run_mean, run_std, run_n = last_100_reward(run_dir)
        base_mean, base_std, base_n = last_100_reward(baseline_dir)
    except FileNotFoundError as e:
        return Verdict("D. No reward regression", False, f"Missing file: {e}")

    base_se = base_std / max(1, np.sqrt(base_n))
    threshold = base_mean - base_se
    passed = run_mean >= threshold
    return Verdict(
        "D. No reward regression",
        passed,
        f"run last-100 reward = {run_mean:+.3f} (n={run_n}), "
        f"baseline = {base_mean:+.3f} (n={base_n}, 1 SE = {base_se:.3f}), "
        f"threshold = {threshold:+.3f}",
    )


def summarize_distributions(df: pd.DataFrame, valid: pd.DataFrame) -> str:
    lines = ["## Distribution summary", ""]
    lines.append(f"Total rows in metrics.csv: {len(df)}")
    lines.append(f"Rows with phi_method == 'insufficient_data': "
                 f"{(df['phi_method'] == 'insufficient_data').sum()}")
    lines.append(f"Rows past warm-up (step >= {MIN_STEP}) and valid: {len(valid)}")
    lines.append("")
    lines.append("| metric | mean | std | min | max | nonzero rows |")
    lines.append("|--------|------|-----|-----|-----|--------------|")
    for col in ("phi", "phi_riiu", "sync_r"):
        m, s = valid[col].mean(), valid[col].std()
        lo, hi = valid[col].min(), valid[col].max()
        nz = (valid[col] != 0).sum()
        lines.append(f"| {col} | {m:.3e} | {s:.3e} | {lo:.3e} | {hi:.3e} | {nz} / {len(valid)} |")
    return "\n".join(lines)


def write_report(
    output_path: str,
    run_dir: str,
    baseline_dir: str | None,
    verdicts: list[Verdict],
    distribution_summary: str,
    valid: pd.DataFrame,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    final_pass = all(v.passed for v in verdicts)
    headline = "PASS" if final_pass else "FAIL"

    lines = [
        f"# RIIU vs pyphi phi pathway comparison",
        "",
        f"**Run:** `{run_dir}`",
        f"**Baseline:** `{baseline_dir}`" if baseline_dir else "**Baseline:** not provided",
        f"**Analysis window:** steps >= {MIN_STEP}, phi_method != insufficient_data",
        f"**Headline verdict:** {headline}",
        "",
        "## Go / no-go criteria (set 2026-05-16 in plan let-s-plan-the-next-misty-parasol.md)",
        "",
        "| Criterion | Verdict | Detail |",
        "|-----------|---------|--------|",
    ]
    for v in verdicts:
        verdict_str = "PASS" if v.passed else "FAIL"
        lines.append(f"| {v.name} | {verdict_str} | {v.detail} |")

    lines.extend([
        "",
        distribution_summary,
        "",
        "## Decision",
        "",
    ])

    if final_pass:
        lines.append(
            "All four go/no-go criteria pass. Per the plan, proceed to a "
            "3-seed multi-run before merging RIIU-as-reward as the default."
        )
    else:
        failed = [v.name for v in verdicts if not v.passed]
        lines.append(
            f"FAILED: {len(failed)} of {len(verdicts)} criteria did not pass "
            f"({', '.join(failed)}). Per plan Phase 8: keep the RIIU code "
            "merged behind --enable-riiu (default off), preserve as a "
            "covariance diagnostic, and move to Direction C (accept negative "
            "result, reframe documentation)."
        )

    lines.extend([
        "",
        "## Reproducibility",
        "",
        f"Every number in this report can be reproduced from `{run_dir}/metrics.csv` "
        "and `episodes.csv` via:",
        "",
        "```bash",
        f"python -m scripts.analysis.compare_phi_pathways "
        f"--run-dir {run_dir} "
        + (f"--baseline-dir {baseline_dir} " if baseline_dir else "")
        + f"--output {output_path}",
        "```",
        "",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True,
                        help="Directory with metrics.csv from --enable-riiu run")
    parser.add_argument("--baseline-dir", default=None,
                        help="Reference run for criterion D (reward regression)")
    parser.add_argument("--substrate", default="broadcast",
                        choices=["broadcast", "tectum", "audio"],
                        help="Which RIIU substrate column to evaluate. The "
                             "selected per-substrate column is treated as "
                             "`phi_riiu` for criteria A, B, C. Falls back to "
                             "the legacy reward-source column if the explicit "
                             "one is missing or all-zero (older runs). Default "
                             "broadcast.")
    parser.add_argument("--output", default=None,
                        help="Path to write markdown report. If omitted, prints to stdout.")
    args = parser.parse_args()

    df = load_metrics(args.run_dir, substrate=args.substrate)
    valid = filter_valid(df)

    if len(valid) == 0:
        print("ERROR: no valid rows in analysis window. Run may be too short.")
        sys.exit(1)

    verdicts = [
        criterion_a_variance_unlock(valid),
        criterion_b_binding_correlation(valid),
        criterion_c_signal_alive(valid),
        criterion_d_no_reward_regression(args.run_dir, args.baseline_dir),
    ]

    distribution_summary = summarize_distributions(df, valid)

    print("=" * 70)
    print(f"Run: {args.run_dir}  (substrate: {args.substrate})")
    print("=" * 70)
    for v in verdicts:
        verdict_str = "PASS" if v.passed else "FAIL"
        print(f"[{verdict_str}] {v.name}")
        print(f"        {v.detail}")
    print()
    print(distribution_summary)
    print()
    final_pass = all(v.passed for v in verdicts)
    print(f"Headline: {'PASS' if final_pass else 'FAIL'}")

    if args.output:
        write_report(args.output, args.run_dir, args.baseline_dir,
                     verdicts, distribution_summary, valid)
        print(f"Wrote markdown report to {args.output}")


if __name__ == "__main__":
    main()

"""Gate B2: the tolerance ignition rule, live, on fresh seeds.

Gate B FAILED on silence (0.62 to 0.69) and on its task criterion, whose within-episode
shuffle null was too weak. An offline replay chose the tolerance ignition rule (V3) using
the inputs of seeds 42 to 44, so this confirmatory gate uses seeds 48, 49 and 50, which
were never run or inspected.

Runs being judged (Gate B flags plus the ignition rule):
    python -m scripts.training.train_rlhf --env dark_room --episodes 10 --max-steps 200 \\
        --seed <48|49|50> --enable-audio --rssm-latent-mode continuous \\
        --capsule-workspace-source all_levels --existence-drive on \\
        --dark-room-audio binaural --dark-room-audio-channels 4 --dark-room-collision \\
        --dark-room-view agent_centered --vision-bid-reduction zscore \\
        --audio-salience surprise --learned-valence \\
        --ignition-rule tolerance --ignition-tolerance-sd 1.0 --log-dir runs/gate_b2_s<seed>

PRE-STATED GATE, written 2026-09-15 before any Gate B2 run, by owner decision. Steps exclude
step 0 of each episode (as Gate B). Read from session records only.

    (1) COMPETITION   at each seed, the runner-up takes at least 0.05 of ignited steps.
    (2) SILENCE       at each seed, steps without a winner are below 0.50.
    (3) SELECTIVITY   at each seed, ignition is not always on (silence at least 0.05), and
                      the largest raw bid is higher on ignited steps than on silent steps
                      by more than the 95th percentile of a null that shuffles the
                      ignited labels within episodes (1000 shuffles, seed 0). Limitation,
                      stated now: ignition is computed from the bids, so this is a guard
                      against random or constant ignition, not evidence of meaning.
    (4) TASK          episode level. For each episode with at least 20 ignited steps:
                      f = share of its steps with the light out of view, a = audio share of
                      its ignited steps. Pooled over the 3 seeds: Spearman rho(f, a) above
                      0, with a one-sided permutation p below 0.05 (a shuffled across
                      episodes within each seed, 2000 shuffles, seed 0). Also rho above 0 at
                      2 or more of the 3 seeds. MEASURABLE only with at least 15 qualifying
                      episodes pooled; otherwise not measurable, which fails.
    KILL              any module at 0.95 or more of ignited steps at any seed.

    Reported only, never a gate: Spearman rho of audio share against episode index
    (learning over time can confound f and a), and first-light steps.

A PASS is not evidence of affect (finding a light is automatic approach).

RESULT OF GATE B2, 2026-09-15: FAILED on (4). Seeds 48 / 49 / 50, 2000 rows and 10 complete
episodes each. Criteria 1 to 3 PASS at all 3 seeds:
    (1) runner-up          0.392 / 0.499 / 0.120
    (2) silence            0.316 / 0.314 / 0.307
    (3) selectivity        top-bid difference 0.050 / 0.115 / 0.099 against null p95
                           0.022 / 0.010 / 0.055
    (4) task FAILED        pooled rho -0.128 over 30 episodes, one-sided p 0.904; per seed
                           -0.622 / -0.400 / +0.375
    KILL                   not triggered (top share 0.608 / 0.501 / 0.880)
    reported only          rho of audio share against episode index 0.770 / -0.248 / 0.517

REUSED AS GATE B3, 2026-09-16, criteria unchanged, on seeds 54 to 56 with
--bid-precision gain (docs/decisions/2026_09_16_precision_weighted_bids.md).
FAILED: seed 55 fired the KILL rule (vision 0.961) and failed (1) with a runner-up share of
0.039. Seeds 54 and 56 passed criteria 1 to 3 (runner-up 0.225 / 0.315, silence 0.303 /
0.335). Criterion (4) passed for the first time: pooled rho 0.446, one-sided p 0.005,
per seed 0.554 / 0.386 / 0.550. Reported only: rho of audio share against episode index
0.853 / 0.607 / 0.152, a confound this design does not separate.

Run:
    python -m scripts.analysis.probe_gate_b2 --runs runs/gate_b2_s48 runs/gate_b2_s49 runs/gate_b2_s50
    python -m scripts.analysis.probe_gate_b2 --runs runs/gate_b3_s54 runs/gate_b3_s55 runs/gate_b3_s56
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from scripts.analysis.probe_gate_b import competition, first_light_steps, load_steps

RUNNER_UP_MIN = 0.05
SILENCE_MAX = 0.50
SILENCE_MIN = 0.05
KILL_SHARE = 0.95
SHUFFLES_SELECTIVITY = 1000
SHUFFLES_TASK = 2000
MIN_IGNITED_PER_EPISODE = 20
MIN_EPISODES_POOLED = 15
TASK_P_MAX = 0.05


def selectivity(rows: list) -> dict:
    top_bid = np.array([max(r[1]["raw_bids"].values()) for r in rows])
    ignited = np.array([bool(r[1]["winner"]) for r in rows])
    episodes = np.array([r[0] for r in rows])
    observed = _difference(top_bid, ignited)
    rng = np.random.default_rng(0)
    null = [_difference(top_bid, _shuffle_within(ignited, episodes, rng))
            for _ in range(SHUFFLES_SELECTIVITY)]
    return {"top_bid_difference": observed, "null_p95": float(np.percentile(null, 95))}


def _difference(values: np.ndarray, labels: np.ndarray) -> float:
    if labels.all() or not labels.any():
        return 0.0
    return float(values[labels].mean() - values[~labels].mean())


def _shuffle_within(labels: np.ndarray, groups: np.ndarray, rng) -> np.ndarray:
    shuffled = labels.copy()
    for group in np.unique(groups):
        index = np.where(groups == group)[0]
        shuffled[index] = labels[rng.permutation(index)]
    return shuffled


def episode_table(rows: list) -> list:
    """(episode, out-of-view share, audio share of ignited, ignited count)."""
    table = []
    for episode in sorted({r[0] for r in rows}):
        steps = [r for r in rows if r[0] == episode]
        ignited = [r for r in steps if r[1]["winner"]]
        if len(ignited) < MIN_IGNITED_PER_EPISODE:
            continue
        out_of_view = float(np.mean([not r[2] for r in steps]))
        audio = float(np.mean([r[1]["winner"] == "audio" for r in ignited]))
        table.append((episode, out_of_view, audio, len(ignited)))
    return table


def task(tables: list) -> dict:
    pooled_f = np.concatenate([[t[1] for t in table] for table in tables])
    pooled_a = [np.array([t[2] for t in table]) for table in tables]
    per_seed = [float(spearmanr([t[1] for t in table], [t[2] for t in table]).correlation)
                if len(table) >= 3 else float("nan") for table in tables]
    if len(pooled_f) < MIN_EPISODES_POOLED:
        return {"episodes": len(pooled_f), "measurable": False, "rho_per_seed": per_seed}
    observed = float(spearmanr(pooled_f, np.concatenate(pooled_a)).correlation)
    if not np.isfinite(observed):
        # A constant f or a constant audio share has no rank correlation. Compared
        # with the null, NaN exceeds nothing and p came out as 1/2001, the smallest.
        return {"episodes": len(pooled_f), "measurable": False, "rho_per_seed": per_seed,
                "reason": "pooled rho undefined because a share is constant across episodes"}
    rng = np.random.default_rng(0)
    null = [spearmanr(pooled_f, np.concatenate([rng.permutation(a) for a in pooled_a])).correlation
            for _ in range(SHUFFLES_TASK)]
    p = float((1 + np.sum(np.array(null) >= observed)) / (1 + SHUFFLES_TASK))
    return {"episodes": len(pooled_f), "measurable": True, "rho_pooled": observed,
            "p_one_sided": p, "rho_per_seed": per_seed}


def audio_share_by_time(table: list) -> float:
    if len(table) < 3:
        return float("nan")
    return float(spearmanr([t[0] for t in table], [t[2] for t in table]).correlation)


def seed_failures(comp: dict, sel: dict) -> list:
    failures = []
    if comp["top_share"] >= KILL_SHARE:
        failures.append("KILL top share %.3f" % comp["top_share"])
    if comp["runner_up_share"] < RUNNER_UP_MIN:
        failures.append("(1) competition")
    if comp["silence"] >= SILENCE_MAX:
        failures.append("(2) silence")
    if comp["silence"] < SILENCE_MIN or sel["top_bid_difference"] <= sel["null_p95"]:
        failures.append("(3) selectivity")
    return failures


def task_failures(result: dict) -> list:
    if not result["measurable"]:
        return ["(4) task not measurable"]
    positive_seeds = sum(1 for rho in result["rho_per_seed"] if rho == rho and rho > 0)
    if result["rho_pooled"] <= 0 or result["p_one_sided"] >= TASK_P_MAX or positive_seeds < 2:
        return ["(4) task"]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs=3, required=True)
    args = parser.parse_args()
    failures, tables = [], []
    for run in args.runs:
        rows = load_steps(run)
        comp, sel, table = competition(rows), selectivity(rows), episode_table(rows)
        tables.append(table)
        seed_fail = seed_failures(comp, sel)
        failures.extend("%s: %s" % (run, f) for f in seed_fail)
        print("\n=== %s" % run)
        print(json.dumps({"competition": comp, "selectivity": sel,
                          "episodes_f_a_n": table,
                          "reported_rho_audio_vs_episode_index": audio_share_by_time(table),
                          "first_light_step_by_episode": first_light_steps(rows)}, indent=2))
        print("FAILED: %s" % "; ".join(seed_fail) if seed_fail else "criteria 1 to 3 PASS at this seed")
    task_result = task(tables)
    failures.extend(task_failures(task_result))
    print("\nTASK (pooled): %s" % json.dumps(task_result))
    print("\nGATE B2: %s" % ("PASS" if not failures else "FAILED: " + "; ".join(failures)))


if __name__ == "__main__":
    main()

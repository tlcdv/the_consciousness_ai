"""Gate B: do vision and hearing compete, and does hearing win when the light is out of view?

Runs being judged (all Step A and Step B flags on):
    python -m scripts.training.train_rlhf --env dark_room --episodes 10 --max-steps 200 \\
        --seed <42|43|44> --enable-audio --rssm-latent-mode continuous \\
        --capsule-workspace-source all_levels --existence-drive on \\
        --dark-room-audio binaural --dark-room-audio-channels 4 --dark-room-collision \\
        --dark-room-view agent_centered --vision-bid-reduction zscore \\
        --audio-salience surprise --learned-valence --log-dir runs/gate_b_s<seed>

PRE-STATED GATE, written 2026-09-15 before any Gate B run, by owner decision. Every
criterion must hold at each of the 3 seeds. Read from session records only.

    Steps: every recorded step of all 10 episodes. Step 0 of each episode is excluded
    (the frame it saw comes from reset, whose info is not recorded). "Ignited" means
    the step has a winner.

    (1) COMPETITION  among ignited steps, the second most frequent winner takes at
                     least 0.05 of them.
    (2) SILENCE      steps without a winner are below 0.50 of all steps.
    (3) TASK         the audio win share among ignited steps is higher when the light was
                     out of view than when it was in view (the view state of the frame the
                     agent saw at step t is `_truth_light_in_view` of step t-1), and the
                     difference exceeds the 95th percentile of a null built by shuffling
                     the view labels across contiguous 20-step blocks within episodes
                     (1000 shuffles, seed 0). MEASURABLE only if each view condition has
                     at least 50 ignited steps; otherwise not measurable, which fails.
    KILL             any single module wins 0.95 or more of ignited steps at any seed.

    Reported only, never a gate: the step at which each episode first reaches the light.

RESULT OF GATE B, 2026-09-15: FAILED at all 3 seeds (42 / 43 / 44), on (2) and (3).
    (1) competition PASS   runner-up audio 0.172 / 0.253 / 0.231 of ignited steps
                           (before Step B vision won 0.99 to 1.00)
    (2) silence FAILED     0.655 / 0.694 / 0.615; seed 43 episodes 8 and 9 never ignite
    (3) task FAILED        audio share out of view minus in view -0.160 / +0.428 / +0.209
                           against null p95 +0.027 / +0.428 / +0.209
    KILL                   not triggered (top share 0.828 / 0.747 / 0.769)
Gate-design weakness found post hoc: the view state is constant for a whole episode in
2 of 10, 4 of 8 and 5 of 10 ignited episodes, so shuffling 20-step blocks within
episodes leaves those labels unchanged; 1, 9 and 12 percent of null draws equal the
observed value. Criterion (3) cannot separate the view state from episode-level changes.
Post hoc pattern, not a result: audio wins in runs of whole episodes (seed 44 episodes
4 to 7: 0.81 to 0.95), not step by step.

A PASS is about competition and a task variable. It is NOT evidence of affect: Feinberg
and Mallatt exclude automatic approach and avoidance, and this task is one.

Run:
    python -m scripts.analysis.probe_gate_b --runs runs/gate_b_s42 runs/gate_b_s43 runs/gate_b_s44
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np

RUNNER_UP_MIN = 0.05
SILENCE_MAX = 0.50
KILL_SHARE = 0.95
BLOCK = 20
SHUFFLES = 1000
MIN_IGNITED_PER_CONDITION = 50


def load_steps(run: str) -> list:
    """(episode, step record, in_view of the frame seen) for every step except step 0."""
    rows = []
    for folder in sorted(glob.glob(os.path.join(run, "episodes", "ep_*"))):
        episode = int(folder.rsplit("_", 1)[1])
        steps = [json.loads(line) for line in open(os.path.join(folder, "steps.jsonl"), encoding="utf-8")]
        for previous, current in zip(steps, steps[1:]):
            in_view = bool(previous["internals"]["info_after_step"]["_truth_light_in_view"])
            rows.append((episode, current, in_view))
    return rows


def competition(rows: list) -> dict:
    winners = [r[1]["winner"] for r in rows]
    ignited = [w for w in winners if w]
    counts = collections.Counter(ignited)
    ranked = counts.most_common()
    return {
        "steps": len(winners),
        "silence": 1.0 - len(ignited) / max(len(winners), 1),
        "winner_shares": {name: n / max(len(ignited), 1) for name, n in ranked},
        "runner_up_share": ranked[1][1] / len(ignited) if len(ranked) > 1 else 0.0,
        "top_share": ranked[0][1] / len(ignited) if ranked else 0.0,
    }


def audio_share_difference(winners: list, in_view: np.ndarray) -> float:
    audio = np.array([w == "audio" for w in winners])
    return float(audio[~in_view].mean() - audio[in_view].mean())


def block_shuffled_labels(in_view: np.ndarray, episodes: np.ndarray, rng) -> np.ndarray:
    """Shuffle contiguous 20-step label blocks within each episode."""
    shuffled = in_view.copy()
    for episode in np.unique(episodes):
        index = np.where(episodes == episode)[0]
        blocks = [index[i:i + BLOCK] for i in range(0, len(index), BLOCK)]
        order = rng.permutation(len(blocks))
        labels = np.concatenate([in_view[blocks[k]] for k in order])[:len(index)]
        shuffled[index] = labels if len(labels) == len(index) else in_view[index]
    return shuffled


def task_variable(rows: list) -> dict:
    ignited = [r for r in rows if r[1]["winner"]]
    winners = [r[1]["winner"] for r in ignited]
    in_view = np.array([r[2] for r in ignited])
    episodes = np.array([r[0] for r in ignited])
    counts = {"ignited_in_view": int(in_view.sum()), "ignited_out_of_view": int((~in_view).sum())}
    if min(counts.values()) < MIN_IGNITED_PER_CONDITION:
        return dict(counts, measurable=False)
    observed = audio_share_difference(winners, in_view)
    rng = np.random.default_rng(0)
    null = [audio_share_difference(winners, block_shuffled_labels(in_view, episodes, rng))
            for _ in range(SHUFFLES)]
    return dict(counts, measurable=True, observed=observed,
                null_p95=float(np.percentile(null, 95)), null_mean=float(np.mean(null)))


def first_light_steps(rows: list) -> dict:
    first = {}
    for episode, step, _ in rows:
        info = step["internals"]["info_after_step"]
        if info["in_light"] and episode not in first:
            first[episode] = step["step"]
    return first


def verdict(comp: dict, task: dict) -> list:
    failures = []
    if comp["top_share"] >= KILL_SHARE:
        failures.append("KILL one module wins %.3f of ignited steps" % comp["top_share"])
    if comp["runner_up_share"] < RUNNER_UP_MIN:
        failures.append("(1) competition")
    if comp["silence"] >= SILENCE_MAX:
        failures.append("(2) silence")
    if not task["measurable"]:
        failures.append("(3) task not measurable")
    elif task["observed"] <= task["null_p95"]:
        failures.append("(3) task variable")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs=3, required=True)
    args = parser.parse_args()
    all_pass = True
    for run in args.runs:
        rows = load_steps(run)
        comp, task = competition(rows), task_variable(rows)
        failures = verdict(comp, task)
        all_pass = all_pass and not failures
        print("\n=== %s" % run)
        print(json.dumps({"competition": comp, "task": task,
                          "first_light_step_by_episode": first_light_steps(rows)}, indent=2))
        print("FAILED: %s" % "; ".join(failures) if failures else "PASS at this seed")
    print("\nGATE B: %s" % ("PASS at all 3 seeds" if all_pass else "FAILED"))


if __name__ == "__main__":
    main()

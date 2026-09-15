"""Gate A: do the repaired dark room senses carry information about the light?

Measured 2026-09-15 before this probe existed: on three seeds (1800 steps) the legacy
dark_room sound carried only noise (docs/results/modality_starvation_2026_09.md,
correction header). The repair adds a binaural tone from the light, a collision event,
and an agent-centered view. This probe checks the repair BEFORE any learning is built
on it.

PRE-STATED GATE, written 2026-09-15 before any Gate A run. Every criterion must pass at
each of the 3 seeds, or Step B does not start.

    (i)   the recorded direction estimate (audio_spatial) has the same sign as the true
          light offset on at least 90 percent of steps, separately for the left/right
          and upper/lower axes. Components smaller than 1 room pixel are excluded.
          Control, printed beside it: the same agreement against offsets shuffled across
          steps, expected near 0.5.
    (ii)  the per-step sound level (root mean square over all channels) correlates with
          the true distance at Spearman rho of -0.8 or stronger.
    (iii) the light is out of view on at least 50 percent of the first episode's steps.
    (iv)  at least one collision is reported.
    (v)   a hand-written controller that hears only the sound level (louder: keep the
          heading; quieter: pick a new random heading) reaches the light in a share of
          episodes at least 0.10 above a controller that picks a random heading every
          step. Same start positions for both, 50 episodes of 200 steps per seed, same
          environment settings as the session runs. No agent, no training.

Pairing: the sound heard at step t was generated from the environment state after step
t-1, so criteria (i) and (ii) compare step t's sound with step t-1's `info_after_step`.

Session records are read from disk only; nothing is trained.

RESULT OF GATE A, 2026-09-15: FAILED. Criterion (ii) at seed 42: rho -0.768. Criterion
(i) was not measurable: the untrained agent stayed at one wall (collisions on 90 to 98
percent of steps), the light's direction changed sign at most 4 times per seed, and the
shuffled control scored 0.79 to 1.00. Criterion (v) passed at all 3 seeds (1.00 against
0.36 to 0.52).

GATE A2, PRE-STATED 2026-09-15 by owner decision, BEFORE any A2 run. It tests the
environment, not the agent, so no agent drives it.

    Data: the random-heading controller moves through the environment, 10 episodes of
    200 steps per seed, seeds 42, 43, 44, same settings as the Gate A sessions. Each
    step's sound is paired with the state that generated it (the same step). The
    direction estimate is SpatialAudioComputer, the estimator the agent uses.
    (i)   sign agreement at least 0.90 per axis, as in Gate A. It is MEASURABLE only if
          the true direction changes sign at least 20 times on that axis and the
          shuffled control is at most 0.60; otherwise it is reported as not measurable,
          which fails the gate.
    (ii)  Spearman rho of sound level against distance at -0.8 or stronger, on steps
          without a collision (the collision sound is a separate event by design).
    (iii) light out of view on at least 50 percent of the first episode's steps.
    (iv)  at least one collision.
    (v)   unchanged from Gate A.
    Raw per-step arrays are saved to runs/senses_gate_a2_s<seed>/rollout.npz.

RESULT OF GATE A2, 2026-09-15: FAILED at seeds 43 and 44. Seed 43: left/right axis not
measurable (18 sign changes). Seed 44: left/right axis not measurable (shuffled control
0.68) and light out of view on 0.095 of the first episode. Where measurable, sign
agreement was 0.997 to 1.000; level against distance on no-collision steps was -0.995 to
-0.998; sound-only steering 1.00 against 0.36 to 0.52. Both failures trace to the gate
design: a shuffled control is not a valid chance level when one sign dominates (0.8
squared plus 0.2 squared is 0.68), and a single episode is one random start position.
Post hoc, and not a pass: balanced sign accuracy 0.997 to 1.000 on every axis and seed;
out of view over all episodes 0.569 / 0.876 / 0.704.

GATE A3, PRE-STATED 2026-09-15 by owner decision, BEFORE any A3 run. Corrects the two
gate-design errors found in A2. Seeds 45, 46, 47, never run or inspected before.

    Data: as Gate A2 (random-heading controller, 10 episodes of 200 steps per seed).
    (i)   BALANCED sign accuracy (mean of the accuracy on positive-sign steps and on
          negative-sign steps) at least 0.90 per axis. Chance is 0.5 whatever the
          imbalance. MEASURABLE only if the true sign changes at least 20 times on
          that axis, each sign has at least 50 usable steps, and the balanced
          accuracy against shuffled offsets is at most 0.60; otherwise not
          measurable, which fails the gate.
    (ii)  as Gate A2: rho of level against distance on no-collision steps, -0.8 or
          stronger.
    (iii) light out of view on at least 50 percent of ALL steps of the seed.
    (iv)  at least one collision.
    (v)   unchanged from Gate A, on seeds 45, 46, 47.
    Raw per-step arrays are saved to runs/senses_gate_a3_s<seed>/rollout.npz.

RESULT OF GATE A3, 2026-09-15: PASS at all 3 seeds (45 / 46 / 47).
    balanced sign accuracy x    0.9994 / 0.9990 / 0.9970   (shuffled 0.506 / 0.496 / 0.514)
    balanced sign accuracy y    0.9984 / 0.9985 / 0.9993   (shuffled 0.523 / 0.521 / 0.477)
    sign changes x, y           47, 111 / 41, 27 / 53, 60
    rho level vs distance       -0.998 / -0.994 / -0.997   (no-collision steps)
    out of view, all steps      0.560 / 0.729 / 0.644
    collisions                  213 / 149 / 134
    steering sound vs random    1.00 vs 0.58 / 1.00 vs 0.56 / 1.00 vs 0.42

Run:
    python -m scripts.analysis.probe_dark_room_senses \\
        --sessions runs/senses_gate_a_s42 runs/senses_gate_a_s43 runs/senses_gate_a_s44
    python -m scripts.analysis.probe_dark_room_senses --environment-only
    python -m scripts.analysis.probe_dark_room_senses --gate-a3 --seeds 45 46 47
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.audio.spatial_audio import SpatialAudioComputer  # noqa: E402
from simulations.environments.simple_visual_env import SimpleVisualEnv  # noqa: E402

SIGN_AGREEMENT_MIN = 0.90
RHO_MAX = -0.8
OUT_OF_VIEW_MIN = 0.50
STEERING_MARGIN_MIN = 0.10
MIN_COMPONENT_PIXELS = 1.0
A2_EPISODES = 10
A2_MAX_STEPS = 200
MIN_SIGN_CHANGES = 20
SHUFFLED_CONTROL_MAX = 0.60
MIN_STEPS_PER_SIGN = 50


def load_episode(folder: str):
    steps = [json.loads(line) for line in open(os.path.join(folder, "steps.jsonl"), encoding="utf-8")]
    vectors = np.load(os.path.join(folder, "vectors.npz"))
    spatial = dict(zip(vectors["audio_spatial__steps"].tolist(),
                       vectors["audio_spatial"].reshape(len(vectors["audio_spatial__steps"]), -1)))
    waves = dict(zip(vectors["audio_waveform__steps"].tolist(), vectors["audio_waveform"]))
    return steps, spatial, waves


def paired_rows(steps, spatial, waves):
    """(estimate [az, el], true offset [dx, dy], sound level, distance) per step t >= 1."""
    rows = []
    for step in steps[1:]:
        t = step["step"]
        previous = steps[t - 1]["internals"]["info_after_step"]
        if t not in spatial or t not in waves:
            continue
        level = float(np.sqrt(np.mean(np.square(waves[t]))))
        rows.append((spatial[t][:2], previous["_truth_light_offset"], level,
                     previous["distance_to_light"]))
    return rows


def sign_agreement(estimates: np.ndarray, offsets: np.ndarray, axis: int) -> float:
    usable = np.abs(offsets[:, axis]) >= MIN_COMPONENT_PIXELS
    return float(np.mean(np.sign(estimates[usable, axis]) == np.sign(offsets[usable, axis])))


def session_criteria(run: str) -> dict:
    """Criteria (i) to (iv) for one session record."""
    folders = sorted(glob.glob(os.path.join(run, "episodes", "ep_*")))
    rows, collisions, first_out_of_view = [], 0, None
    for index, folder in enumerate(folders):
        steps, spatial, waves = load_episode(folder)
        rows.extend(paired_rows(steps, spatial, waves))
        infos = [s["internals"]["info_after_step"] for s in steps]
        collisions += sum(bool(i.get("collision")) for i in infos)
        if index == 0:
            first_out_of_view = float(np.mean([not i["_truth_light_in_view"] for i in infos]))
    return _criteria_from_rows(rows, collisions, first_out_of_view)


def _criteria_from_rows(rows, collisions, first_out_of_view) -> dict:
    estimates = np.array([r[0] for r in rows], dtype=float)
    offsets = np.array([r[1] for r in rows], dtype=float)
    shuffled = offsets[np.random.default_rng(0).permutation(len(offsets))]
    return {
        "steps": len(rows),
        "sign_x": sign_agreement(estimates, offsets, 0),
        "sign_y": sign_agreement(estimates, offsets, 1),
        "sign_x_shuffled": sign_agreement(estimates, shuffled, 0),
        "sign_y_shuffled": sign_agreement(estimates, shuffled, 1),
        "rho_level_distance": float(spearmanr([r[2] for r in rows], [r[3] for r in rows]).correlation),
        "first_episode_out_of_view": first_out_of_view,
        "collisions": collisions,
    }


def gate_environment(episode_seed: int) -> SimpleVisualEnv:
    """The Gate A environment settings, reset to a start fixed by `episode_seed`."""
    np.random.seed(episode_seed)  # start positions come from the global generator
    env = SimpleVisualEnv(width=224, height=224, audio="binaural", audio_channels=4,
                          report_collision=True, view="agent_centered")
    env.seed_audio(episode_seed)
    env.reset()
    return env


def steering_success(policy: str, seed: int, episodes: int, max_steps: int) -> float:
    """Share of episodes in which the controller reaches the light."""
    rng = np.random.default_rng(seed)
    reached = 0
    for episode in range(episodes):
        reached += _run_controller(gate_environment(seed * 1000 + episode), policy, rng, max_steps)
    return reached / episodes


def _run_controller(env, policy: str, rng, max_steps: int) -> int:
    heading = _random_heading(rng)
    previous_level = None
    for _ in range(max_steps):
        _, _, terminated, _, info = env.step(heading)
        if info["in_light"]:
            return 1
        level = float(np.sqrt(np.mean(np.square(info["audio_waveform"]))))
        quieter = previous_level is not None and level < previous_level
        if policy == "random" or quieter:
            heading = _random_heading(rng)
        previous_level = level
        if terminated:
            break
    return 0


def _random_heading(rng) -> np.ndarray:
    angle = rng.uniform(0.0, 2.0 * np.pi)
    return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)


def rollout_arrays(seed: int) -> dict:
    """Gate A2 data: the random-heading controller moves; each step's sound is paired
    with the state that generated it."""
    estimator, rng = SpatialAudioComputer(), np.random.default_rng(seed)
    columns = {k: [] for k in ("estimate", "offset", "level", "distance",
                               "collision", "in_view", "episode")}
    for episode in range(A2_EPISODES):
        env = gate_environment(seed * 1000 + episode)
        for _ in range(A2_MAX_STEPS):
            _, _, terminated, _, info = env.step(_random_heading(rng))
            _append_step(columns, estimator, info, episode)
            if terminated:
                break
    return {k: np.asarray(v) for k, v in columns.items()}


def _append_step(columns: dict, estimator, info: dict, episode: int) -> None:
    wave = info["audio_waveform"]
    with torch.no_grad():
        estimate = estimator(torch.tensor(wave[None], dtype=torch.float32))[0].numpy()
    columns["estimate"].append(estimate)
    columns["offset"].append(info["_truth_light_offset"])
    columns["level"].append(float(np.sqrt(np.mean(np.square(wave)))))
    columns["distance"].append(info["distance_to_light"])
    columns["collision"].append(bool(info["collision"]))
    columns["in_view"].append(bool(info["_truth_light_in_view"]))
    columns["episode"].append(episode)


def a2_criteria(arrays: dict) -> dict:
    estimate, offset = arrays["estimate"].astype(float), arrays["offset"].astype(float)
    shuffled = offset[np.random.default_rng(0).permutation(len(offset))]
    quiet = ~arrays["collision"]
    first = arrays["episode"] == 0
    return {
        "steps": int(len(offset)),
        "sign_changes_x": int(np.sum(np.diff(np.sign(offset[:, 0])) != 0)),
        "sign_changes_y": int(np.sum(np.diff(np.sign(offset[:, 1])) != 0)),
        "sign_x": sign_agreement(estimate, offset, 0),
        "sign_y": sign_agreement(estimate, offset, 1),
        "sign_x_shuffled": sign_agreement(estimate, shuffled, 0),
        "sign_y_shuffled": sign_agreement(estimate, shuffled, 1),
        "steps_without_collision": int(quiet.sum()),
        "rho_level_distance_without_collision": float(
            spearmanr(arrays["level"][quiet], arrays["distance"][quiet]).correlation),
        "first_episode_out_of_view": float(np.mean(~arrays["in_view"][first])),
        "collisions": int(arrays["collision"].sum()),
    }


def a2_verdict(criteria: dict, steering: float, random_rate: float) -> list:
    failures = []
    for axis in ("x", "y"):
        measurable = (criteria["sign_changes_" + axis] >= MIN_SIGN_CHANGES
                      and criteria["sign_%s_shuffled" % axis] <= SHUFFLED_CONTROL_MAX)
        if not measurable:
            failures.append("(i) axis %s not measurable" % axis)
        elif criteria["sign_" + axis] < SIGN_AGREEMENT_MIN:
            failures.append("(i) axis %s sign agreement" % axis)
    if criteria["rho_level_distance_without_collision"] > RHO_MAX:
        failures.append("(ii) level against distance")
    if criteria["first_episode_out_of_view"] < OUT_OF_VIEW_MIN:
        failures.append("(iii) light out of view")
    if criteria["collisions"] < 1:
        failures.append("(iv) collisions")
    if steering - random_rate < STEERING_MARGIN_MIN:
        failures.append("(v) sound-only steering")
    return failures


def run_gate_a2(seeds, steering_episodes: int, steering_max_steps: int) -> bool:
    all_pass = True
    for seed in seeds:
        arrays = rollout_arrays(seed)
        out_dir = Path("runs") / ("senses_gate_a2_s%d" % seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "rollout.npz", **arrays)
        criteria = a2_criteria(arrays)
        steering = steering_success("sound", seed, steering_episodes, steering_max_steps)
        random_rate = steering_success("random", seed, steering_episodes, steering_max_steps)
        failures = a2_verdict(criteria, steering, random_rate)
        all_pass = all_pass and not failures
        print("\n=== Gate A2, seed %d" % seed)
        print(json.dumps(criteria, indent=2))
        print("steering: sound-only %.2f  random %.2f  margin %.2f" % (
            steering, random_rate, steering - random_rate))
        print("FAILED: %s" % ", ".join(failures) if failures else "PASS at this seed")
    return all_pass


def balanced_sign_accuracy(estimates: np.ndarray, offsets: np.ndarray, axis: int):
    """(balanced accuracy, smaller sign count). Chance is 0.5 whatever the imbalance."""
    usable = np.abs(offsets[:, axis]) >= MIN_COMPONENT_PIXELS
    truth = np.sign(offsets[usable, axis])
    guess = np.sign(estimates[usable, axis])
    positive, negative = truth > 0, truth < 0
    if not positive.any() or not negative.any():
        return float("nan"), 0
    accuracy = 0.5 * (np.mean(guess[positive] > 0) + np.mean(guess[negative] < 0))
    return float(accuracy), int(min(positive.sum(), negative.sum()))


def a3_criteria(arrays: dict) -> dict:
    estimate, offset = arrays["estimate"].astype(float), arrays["offset"].astype(float)
    shuffled = offset[np.random.default_rng(0).permutation(len(offset))]
    quiet = ~arrays["collision"]
    criteria = {"steps": int(len(offset)), "collisions": int(arrays["collision"].sum()),
                "out_of_view_all_steps": float(np.mean(~arrays["in_view"])),
                "rho_level_distance_without_collision": float(
                    spearmanr(arrays["level"][quiet], arrays["distance"][quiet]).correlation)}
    for axis, name in ((0, "x"), (1, "y")):
        accuracy, smaller = balanced_sign_accuracy(estimate, offset, axis)
        criteria["balanced_" + name] = accuracy
        criteria["balanced_%s_shuffled" % name] = balanced_sign_accuracy(estimate, shuffled, axis)[0]
        criteria["smaller_sign_steps_" + name] = smaller
        criteria["sign_changes_" + name] = int(np.sum(np.diff(np.sign(offset[:, axis])) != 0))
    return criteria


def a3_verdict(criteria: dict, steering: float, random_rate: float) -> list:
    failures = []
    for axis in ("x", "y"):
        measurable = (criteria["sign_changes_" + axis] >= MIN_SIGN_CHANGES
                      and criteria["smaller_sign_steps_" + axis] >= MIN_STEPS_PER_SIGN
                      and criteria["balanced_%s_shuffled" % axis] <= SHUFFLED_CONTROL_MAX)
        if not measurable:
            failures.append("(i) axis %s not measurable" % axis)
        elif criteria["balanced_" + axis] < SIGN_AGREEMENT_MIN:
            failures.append("(i) axis %s balanced accuracy" % axis)
    if criteria["rho_level_distance_without_collision"] > RHO_MAX:
        failures.append("(ii) level against distance")
    if criteria["out_of_view_all_steps"] < OUT_OF_VIEW_MIN:
        failures.append("(iii) light out of view")
    if criteria["collisions"] < 1:
        failures.append("(iv) collisions")
    if steering - random_rate < STEERING_MARGIN_MIN:
        failures.append("(v) sound-only steering")
    return failures


def run_environment_gate(label: str, seeds, criteria_fn, verdict_fn, steering_episodes: int,
                         steering_max_steps: int) -> bool:
    all_pass = True
    for seed in seeds:
        arrays = rollout_arrays(seed)
        out_dir = Path("runs") / ("senses_gate_%s_s%d" % (label, seed))
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "rollout.npz", **arrays)
        criteria = criteria_fn(arrays)
        steering = steering_success("sound", seed, steering_episodes, steering_max_steps)
        random_rate = steering_success("random", seed, steering_episodes, steering_max_steps)
        failures = verdict_fn(criteria, steering, random_rate)
        all_pass = all_pass and not failures
        print("\n=== Gate %s, seed %d" % (label.upper(), seed))
        print(json.dumps(criteria, indent=2))
        print("steering: sound-only %.2f  random %.2f  margin %.2f" % (
            steering, random_rate, steering - random_rate))
        print("FAILED: %s" % ", ".join(failures) if failures else "PASS at this seed")
    return all_pass


def verdict(criteria: dict, steering: float, random_rate: float) -> list:
    failures = []
    if min(criteria["sign_x"], criteria["sign_y"]) < SIGN_AGREEMENT_MIN:
        failures.append("(i) direction sign agreement")
    if criteria["rho_level_distance"] > RHO_MAX:
        failures.append("(ii) level against distance")
    if criteria["first_episode_out_of_view"] < OUT_OF_VIEW_MIN:
        failures.append("(iii) light out of view")
    if criteria["collisions"] < 1:
        failures.append("(iv) collisions")
    if steering - random_rate < STEERING_MARGIN_MIN:
        failures.append("(v) sound-only steering")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gate-a3", action="store_true",
                        help="Run Gate A3 (corrected statistics, environment rollouts).")
    parser.add_argument("--environment-only", action="store_true",
                        help="Run Gate A2 (environment rollouts, no agent) instead of Gate A.")
    parser.add_argument("--sessions", nargs=3, default=None,
                        help="Three session record folders, seeds 42, 43, 44 in that order.")
    parser.add_argument("--seeds", nargs=3, type=int, default=[42, 43, 44])
    parser.add_argument("--steering-episodes", type=int, default=50)
    parser.add_argument("--steering-max-steps", type=int, default=200)
    args = parser.parse_args()
    if args.gate_a3:
        passed = run_environment_gate("a3", args.seeds, a3_criteria, a3_verdict,
                                      args.steering_episodes, args.steering_max_steps)
        print("\nGATE A3: %s" % ("PASS at all 3 seeds" if passed else "FAILED"))
        return
    if args.environment_only:
        passed = run_gate_a2(args.seeds, args.steering_episodes, args.steering_max_steps)
        print("\nGATE A2: %s" % ("PASS at all 3 seeds" if passed else "FAILED"))
        return
    if args.sessions is None:
        parser.error("--sessions is required unless --environment-only is given")
    all_pass = True
    for run, seed in zip(args.sessions, args.seeds):
        criteria = session_criteria(run)
        steering = steering_success("sound", seed, args.steering_episodes, args.steering_max_steps)
        random_rate = steering_success("random", seed, args.steering_episodes, args.steering_max_steps)
        failures = verdict(criteria, steering, random_rate)
        all_pass = all_pass and not failures
        print("\n=== seed %d  %s" % (seed, run))
        print(json.dumps(criteria, indent=2))
        print("steering: sound-only %.2f  random %.2f  margin %.2f" % (
            steering, random_rate, steering - random_rate))
        print("FAILED: %s" % ", ".join(failures) if failures else "PASS at this seed")
    print("\nGATE A: %s" % ("PASS at all 3 seeds" if all_pass else "FAILED"))


if __name__ == "__main__":
    main()

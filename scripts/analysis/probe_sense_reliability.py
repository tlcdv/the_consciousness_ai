"""Feasibility check: can "how well a sense localizes something now" track the light's view state?

Gate B2 (docs/results/dark_room_senses_2026_09.md) FAILED on the task criterion, and the
bids moved against it: the audio bid was higher with the light in view at all 3 seeds,
because surprise measures change, not usefulness. The multisensory literature weights a
cue by its reliability: blurred, poorly localized vision lets sound dominate (Alais and
Burr 2004); cues combine in proportion to inverse variance (Ernst and Banks 2002);
MSTd activity predicts reliability weights (Fetsch et al. 2012). This probe checks, from
records only, whether two task-agnostic localization-reliability measures separate the
light in view from out of view BEFORE any model code is built.

MEASURES, defined 2026-09-15 before computing any value.
    Vision r_v  from the recorded frame the agent saw: grayscale, pooled to the tectum's
                16x16 grid (14x14 pixel blocks), minus a running mean of the pooled frames
                (factor 0.95, read before update, reset each episode); energy per cell is
                the squared deviation; r_v = 1 - H(p) / log(256), where p is the energy
                share per cell. A localized change gives r_v near 1, a spread or absent
                change near 0. The running mean removes the agent's own disc, which is
                always at the centre. (A model implementation would use the tectum's visual
                grid before audio fusion; the recorded obs_map is already fused with audio.)
    Audio r_a   from the recorded 4-channel waveform: for each ear pair, the maximum over
                lags within +/-9 samples of the normalized cross-correlation between the
                two ears; r_a = mean over the 2 pairs, clipped to [0, 1]. A coherent tone
                gives r_a near 1; independent noise near 0.
    Weight w_a  = r_a / (r_a + r_v), the audio share of an inverse-noise weighting.

PRE-STATED GATE, 2026-09-15, by owner decision. Data: Gate B2 session records, seeds 48,
49, 50, all 10 episodes. The view state of the frame and sound at step t is
`_truth_light_in_view` of step t-1. Steps 0 to 20 of each episode are excluded (running
mean warm-up).
    (R1) AUC of r_v for "light in view" (higher r_v = in view) at least 0.70 at each seed.
    (R2) AUC of w_a for "light out of view" (higher w_a = out of view) at least 0.70 at
         each seed.
    Reported only: AUC of r_a alone, and both AUCs against shuffled view labels.
PASS on R1 and R2 at all 3 seeds allows building reliability-weighted bids. FAIL means no
build. These seeds were used for Gate B2, but not for choosing these measures.

RESULT, 2026-09-15: FAILED at seeds 48 and 50; nothing is built.
    (R1) AUC r_v in view        0.618 / 0.990 / 0.554   (shuffled 0.502 / 0.494 / 0.499)
    (R2) AUC w_a out of view    0.512 / 0.987 / 0.528   (shuffled 0.498 / 0.493 / 0.498)
    reported AUC r_a out of view 0.101 / 0.673 / 0.300  (audio coherence is higher in view)
Post hoc hypothesis, not a result: the running mean that removes the agent's own disc also
removes a light that stays in view, so a stationary light stops counting as localized.
Seeds 48 and 50 have 374 and 449 in-view steps, seed 49 has 184. A revised measure must be
pre-stated and tested on new seeds, never re-scored on these.

Run:
    python -m scripts.analysis.probe_sense_reliability --runs runs/gate_b2_s48 runs/gate_b2_s49 runs/gate_b2_s50
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

GRID = 16
BLOCK = 14
MEAN_FACTOR = 0.95
WARMUP_STEPS = 20
MAX_LAG = 9
AUC_MIN = 0.70


def pooled_gray(frame: np.ndarray) -> np.ndarray:
    gray = frame.astype(np.float64).mean(axis=2)
    return gray.reshape(GRID, BLOCK, GRID, BLOCK).mean(axis=(1, 3))


def concentration(energy: np.ndarray) -> float:
    total = energy.sum()
    if total <= 0:
        return 0.0
    p = energy.ravel() / total
    nonzero = p[p > 0]
    return float(1.0 - (-(nonzero * np.log(nonzero)).sum()) / np.log(GRID * GRID))


def vision_reliability(frames: np.ndarray) -> list:
    """r_v per step; the running mean is read before it is updated."""
    values, mean = [], None
    for frame in frames:
        pooled = pooled_gray(frame)
        values.append(0.0 if mean is None else concentration((pooled - mean) ** 2))
        mean = pooled if mean is None else MEAN_FACTOR * mean + (1 - MEAN_FACTOR) * pooled
    return values


def pair_coherence(first: np.ndarray, second: np.ndarray) -> float:
    norm = np.sqrt(np.sum(first ** 2) * np.sum(second ** 2))
    if norm <= 0:
        return 0.0
    corr = np.correlate(first, second, mode="full") / norm
    mid = len(corr) // 2
    return float(np.max(corr[mid - MAX_LAG: mid + MAX_LAG + 1]))


def audio_reliability(wave: np.ndarray) -> float:
    pairs = [(0, 1), (2, 3)] if wave.shape[0] >= 4 else [(0, 1)]
    return float(np.clip(np.mean([pair_coherence(wave[a], wave[b]) for a, b in pairs]), 0.0, 1.0))


def episode_rows(folder: str) -> list:
    steps = [json.loads(line) for line in open(os.path.join(folder, "steps.jsonl"), encoding="utf-8")]
    frames = np.load(os.path.join(folder, "frames.npz"))["frames"]
    vectors = np.load(os.path.join(folder, "vectors.npz"))
    waves = dict(zip(vectors["audio_waveform__steps"].tolist(), vectors["audio_waveform"]))
    r_v = vision_reliability(frames)
    rows = []
    for t in range(WARMUP_STEPS + 1, len(steps)):
        if t not in waves:
            continue
        in_view = bool(steps[t - 1]["internals"]["info_after_step"]["_truth_light_in_view"])
        rows.append((in_view, r_v[t], audio_reliability(waves[t])))
    return rows


def seed_result(run: str) -> dict:
    rows = [row for folder in sorted(glob.glob(os.path.join(run, "episodes", "ep_*")))
            for row in episode_rows(folder)]
    in_view = np.array([r[0] for r in rows])
    r_v = np.array([r[1] for r in rows])
    r_a = np.array([r[2] for r in rows])
    w_a = r_a / np.maximum(r_a + r_v, 1e-12)
    shuffled = np.random.default_rng(0).permutation(in_view)
    return {
        "steps": len(rows), "in_view_steps": int(in_view.sum()),
        "auc_r_v_in_view": float(roc_auc_score(in_view, r_v)),
        "auc_w_a_out_of_view": float(roc_auc_score(~in_view, w_a)),
        "reported_auc_r_a_out_of_view": float(roc_auc_score(~in_view, r_a)),
        "shuffled_auc_r_v": float(roc_auc_score(shuffled, r_v)),
        "shuffled_auc_w_a": float(roc_auc_score(~shuffled, w_a)),
        "median_r_v_in_out": [float(np.median(r_v[in_view])), float(np.median(r_v[~in_view]))],
        "median_r_a_in_out": [float(np.median(r_a[in_view])), float(np.median(r_a[~in_view]))],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs=3, required=True)
    args = parser.parse_args()
    passed = True
    for run in args.runs:
        result = seed_result(run)
        failures = []
        if result["auc_r_v_in_view"] < AUC_MIN:
            failures.append("(R1)")
        if result["auc_w_a_out_of_view"] < AUC_MIN:
            failures.append("(R2)")
        passed = passed and not failures
        print("\n=== %s\n%s" % (run, json.dumps(result, indent=2)))
        print("FAILED: %s" % " ".join(failures) if failures else "PASS at this seed")
    print("\nRELIABILITY CHECK: %s" % ("PASS at all 3 seeds" if passed else "FAILED"))


if __name__ == "__main__":
    main()

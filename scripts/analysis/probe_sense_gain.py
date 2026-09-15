"""Feasibility check 2: does the GAIN of current activity track which sense locates the light?

The first check (probe_sense_reliability.py) FAILED at seeds 48 and 50. Its vision measure
was the concentration of CHANGE after a running mean, which is a salience quantity: it
falls when an input stays constant. The sources separate two quantities. Salience
(prediction error) falls for a constant stimulus, by habituation; reliability (precision)
stays high (Feldman and Friston 2010; Stein and Stanford 2008). A population shows
reliability by the gain of its current activity, not by its change (Ma, Beck, Latham and
Pouget 2006), and cue weights follow reliability within each trial, read from the current
input, with no learning (Fetsch et al. 2009, 2012). This probe checks gain measures with no
running mean, from records only, before any model code is built.

MEASURES, defined 2026-09-15 before the runs below existed.
    Vision g_v  from the recorded frame: grayscale, pooled to the tectum's 16x16 grid
                (14x14 pixel blocks); cells whose centre lies within 56 pixels of the frame
                centre are masked (the agent's own disc, radius about 47 pixels in the
                agent-centered view, is always there); g_v = the maximum unmasked cell /
                255. No running mean, no memory: the strongest visual response outside the
                self, now.
    Audio g_a   from the recorded 4-channel waveform: RMS over all channels and samples,
                divided by the RMS of the tone at the source (0.5 / sqrt 2), clipped to
                [0, 1]. No running mean.
    Weight w_a  = g_a / (g_a + g_v), the audio share (divisive normalization with a zero
                constant, Ohshiro, Angelaki and DeAngelis 2011).

PRE-STATED GATE, 2026-09-15, by owner decision. Data: new seeds 51, 52, 53, never run
before, with the Gate B2 flags:
    python -m scripts.training.train_rlhf --env dark_room --episodes 10 --max-steps 200 \\
        --seed <51|52|53> --enable-audio --rssm-latent-mode continuous \\
        --capsule-workspace-source all_levels --existence-drive on \\
        --dark-room-audio binaural --dark-room-audio-channels 4 --dark-room-collision \\
        --dark-room-view agent_centered --vision-bid-reduction zscore \\
        --audio-salience surprise --learned-valence \\
        --ignition-rule tolerance --ignition-tolerance-sd 1.0 --log-dir runs/gain_check_s<seed>
The view state of the frame and sound at step t is `_truth_light_in_view` of step t-1, as
in the first check. Steps 0 to 20 of each episode are excluded, the same step set as the
first check.
    (R1) AUC of g_v for "light in view" at least 0.70 at each seed.
    (R2) AUC of w_a for "light out of view" at least 0.70 at each seed.
    (R3) No habituation. An in-view run is a maximal block of consecutive included steps
         with the light in view, at least 20 steps long. For each run, ratio = median g_v
         of its steps 10 to 19 / median g_v of its steps 0 to 9 (runs with a zero
         denominator are excluded and counted). The median ratio over runs is at least
         0.80 at each seed. Fewer than 3 usable runs at a seed is FAILED (not measurable).
    Reported only: AUC of g_a alone; AUCs against shuffled view labels; the first check's
    r_v and w_a AUCs on these seeds; the R3 ratio of the first check's r_v, which tests the
    earlier post hoc hypothesis that a light staying in view stops counting (that
    hypothesis predicts a ratio below 0.80).
PASS on R1, R2 and R3 at all 3 seeds allows building precision-weighted bids behind a
default-off flag, judged later by a live gate on further new seeds. FAIL means no build.

Written before the runs, as a limit of this check: in this room the light (gray 237) is
brighter than the walls (60) and the agent is masked, so R1 can pass by the construction of
the environment. R1 alone is weak evidence; R2 and R3 carry the test.

RESULT, 2026-09-15: FAILED on R3 at seed 51 (1 usable in-view run, 3 required: not
measurable). Nothing is built.
                                   seed 51   seed 52   seed 53
    in-view steps (of 1790)            181       912       317
    (R1) AUC g_v in view             0.997     0.981     0.904   PASS
    (R2) AUC w_a out of view         0.996     0.961     0.857   PASS
    (R3) usable runs                     1         7         5   FAILED at 51
    (R3) median ratio g_v            1.000     1.000     1.000
    reported: AUC g_a out of view    0.003     0.020     0.089   (sound is louder in view)
    reported: shuffled AUC g_v       0.512     0.515     0.506
    reported: shuffled AUC w_a       0.491     0.514     0.514
    reported: first check r_v AUC    0.865     0.559     0.677
    reported: first check w_a AUC    0.785     0.465     0.613
    reported: first check r_v ratio  1.000     0.996     0.991
    median g_v in / out of view: 0.928 / 0.235, 0.928 / 0.235, 0.729 / 0.235
Post hoc, not results: the earlier hypothesis (a light that stays in view stops counting
for the first check's r_v) is not supported where measurable, since that ratio is 0.99 to
1.00. The g_v ratio of 1.000 is also expected by construction: a light that fills a cell
gives the same maximum every step. The median out-of-view g_v equals the wall gray (60/255),
so the agent sees a wall on most out-of-view steps.

Run:
    python -m scripts.analysis.probe_sense_gain --runs runs/gain_check_s51 runs/gain_check_s52 runs/gain_check_s53
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.analysis.probe_sense_reliability import (
    audio_reliability, pooled_gray, vision_reliability,
)

GRID = 16
BLOCK = 14
SELF_MASK_PX = 56
TONE_RMS_AT_SOURCE = 0.5 / np.sqrt(2.0)
WARMUP_STEPS = 20
AUC_MIN = 0.70
RUN_MIN_STEPS = 20
HALF = 10
RATIO_MIN = 0.80
RUNS_MIN = 3


def _self_mask() -> np.ndarray:
    centres = BLOCK / 2.0 + BLOCK * np.arange(GRID)
    x, y = np.meshgrid(centres, centres)
    return np.hypot(x - GRID * BLOCK / 2.0, y - GRID * BLOCK / 2.0) < SELF_MASK_PX


SELF_MASK = _self_mask()


def vision_gain(frame: np.ndarray) -> float:
    """The strongest pooled response outside the agent's own disc, in [0, 1]."""
    return float(pooled_gray(frame)[~SELF_MASK].max() / 255.0)


def audio_gain(wave: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.asarray(wave, dtype=np.float64) ** 2))
    return float(min(1.0, rms / TONE_RMS_AT_SOURCE))


def audio_share(g_a: np.ndarray, g_v: np.ndarray) -> np.ndarray:
    return g_a / np.maximum(g_a + g_v, 1e-12)


def in_view_runs(steps: np.ndarray, in_view: np.ndarray) -> list:
    """Index lists of maximal blocks of consecutive steps with the light in view."""
    runs, current = [], []
    for index, (step, visible) in enumerate(zip(steps, in_view)):
        contiguous = bool(current) and step == steps[current[-1]] + 1
        if visible and (contiguous or not current):
            current.append(index)
            continue
        if len(current) >= RUN_MIN_STEPS:
            runs.append(current)
        current = [index] if visible else []
    if len(current) >= RUN_MIN_STEPS:
        runs.append(current)
    return runs


def habituation_ratios(measure: np.ndarray, runs: list) -> tuple:
    """(ratios of usable runs, count excluded for a zero denominator)."""
    ratios, excluded = [], 0
    for run in runs:
        early = np.median(measure[run[:HALF]])
        late = np.median(measure[run[HALF:2 * HALF]])
        if early <= 0:
            excluded += 1
            continue
        ratios.append(float(late / early))
    return ratios, excluded


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
        rows.append((t, in_view, vision_gain(frames[t]), audio_gain(waves[t]),
                     r_v[t], audio_reliability(waves[t])))
    return rows


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def _median_or_none(ratios: list):
    return float(np.median(ratios)) if ratios else None


def seed_result(run: str) -> dict:
    per_episode = [np.array(episode_rows(folder), dtype=object)
                   for folder in sorted(glob.glob(os.path.join(run, "episodes", "ep_*")))]
    runs_g, runs_r = [], []
    for rows in per_episode:
        steps, in_view = rows[:, 0].astype(int), rows[:, 1].astype(bool)
        blocks = in_view_runs(steps, in_view)
        runs_g.append(habituation_ratios(rows[:, 2].astype(float), blocks))
        runs_r.append(habituation_ratios(rows[:, 4].astype(float), blocks))
    rows = np.concatenate(per_episode)
    in_view = rows[:, 1].astype(bool)
    g_v, g_a = rows[:, 2].astype(float), rows[:, 3].astype(float)
    r_v, r_a = rows[:, 4].astype(float), rows[:, 5].astype(float)
    w_a = audio_share(g_a, g_v)
    shuffled = np.random.default_rng(0).permutation(in_view)
    ratios_g = [ratio for ratios, _ in runs_g for ratio in ratios]
    ratios_r = [ratio for ratios, _ in runs_r for ratio in ratios]
    return {
        "steps": len(rows), "in_view_steps": int(in_view.sum()),
        "auc_g_v_in_view": _auc(in_view, g_v),
        "auc_w_a_out_of_view": _auc(~in_view, w_a),
        "r3_usable_runs": len(ratios_g),
        "r3_excluded_zero_denominator": int(sum(excluded for _, excluded in runs_g)),
        "r3_median_ratio_g_v": _median_or_none(ratios_g),
        "reported_auc_g_a_out_of_view": _auc(~in_view, g_a),
        "reported_shuffled_auc_g_v": _auc(shuffled, g_v),
        "reported_shuffled_auc_w_a": _auc(~shuffled, w_a),
        "reported_first_check_auc_r_v_in_view": _auc(in_view, r_v),
        "reported_first_check_auc_w_a_out_of_view": _auc(~in_view, audio_share(r_a, r_v)),
        "reported_first_check_r3_median_ratio_r_v": _median_or_none(ratios_r),
        "median_g_v_in_out": [float(np.median(g_v[in_view])), float(np.median(g_v[~in_view]))],
        "median_g_a_in_out": [float(np.median(g_a[in_view])), float(np.median(g_a[~in_view]))],
    }


def gate_failures(result: dict) -> list:
    failures = []
    if result["auc_g_v_in_view"] < AUC_MIN:
        failures.append("(R1)")
    if result["auc_w_a_out_of_view"] < AUC_MIN:
        failures.append("(R2)")
    ratio = result["r3_median_ratio_g_v"]
    if result["r3_usable_runs"] < RUNS_MIN or ratio is None or ratio < RATIO_MIN:
        failures.append("(R3)")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs=3, required=True)
    args = parser.parse_args()
    passed = True
    for run in args.runs:
        result = seed_result(run)
        failures = gate_failures(result)
        passed = passed and not failures
        print("\n=== %s\n%s" % (run, json.dumps(result, indent=2)))
        print("FAILED: %s" % " ".join(failures) if failures else "PASS at this seed")
    print("\nGAIN CHECK: %s" % ("PASS at all 3 seeds" if passed else "FAILED"))


if __name__ == "__main__":
    main()

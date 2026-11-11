"""Does the broadcast reach the action, and does the task link of the dark room gates
survive when the episode index is held or removed?

Read-only, on saved runs. No training and no weight change. Result:
docs/results/workspace_action_2026_11.md.

RULES FOR CHECKS A TO C, written 2026-11-11 before the first run of this probe.

Check A, action path per run. From session.json argv: --tectal-orienting, --policy,
    --policy-input, --broadcast-merge. A description, no rule.

Check B, policy replay (method of docs/results/action_core_relay_defect_2026_10.md).
    For each gate run with saved weights of its last recorded episode, the policy is
    rebuilt from those weights and fed the recorded policy_state of that episode.
    ratio = (largest per-axis spread of the replayed action mean over the steps)
            / (smallest per-axis spread of recorded action minus replayed mean).
    "The policy input reaches the policy output" is SUPPORTED at a run when ratio >= 0.5,
    EXCLUDED when ratio <= 0.1, UNDECIDED between. Control K1: the largest absolute
    difference of recorded policy_state and recorded broadcast must be 0, otherwise the
    replay input is not the broadcast and the run is VOID. Control K2: the replayed mean
    for an all-zero input is printed beside tanh(relay bias).
    Known limits, stated now: the weights are those of the end of the episode, and the
    working memory of the policy starts at zero in the replay, while the live loop never
    resets it.

Check C, the task link against the episode index. f and a are the gate's own values
    (probe_gate_b2.episode_table). Control K3: the pooled rho and the index rho of Gate B4
    must reproduce the document (0.458; -0.194 / 0.512 / 0.815), or the check is VOID.
    C1 partial link. Inside each seed, f, a and the episode index are ranked, and the
        ranks of f and a are replaced by their residuals of a straight line on the index
        rank. Statistic: Pearson r of the pooled residuals. Null: residuals of a shuffled
        inside each seed, 5000 shuffles, seed 0, one-sided.
    C2 inside one episode. For each episode with at least 10 ignited steps in view and 10
        out of view: d = audio share out of view minus audio share in view. Statistic:
        mean d over the episodes of a gate. Null: the in-view labels of each episode are
        rotated by a random offset, 2000 times, seed 0, one-sided. The index is constant
        inside an episode.
    C3 same index, other layout. Gate B4 seeds 57 to 59 against the same seeds with the
        layouts moved by one episode. For each seed and episode present in both:
        df and da. Statistic: Spearman rho(df, da). Null: da shuffled inside each seed,
        5000 shuffles, one-sided.
    Reading. The index explanation is EXCLUDED for a gate when C1 and C2 both have p below
    0.05 with a positive statistic. It is SUPPORTED when both have p of 0.05 or more. One
    of each is UNDECIDED. C3 is one small set and is reported only.

RULES FOR CHECK D, written 2026-11-11 after checks A to C were read and before the first
run of check D.

Check D, the gain weighting. Hypothesis: with --bid-precision gain the task link is the
    arithmetic of the weighting in the same step. The gains are not recorded, so they are
    computed again from the recorded frame and waveform with the functions of
    models/core/precision_weighting.py. Only the episodes with recorded frames are used
    (the first and the last of each run). Steps 1 to 199, ignited steps only.
    Weighted order: the larger of the recorded raw bids of vision and audio.
    Unweighted order: each recorded bid divided by twice its share, then clipped to
    [0, 1]. A step is RECOVERABLE when its share is above 0 and the recorded bid is below
    1, or the recorded bid is 1 and the share is 0.5 or less (the clipped value is then
    known to be 1). Other steps are left out of both orders.
    Control K4: over the ignited recoverable steps of a gate, the weighted order names
    the recorded winner at 0.9 or more of the steps. Otherwise the raw order does not
    stand for the winner and the gate is VOID.
    Control K5: recoverable steps are 0.8 or more of the ignited steps, or the gate is VOID.
    Statistic, for both orders: d of check C2 with at least 10 steps in each view state,
    mean over the episodes of a gate, same rotation null.
    Reading. SUPPORTED for a gate when d weighted is above 0 with p below 0.05 and d
    unweighted has p of 0.05 or more. EXCLUDED when d unweighted is above 0 with p below
    0.05 and is at least half of d weighted. UNDECIDED otherwise, and NOT MEASURABLE when
    d weighted itself has p of 0.05 or more or fewer than 5 episodes qualify.
    Limits, stated now. The ignition is the recorded one. The weights learned with the
    weighting in place, so this removes the weighting from one step and not from the
    history. Bound bids, the learned valence and the settle cycles are not replayed.

RULES FOR CHECK E, written 2026-11-11 after check B was read and before the first run of
check E.

Check E, the action before the relay layer. Question: if the relay layer is taken out of
    the path, does the policy output depend on the broadcast? Same replay as check B. The
    value read is go * (1 - no-go) * (1 - STN brake), the input of the relay layer, with
    dopamine at 0 as in the live call.
    ratio = (largest per-axis spread of that value over the steps)
            / (smallest per-axis spread of recorded action minus the replayed action mean).
    The denominator is the exploration noise of check B, because the same noise would be
    added to this value. SUPPORTED at a run when ratio >= 0.5, EXCLUDED when ratio <= 0.1,
    UNDECIDED between. Control K1 of check B applies. Over the runs: the reading "the relay
    layer alone is the cut" is SUPPORTED when 9 or more of the 12 gate runs are SUPPORTED,
    EXCLUDED when 9 or more are EXCLUDED, UNDECIDED otherwise.
    Reported only: the spread of the go signal alone, the means of no-go and of the brake,
    and the same value for an input of zeros.
    Limits, stated now. The weights learned with the relay layer in place and with a loss
    that does not read it, so this shows what the trained pathways hold today. It does not
    show what they would learn with the layer removed.

Run:
    python -m scripts.analysis.probe_workspace_action --runs-root runs --checks a b c d e
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from models.core.precision_weighting import audio_gain, precision_shares, vision_gain
from models.self_model.action_selection_core import BasalGanglia, PrefrontalCortex
from scripts.analysis.probe_gate_b import load_steps
from scripts.analysis.probe_gate_b2 import episode_table

GATES = {
    "B2": ["gate_b2_s%d" % s for s in (48, 49, 50)],
    "B3": ["gate_b3_s%d" % s for s in (54, 55, 56)],
    "B4": ["gate_b4_s%d" % s for s in (57, 58, 59)],
    "B4 moved": ["gate_b4_legacy_ctrl_s%d" % s for s in (57, 58, 59)],
    "B5": ["gate_b5_s%d" % s for s in (60, 61, 62)],
    "B6": ["gate_b6_s%d" % s for s in range(63, 73)],
    "B7": ["gate_b7_s%d" % s for s in range(73, 83)],
    "B8": ["gate_b8_s%d" % s for s in range(83, 93)],
}
REPLAY_GATES = ("B2", "B3", "B4", "B7")
LEARNING_RUNS = ["vg_learn_s%d" % seed for seed in (601, 602, 603)]
GAIN_GATES = ("B3", "B4", "B4 moved", "B5", "B6", "B7")
MIN_STATE_STEPS = 10
MIN_EPISODES_D = 5
SUPPORTED_RATIO, EXCLUDED_RATIO = 0.5, 0.1
WINNER_AGREEMENT_MIN, RECOVERABLE_MIN = 0.9, 0.8


def flag(argv: list, name: str, default: str) -> str:
    return argv[argv.index(name) + 1] if name in argv else default


def action_path(argv: list) -> tuple:
    """The flags that decide which action reaches the room."""
    return (flag(argv, "--env", "?"), flag(argv, "--tectal-orienting", "off"),
            flag(argv, "--tectal-orienting-idle", "policy"),
            "covert" if "--tectal-orienting-covert" in argv else "overt",
            flag(argv, "--policy", "gonogo"), flag(argv, "--policy-input", "broadcast"),
            flag(argv, "--broadcast-merge", "legacy"))


def check_a(root: str) -> None:
    print("== Check A: action path per run family ==")
    counts = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "session.json"))):
        with open(path, encoding="utf-8") as handle:
            argv = json.load(handle)["run"].get("argv", [])
        counts.setdefault(action_path(argv), []).append(os.path.basename(os.path.dirname(path)))
    print("env | orienting | idle | covert | policy | policy input | merge | runs | examples")
    for key, names in sorted(counts.items()):
        print(" | ".join(key), "|", len(names), "|", ", ".join(names[:3]))


def last_recorded_episode(run_folder: str) -> str:
    folders = sorted(glob.glob(os.path.join(run_folder, "episodes", "ep_*")))
    complete = [f for f in folders if os.path.exists(os.path.join(f, "weights.pt"))
                and os.path.exists(os.path.join(f, "maps.npz"))]
    if not complete:
        raise FileNotFoundError("no episode with weights and maps in %s" % run_folder)
    return complete[-1]


def build_policy(weights: dict) -> tuple:
    pfc_state, bg_state = weights["action_core.pfc"], weights["action_core.bg"]
    context = pfc_state["striatum_projection.weight"].shape[0]
    width = pfc_state["working_memory.weight_ih"].shape[1]
    actions = bg_state["thalamic_relay.0.weight"].shape[0]
    pfc, bg = PrefrontalCortex(width, context), BasalGanglia(context, actions)
    pfc.load_state_dict(pfc_state)
    bg.load_state_dict(bg_state)
    return pfc.eval(), bg.eval(), context


def replay_mean(pfc, bg, context: int, inputs: torch.Tensor) -> np.ndarray:
    """The action mean of the policy at each step, with no exploration noise."""
    hidden, means = torch.zeros(1, context), []
    with torch.no_grad():
        for row in inputs:
            state, hidden = pfc(row.reshape(1, -1), hidden)
            means.append(bg(state)[0].squeeze(0).numpy())
    return np.array(means)


def recorded_actions(folder: str, steps: np.ndarray) -> np.ndarray:
    by_step = {}
    with open(os.path.join(folder, "steps.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            by_step[int(record["step"])] = record["action"]
    return np.array([by_step[int(s)] for s in steps], dtype=float)


def reach_ratio(mean: np.ndarray, noise: np.ndarray) -> float:
    return float(mean.std(axis=0).max() / noise.std(axis=0).min())


def reach_label(ratio: float, input_gap: float) -> str:
    if input_gap != 0.0:
        return "VOID"
    if ratio >= SUPPORTED_RATIO:
        return "SUPPORTED"
    return "EXCLUDED" if ratio <= EXCLUDED_RATIO else "UNDECIDED"


def replay_run(root: str, run: str) -> None:
    """One line of check B. Longer than 20 lines because it prints every control beside the ratio."""
    folder = last_recorded_episode(os.path.join(root, run))
    weights = torch.load(os.path.join(folder, "weights.pt"), map_location="cpu", weights_only=False)
    maps, vectors = np.load(os.path.join(folder, "maps.npz")), np.load(os.path.join(folder, "vectors.npz"))
    policy_state = torch.tensor(maps["policy_state"]).float().reshape(len(maps["policy_state"]), -1)
    input_gap = float(np.abs(maps["policy_state"] - vectors["broadcast"]).max())
    pfc, bg, context = build_policy(weights)
    mean = replay_mean(pfc, bg, context, policy_state)
    order = np.random.default_rng(0).permutation(len(policy_state))
    shuffled = replay_mean(pfc, bg, context, policy_state[order])
    zeros = replay_mean(pfc, bg, context, torch.zeros_like(policy_state))[-1]
    noise = recorded_actions(folder, maps["policy_state__steps"]) - mean
    ratio = reach_ratio(mean, noise)
    relay_bias = np.tanh(weights["action_core.bg"]["thalamic_relay.0.bias"].numpy())
    cosine = torch.nn.functional.cosine_similarity(policy_state, policy_state.mean(0, keepdim=True)).mean()
    print("%s %s | K1 gap %.3g | mean %s | spread %s | shuffled mean %s | zero input %s | tanh(bias) %s | "
          "noise spread %s | input cosine to its mean %.3f | ratio %.4f | %s"
          % (run, os.path.basename(folder), input_gap, np.round(mean.mean(0), 4), np.round(mean.std(0), 5),
             np.round(shuffled.mean(0), 4), np.round(zeros, 4), np.round(relay_bias, 4),
             np.round(noise.std(0), 4), float(cosine), ratio, reach_label(ratio, input_gap)))


def pathway_signals(pfc, bg, context: int, inputs: torch.Tensor) -> dict:
    """Per step: the go signal, the no-go signal, the brake, and the input of the relay layer."""
    hidden, rows = torch.zeros(1, context), {"go": [], "no_go": [], "brake": [], "before_relay": []}
    with torch.no_grad():
        for row in inputs:
            state, hidden = pfc(row.reshape(1, -1), hidden)
            go, no_go, brake = bg.direct_pathway(state), bg.indirect_pathway(state), bg.stn_pathway(state)
            rows["go"].append(go.squeeze(0).numpy())
            rows["no_go"].append(no_go.squeeze(0).numpy())
            rows["brake"].append(brake.reshape(-1).numpy())
            rows["before_relay"].append((go * (1.0 - no_go) * (1.0 - brake)).squeeze(0).numpy())
    return {name: np.array(series) for name, series in rows.items()}


def before_relay_run(root: str, run: str) -> str:
    """One line of check E, and its label."""
    folder = last_recorded_episode(os.path.join(root, run))
    weights = torch.load(os.path.join(folder, "weights.pt"), map_location="cpu", weights_only=False)
    maps, vectors = np.load(os.path.join(folder, "maps.npz")), np.load(os.path.join(folder, "vectors.npz"))
    policy_state = torch.tensor(maps["policy_state"]).float().reshape(len(maps["policy_state"]), -1)
    input_gap = float(np.abs(maps["policy_state"] - vectors["broadcast"]).max())
    pfc, bg, context = build_policy(weights)
    signals = pathway_signals(pfc, bg, context, policy_state)
    zeros = pathway_signals(pfc, bg, context, torch.zeros_like(policy_state))["before_relay"][-1]
    noise = recorded_actions(folder, maps["policy_state__steps"]) - replay_mean(pfc, bg, context, policy_state)
    ratio = reach_ratio(signals["before_relay"], noise)
    label = reach_label(ratio, input_gap)
    print("%s %s | K1 gap %.3g | before relay mean %s, spread %s | zero input %s | go spread %s | "
          "no-go mean %s | brake mean %.4f | noise spread %s | ratio %.4f | %s"
          % (run, os.path.basename(folder), input_gap, np.round(signals["before_relay"].mean(0), 4),
             np.round(signals["before_relay"].std(0), 5), np.round(zeros, 4),
             np.round(signals["go"].std(0), 5), np.round(signals["no_go"].mean(0), 4),
             float(signals["brake"].mean()), np.round(noise.std(0), 4), ratio, label))
    return label


def relay_alone_reading(labels: list) -> str:
    """The reading over the gate runs: 9 of 12, or the same share of another count, rounded up."""
    needed = int(np.ceil(0.75 * len(labels)))
    if labels.count("SUPPORTED") >= needed:
        return "SUPPORTED"
    return "EXCLUDED" if labels.count("EXCLUDED") >= needed else "UNDECIDED"


def check_e(root: str) -> None:
    print("\n== Check E: the policy output before the relay layer ==")
    labels = [before_relay_run(root, run) for gate in REPLAY_GATES for run in GATES[gate][:3]]
    print("the relay layer alone is the cut: %s (%d SUPPORTED, %d EXCLUDED, %d UNDECIDED, %d VOID of %d)"
          % (relay_alone_reading(labels), labels.count("SUPPORTED"), labels.count("EXCLUDED"),
             labels.count("UNDECIDED"), labels.count("VOID"), len(labels)))
    print("reported only, runs with the value path, where the room never received the policy action:")
    for run in LEARNING_RUNS:
        if os.path.isdir(os.path.join(root, run, "episodes")):
            before_relay_run(root, run)


def check_b(root: str) -> None:
    print("\n== Check B: policy replay on the recorded policy input ==")
    for gate in REPLAY_GATES:
        for run in GATES[gate][:3]:
            replay_run(root, run)


def ranks(series) -> np.ndarray:
    return np.argsort(np.argsort(np.asarray(series, dtype=float))).astype(float)


def residual_on_index(series, index) -> np.ndarray:
    """Ranks of the series minus their straight line on the ranks of the index."""
    y, x = ranks(series), ranks(index)
    slope, intercept = np.polyfit(x, y, 1)
    return y - (slope * x + intercept)


def one_sided_p(null, observed: float) -> float:
    return float((1 + np.sum(np.asarray(null) >= observed)) / (1 + len(null)))


def partial_link(tables: list, shuffles: int = 5000) -> dict:
    res_f = [residual_on_index([t[1] for t in tb], [t[0] for t in tb]) for tb in tables]
    res_a = [residual_on_index([t[2] for t in tb], [t[0] for t in tb]) for tb in tables]
    pooled_f = np.concatenate(res_f)
    if np.allclose(pooled_f, 0.0) or np.allclose(np.concatenate(res_a), 0.0):
        # f or a is the index itself: nothing is left after the index is removed.
        return {"measurable": False, "seeds": len(tables)}
    observed = float(np.corrcoef(pooled_f, np.concatenate(res_a))[0, 1])
    rng = np.random.default_rng(0)
    null = [np.corrcoef(pooled_f, np.concatenate([rng.permutation(a) for a in res_a]))[0, 1]
            for _ in range(shuffles)]
    per_seed = [float(np.corrcoef(f, a)[0, 1]) for f, a in zip(res_f, res_a)]
    return {"r": observed, "p": one_sided_p(null, observed),
            "positive_seeds": int(np.sum(np.array(per_seed) > 0)), "seeds": len(tables)}


def state_difference(audio: np.ndarray, in_view: np.ndarray) -> float:
    return float(audio[~in_view].mean() - audio[in_view].mean())


def qualifies(in_view: np.ndarray) -> bool:
    return in_view.sum() >= MIN_STATE_STEPS and (~in_view).sum() >= MIN_STATE_STEPS


def rotation_test(episodes: list, shuffles: int = 2000) -> dict:
    """Mean d over (audio, in_view) episodes against in-view labels rotated inside each episode."""
    episodes = [(a, v) for a, v in episodes if qualifies(v)]
    if not episodes:
        return {"episodes": 0}
    observed = float(np.mean([state_difference(a, v) for a, v in episodes]))
    rng = np.random.default_rng(0)
    null = [np.mean([state_difference(a, np.roll(v, int(rng.integers(1, len(v))))) for a, v in episodes])
            for _ in range(shuffles)]
    positive = int(np.sum([state_difference(a, v) > 0 for a, v in episodes]))
    return {"episodes": len(episodes), "mean_d": observed, "positive_episodes": positive,
            "p": one_sided_p(null, observed)}


def ignited_episodes(rows: list) -> list:
    episodes = []
    for episode in sorted({r[0] for r in rows}):
        ignited = [r for r in rows if r[0] == episode and r[1]["winner"]]
        audio = np.array([r[1]["winner"] == "audio" for r in ignited], dtype=float)
        episodes.append((audio, np.array([r[2] for r in ignited], dtype=bool)))
    return episodes


def paired_layouts(first: list, second: list, shuffles: int = 5000) -> dict:
    df, da = [], []
    for table_a, table_b in zip(first, second):
        other = {t[0]: t for t in table_b}
        pairs = [(t, other[t[0]]) for t in table_a if t[0] in other]
        df.append(np.array([b[1] - a[1] for a, b in pairs]))
        da.append(np.array([b[2] - a[2] for a, b in pairs]))
    pooled = np.concatenate(df)
    observed = float(spearmanr(pooled, np.concatenate(da)).correlation)
    rng = np.random.default_rng(0)
    null = [spearmanr(pooled, np.concatenate([rng.permutation(a) for a in da])).correlation
            for _ in range(shuffles)]
    return {"pairs": len(pooled), "rho": observed, "p": one_sided_p(null, observed)}


def runs_on_disk(root: str, gate: str) -> list:
    present = [r for r in GATES[gate] if os.path.isdir(os.path.join(root, r, "episodes"))]
    if len(present) != len(GATES[gate]):
        print("%s: %d of %d run folders on disk, skipped" % (gate, len(present), len(GATES[gate])))
        return []
    return present


def print_gate_link(gate: str, rows: list, tables: list) -> None:
    pooled_f = np.concatenate([[t[1] for t in tb] for tb in tables])
    pooled_a = np.concatenate([[t[2] for t in tb] for tb in tables])
    index_a = [round(float(spearmanr([t[0] for t in tb], [t[2] for t in tb]).correlation), 3) for tb in tables]
    index_f = [round(float(spearmanr([t[0] for t in tb], [t[1] for t in tb]).correlation), 3) for tb in tables]
    print("%s | episodes %d | raw pooled rho %.3f | rho(a, index) %s | rho(f, index) %s"
          % (gate, len(pooled_f), float(spearmanr(pooled_f, pooled_a).correlation), index_a, index_f))
    print("   C1 partial:", partial_link(tables))
    print("   C2 inside episode:", rotation_test([e for r in rows for e in ignited_episodes(r)]))


def check_c(root: str) -> None:
    print("\n== Check C: the task link against the episode index ==")
    tables = {}
    for gate in GATES:
        present = runs_on_disk(root, gate)
        if not present:
            continue
        rows = [load_steps(os.path.join(root, r)) for r in present]
        tables[gate] = [episode_table(r) for r in rows]
        print_gate_link(gate, rows, tables[gate])
    if "B4" in tables and "B4 moved" in tables:
        print("C3 same index, other layout (B4 against B4 moved):", paired_layouts(tables["B4"], tables["B4 moved"]))


def unweighted_bid(recorded: float, share: float):
    """The bid before the weighting, clipped to [0, 1], or None when it cannot be recovered."""
    if recorded >= 1.0:
        return 1.0 if share <= 0.5 else None
    if share <= 0.0:
        return None
    return min(1.0, recorded / (2.0 * share))


def step_gains(folder: str) -> dict:
    """step -> (vision share, audio share), computed again from the recorded frame and waveform."""
    frames = np.load(os.path.join(folder, "frames.npz"))["frames"]
    vectors = np.load(os.path.join(folder, "vectors.npz"))
    waves = {int(s): w for s, w in zip(vectors["audio_waveform__steps"], vectors["audio_waveform"])}
    shares = {}
    for step, frame in enumerate(frames):
        tensor = torch.from_numpy(frame).float().div(255.0).permute(2, 0, 1).unsqueeze(0)
        wave = torch.from_numpy(waves[step]).float().unsqueeze(0) if step in waves else None
        shares[step] = precision_shares(vision_gain(tensor), audio_gain(wave))
    return shares


def gain_episode(folder: str) -> list:
    """Ignited steps from step 1: (winner, in_view, weighted audio top, unweighted audio top or None)."""
    with open(os.path.join(folder, "steps.jsonl"), encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    shares, steps = step_gains(folder), []
    for previous, current in zip(records, records[1:]):
        if not current["winner"]:
            continue
        share_v, share_a = shares[int(current["step"])]
        vision, audio = current["raw_bids"]["vision"], current["raw_bids"]["audio"]
        plain_v, plain_a = unweighted_bid(vision, share_v), unweighted_bid(audio, share_a)
        plain = None if plain_v is None or plain_a is None else plain_a > plain_v
        in_view = bool(previous["internals"]["info_after_step"]["_truth_light_in_view"])
        steps.append((current["winner"], in_view, audio > vision, plain, share_v))
    return steps


def gain_gate(root: str, gate: str) -> None:
    """One gate of check D. Longer than 20 lines because both controls and both orders print together."""
    present = runs_on_disk(root, gate)
    episodes = []
    for run in present:
        for folder in sorted(glob.glob(os.path.join(root, run, "episodes", "ep_*"))):
            if os.path.exists(os.path.join(folder, "frames.npz")):
                episodes.append(gain_episode(folder))
    if not episodes:
        return
    ignited = sum(len(e) for e in episodes)
    kept = [[s for s in e if s[3] is not None] for e in episodes]
    recovered = sum(len(e) for e in kept)
    agree = np.mean([(s[0] == "audio") == s[2] for e in kept for s in e if s[0] in ("audio", "vision")])
    share_in = np.mean([s[4] for e in kept for s in e if s[1]])
    share_out = np.mean([s[4] for e in kept for s in e if not s[1]])
    views = [np.array([s[1] for s in e], dtype=bool) for e in kept]
    weighted = rotation_test([(np.array([s[2] for s in e], dtype=float), v) for e, v in zip(kept, views)])
    plain = rotation_test([(np.array([s[3] for s in e], dtype=float), v) for e, v in zip(kept, views)])
    print("%s | episodes with frames %d | ignited %d | recoverable %.3f (K5) | weighted order names the "
          "winner %.3f (K4) | vision share in view %.3f, out of view %.3f"
          % (gate, len(episodes), ignited, recovered / ignited, float(agree), float(share_in), float(share_out)))
    print("   weighted order:", weighted)
    print("   unweighted order:", plain)
    print("   reading:", gain_reading(weighted, plain, recovered / ignited, float(agree)))


def gain_reading(weighted: dict, plain: dict, recoverable: float, agreement: float) -> str:
    if agreement < WINNER_AGREEMENT_MIN or recoverable < RECOVERABLE_MIN:
        return "VOID"
    if weighted["episodes"] < MIN_EPISODES_D or weighted["p"] >= 0.05 or weighted["mean_d"] <= 0:
        return "NOT MEASURABLE"
    if plain["p"] >= 0.05:
        return "SUPPORTED"
    if plain["mean_d"] > 0 and plain["mean_d"] >= 0.5 * weighted["mean_d"]:
        return "EXCLUDED"
    return "UNDECIDED"


def check_d(root: str) -> None:
    print("\n== Check D: the task link with the gain weighting removed from the step ==")
    for gate in GAIN_GATES:
        gain_gate(root, gate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--checks", nargs="+", choices=["a", "b", "c", "d", "e"], default=["a", "b", "c", "d", "e"])
    args = parser.parse_args()
    for name, check in (("a", check_a), ("b", check_b), ("c", check_c), ("d", check_d), ("e", check_e)):
        if name in args.checks:
            check(args.runs_root)


if __name__ == "__main__":
    main()

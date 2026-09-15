"""Replay the Gate B inputs through a real workspace to compare ignition rules offline.

Gate B (docs/results/dark_room_senses_2026_09.md) FAILED on silence: 0.656 / 0.695 /
0.616 of steps never ignited, and silence feeds itself inside each step (the running
average moves on every settle cycle, and an empty broadcast lowers every bid by 5
percent). This probe replays the recorded per-step inputs (raw bids, module contents,
affect, interoception, learned valence values) through a real GlobalWorkspace,
AffectiveModulator and ReentrantProcessor. The specialists' feedback uses the real
`receive_broadcast` rules, which need only each module's last content, which is
recorded. Read-only: no training, no model file changes.

Not reproduced exactly: the oscillator phases and the trained binding coupling (the
workspace is rebuilt from the run's config with the run seed). The fidelity gate decides
whether that matters.

PRE-STATED GATES, written 2026-09-15 before any replay, by owner decision.

    FIDELITY (the default rule, replayed): at each of the 3 seeds the replay reproduces
    the recorded `ignited` flag on at least 0.90 of steps, and the replayed silence share
    is within 0.05 of the recorded share. If fidelity fails, no variant result is used.

    VARIANTS
        V1   the running average moves once per environment step, and salience is
             computed against the average from BEFORE this step
        V2   an empty broadcast leaves a specialist's bid unchanged (no 5 percent cut)
        V3   V1 with a tolerance: ignite if input >= average - 1.0 * running sd (the sd is
             an average with the same 0.95 factor)
        V12  V1 and V2 together
    A variant is PROMISING only if, at each of the 3 seeds: silence below 0.50, the
    runner-up module at least 0.05 of ignited steps, and no module at 0.95 or more of
    ignited steps. A promising variant is a candidate for a live Gate B rerun, never a
    result: the replay is open-loop (actions and learning do not respond to it).

RESULT, 2026-09-15 (seeds 42 / 43 / 44, 2000 steps each):
    FIDELITY PASS   ignition agreement 0.973 / 0.970 / 0.997; silence replayed against
                    recorded 0.655 vs 0.656 / 0.726 vs 0.695 / 0.615 vs 0.616
    V1   not promising   silence 0.681 / 0.305 / 0.621
    V2   not promising   silence 0.692 / 0.714 / 0.655
    V3   PROMISING       silence 0.289 / 0.130 / 0.286; runner-up (audio) 0.181 / 0.205 /
                         0.174; top share 0.819 / 0.795 / 0.826
    V12  not promising   silence 0.720 / 0.309 / 0.654
Not tested: whether V3 ignition still carries information (it ignites on 0.71 to 0.87 of
steps), the task variable, and closed-loop behavior.

Run:
    python -m scripts.analysis.probe_ignition_replay --runs runs/gate_b_s42 runs/gate_b_s43 runs/gate_b_s44
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.audio.auditory_specialist import AuditorySpecialist  # noqa: E402
from models.core.sensory_tectum import SensoryTectum  # noqa: E402
from scripts.training.train_rlhf import init_components  # noqa: E402

FIDELITY_AGREEMENT_MIN = 0.90
FIDELITY_SILENCE_TOLERANCE = 0.05
SILENCE_MAX = 0.50
RUNNER_UP_MIN = 0.05
TOP_SHARE_MAX = 0.95
BASELINE_FACTOR = 0.95
TOLERANCE_SD = 1.0
VARIANTS = ("default", "V1", "V2", "V3", "V12")
GOAL = [1.0, -1.0, 1.0]


class ReplaySpecialist:
    """Holds a recorded last content and applies a real receive_broadcast rule."""

    def __init__(self, rule, keep_bid_on_empty: bool):
        self._rule, self._keep = rule, keep_bid_on_empty
        self._last_content = None

    def receive_broadcast(self, broadcast_content, current_bid):
        if self._keep and not _has_tensor(broadcast_content):
            return current_bid
        return self._rule(self, broadcast_content, current_bid)


def _has_tensor(broadcast) -> bool:
    if isinstance(broadcast, torch.Tensor):
        return True
    return isinstance(broadcast, dict) and any(
        isinstance(broadcast.get(k), torch.Tensor) for k in ("_fused", "tensor"))


def load_run(run: str):
    config = json.load(open(os.path.join(run, "config.json"), encoding="utf-8"))
    seed = json.load(open(os.path.join(run, "session.json"), encoding="utf-8"))["run"]["seed"]
    episodes = []
    for folder in sorted(glob.glob(os.path.join(run, "episodes", "ep_*"))):
        steps = [json.loads(line) for line in open(os.path.join(folder, "steps.jsonl"), encoding="utf-8")]
        vectors = np.load(os.path.join(folder, "vectors.npz"))
        episodes.append((steps, {k: vectors[k] for k in vectors.files}))
    return config, seed, episodes


def row_for_step(vectors: dict, name: str, step: int):
    steps = vectors.get(name + "__steps")
    if steps is None:
        return None
    hits = np.where(steps == step)[0]
    return None if len(hits) == 0 else vectors[name][hits[0]]


def build_chain(config: dict, seed: int, variant: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    components = init_components(config)
    workspace, reentrant, modulator = components[1], components[2], components[3]
    keep = variant in ("V2", "V12")
    specialists = {"vision": ReplaySpecialist(SensoryTectum.receive_broadcast, keep),
                   "audio": ReplaySpecialist(AuditorySpecialist.receive_broadcast, keep)}
    return workspace, reentrant, modulator, specialists


def replay_step(chain, step: dict, vectors: dict, device, state: dict, variant: str):
    workspace, reentrant, modulator, specialists = chain
    index = step["step"]
    payloads, active = _payloads(vectors, index, step, specialists, device, workspace)
    if modulator.learned_valence is not None and step["internals"].get("learned_valence"):
        modulator.learned_valence.values = dict(step["internals"]["learned_valence"]["values"])
    _before_settle(workspace, state, variant)
    reentrant.settle(workspace=workspace, specialists=active, initial_bids=dict(step["raw_bids"]),
                     payloads=payloads, goal_vector=torch.tensor(GOAL, device=device),
                     pad_state=step["internals"].get("emotion"),
                     interoceptive_state=step.get("interoception") or None)
    _after_settle(workspace, state, variant)
    winners = getattr(workspace.state, "winners", []) or []
    return bool(workspace.state.is_conscious), (winners[0] if winners else "")


def _payloads(vectors, index, step, specialists, device, workspace):
    dim = workspace.workspace_dim if hasattr(workspace, "workspace_dim") else 256
    zero = torch.zeros(1, dim, device=device)
    content = {}
    for name in ("tectum_content", "audio_content"):
        row = row_for_step(vectors, name, index)
        content[name] = zero if row is None else torch.tensor(row, dtype=torch.float32, device=device)
    vision = {"tensor": content["tectum_content"], "source": "tectum"}
    for name in ("capsule_poses", "capsule_activities"):
        row = row_for_step(vectors, name, index)
        if row is not None:
            vision[name] = torch.tensor(row, dtype=torch.float32, device=device)
    specialists["vision"]._last_content = content["tectum_content"]
    specialists["audio"]._last_content = content["audio_content"]
    active = {"vision": specialists["vision"]}
    if step["raw_bids"].get("audio", 0.0) > 0.0:
        active["audio"] = specialists["audio"]
    payloads = {"vision": vision, "audio": {"tensor": content["audio_content"], "source": "audio"},
                "semantic": {"tensor": zero, "source": "semantic"}}
    return payloads, active


def _before_settle(workspace, state: dict, variant: str) -> None:
    if variant not in ("V1", "V3", "V12"):
        return
    workspace.baseline_alpha = 1.0  # frozen inside the settle cycles
    if state.get("mean") is None:
        return
    offset = TOLERANCE_SD * state["var"] ** 0.5 if variant == "V3" else 0.0
    workspace._energy_baseline = state["mean"] - offset


def _after_settle(workspace, state: dict, variant: str) -> None:
    if variant not in ("V1", "V3", "V12"):
        return
    energy = max(workspace.state.competition_results.values())
    if state.get("mean") is None:
        state["mean"], state["var"] = energy, 0.0
        return
    delta = energy - state["mean"]
    state["mean"] += (1.0 - BASELINE_FACTOR) * delta
    state["var"] = BASELINE_FACTOR * (state["var"] + (1.0 - BASELINE_FACTOR) * delta * delta)


def replay_run(run: str, variant: str) -> dict:
    config, seed, episodes = load_run(run)
    chain = build_chain(config, seed, variant)
    device = config.get("device", "cpu")
    state, agree, recorded, replayed = {}, [], [], []
    with torch.no_grad():
        for steps, vectors in episodes:
            for step in steps:
                ignited, winner = replay_step(chain, step, vectors, device, state, variant)
                agree.append(ignited == bool(step["ignited"]))
                recorded.append(step["winner"])
                replayed.append(winner)
    return summarize(replayed, recorded, agree)


def summarize(replayed: list, recorded: list, agree: list) -> dict:
    def shares(winners):
        ignited = [w for w in winners if w]
        ranked = collections.Counter(ignited).most_common()
        return {"silence": 1.0 - len(ignited) / len(winners),
                "top_share": ranked[0][1] / len(ignited) if ranked else 0.0,
                "runner_up": ranked[1][1] / len(ignited) if len(ranked) > 1 else 0.0,
                "winners": dict(ranked)}
    return {"steps": len(agree), "ignition_agreement": float(np.mean(agree)),
            "replayed": shares(replayed), "recorded": shares(recorded)}


def fidelity_ok(summary: dict) -> bool:
    gap = abs(summary["replayed"]["silence"] - summary["recorded"]["silence"])
    return summary["ignition_agreement"] >= FIDELITY_AGREEMENT_MIN and gap <= FIDELITY_SILENCE_TOLERANCE


def promising(summary: dict) -> bool:
    r = summary["replayed"]
    return r["silence"] < SILENCE_MAX and r["runner_up"] >= RUNNER_UP_MIN and r["top_share"] < TOP_SHARE_MAX


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs=3, required=True)
    args = parser.parse_args()
    results = {variant: [replay_run(run, variant) for run in args.runs] for variant in VARIANTS}
    for variant, summaries in results.items():
        for run, summary in zip(args.runs, summaries):
            print("%-7s %s %s" % (variant, run, json.dumps(summary)))
    fidelity = all(fidelity_ok(s) for s in results["default"])
    print("\nFIDELITY: %s" % ("PASS at all 3 seeds" if fidelity else "FAILED"))
    if not fidelity:
        print("No variant result is used.")
        return
    for variant in VARIANTS[1:]:
        ok = all(promising(s) for s in results[variant])
        print("%-4s %s" % (variant, "PROMISING at all 3 seeds" if ok else "not promising"))


if __name__ == "__main__":
    main()

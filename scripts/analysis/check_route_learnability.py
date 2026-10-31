"""Can a policy use a selected place in its input? A cheap check before any gate of the direct route.

docs/decisions/2026_10_31_workspace_link_design.md, design B. A decode gate of the route passes by
construction, so it is not worth a run on its own. The open question is if a policy can use the route.
This check answers it with no tectum, workspace or binder. It runs the real room and the real
`StandardActorCritic`, with the update cadence of the training loop (every 10 steps) and the raw reward.
The input is built from the truth of the room, so the selection here is perfect, and the live route can
only be worse. The light is the louder object and the food. Written and pushed before its first run.

Arms, each a 256-D input through one fixed orthonormal projection:
  direction  the unit direction to the light, at every step. A control on the policy itself.
  route      the map cells on the light, as a place mask, and all zeros when the light is out of view.
  blank      one constant vector. The null.

PRE-STATED RULE, seeds 1001 to 1003, 40 episodes of 200 steps. The reading is light first in episodes 21
to 40, as in the value gate: within the first 60 steps, the first object closer than 60 room pixels (the
nearer one on a tie), and no contact counts as not light. The share of steps in the food is reported and
not judged, because one untrained episode on a discarded seed gave 0.82. On that seed (999, 20 episodes)
the blank arm gave light first 0.55 and a food share of 0.225.
  C1  direction at least 0.8, and at least 0.2 above blank, at 3 of 3 seeds. If it fails, this policy does
      not learn the task even from a perfect input, so a live gate with this policy would waste its cost.
      The run stops at the first seed where C1 fails.
  C2  route at least 0.8, and at least 0.2 above blank, at 3 of 3 seeds. If it fails while C1 holds, a place
      mask is not a usable input for this policy, so the route needs another form before any live run.
Only if both hold is a live test of the route worth its cost.

RESULT OF THAT RULE, seed 1001. C1 FAILED and the run stopped (direction 0.55, route 1.00, blank 0.40). The
reason stated for C1 was wrong, because the route arm did learn, so the direction arm was a weak control.

FOLLOW-UP, added after that result and pushed before its run (--follow-up). Seeds 1002 to 1006, fresh. Arms
route, blank, and route_cut, which is the route until episode 30 and the blank input from episode 31 on.
  F1  route at least 0.8, and at least 0.2 above blank, in episodes 21 to 40, at 5 of 5 seeds.
  F2  in episodes 31 to 40, route_cut at least 0.2 below route, at 5 of 5 seeds. The cut is the causal test.
The direction arm is not run again. Why it fails is not known.

RESULT OF THE FOLLOW-UP. F1 FAILED, 4 of 5 seeds (seed 1005 reached 0.75). F2 held at 5 of 5.

BINDER CHECK, added after that result and pushed before its run (--binder). The same test with the real
rhythm binder, which selects the cells from the real frames and the real sound. No tectum and no workspace.
The input is the mask of the group of the current step, and all zeros without a group. The binder gives no
group for its first 12 steps of sound, so in all arms the agent holds still for the first 13 steps, as the
earlier gates did, and the policy neither acts nor learns there. Seeds 1007 to 1009, arms route, blank,
route_cut.
  B1  route at least 0.8, and at least 0.2 above blank, in episodes 21 to 40, at 3 of 3 seeds.
  B2  in episodes 31 to 40, route_cut at least 0.2 below route, at 3 of 3 seeds.
Reported only: the share of steps with a group, and the share of those where the group is on the light.
If B1 or B2 fails, no live run follows without a new decision.

RESULT OF THE BINDER CHECK. FAILED. B1 at 0 of 3 seeds (route 0.40, 0.40, 0.50), B2 at 1 of 3. The group lay on
the light at 0.47 to 0.68 of steps. scripts/analysis/diagnose_binder_motion.py then gave 0.95 to 1.00 with the
agent still or moving smoothly, and 0.45 to 0.75 under noise of standard deviation 1.0.

LOW NOISE CHECK, added after that result and pushed before its run (--binder-low-noise). The binder check with
one change. The exploration noise of the policy starts at standard deviation 0.1, the noise of the motor map,
and it stays a learned parameter. Seeds 1010 to 1012, the same arms and the same thresholds.
  N1  route at least 0.8, and at least 0.2 above blank, in episodes 21 to 40, at 3 of 3 seeds.
  N2  in episodes 31 to 40, route_cut at least 0.2 below route, at 3 of 3 seeds.
If N1 or N2 fails, design B stops, and no live run follows. This is the last variation of this check. A third
change of the check after seeing its results would be a search for a pass.

Limits, stated before the run. The raw reward leaves out the emotion shaping of the live loop. The
exploration level is fixed at arousal 0.5. One policy, one learning rate.

Usage:
    python -m scripts.analysis.check_route_learnability --output-json out.json
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

import numpy as np
import torch

from models.audio.auditory_specialist import AuditorySpecialist
from models.core.rhythm_binding import RhythmMapBinding, fresh_band_envelope
from models.core.tectal_motor_map import cell_offsets
from models.self_model.standard_actor_critic import StandardActorCritic
from scripts.analysis.diagnose_attention_modes import Cochlea, as_image
from scripts.analysis.probe_tectal_attention import selected_object
from simulations.environments.simple_visual_env import SimpleVisualEnv

ARMS = ("direction", "route", "blank")
SEEDS = (1001, 1002, 1003)
FOLLOW_UP_ARMS = ("route", "blank", "route_cut")
FOLLOW_UP_SEEDS = (1002, 1003, 1004, 1005, 1006)
CUT_AFTER = 30
BINDER_SEEDS = (1007, 1008, 1009)
HOLD = 13
LOW_NOISE_SEEDS = (1010, 1011, 1012)
LOW_NOISE = 0.1
EPISODES, STEPS, UPDATE_EVERY, DIM = 40, 200, 10, 256
LEARNING_RATE, CONTACT, LIMIT, SCORE, MARGIN = 1e-3, 60.0, 60, 0.8, 0.2


class RawReward:
    def compute_emotional_reward(self, emotion_values, base_reward, context):
        return base_reward


class NoMemory:
    def store_experience(self, **kwargs) -> None:
        return None


def build_room(seed: int) -> SimpleVisualEnv:
    """The room of the gates, with the decoy always the quieter object, so the light is selected."""
    np.random.seed(seed)
    room = SimpleVisualEnv(width=224, height=224, audio="binaural", audio_channels=4, report_collision=True,
                           view="agent_centered", agent_mark="ring", agent_colour=(217, 119, 87),
                           decoy_sound=True, decoy_level=0.5, decoy_visible=True, start_between=True,
                           pulse=True, sound_direction=False)
    room.seed_audio(seed)
    room.seed_decoy(seed)
    return room


def place_mask(offset: np.ndarray, radius: float) -> np.ndarray:
    """256 values, 1 on the map cells within `radius` of the object, unit norm, all zeros out of view."""
    dx, dy = cell_offsets()
    mask = (np.hypot(dx - offset[0], dy - offset[1]) < radius).astype(np.float64).reshape(-1)
    size = np.linalg.norm(mask)
    return mask / size if size > 0 else mask


def projection(seed: int) -> np.ndarray:
    """A fixed orthonormal matrix, the same for all arms of a seed."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(DIM, DIM)))
    return q


def arm_input(arm: str, offset: np.ndarray, radius: float, basis: np.ndarray) -> torch.Tensor:
    if arm == "direction":
        code = np.zeros(DIM)
        code[:2] = offset / max(np.linalg.norm(offset), 1e-9)
    elif arm == "route":
        code = place_mask(offset, radius)
    elif arm == "blank":
        code = np.zeros(DIM)
        code[0] = 1.0
    else:
        raise ValueError("unknown arm %r" % (arm,))
    return torch.as_tensor(basis @ code, dtype=torch.float32)[None]


def first_contact(room) -> str:
    """The object closer than CONTACT now, the nearer one on a tie, or an empty name."""
    light = float(np.linalg.norm(room.light_pos - room.agent_pos))
    decoy = float(np.linalg.norm(room.decoy_pos - room.agent_pos))
    if min(light, decoy) >= CONTACT:
        return ""
    return "light" if light <= decoy else "decoy"


def effective_arm(arm: str, episode: int) -> str:
    """The input of an arm in one episode (counted from 0). route_cut loses the route after CUT_AFTER."""
    if arm == "route_cut":
        return "route" if episode < CUT_AFTER else "blank"
    return arm


def run_arm(arm: str, seed: int) -> Dict[str, List[float]]:
    """Per episode, 1 if the light came first, and the share of steps in the food."""
    room, basis = build_room(seed), projection(seed)
    torch.manual_seed(seed)
    policy = StandardActorCritic({"workspace_dim": DIM, "action_dim": 2, "learning_rate": LEARNING_RATE,
                                  "device": "cpu"}, RawReward(), NoMemory())
    radius = float(room.light_radius)
    firsts, shares = [], []
    for episode in range(EPISODES):
        room.reset()
        shown = effective_arm(arm, episode)
        state, hits, first = arm_input(shown, room.light_pos - room.agent_pos, radius, basis), 0, ""
        for step in range(STEPS):
            action, _ = policy.select_action(state, emotion_arousal=0.5)
            _, reward, done, cut, _ = room.step(action)
            hits += int(reward > 0)
            if not first and step < LIMIT:
                first = first_contact(room)
            following = arm_input(shown, room.light_pos - room.agent_pos, radius, basis)
            policy.step(state, action, float(reward), following, bool(done or cut), {}, 0.0)
            if step > 0 and step % UPDATE_EVERY == 0:
                policy.update_policy()
            state = following
        firsts.append(float(first == "light"))
        shares.append(hits / STEPS)
    return {"light_first": firsts, "food_share": shares}


def late(values: List[float]) -> float:
    """Episodes 21 to 40."""
    return float(np.mean(values[20:40]))


def holds_for(arm_firsts: List[float], blank_firsts: List[float]) -> bool:
    return bool(late(arm_firsts) >= SCORE and late(arm_firsts) >= late(blank_firsts) + MARGIN)


def after_cut(values: List[float]) -> float:
    """Episodes 31 to 40."""
    return float(np.mean(values[CUT_AFTER:EPISODES]))


def follow_up_holds(cells: dict) -> Dict[str, bool]:
    """F1 and F2 at one seed."""
    firsts = {arm: cells[arm]["light_first"] for arm in FOLLOW_UP_ARMS}
    return {"F1": holds_for(firsts["route"], firsts["blank"]),
            "F2": bool(after_cut(firsts["route_cut"]) <= after_cut(firsts["route"]) - MARGIN)}


def group_mask(cells: List[int]) -> np.ndarray:
    """256 values, 1 on the cells of the group, unit norm, all zeros without a group."""
    mask = np.zeros(DIM)
    mask[list(cells)] = 1.0
    size = np.linalg.norm(mask)
    return mask / size if size > 0 else mask


def binder_input(arm: str, episode: int, cells: List[int], basis: np.ndarray) -> torch.Tensor:
    if effective_arm(arm, episode) == "route":
        code = group_mask(cells)
    else:
        code = np.zeros(DIM)
        code[0] = 1.0
    return torch.as_tensor(basis @ code, dtype=torch.float32)[None]


def run_binder_arm(arm: str, seed: int, specialist, start_noise: Optional[float] = None) -> Dict[str, List[float]]:
    """As run_arm, with the group of the real binder as the input, and the hold at the start."""
    room, basis = build_room(seed), projection(seed)
    torch.manual_seed(seed)
    policy = StandardActorCritic({"workspace_dim": DIM, "action_dim": 2, "learning_rate": LEARNING_RATE,
                                  "device": "cpu"}, RawReward(), NoMemory())
    if start_noise is not None:
        policy.log_std.data.fill_(float(np.log(start_noise)))
    ear = Cochlea(specialist)
    binder = RhythmMapBinding(fresh_band_envelope(ear), bands=specialist.num_bands)
    firsts, shares, grouped, on_light = [], [], 0, 0
    for episode in range(EPISODES):
        frame, facts = room.reset()
        hits, first, state, action, reward = 0, "", None, None, 0.0
        for step in range(STEPS):
            if "audio_waveform" in facts:
                ear.hear(facts["audio_waveform"])
            image = as_image(frame)
            _, record = binder.observe(image, None, image)
            cells = record.get("group", [])
            current = binder_input(arm, episode, cells, basis)
            if state is not None:
                policy.step(state, action, float(reward), current, False, {}, 0.0)
                if step % UPDATE_EVERY == 0:
                    policy.update_policy()
            if cells:
                grouped += 1
                truth = {"_truth_light_offset": (room.light_pos - room.agent_pos).tolist(),
                         "_truth_decoy_offset": (room.decoy_pos - room.agent_pos).tolist()}
                on_light += int(selected_object(cells, truth) == "light")
            if step >= HOLD:
                state, action = current, policy.select_action(current, emotion_arousal=0.5)[0]
            else:
                state, action = None, np.zeros(2, dtype=np.float32)
            frame, reward, _, _, facts = room.step(action)
            hits += int(reward > 0)
            if not first and step < LIMIT:
                first = first_contact(room)
        firsts.append(float(first == "light"))
        shares.append(hits / STEPS)
    return {"light_first": firsts, "food_share": shares, "group_share": [grouped / (EPISODES * STEPS)],
            "on_light": [on_light / max(grouped, 1)]}


def run_binder(output_json: str, seeds: tuple = BINDER_SEEDS, start_noise: Optional[float] = None) -> None:
    specialist = AuditorySpecialist({})
    specialist.eval()
    table: dict = {}
    holds: Dict[str, List[bool]] = {"B1": [], "B2": []}
    for seed in seeds:
        cells = {arm: run_binder_arm(arm, seed, specialist, start_noise) for arm in FOLLOW_UP_ARMS}
        table[str(seed)] = cells
        print(seed, " | ".join("%s: 21-40 %.2f, 31-40 %.2f, food %.3f, group %.2f, on light %.2f" % (
            arm, late(cells[arm]["light_first"]), after_cut(cells[arm]["light_first"]), after_cut(cells[arm]["food_share"]),
            cells[arm]["group_share"][0], cells[arm]["on_light"][0]) for arm in FOLLOW_UP_ARMS), flush=True)
        verdicts = follow_up_holds(cells)
        holds["B1"].append(verdicts["F1"])
        holds["B2"].append(verdicts["F2"])
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump({"cells": table, "holds": holds}, handle, indent=2)
    print("HOLDS:", holds, flush=True)


def run_follow_up(output_json: str) -> None:
    table: dict = {}
    holds: Dict[str, List[bool]] = {"F1": [], "F2": []}
    for seed in FOLLOW_UP_SEEDS:
        cells = {arm: run_arm(arm, seed) for arm in FOLLOW_UP_ARMS}
        table[str(seed)] = cells
        print(seed, " | ".join("%s: 21-40 %.2f, 31-40 %.2f, food %.3f" % (
            arm, late(cells[arm]["light_first"]), after_cut(cells[arm]["light_first"]), after_cut(cells[arm]["food_share"]))
            for arm in FOLLOW_UP_ARMS), flush=True)
        for check, held in follow_up_holds(cells).items():
            holds[check].append(held)
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump({"cells": table, "holds": holds}, handle, indent=2)
    print("HOLDS:", holds, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--follow-up", action="store_true")
    parser.add_argument("--binder", action="store_true")
    parser.add_argument("--binder-low-noise", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.binder_low_noise:
        run_binder(args.output_json, LOW_NOISE_SEEDS, LOW_NOISE)
        return
    if args.binder:
        run_binder(args.output_json)
        return
    if args.follow_up:
        run_follow_up(args.output_json)
        return
    table: dict = {}
    holds: Dict[str, List[bool]] = {"C1": [], "C2": []}
    for seed in SEEDS:
        table[str(seed)] = {arm: run_arm(arm, seed) for arm in ARMS}
        cells = table[str(seed)]
        for arm in ARMS:
            print(seed, arm, "light first, episodes 21 to 40: %.2f; food share %.3f" % (
                late(cells[arm]["light_first"]), late(cells[arm]["food_share"])), flush=True)
        holds["C1"].append(holds_for(cells["direction"]["light_first"], cells["blank"]["light_first"]))
        holds["C2"].append(holds_for(cells["route"]["light_first"], cells["blank"]["light_first"]))
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump({"cells": table, "holds": holds}, handle, indent=2)
        if not holds["C1"][-1]:
            print("C1 FAILED at seed %d, the run stops" % seed, flush=True)
            break
    print("HOLDS:", holds, flush=True)


if __name__ == "__main__":
    main()

"""Does movement of the agent break the rhythm binder's selection? After the binder check FAILED, reported only.

scripts/analysis/check_route_learnability.py --binder gave a group on the light at 0.47 to 0.68 of steps,
while the agent moved under an untrained policy (noise of standard deviation about 1 per axis). The motor map
gates moved it toward the target with noise of 0.1. This script has no policy and no learning. The real room,
cochlea and binder, the light always the louder object, 20 episodes of 60 steps per motion, seeds 991 to 993.

Motions after a still start of 13 steps:
  still    the agent does not move.
  smooth   0.5 times the unit vector to the light, plus noise of standard deviation 0.1.
  jitter   noise of standard deviation 1.0, clipped to the range of an action, as the untrained policy.

The reading is the share of steps with a group on the light, at step 13, in steps 14 to 30 and in steps 31 to 60.

Usage:
    python -m scripts.analysis.diagnose_binder_motion
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from models.audio.auditory_specialist import AuditorySpecialist
from models.core.rhythm_binding import RhythmMapBinding, fresh_band_envelope
from scripts.analysis.check_route_learnability import HOLD, build_room
from scripts.analysis.diagnose_attention_modes import Cochlea, as_image
from scripts.analysis.probe_tectal_attention import selected_object

MOTIONS = ("still", "smooth", "jitter")
SEEDS = (991, 992, 993)
EPISODES, STEPS = 20, 60
PHASES = (("step 13", HOLD, HOLD + 1), ("steps 14 to 30", HOLD + 1, 31), ("steps 31 to 60", 31, STEPS))


def motion_action(motion: str, step: int, offset: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The action of one step. All motions hold still for the first HOLD steps."""
    if motion == "still" or step < HOLD:
        return np.zeros(2, dtype=np.float32)
    if motion == "smooth":
        unit = offset / max(np.linalg.norm(offset), 1e-9)
        return np.clip(0.5 * unit + rng.normal(0.0, 0.1, 2), -1.0, 1.0).astype(np.float32)
    if motion == "jitter":
        return np.clip(rng.normal(0.0, 1.0, 2), -1.0, 1.0).astype(np.float32)
    raise ValueError("unknown motion %r" % (motion,))


def phase_shares(on_light: List[float], grouped: List[int]) -> Dict[str, Dict[str, float]]:
    """Per phase, the share of steps with a group, and the share of those on the light. Rows are [episode, step]."""
    hits, groups = np.array(on_light), np.array(grouped)
    shares = {}
    for name, start, end in PHASES:
        seen = groups[:, start:end]
        shares[name] = {"group": float(seen.mean()), "on_light": float(hits[:, start:end].sum() / max(seen.sum(), 1))}
    return shares


def run_motion(motion: str, seed: int, specialist) -> Dict[str, Dict[str, float]]:
    room, rng = build_room(seed), np.random.default_rng(seed)
    ear = Cochlea(specialist)
    binder = RhythmMapBinding(fresh_band_envelope(ear), bands=specialist.num_bands)
    on_light, grouped = [], []
    for _ in range(EPISODES):
        frame, facts = room.reset()
        hits, seen = [], []
        for step in range(STEPS):
            if "audio_waveform" in facts:
                ear.hear(facts["audio_waveform"])
            image = as_image(frame)
            cells = binder.observe(image, None, image)[1].get("group", [])
            truth = {"_truth_light_offset": (room.light_pos - room.agent_pos).tolist(),
                     "_truth_decoy_offset": (room.decoy_pos - room.agent_pos).tolist()}
            seen.append(int(bool(cells)))
            hits.append(int(selected_object(cells, truth) == "light") if cells else 0)
            frame, _, _, _, facts = room.step(motion_action(motion, step, room.light_pos - room.agent_pos, rng))
        on_light.append(hits)
        grouped.append(seen)
    return phase_shares(on_light, grouped)


def main() -> None:
    specialist = AuditorySpecialist({})
    specialist.eval()
    for motion in MOTIONS:
        for seed in SEEDS:
            shares = run_motion(motion, seed, specialist)
            print(motion, seed, " | ".join("%s: group %.2f, on light %.2f" % (name, cell["group"], cell["on_light"])
                                           for name, cell in shares.items()), flush=True)


if __name__ == "__main__":
    main()

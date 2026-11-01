"""Re-check of RPT-1, GWT-1 and GWT-3 against the empirical rule. Read-only, on recorded runs.

THE QUESTION. `docs/consciousness_indicators_butlin.md`, "What IMPLEMENTED means" (2026-08-11),
asks for empirical evidence that the system has the property. These three rows got their score
before that rule and were not examined under it. For each row this probe asks if that evidence
exists in the dark room configuration. It changes no score. The owner decides every score.

WHAT THE SOURCES ASK, paraphrased. Butlin et al. 2023 (arXiv 2308.08708v3), sections 2.1.3, 2.2.3,
2.4.6, 2.5, 3.1.1 and 3.1.2 read in full. Butlin et al. 2025 (doi 10.1016/j.tics.2025.10.011), the
whole article read in the published version at the repository of the London School of Economics.
  RPT-1. Input modules use algorithmic recurrence. The same operation is applied again, and what
         is retained from the past acts on present processing.
  GWT-1. Several specialised systems that can work in parallel. They need not be encapsulated, and
         their exact properties are not the main point of the theory.
  GWT-3. What the workspace holds is available to all modules. All modules take workspace states
         as input, the modules that process input included, so information flows back and
         influences their processing.
The 2025 article says that many indicators can be inferred from architecture and training. The
local rule asks for evidence that the system has the property, so this probe measures it.

BEFORE THESE RULES I READ, from the record of the three runs named below. `bid_memory`, `bid_body`
and `bid_semantic` have 1 distinct value at 8000 of 8000 steps. `bid_vision` and `bid_audio` have
7499 to 7929. Episodes with decoy level 0.5 use 2 settle cycles at nearly every step, and episodes
with level 3.0 use 2 to 4. The action is zero at 24000 of 24000 steps. I also read the training
loop. `specialists` holds vision and audio only, each `receive_broadcast` returns a new
bid and writes no state, and no module forward reads the broadcast. My expectation from that read,
which is no rule, is that the carried state changes the vision content, that the audio content has
no history, that the settle cycles leave every content unchanged, and that a perturbed broadcast
leaves the next content and raw bid of every module unchanged.

DATA. `runs/ta_nolink_s801`, `_s802`, `_s803`, the no gain arm of the tectal attention gate. Their
flags are in the run's `session.json`. Episodes 38 (decoy level 3.0) and 39 (level 0.5) replay with
the weights saved at the end of episode 39. The probe reads 2 levels because the settle cycles
differ between them. No run is trained and no file in the run folders is written.

THE REPLAY. `run_episode` of `scripts/training/train_rlhf.py` runs unchanged, on the components of
`init_components`, with the recorded config and map binding. `ReplayEnv` gives it the recorded
frames, sounds, rewards and step facts in order. Wrappers on the live objects read each step and
apply the one change of an arm. Differences from a live run, all of them chosen and stated here.
  1. Open loop. The environment ignores the action, so a change cannot alter the next frame.
     In these runs the policy output was never applied, so this removes no recorded path.
  2. Frozen weights. No optimizer runs, `update_policy` only clears its buffer, and the loop runs
     under no_grad so the sound predictor does not learn. `replay_arm` raises if a weight changes.
  3. The recorded action goes to the loop. The policy still runs and its output is read.
  4. The motor map is not wired, because the recorded action replaces its command.
  5. The vision and audio bid statistics and the ignition statistics start at the values of the
     live run, computed from the recorded series of all earlier episodes.
  6. Common noise. The torch stream is seeded with base + step before the tectum and before the
     policy at every step, so two arms draw the same noise however much each consumes in between.
  7. The memory list starts empty, with retrieval off as in the run.
  8. pyphi runs as in the loop, and a repeated TPM and state take its earlier output (`remember_phi`).
Modes. Training mode is the mode of the live runs, the RSSM samples, and it decides the verdicts.
Evaluation mode, where the RSSM takes its mean, is reported beside it, with the arms of EVAL_ARMS.
Steps 0 to 29 are left out of every measure (WARM_IN), because the oscillators of the map binding
start from zero.

ARMS, each compared with `base` at the same step and the same noise.
  replica       base again. Control V0.
  noise         base with another noise stream. Its change is the sampling noise, the floor.
  zero_h        h_state zeroed before every step, z_state kept.
  zero_hz       h_state and z_state zeroed before every step. No state crosses a step.
  capsule_off   the top down iterations of the capsule hierarchy set to 0.
  settle_off    one settle cycle per step.
  feedback_off  settle cycles run, and `receive_broadcast` returns the bid unchanged.
  history_swap  at steps 40, 60, ... 180, h_state and z_state are replaced by those of the base
                replay of the other episode at the same step (another layout and decoy level).
                The frames after it are unchanged. Read at lags 0 to 19 after the swap.
  audio_history the sound of the step before each of those steps comes from the other episode.
                Read at the step after the swapped sound.
  frame_other   all frames from the other episode, sound unchanged.
  wave_other    all sounds from the other episode, frames unchanged.
  battery_low   the battery reads 0.2 at every step, against 0.8 to 1.0 in the record.
  bc_random     at ignited steps from 30, every 10 steps, the broadcast tensor is replaced in every
                settle cycle by a random direction of the same size. Read at lags 0 to 9.
  bc_other      the same steps, the broadcast tensor is replaced by the vector of the other sense.

MEASURES. For vectors, |base - arm| / |base| at each step. For bids and the policy mean, the
absolute change (for the policy the largest of its dimensions). Medians over the steps of the arm,
pooled over both episodes of a seed. LABELS of a change against the floor at the same steps.
  absent        at or below TOLERANCE, 1e-6.
  below_noise   above TOLERANCE and below the median change of the `noise` arm.
  above_noise   at or above that floor. In evaluation mode the floor is 0, so any path is above it.
The policy floor is its exploration noise at those steps, 0.1 * (1 + max(0.1, 2 * arousal)).

CONTROLS.
  V0. `replica` changes nothing above TOLERANCE at any step in any reading. If not, VOID.
  V1. In training mode the median cosine of replay and record is at least 0.95 for `audio_content`
      and `obs_map` (episode 39, the only one with maps) and at least 0.90 for `tectum_content`,
      at every seed. If not, VOID for that seed. Agreement of winner and ignition with the record
      is reported only, because the arms are compared with the replay.
  V2. For GWT-3, `gate` changes above TOLERANCE at lag 0 in both broadcast arms at every seed. The
      gate reads the broadcast, so this shows that the perturbation reached a reader. If not, VOID.

TESTS AND PASS RULES. Training mode, per seed. A claim is PASS when it holds at 3 of 3 seeds,
FAILED when it fails at 2 or more, UNDECIDED at 1 (`claim_verdict`).
  RPT-1, the claim for the tectum. Both hold (`rpt1_holds`).
    R1. `zero_h`, `vision_content` is above_noise. Zeroing can move an offset, so R1 alone is not
        enough, and the next line uses states that the module itself produced.
    R2. `history_swap`, `vision_content` at lag 0 is above_noise. The last lag where it is
        above_noise is reported as the reach of the history.
  RPT-1, other claims of the rubric row, each reported with its own verdict.
    R2a. Audio. `audio_history`, `audio_content` is above_noise. The bid is reported beside it.
    R3.  Settle cycles. `settle_off` or `feedback_off` has `vision_content` or `audio_content`
         above_noise. The share of ignited steps with another broadcast or winner is reported.
    R4.  Capsule feedback. `capsule_off`, `vision_content` is above_noise.
  GWT-1, the indicator text. At least 2 modules count and are specialised (`gwt1_specialised`). A
    module counts when its output follows its own input over the inputs of this configuration.
    Vision. `frame_other`, content or raw bid above_noise.
    Audio. `wave_other`, content or raw bid above_noise.
    Memory. Its input is the previous broadcast. `frame_other` or `bc_random` at lag 1, raw bid
            above_noise.
    Body. Its input is the energy. The raw bid has more than 1 distinct value in the record.
          `battery_low`, which is outside the recorded range, is reported beside it and does not
          count.
    Semantic. The raw bid has more than 1 distinct value in the record, and the pathway ran.
    Specialised. The content change is larger under the module's own input than under the input of
    the other sense.
  GWT-1, the row text, "vision, audio, memory, body and semantic". All 5 modules count.
  GWT-3. Every module that counts for GWT-1 responds at the next step, and the policy responds.
    Module. In `bc_random` or `bc_other`, content or raw bid at lag 1 is above_noise.
    Policy. In `bc_random` or `bc_other`, the policy mean at lag 0 or lag 1 is at or above its
            exploration noise.
    The change of the bids in the same step (lag 0, `bid_final`), which is the feedback path of
    `reentrant_processor.py`, and the calls of each `receive_broadcast`, are reported beside it.

Limits, stated before the run. Three seeds and one configuration, the default merge is not tested.
Weights of the end of episode 39 also drive episode 38. Open loop, so a path through the room is
not measured. A statement about these rows and no claim about consciousness.

ADDITION, 2026-10-31. Written after the verdicts of the first group were read, and before any replay
of the second group. No rule above changes, and the first group keeps its verdicts.
  Why. `docs/results/content_map_sensitivity_2026_11.md`, merged after the rules above were pushed,
  shows that the first group stayed near its starting weights, with a content equal to the bias of
  the projection. So a content test at those weights says little about any one input. The same
  document shows a second regime in runs with trained weights.
  Second group. `runs/vg_learn_s601`, `_s602`, `_s603`, the learning arm of the value gate. The same
  arms, episodes 38 and 39, weights of episode 39, rules and controls. Each group gets its own
  verdicts, and no number is pooled over groups. These runs use the default merge and two bound
  bands (`map_settings`). Their action came from the motor map and the value gate, and the replay
  feeds the recorded action as it does for the first group.
  Upstream readings, reported only. No verdict uses them. In `base`, `noise`, `replica`,
  `history_swap` and `frame_other`, the relative change of `obs_map`, the posterior mean and
  `h_state` at the steps of the history lags. They show how far a change reaches inside the vision
  module before the capsules. The floor is `noise` in training mode and `replica` in evaluation mode.
  The first group runs these arms again at this revision (`--arms-train`, `--arms-eval`). Its `base`
  readings are compared with the earlier run, and any difference is reported.
  Read from the second group before this addition. Its flags, its settle cycles, winners and actions
  per episode, the distinct bids in `metrics.csv`, and the tables of the document named above.
  My expectation, which is no rule. A perturbed broadcast leaves the next content and raw bid of every
  module unchanged in this group too, because the code path is the same. The labels of R1, R2 and of
  the vision module in GWT-1 can differ between the groups and between seeds, because the document
  above finds differences of 1000 times between seeds.

Usage:
    python -m scripts.analysis.probe_rubric_recheck --runs runs/ta_nolink_s801 runs/ta_nolink_s802 \
        runs/ta_nolink_s803 --output-json runs/_audit_followup/rubric_recheck/result.json \
        --arrays-dir runs/_audit_followup/rubric_recheck
    python -m scripts.analysis.probe_rubric_recheck --runs runs/ta_nolink_s801 --controls-only
    python -m scripts.analysis.probe_rubric_recheck --runs runs/vg_learn_s601 runs/vg_learn_s602 \
        runs/vg_learn_s603 --output-json runs/_audit_followup/rubric_recheck/learning/result.json \
        --arrays-dir runs/_audit_followup/rubric_recheck/learning
    python -m scripts.analysis.probe_rubric_recheck --runs runs/ta_nolink_s801 runs/ta_nolink_s802 \
        runs/ta_nolink_s803 --arms-train base,replica,noise,history_swap,frame_other \
        --arms-eval base,replica,history_swap,frame_other \
        --output-json runs/_audit_followup/rubric_recheck/covert_upstream/result.json \
        --arrays-dir runs/_audit_followup/rubric_recheck/covert_upstream
    python -m scripts.analysis.probe_rubric_recheck --count-arrays runs/_audit_followup/rubric_recheck \
        runs/_audit_followup/rubric_recheck/learning \
        --output-json runs/_audit_followup/rubric_recheck/exact_counts.json
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.core.semantic_pathway import SemanticPathway  # noqa: E402
from models.self_model.embodiment_core import ProprioceptiveProcessor  # noqa: E402
from scripts.analysis.run_unit_feedback import extract_tensor  # noqa: E402
from scripts.training.train_rlhf import (  # noqa: E402
    _set_global_seed,
    _wire_map_binding,
    init_components,
    run_episode,
)

ABSENT, BELOW, ABOVE = "absent", "below_noise", "above_noise"
MODULES = ("vision", "audio", "memory", "body", "semantic")
BID_STAGES = ("raw", "modulated", "bound", "final")
VECTORS = ("vision_content", "audio_content", "semantic_content", "broadcast", "gate")
SHARES = ("winner", "ignited", "broadcast_changed", "cycles")
EPISODES = (38, 39)
WEIGHT_EPISODE = 39
WARM_IN = 30
HISTORY_FROM, HISTORY_EVERY = 40, 20
HISTORY_LAGS = (0, 1, 2, 3, 5, 10, 19)
BROADCAST_FROM, BROADCAST_EVERY = 30, 10
BROADCAST_LAGS = (0, 1, 2, 5, 9)
TOLERANCE = 1e-6
AUDIO_COSINE, OBS_MAP_COSINE, CONTENT_COSINE = 0.95, 0.95, 0.90
BATTERY_LOW = 0.2
ROOM_RESET_BATTERY = 1.0
NOISE_BASE, OTHER_NOISE_BASE, POLICY_OFFSET = 1_000_000, 2_000_000, 500_000
PERTURBATION_SEED = 3_000_000


@dataclass(frozen=True)
class ArmSpec:
    """The one change of an arm. The defaults are the base replay."""
    noise_base: int = NOISE_BASE
    carry: str = "both"                       # "both", "z" (h zeroed) or "none" (h and z zeroed)
    capsule_iterations: Optional[int] = None
    max_cycles: Optional[int] = None
    feedback: bool = True
    history_from_other: bool = False
    frames_from_other: bool = False
    waves_from_other: bool = False
    swap_wave_before_schedule: bool = False
    battery: Optional[float] = None
    broadcast: Optional[str] = None           # "random" or "other"
    keep_states: bool = False


ARMS = {
    "base": ArmSpec(keep_states=True),
    "replica": ArmSpec(),
    "noise": ArmSpec(noise_base=OTHER_NOISE_BASE),
    "zero_h": ArmSpec(carry="z"),
    "zero_hz": ArmSpec(carry="none"),
    "capsule_off": ArmSpec(capsule_iterations=0),
    "settle_off": ArmSpec(max_cycles=1),
    "feedback_off": ArmSpec(feedback=False),
    "history_swap": ArmSpec(history_from_other=True),
    "audio_history": ArmSpec(swap_wave_before_schedule=True),
    "frame_other": ArmSpec(frames_from_other=True),
    "wave_other": ArmSpec(waves_from_other=True),
    "battery_low": ArmSpec(battery=BATTERY_LOW),
    "bc_random": ArmSpec(broadcast="random"),
    "bc_other": ArmSpec(broadcast="other"),
}
HISTORY_ARMS = ("history_swap", "audio_history")
BROADCAST_ARMS = ("bc_random", "bc_other")
WINDOW_ARMS = tuple(name for name in ARMS if name not in HISTORY_ARMS + BROADCAST_ARMS + ("base",))
CONTROL_ARMS = ("base", "replica")
# Evaluation mode is reported beside the verdicts and decides none. The RSSM takes its mean there,
# so `replica` gives the floor, and these arms cover each rule.
EVAL_ARMS = ("base", "replica", "zero_h", "capsule_off", "settle_off", "history_swap", "frame_other",
             "wave_other", "bc_random", "bc_other")
# The upstream readings of the addition of 2026-10-31. Reported only.
UPSTREAM_ARMS = ("noise", "replica", "history_swap", "frame_other")
UPSTREAM_TAPS = ("obs_map", "posterior", "h_state")
PHI_MEMO: Dict[tuple, tuple] = {}


def relative_change(base: np.ndarray, other: np.ndarray) -> float:
    """|base - other| / |base|. 0 for two zero vectors, infinite from a zero base to a vector."""
    distance = float(np.linalg.norm(np.asarray(base, dtype=np.float64) - np.asarray(other, dtype=np.float64)))
    if distance == 0.0:
        return 0.0
    size = float(np.linalg.norm(np.asarray(base, dtype=np.float64)))
    return distance / size if size > 0.0 else float("inf")


def label_effect(effect: float, floor: float) -> str:
    """absent, below_noise or above_noise. A number that is not a number is an error, never a label."""
    if np.isnan(effect) or np.isnan(floor):
        raise ValueError("cannot label effect %r against floor %r" % (effect, floor))
    if effect <= TOLERANCE:
        return ABSENT
    return ABOVE if effect >= floor else BELOW


def random_same_norm(tensor: torch.Tensor, seed: int) -> torch.Tensor:
    """A random direction with the size of `tensor`. Its own generator, so no stream moves."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction = torch.randn(tensor.numel(), generator=generator, dtype=torch.float64)
    scaled = direction / direction.norm() * float(tensor.detach().norm())
    return scaled.view_as(tensor).to(tensor.device, dtype=tensor.dtype)


def exploration_std(arousal: float) -> float:
    """The noise of the policy, `select_action` in models/self_model/action_selection_core.py."""
    return 0.1 * (1.0 + max(0.1, arousal * 2.0))


def schedule_steps(start: int, every: int, length: int) -> List[int]:
    return list(range(start, length, every))


def lag_steps(steps: Sequence[int], lag: int, length: int) -> List[int]:
    return [step + lag for step in steps if step + lag < length]


def upstream_steps(length: int) -> List[int]:
    """Every step that a history lag reads, for the upstream readings."""
    schedule = schedule_steps(HISTORY_FROM, HISTORY_EVERY, length)
    return sorted({step for lag in HISTORY_LAGS for step in lag_steps(schedule, lag, length)})


def map_settings(argv: Sequence[str]) -> tuple:
    """The map binding of a recorded run and its bound bands, as `main` of the training loop sets them."""
    mode = argv[argv.index("--map-binding") + 1]
    orienting = argv[argv.index("--tectal-orienting") + 1] if "--tectal-orienting" in argv else "off"
    return mode, 2 if orienting == "value" else 1


def claim_verdict(holds: Sequence[bool]) -> str:
    """PASS at 3 of 3 seeds, FAILED at 2 or more failing seeds, UNDECIDED at 1."""
    if len(holds) != 3:
        raise ValueError("a verdict needs 3 seeds, got %d" % len(holds))
    failing = sum(1 for held in holds if not held)
    if failing == 0:
        return "PASS"
    return "FAILED" if failing >= 2 else "UNDECIDED"


def rpt1_holds(carry_label: str, history_label: str) -> bool:
    return carry_label == ABOVE and history_label == ABOVE


def gwt1_module_counts(labels: Dict[str, Sequence[str]]) -> Dict[str, bool]:
    """A module counts when any of its labels is above_noise."""
    return {module: ABOVE in found for module, found in labels.items()}


def gwt1_specialised(counts: Dict[str, bool], own: Dict[str, float], cross: Dict[str, float]) -> bool:
    """At least 2 counted modules whose content follows their own input more than the other input."""
    specialised = [module for module, counted in counts.items()
                   if counted and module in own and own[module] > cross[module]]
    return len(specialised) >= 2


def gwt3_module_holds(labels: Sequence[str]) -> bool:
    return ABOVE in labels


def gwt3_policy_holds(labels: Sequence[str]) -> bool:
    return ABOVE in labels


@dataclass
class EpisodeRecord:
    """What the replay reads from one recorded episode."""
    index: int
    frames: np.ndarray
    waveforms: Dict[int, np.ndarray]
    rewards: List[float]
    facts: List[dict]
    actions: List[list]
    vectors: dict
    obs_map: Optional[np.ndarray]


def read_steps(folder: str) -> List[dict]:
    with open(os.path.join(folder, "steps.jsonl"), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_episode(run: str, index: int) -> EpisodeRecord:
    folder = os.path.join(run, "episodes", "ep_%04d" % index)
    steps = read_steps(folder)
    recorded = np.load(os.path.join(folder, "vectors.npz"))
    maps_path = os.path.join(folder, "maps.npz")
    return EpisodeRecord(
        index=index,
        frames=np.load(os.path.join(folder, "frames.npz"))["frames"],
        waveforms={int(s): w for s, w in zip(recorded["audio_waveform__steps"], recorded["audio_waveform"])},
        rewards=[row["internals"]["rewards"]["env_reward"] for row in steps],
        facts=[row["internals"]["info_after_step"] for row in steps],
        actions=[row["action"] for row in steps],
        vectors={"steps": steps, **{name: recorded[name] for name in ("tectum_content", "audio_content")}},
        obs_map=np.load(maps_path)["obs_map"] if os.path.exists(maps_path) else None)


class ReplayEnv:
    """The recorded frames, sounds and step facts in order. Actions are kept and do nothing else."""

    def __init__(self, frames: np.ndarray, waveforms: Dict[int, np.ndarray], rewards: Sequence[float],
                 facts: Sequence[dict], reset_battery: float = ROOM_RESET_BATTERY):
        self.frames, self.waveforms, self.rewards, self.facts = frames, waveforms, rewards, facts
        self.reset_battery = reset_battery
        self.received_actions: List[np.ndarray] = []
        self.step_index = 0

    def reset(self, **_) -> tuple:
        """The room gives no sound at a reset, so step 0 has no input for the audio module."""
        self.step_index = 0
        return self.frames[0], {"battery": self.reset_battery}

    def step(self, action) -> tuple:
        self.received_actions.append(np.array(action, copy=True))
        done_step = self.step_index
        self.step_index += 1
        ended = self.step_index >= len(self.frames)
        facts = dict(self.facts[done_step])
        sound = self.waveforms.get(self.step_index)
        if sound is not None:
            facts["audio_waveform"] = sound
        frame = self.frames[min(self.step_index, len(self.frames) - 1)]
        return frame, self.rewards[done_step], False, ended, facts

    def close(self) -> None:
        return None


def arm_env(spec: ArmSpec, record: EpisodeRecord, other: EpisodeRecord, schedule: Sequence[int]) -> ReplayEnv:
    """The input streams of one arm."""
    waves = dict(other.waveforms if spec.waves_from_other else record.waveforms)
    if spec.swap_wave_before_schedule:
        for step in schedule:
            if step - 1 in other.waveforms:
                waves[step - 1] = other.waveforms[step - 1]
    facts = record.facts
    if spec.battery is not None:
        facts = [dict(row, battery=spec.battery) for row in record.facts]
    frames = other.frames if spec.frames_from_other else record.frames
    return ReplayEnv(frames, waves, record.rewards, facts,
                     reset_battery=ROOM_RESET_BATTERY if spec.battery is None else spec.battery)


def history_rows(steps: Sequence[dict]) -> List[dict]:
    """The recorded series behind the running statistics, in the order of the run."""
    rows = []
    for row in steps:
        error = row["internals"].get("audio_surprise_error")
        rows.append({"kl": row["internals"]["kl_div_pre_tanh"],
                     "audio_error": error if row["input_source"]["audio"] == "sound" else None,
                     "top_bound_bid": max(row["bound_bids"].values())})
    return rows


def prewarm_statistics(agent, rows: Sequence[dict]) -> None:
    """Move the running statistics of the bids and of the ignition as the live run moved them."""
    normalizer = agent.tectum.bid_normalizer
    surprise = getattr(agent.auditory, "surprise", None)
    for row in rows:
        if normalizer is not None:
            normalizer.bid(row["kl"])
        if surprise is not None and row["audio_error"] is not None:
            surprise.normalizer.bid(row["audio_error"])
        agent.workspace._update_tolerance(row["top_bound_bid"])


@dataclass
class Agent:
    config: dict
    tectum: object
    workspace: object
    reentrant: object
    modulator: object
    memory: object
    action_core: object
    gate: object
    auditory: object
    self_model: object
    rnd: object


def set_mode(agent: Agent, mode: str) -> None:
    training = mode == "train"
    for module in (agent.tectum, agent.auditory, agent.gate, agent.rnd, agent.workspace.binding_system):
        module.train(training)


def build_agent(config: dict, map_binding: str, seed: int, mode: str, bands_to_bind: int = 1) -> Agent:
    """The components of `init_components` and the map binding of the run. Nothing writes a record."""
    _set_global_seed(seed)
    parts = init_components(config)
    agent = Agent(config=config, tectum=parts[0], workspace=parts[1], reentrant=parts[2], modulator=parts[3],
                  memory=parts[5], action_core=parts[6], gate=parts[8], auditory=parts[13],
                  self_model=parts[14], rnd=parts[15])
    if map_binding != "off":
        with mock.patch("atexit.register"):
            _wire_map_binding(agent.tectum, agent.workspace, None, map_binding,
                              auditory_specialist=agent.auditory, seed=seed, bands_to_bind=bands_to_bind)
    set_mode(agent, mode)
    return agent


def load_weights(agent: Agent, weights: dict) -> None:
    """The saved modules, strict. The carried states are not parameters, and the loop resets them."""
    tectum_state = {k: v for k, v in weights["tectum"].items() if k not in ("h_state", "z_state")}
    agent.tectum.load_state_dict(tectum_state, strict=True)
    binding = agent.workspace.binding_system
    binding_state = dict(weights["workspace_binding"])
    phases = binding_state.pop("current_phases", None)
    binding.load_state_dict(binding_state, strict=True)
    if phases is not None:
        # The live loop keeps the binding system on the CPU, whatever the device of the rest.
        binding.current_phases = phases.to(next(binding.parameters()).device)
    agent.gate.load_state_dict(weights["gate"], strict=True)
    agent.rnd.load_state_dict(weights["rnd"], strict=True)
    agent.auditory.load_state_dict(weights["auditory_specialist"], strict=True)
    agent.action_core.emotion_shaper.load_state_dict(weights["action_core.emotion_shaper"], strict=True)
    agent.action_core.pfc.load_state_dict(weights["action_core.pfc"], strict=True)
    agent.action_core.bg.load_state_dict(weights["action_core.bg"], strict=True)


def remember_phi(iit_metrics) -> None:
    """The same TPM and state give the same phi, so a repeated input skips pyphi. The value is pyphi's."""
    calculate = iit_metrics.calculate_phi

    def remembered(tpm, current_state, cm=None):
        key = (tpm.tobytes(), tuple(current_state))
        if key not in PHI_MEMO:
            PHI_MEMO[key] = (calculate(tpm, current_state, cm=cm), iit_metrics.last_phi_error)
        phi, iit_metrics.last_phi_error = PHI_MEMO[key]
        return phi
    iit_metrics.calculate_phi = remembered


def weights_fingerprint(agent: Agent) -> float:
    """The summed size of the weights a replay could move. Compared exactly before and after."""
    modules = (agent.tectum, agent.auditory, agent.gate, agent.rnd, agent.action_core.pfc,
               agent.action_core.bg, agent.workspace.binding_system)
    return float(sum(parameter.detach().double().abs().sum().item()
                     for module in modules for parameter in module.parameters()))


@contextlib.contextmanager
def count_class_calls(calls: collections.Counter):
    """Counts calls of the modules that the rubric names and the loop may never build."""
    targets = ((SemanticPathway, "forward", "semantic_forward"),
               (SemanticPathway, "receive_broadcast", "semantic_feedback"),
               (ProprioceptiveProcessor, "forward", "body_forward"),
               (ProprioceptiveProcessor, "receive_broadcast", "body_feedback"))
    originals = [(owner, name, getattr(owner, name)) for owner, name, _ in targets]

    def counted(original: Callable, key: str) -> Callable:
        def wrapper(*args, **kwargs):
            calls[key] += 1
            return original(*args, **kwargs)
        return wrapper

    try:
        for owner, name, key in targets:
            setattr(owner, name, counted(getattr(owner, name), key))
            calls[key] += 0
        yield
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)


def as_vector(tensor: Optional[torch.Tensor], size: int) -> np.ndarray:
    if tensor is None:
        return np.zeros(size, dtype=np.float32)
    return tensor.detach().cpu().numpy().ravel().astype(np.float32)


def bid_vector(bids: dict) -> np.ndarray:
    return np.array([float(bids.get(module, 0.0)) for module in MODULES], dtype=np.float64)


class Tap:
    """Wraps the live objects of one replay. It reads every step and applies the change of the arm."""

    def __init__(self, agent: Agent, spec: ArmSpec, record: EpisodeRecord, schedule: Sequence[int],
                 other_states: Optional[dict], tap_steps: Sequence[int] = (), tap_reference: Optional[dict] = None):
        self.agent, self.spec, self.record = agent, spec, record
        self.schedule, self.other_states = set(schedule), other_states
        self.tap_steps, self.tap_reference = set(tap_steps), tap_reference
        self.kept_taps: dict = {}
        self.tap_changes: Dict[str, dict] = {name: {} for name in UPSTREAM_TAPS}
        self.step = -1
        self.rows: List[dict] = []
        self.row: dict = {}
        self.kept_states: dict = {}
        self.injected: set = set()
        self.calls: collections.Counter = collections.Counter()
        self.memory_error: Optional[str] = None
        self.workspace_dim = agent.config["workspace_dim"]

    def install(self) -> None:
        self._apply_settings()
        self._wrap_tectum()
        self._wrap_audio()
        self._wrap_competition()
        self._wrap_settle()
        self._wrap_feedback()
        self._wrap_policy()
        self._wrap_memory()

    def _apply_settings(self) -> None:
        if self.spec.capsule_iterations is not None:
            self.agent.tectum.capsule_layer.reentrant_iterations = self.spec.capsule_iterations
        if self.spec.max_cycles is not None:
            self.agent.reentrant.max_cycles = self.spec.max_cycles
        core = self.agent.action_core
        core.update_policy = core.rollout_buffer.clear
        remember_phi(self.agent.workspace.iit_metrics)

    def _wrap_tectum(self) -> None:
        tectum = self.agent.tectum
        forward = tectum.forward

        def wrapped(*args, **kwargs):
            self.step += 1
            self.calls["vision_forward"] += 1
            self._prepare_state()
            torch.manual_seed(self.spec.noise_base + self.step)
            return forward(*args, **kwargs)
        tectum.forward = wrapped

    def _wrap_audio(self) -> None:
        auditory = self.agent.auditory
        forward = auditory.forward

        def wrapped(*args, **kwargs):
            self.calls["audio_forward"] += 1
            return forward(*args, **kwargs)
        auditory.forward = wrapped

    def _prepare_state(self) -> None:
        tectum = self.agent.tectum
        if tectum.h_state is None:
            return
        if self.spec.keep_states and self.step in self.schedule:
            self.kept_states[self.step] = (tectum.h_state.detach().clone(), tectum.z_state.detach().clone())
        if self.spec.history_from_other and self.step in self.schedule:
            tectum.h_state, tectum.z_state = self.other_states[self.step]
        if self.spec.carry != "both":
            tectum.h_state = torch.zeros_like(tectum.h_state)
        if self.spec.carry == "none":
            tectum.z_state = torch.zeros_like(tectum.z_state)

    def _wrap_competition(self) -> None:
        if self.spec.broadcast is None:
            return
        compete = self.agent.workspace.run_competition

        def wrapped(*args, **kwargs):
            content, bids = compete(*args, **kwargs)
            if self.step in self.schedule:
                content = self._perturbed(content, kwargs["payloads"])
            return content, bids
        self.agent.workspace.run_competition = wrapped

    def _perturbed(self, content: dict, payloads: dict) -> dict:
        tensor = extract_tensor(content)
        if tensor is None:
            return content
        self.injected.add(self.step)
        if self.spec.broadcast == "random":
            return dict(content, tensor=random_same_norm(tensor, PERTURBATION_SEED + self.step))
        carrier = "audio" if content.get("source") == "tectum" else "vision"
        return dict(content, tensor=payloads[carrier]["tensor"].detach())

    def _wrap_settle(self) -> None:
        settle = self.agent.reentrant.settle

        def wrapped(**kwargs):
            payloads = kwargs["payloads"]
            self.row = {"bid_raw": bid_vector(kwargs["initial_bids"]),
                        "vision_content": as_vector(payloads["vision"]["tensor"], self.workspace_dim),
                        "audio_content": as_vector(payloads["audio"]["tensor"], self.workspace_dim),
                        "semantic_content": as_vector(payloads["semantic"]["tensor"], self.workspace_dim)}
            outcome = settle(**kwargs)
            self._read_outcome(outcome)
            return outcome
        self.agent.reentrant.settle = wrapped

    def _read_outcome(self, outcome) -> None:
        workspace = self.agent.workspace
        winners = list(workspace.state.winners or [])
        self.row.update(bid_final=bid_vector(outcome.final_bids),
                        bid_modulated=bid_vector(workspace.last_modulated_bids),
                        bid_bound=bid_vector(workspace.state.competition_results),
                        winner=winners[0] if winners else "",
                        ignited=bool(outcome.is_conscious), cycles=int(outcome.cycles_used),
                        broadcast=as_vector(extract_tensor(outcome.broadcast_content), self.workspace_dim))

    def _wrap_feedback(self) -> None:
        for name, module in (("vision", self.agent.tectum), ("audio", self.agent.auditory)):
            module.receive_broadcast = self._counted_rule(name, module.receive_broadcast)

    def _counted_rule(self, name: str, rule: Callable) -> Callable:
        self.calls[name + "_feedback"] += 0

        def wrapped(content, bid):
            self.calls[name + "_feedback"] += 1
            return rule(content, bid) if self.spec.feedback else bid
        return wrapped

    def _wrap_policy(self) -> None:
        core = self.agent.action_core
        select = core.select_action

        def wrapped(policy_state, **kwargs):
            captured = {}
            hook = core.bg.register_forward_hook(
                lambda _module, _inputs, output: captured.update(mean=output[0], value=output[1]))
            torch.manual_seed(self.spec.noise_base + POLICY_OFFSET + self.step)
            try:
                _, estimate = select(policy_state, **kwargs)
            finally:
                hook.remove()
            self._finish_row(captured["mean"], kwargs["emotion_arousal"])
            return np.asarray(self.record.actions[self.step], dtype=np.float32), estimate
        core.select_action = wrapped

    def _finish_row(self, action_mean: torch.Tensor, arousal: float) -> None:
        self.row.update(action_mean=action_mean.detach().cpu().numpy().ravel().astype(np.float64),
                        arousal=float(arousal),
                        gate=as_vector(self.agent.gate.last_gate_values_tensor, 5),
                        obs_map_cosine=self._obs_map_cosine())
        self.rows.append(self.row)
        if self.step in self.tap_steps:
            self._read_taps()

    def _read_taps(self) -> None:
        """The upstream readings. Nothing branches on them."""
        tectum = self.agent.tectum
        found = {"obs_map": tectum._last_obs_map, "posterior": tectum._last_post_logits, "h_state": tectum.h_state}
        found = {name: as_vector(tensor, 0) for name, tensor in found.items()}
        if self.tap_reference is None:
            self.kept_taps[self.step] = found
            return
        for name, reading in found.items():
            self.tap_changes[name][self.step] = relative_change(self.tap_reference[self.step][name], reading)

    def _obs_map_cosine(self) -> float:
        if not self.spec.keep_states or self.record.obs_map is None:
            return float("nan")
        replayed = self.agent.tectum._last_obs_map.detach().cpu().numpy().ravel()
        return cosine(replayed, self.record.obs_map[self.step].ravel())

    def _wrap_memory(self) -> None:
        memory = self.agent.memory
        search = memory.get_similar_experiences

        def wrapped(*args, **kwargs):
            self.calls["memory_queries"] += 1
            try:
                found = search(*args, **kwargs)
            except Exception as caught:
                self.calls["memory_query_errors"] += 1
                self.memory_error = "%s: %s" % (type(caught).__name__, caught)
                raise
            self.calls["memory_positive_scores"] += sum(1 for match in found if match.get("score", 0.0) > 0.0)
            return found
        memory.get_similar_experiences = wrapped

    def readings(self) -> dict:
        names = VECTORS + ("bid_raw", "bid_modulated", "bid_bound", "bid_final", "action_mean", "arousal",
                           "winner", "ignited", "cycles", "obs_map_cosine")
        found = {name: np.array([row[name] for row in self.rows]) for name in names}
        found.update(injected=sorted(self.injected), calls=dict(self.calls), kept_states=self.kept_states,
                     memory_error=self.memory_error, kept_taps=self.kept_taps)
        for name, changes in self.tap_changes.items():
            if changes:
                found["tap_" + name] = self._tap_array(changes)
        return found

    def _tap_array(self, changes: Dict[int, float]) -> np.ndarray:
        array = np.full(len(self.rows), np.nan)
        for step, change in changes.items():
            array[step] = change
        return array


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    first, second = np.asarray(first, dtype=np.float64).ravel(), np.asarray(second, dtype=np.float64).ravel()
    size = float(np.linalg.norm(first) * np.linalg.norm(second))
    if size == 0.0:
        raise ValueError("the cosine of a zero vector is not defined")
    return float(np.dot(first, second) / size)


def replay_arm(make_agent: Callable[[], Agent], spec: ArmSpec, record: EpisodeRecord, other: EpisodeRecord,
               schedule: Sequence[int], other_states: Optional[dict] = None, history: Sequence[dict] = (),
               tap_steps: Sequence[int] = (), tap_reference: Optional[dict] = None) -> dict:
    """One episode through the real loop with the change of one arm. Raises if a weight moved."""
    agent = make_agent()
    prewarm_statistics(agent, history)
    env = arm_env(spec, record, other, schedule)
    tap = Tap(agent, spec, record, schedule, other_states, tap_steps, tap_reference)
    tap.install()
    before = weights_fingerprint(agent)
    with torch.no_grad(), count_class_calls(tap.calls):
        run_episode(record.index, agent.config, agent.tectum, agent.workspace, agent.reentrant, agent.modulator,
                    agent.action_core, env, gate=agent.gate, memory=agent.memory,
                    auditory_specialist=agent.auditory, self_model=agent.self_model, rnd=agent.rnd)
    if weights_fingerprint(agent) != before:
        raise RuntimeError("a weight changed during the replay, which must be read-only")
    if len(tap.rows) != len(record.frames):
        raise RuntimeError("the loop ran %d steps of %d" % (len(tap.rows), len(record.frames)))
    found = tap.readings()
    found["received_actions"] = env.received_actions
    return found


def step_changes(base: dict, other: dict) -> Dict[str, np.ndarray]:
    """Per step change of every reading between two replays of one episode."""
    changes = {name: np.array([relative_change(a, b) for a, b in zip(base[name], other[name])]) for name in VECTORS}
    changes["broadcast_changed"] = (changes["broadcast"] > TOLERANCE).astype(np.float64)
    for stage in BID_STAGES:
        difference = np.abs(base["bid_" + stage] - other["bid_" + stage])
        for index, module in enumerate(MODULES):
            changes["bid_%s_%s" % (stage, module)] = difference[:, index]
    changes["action_mean"] = np.abs(base["action_mean"] - other["action_mean"]).max(axis=1)
    changes["winner"] = (base["winner"] != other["winner"]).astype(np.float64)
    changes["ignited"] = (base["ignited"] != other["ignited"]).astype(np.float64)
    if "cycles" in base:
        changes["cycles"] = (base["cycles"] != other["cycles"]).astype(np.float64)
    return changes


def pooled(changes: dict, quantity: str, steps: Dict[int, Sequence[int]]) -> np.ndarray:
    return np.concatenate([changes[episode][quantity][list(chosen)] for episode, chosen in steps.items()])


def effect_table(changes: dict, floors: dict, steps: Dict[int, Sequence[int]]) -> dict:
    """quantity -> effect, floor, label at the given steps. Each argument is keyed by episode."""
    if sum(len(chosen) for chosen in steps.values()) == 0:
        raise ValueError("no steps to measure")
    table = {}
    for quantity in changes[next(iter(steps))]:
        own, floor = pooled(changes, quantity, steps), pooled(floors, quantity, steps)
        if quantity in SHARES:
            table[quantity] = {"effect": float(np.mean(own)), "floor": float(np.mean(floor)), "steps": len(own)}
            continue
        effect_value, floor_value = float(np.median(own)), float(np.median(floor))
        table[quantity] = {"effect": effect_value, "floor": floor_value,
                           "label": label_effect(effect_value, floor_value), "steps": len(own)}
    return table


def policy_table(changes: dict, arousal: Dict[int, np.ndarray], steps: Dict[int, Sequence[int]]) -> dict:
    """The change of the policy mean against its exploration noise at the same steps."""
    own = pooled(changes, "action_mean", steps)
    noise = np.concatenate([[exploration_std(float(arousal[episode][t])) for t in chosen]
                            for episode, chosen in steps.items()])
    effect_value, floor_value = float(np.median(own)), float(np.median(noise))
    return {"effect": effect_value, "floor": floor_value, "label": label_effect(effect_value, floor_value)}


def window_steps(length: int) -> List[int]:
    return list(range(WARM_IN, length))


def run_config(run: str, device: str) -> tuple:
    """The recorded config and map binding of a run, on the chosen device."""
    with open(os.path.join(run, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    with open(os.path.join(run, "session.json"), encoding="utf-8") as handle:
        argv = json.load(handle)["run"]["argv"]
    config["device"] = device
    config["action_selection"]["device"] = device
    if not config.get("ablate_gate_diversity", False):
        raise ValueError("this probe assumes runs without the gate diversity loss, %s has it" % run)
    mode, bands_to_bind = map_settings(argv)
    return config, mode, int(config["seed"]), bands_to_bind


def run_history(run: str, before: int) -> List[dict]:
    rows = []
    for index in range(before):
        rows.extend(history_rows(read_steps(os.path.join(run, "episodes", "ep_%04d" % index))))
    return rows


def replay_seed(run: str, mode: str, device: str, arms: Sequence[str]) -> tuple:
    """Every arm on both episodes of one run in one mode. Serial, one replay at a time."""
    config, map_binding, seed, bands_to_bind = run_config(run, device)
    path = os.path.join(run, "episodes", "ep_%04d" % WEIGHT_EPISODE, "weights.pt")
    weights = torch.load(path, map_location="cpu", weights_only=False)

    def make_agent() -> Agent:
        agent = build_agent(config, map_binding, seed, mode, bands_to_bind)
        load_weights(agent, weights)
        return agent

    records = {episode: load_episode(run, episode) for episode in EPISODES}
    histories = {episode: run_history(run, episode) for episode in EPISODES}
    history_schedule = schedule_steps(HISTORY_FROM, HISTORY_EVERY, len(records[EPISODES[0]].frames))
    broadcast_schedule = schedule_steps(BROADCAST_FROM, BROADCAST_EVERY, len(records[EPISODES[0]].frames))
    replays: Dict[str, dict] = {}
    for name in arms:
        spec = ARMS[name]
        replays[name] = {}
        schedule = broadcast_schedule if spec.broadcast else history_schedule
        for episode in EPISODES:
            other_episode = next(e for e in EPISODES if e != episode)
            states = replays["base"][other_episode]["kept_states"] if spec.history_from_other else None
            tap_reference = replays["base"][episode]["kept_taps"] if name in UPSTREAM_ARMS else None
            tap_steps = upstream_steps(len(records[episode].frames)) if name == "base" or name in UPSTREAM_ARMS else ()
            started = time.time()
            replays[name][episode] = replay_arm(make_agent, spec, records[episode], records[other_episode],
                                                schedule, other_states=states, history=histories[episode],
                                                tap_steps=tap_steps, tap_reference=tap_reference)
            print("%s %s %s episode %d, %.0f s" % (seed, mode, name, episode, time.time() - started), flush=True)
    return replays, records


def median_cosine(replayed: np.ndarray, recorded: np.ndarray, steps: Sequence[int]) -> float:
    return float(np.median([cosine(replayed[t], recorded[t]) for t in steps]))


def median_bid_change(replayed: np.ndarray, rows: Sequence[dict], module: str, steps: Sequence[int]) -> float:
    index = MODULES.index(module)
    return float(np.median([abs(replayed[t, index] - rows[t]["raw_bids"][module]) for t in steps]))


def reproduction(base: Dict[int, dict], records: Dict[int, EpisodeRecord]) -> dict:
    """Replay against record, over the window. Cosines for V1, the other numbers are reported only."""
    report = {}
    for episode, record in records.items():
        steps = window_steps(len(record.frames))
        replayed, rows, recorded = base[episode], record.vectors["steps"], record.vectors["tectum_content"]
        report[episode] = {
            "audio_content": median_cosine(replayed["audio_content"], record.vectors["audio_content"], steps),
            "tectum_content": median_cosine(replayed["vision_content"], recorded, steps),
            "tectum_relative_change": float(np.median([relative_change(recorded[t].ravel(), replayed["vision_content"][t])
                                                       for t in steps])),
            "bid_vision_change": median_bid_change(replayed["bid_raw"], rows, "vision", steps),
            "bid_audio_change": median_bid_change(replayed["bid_raw"], rows, "audio", steps),
            "winner_agreement": float(np.mean([replayed["winner"][t] == rows[t]["winner"] for t in steps])),
            "ignition_agreement": float(np.mean([bool(replayed["ignited"][t]) == bool(rows[t]["ignited"])
                                                 for t in steps])),
        }
        if record.obs_map is not None:
            report[episode]["obs_map"] = float(np.median(replayed["obs_map_cosine"][steps]))
    return report


def reproduction_holds(report: dict) -> bool:
    """V1, applied in training mode."""
    for row in report.values():
        if row["audio_content"] < AUDIO_COSINE or row["tectum_content"] < CONTENT_COSINE:
            return False
        if "obs_map" in row and row["obs_map"] < OBS_MAP_COSINE:
            return False
    return True


def largest_change(changes: Dict[str, np.ndarray]) -> float:
    return float(max(np.max(change) for change in changes.values()))


def arm_tables(replays: Dict[str, dict]) -> dict:
    """Every comparison with the base replay, as the rules above state them."""
    base = replays["base"]
    changes = {name: {e: step_changes(base[e], replays[name][e]) for e in base} for name in replays if name != "base"}
    arousal = {e: base[e]["arousal"] for e in base}
    ignited = {e: base[e]["ignited"] for e in base}
    length = len(next(iter(base.values()))["winner"])
    tables: dict = {}
    for name in changes:
        floors = changes["noise"] if "noise" in changes else changes["replica"]
        if name in WINDOW_ARMS:
            tables[name] = window_table(changes[name], floors, arousal, length)
            tables[name]["ignited_shares"] = ignited_shares(changes[name], ignited, length)
        elif name == "history_swap":
            schedule = schedule_steps(HISTORY_FROM, HISTORY_EVERY, length)
            tables[name] = lag_table(changes[name], floors, arousal, {e: schedule for e in base}, HISTORY_LAGS, length)
        elif name == "audio_history":
            schedule = schedule_steps(HISTORY_FROM, HISTORY_EVERY, length)
            tables[name] = lag_table(changes[name], floors, arousal, {e: schedule for e in base}, (0,), length)
        else:
            injected = {e: replays[name][e]["injected"] for e in base}
            tables[name] = lag_table(changes[name], floors, arousal, injected, BROADCAST_LAGS, length)
    return tables


def window_table(changes: dict, floors: dict, arousal: dict, length: int) -> dict:
    steps = {episode: window_steps(length) for episode in changes}
    by_episode = {episode: effect_table(changes, floors, {episode: steps[episode]}) for episode in changes}
    return {"window": effect_table(changes, floors, steps), "policy": policy_table(changes, arousal, steps),
            "by_episode": by_episode}


def ignited_shares(changes: dict, ignited: Dict[int, np.ndarray], length: int) -> dict:
    """Shares of the window steps where the base replay ignited with another broadcast or winner."""
    chosen = {episode: [t for t in window_steps(length) if ignited[episode][t]] for episode in changes}
    if sum(len(steps) for steps in chosen.values()) == 0:
        raise ValueError("the base replay never ignited in the window")
    return {quantity: float(np.mean(pooled(changes, quantity, chosen))) for quantity in ("broadcast_changed", "winner")}


def lag_table(changes: dict, floors: dict, arousal: dict, schedule: dict, lags: Sequence[int], length: int) -> dict:
    tables: dict = {}
    for lag in lags:
        steps = {episode: lag_steps(chosen, lag, length) for episode, chosen in schedule.items()}
        tables[str(lag)] = {"effects": effect_table(changes, floors, steps),
                            "policy": policy_table(changes, arousal, steps)}
    return {"lags": tables, "injected": {str(e): len(chosen) for e, chosen in schedule.items()}}


def upstream_table(replays: Dict[str, dict], length: int) -> dict:
    """Reported only. How far a changed history or a changed frame reaches before the capsules."""
    floors = replays["noise"] if "noise" in replays else replays["replica"]
    schedule = schedule_steps(HISTORY_FROM, HISTORY_EVERY, length)
    groups = {"history_swap, lag %d" % lag: ("history_swap", lag_steps(schedule, lag, length)) for lag in HISTORY_LAGS}
    groups["frame_other"] = ("frame_other", upstream_steps(length))
    table = {}
    for label, (arm, steps) in groups.items():
        row = {}
        for tap in UPSTREAM_TAPS:
            effect = float(np.median(np.concatenate([replays[arm][e]["tap_" + tap][steps] for e in replays[arm]])))
            floor = float(np.median(np.concatenate([floors[e]["tap_" + tap][steps] for e in floors])))
            row[tap] = {"effect": effect, "floor": floor, "label": label_effect(effect, floor)}
        table[label] = row
    return table


def history_reach(table: dict) -> Optional[int]:
    """The last lag where the vision content is above the sampling noise, None if no lag is."""
    above = [int(lag) for lag, row in table["lags"].items() if row["effects"]["vision_content"]["label"] == ABOVE]
    return max(above) if above else None


def record_facts(run: str) -> dict:
    """What the record of a whole run says, without any replay."""
    with open(os.path.join(run, "metrics.csv"), newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    facts = {"bid_distinct": {m: len({float(row["bid_" + m]) for row in rows}) for m in MODULES},
             "metric_steps": len(rows)}
    cycles, sources, winners = collections.Counter(), collections.Counter(), collections.Counter()
    moved_action, changed_after_feedback, steps_read = 0, 0, 0
    for folder in sorted(os.listdir(os.path.join(run, "episodes"))):
        for row in read_steps(os.path.join(run, "episodes", folder)):
            settle = row["internals"]["settle"]
            cycles[settle["cycles_used"]] += 1
            sources[row["input_source"]["memory"] + "/" + row["input_source"]["semantic"]] += 1
            winners[row["winner"]] += 1
            moved_action += any(float(part) != 0.0 for part in row["action"])
            errors = settle["prediction_errors"]
            changed_after_feedback += len(errors) > 1 and errors[1] > 0.0
            steps_read += 1
    facts.update(steps_read=steps_read, cycles_used={str(k): v for k, v in sorted(cycles.items())},
                 memory_and_semantic_input=dict(sources), winners=dict(winners), steps_with_action=moved_action,
                 steps_where_feedback_changed_the_broadcast=changed_after_feedback)
    return facts


def rpt1_claims(tables: dict) -> dict:
    swap = tables["history_swap"]["lags"]["0"]["effects"]["vision_content"]["label"]
    carry = tables["zero_h"]["window"]["vision_content"]["label"]
    settle_labels = [tables[arm]["window"][content]["label"] for arm in ("settle_off", "feedback_off")
                     for content in ("vision_content", "audio_content")]
    return {"carry_label": carry, "history_label": swap, "tectum": rpt1_holds(carry, swap),
            "history_reach": history_reach(tables["history_swap"]),
            "audio": tables["audio_history"]["lags"]["0"]["effects"]["audio_content"]["label"] == ABOVE,
            "settle": ABOVE in settle_labels,
            "capsule": tables["capsule_off"]["window"]["vision_content"]["label"] == ABOVE}


def gwt1_claims(tables: dict, facts: dict, base_calls: dict) -> dict:
    distinct = facts["bid_distinct"]
    frame, wave = tables["frame_other"]["window"], tables["wave_other"]["window"]
    labels = {
        "vision": [frame["vision_content"]["label"], frame["bid_raw_vision"]["label"]],
        "audio": [wave["audio_content"]["label"], wave["bid_raw_audio"]["label"]],
        "memory": [frame["bid_raw_memory"]["label"],
                   tables["bc_random"]["lags"]["1"]["effects"]["bid_raw_memory"]["label"]],
        "body": [ABOVE if distinct["body"] > 1 else ABSENT],
        "semantic": [ABOVE if distinct["semantic"] > 1 and base_calls.get("semantic_forward", 0) > 0 else ABSENT],
    }
    counts = gwt1_module_counts(labels)
    own = {"vision": frame["vision_content"]["effect"], "audio": wave["audio_content"]["effect"]}
    cross = {"vision": wave["vision_content"]["effect"], "audio": frame["audio_content"]["effect"]}
    return {"labels": labels, "counts": counts, "count": sum(counts.values()), "own": own, "cross": cross,
            "indicator": gwt1_specialised(counts, own, cross), "row_text": all(counts.values()),
            "body_forced": tables["battery_low"]["window"]["bid_raw_body"]}


def gwt3_claims(tables: dict, counts: Dict[str, bool]) -> dict:
    responses = {}
    for module in ("vision", "audio"):
        labels = [tables[arm]["lags"]["1"]["effects"][quantity]["label"] for arm in BROADCAST_ARMS
                  for quantity in ("%s_content" % module, "bid_raw_%s" % module)]
        responses[module] = gwt3_module_holds(labels)
    for module in ("memory", "body", "semantic"):
        labels = [tables[arm]["lags"]["1"]["effects"]["bid_raw_%s" % module]["label"] for arm in BROADCAST_ARMS]
        responses[module] = gwt3_module_holds(labels)
    policy_labels = [tables[arm]["lags"][lag]["policy"]["label"] for arm in BROADCAST_ARMS for lag in ("0", "1")]
    counted_respond = all(responses[module] for module, counted in counts.items() if counted)
    policy = gwt3_policy_holds(policy_labels)
    return {"responses": responses, "policy": policy, "holds": counted_respond and policy, "policy_labels": policy_labels,
            "gate_reached": all(tables[arm]["lags"]["0"]["effects"]["gate"]["effect"] > TOLERANCE for arm in BROADCAST_ARMS)}


def jsonable(item):
    if isinstance(item, dict):
        return {str(key): jsonable(entry) for key, entry in item.items()}
    if isinstance(item, (list, tuple)):
        return [jsonable(entry) for entry in item]
    if isinstance(item, (np.floating, np.integer, np.bool_)):
        return item.item()
    return item


def save_arrays(folder: str, seed: int, mode: str, replays: Dict[str, dict]) -> None:
    """The per step readings, for a second count with separate code."""
    os.makedirs(folder, exist_ok=True)
    arrays = {}
    for arm, episodes in replays.items():
        for episode, found in episodes.items():
            for name, array in found.items():
                if isinstance(array, np.ndarray):
                    arrays["%s__%d__%s" % (arm, episode, name)] = array
            arrays["%s__%d__injected" % (arm, episode)] = np.array(found["injected"], dtype=np.int64)
    np.savez_compressed(os.path.join(folder, "readings_s%d_%s.npz" % (seed, mode)), **arrays)


def measure_run(run: str, mode: str, device: str, arms: Sequence[str], arrays_dir: Optional[str]) -> tuple:
    started = time.time()
    replays, records = replay_seed(run, mode, device, arms)
    seed = int(run_config(run, device)[2])
    if arrays_dir:
        save_arrays(arrays_dir, seed, mode, replays)
    base = replays["base"]
    report = {"seconds": time.time() - started, "reproduction": reproduction(base, records),
              "replica_largest_change": max(largest_change(step_changes(base[e], replays["replica"][e])) for e in base),
              "base_calls": {e: base[e]["calls"] for e in base},
              "memory_error": {e: base[e]["memory_error"] for e in base}}
    return report, replays


def control_lines(seed: int, mode: str, report: dict) -> None:
    print("seed %d %s, %.0f s. V0, replica largest change %.3g. Calls %s" % (
        seed, mode, report["seconds"], report["replica_largest_change"], report["base_calls"]))
    for episode, row in report["reproduction"].items():
        print("  episode %s, replay against record %s" % (episode, json.dumps(row)))


def ruled(verdict: str, controls_hold: bool) -> str:
    """A claim whose control failed is VOID, whatever its count says."""
    return verdict if controls_hold else "VOID"


def verdict_lines(by_seed: dict) -> dict:
    """The verdicts of the rules above, in training mode."""
    seeds = sorted(by_seed)
    rpt1, gwt1, gwt3 = {}, {}, {}
    for seed in seeds:
        tables, facts = by_seed[seed]["train"]["tables"], by_seed[seed]["facts"]
        calls = by_seed[seed]["train"]["base_calls"][EPISODES[-1]]
        rpt1[seed] = rpt1_claims(tables)
        gwt1[seed] = gwt1_claims(tables, facts, calls)
        gwt3[seed] = gwt3_claims(tables, gwt1[seed]["counts"])
    v0 = all(by_seed[s][m]["replica_largest_change"] <= TOLERANCE for s in seeds for m in by_seed[s] if m != "facts")
    v1 = {s: reproduction_holds(by_seed[s]["train"]["reproduction"]) for s in seeds}
    v2 = {s: gwt3[s]["gate_reached"] for s in seeds}
    replay_holds = v0 and all(v1.values())
    return {
        "V0": v0, "V1": v1, "V2": v2,
        "RPT-1 tectum": ruled(claim_verdict([rpt1[s]["tectum"] for s in seeds]), replay_holds),
        "RPT-1 audio": ruled(claim_verdict([rpt1[s]["audio"] for s in seeds]), replay_holds),
        "RPT-1 settle cycles": ruled(claim_verdict([rpt1[s]["settle"] for s in seeds]), replay_holds),
        "RPT-1 capsule feedback": ruled(claim_verdict([rpt1[s]["capsule"] for s in seeds]), replay_holds),
        "GWT-1 indicator text": ruled(claim_verdict([gwt1[s]["indicator"] for s in seeds]), replay_holds),
        "GWT-1 row text": ruled(claim_verdict([gwt1[s]["row_text"] for s in seeds]), replay_holds),
        "GWT-3": ruled(claim_verdict([gwt3[s]["holds"] for s in seeds]), replay_holds and all(v2.values())),
        "details": {"rpt1": rpt1, "gwt1": gwt1, "gwt3": gwt3}}


def changed_steps(base: np.ndarray, other: np.ndarray) -> int:
    """Steps where any element differs, with no tolerance."""
    return int(np.sum(np.any(base.reshape(len(base), -1) != other.reshape(len(other), -1), axis=1)))


def broadcast_arm_counts(readings, arm: str) -> dict:
    """Exact counts of one broadcast arm, over both episodes of one saved file of readings."""
    counts = collections.Counter()
    for episode in EPISODES:
        base = {name: readings["base__%d__%s" % (episode, name)] for name in ("broadcast", "gate", "winner")}
        moved = {name: readings["%s__%d__%s" % (arm, episode, name)] for name in base}
        injected = [int(step) for step in readings["%s__%d__injected" % (arm, episode)]]
        same_broadcast = np.all(base["broadcast"][injected] == moved["broadcast"][injected], axis=1)
        other_winner = base["winner"][injected] != moved["winner"][injected]
        counts["injected"] += len(injected)
        counts["gate_moved"] += changed_steps(base["gate"][injected], moved["gate"][injected])
        counts["broadcast_moved"] += int(np.sum(~same_broadcast))
        counts["same_broadcast_other_winner"] += int(np.sum(same_broadcast & other_winner))
        for name in ("vision_content", "audio_content", "semantic_content", "bid_raw"):
            counts["steps_compared_" + name] += len(readings["base__%d__%s" % (episode, name)])
            counts["steps_changed_" + name] += changed_steps(readings["base__%d__%s" % (episode, name)],
                                                             readings["%s__%d__%s" % (arm, episode, name)])
    return dict(counts)


def settle_arm_counts(readings, arm: str) -> dict:
    """Exact counts of one settle arm per episode. The broadcast is read at ignited steps of the window."""
    counts = {}
    for episode in EPISODES:
        ignited = readings["base__%d__ignited" % episode]
        window = [t for t in window_steps(len(ignited)) if ignited[t]]
        counts[episode] = {
            "ignited_steps": len(window),
            "broadcast_moved": changed_steps(readings["base__%d__broadcast" % episode][window],
                                             readings["%s__%d__broadcast" % (arm, episode)][window]),
            "vision_content_moved": changed_steps(readings["base__%d__vision_content" % episode],
                                                  readings["%s__%d__vision_content" % (arm, episode)]),
            "audio_content_moved": changed_steps(readings["base__%d__audio_content" % episode],
                                                 readings["%s__%d__audio_content" % (arm, episode)])}
    return counts


def exact_counts(folders: Sequence[str]) -> dict:
    """The exact counts of every file of readings in the folders. No replay runs."""
    report = {}
    for folder in folders:
        for name in sorted(os.listdir(folder)):
            if not (name.startswith("readings_") and name.endswith(".npz")):
                continue
            readings = np.load(os.path.join(folder, name))
            found = {arm: broadcast_arm_counts(readings, arm) for arm in BROADCAST_ARMS
                     if "%s__%d__broadcast" % (arm, EPISODES[0]) in readings.files}
            found.update({arm: settle_arm_counts(readings, arm) for arm in ("settle_off", "feedback_off")
                          if "%s__%d__broadcast" % (arm, EPISODES[0]) in readings.files})
            report[name[len("readings_"):-len(".npz")]] = found
    return report


def mode_arms(args: argparse.Namespace, mode: str) -> tuple:
    """The arms of one mode. A chosen list must start at `base`, the reference of every arm."""
    chosen = args.arms_train if mode == "train" else args.arms_eval
    if chosen:
        arms = tuple(chosen.split(","))
        if arms[0] != "base" or "replica" not in arms:
            raise ValueError("a list of arms starts at base and holds replica, got %s" % chosen)
        return arms
    if args.controls_only:
        return CONTROL_ARMS
    return tuple(ARMS) if mode == "train" else EVAL_ARMS


def run_all(args: argparse.Namespace) -> dict:
    by_seed: dict = {}
    for run in args.runs:
        seed = int(run_config(run, args.device)[2])
        by_seed[seed] = {"facts": record_facts(run)}
        for mode in args.modes:
            arms = mode_arms(args, mode)
            report, replays = measure_run(run, mode, args.device, arms, args.arrays_dir)
            control_lines(seed, mode, report)
            if not args.controls_only:
                report["tables"] = arm_tables(replays)
                if set(UPSTREAM_ARMS[1:]) <= set(arms):
                    report["upstream"] = upstream_table(replays, len(replays["base"][EPISODES[0]]["winner"]))
            by_seed[seed][mode] = report
            del replays
            if args.output_json:
                write_report(args.output_json + ".partial", {"by_seed": by_seed})
    return by_seed


def write_report(path: str, report: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(jsonable(report), handle, indent=1)


def code_revision() -> str:
    """The commit the probe ran at, with a mark when tracked files were changed."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    revision = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True).stdout
    changed = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                             capture_output=True, text=True).stdout
    return revision.strip() + (" with changes" if changed.strip() else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--count-arrays", nargs="+", help="Folders of saved readings. Prints exact counts and runs no replay.")
    parser.add_argument("--modes", nargs="+", default=["train", "eval"], choices=["train", "eval"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--controls-only", action="store_true",
                        help="Only base and replica, for V0, V1 and the time per replay. No arm is compared.")
    parser.add_argument("--arms-train", help="Comma list of arms instead of all of them. No verdict is computed.")
    parser.add_argument("--arms-eval", help="Comma list of arms instead of EVAL_ARMS. No verdict is computed.")
    parser.add_argument("--output-json")
    parser.add_argument("--arrays-dir")
    args = parser.parse_args()
    if args.count_arrays:
        counts = exact_counts(args.count_arrays)
        print(json.dumps(jsonable(counts), indent=1))
        if args.output_json:
            write_report(args.output_json, counts)
        return
    if not args.runs:
        parser.error("--runs is needed unless --count-arrays is given")
    print("revision %s, device %s, modes %s" % (code_revision(), args.device, args.modes))
    by_seed = run_all(args)
    report = {"revision": code_revision(), "device": args.device, "by_seed": by_seed}
    if not (args.controls_only or args.arms_train or args.arms_eval):
        report["verdicts"] = verdict_lines(by_seed)
        print(json.dumps({k: v for k, v in jsonable(report["verdicts"]).items() if k != "details"}, indent=2))
    if args.output_json:
        write_report(args.output_json, report)


if __name__ == "__main__":
    main()

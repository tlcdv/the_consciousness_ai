"""Offline diagnosis of the competing goals gate, check 2, with the rule written before the first run.

`docs/decisions/2026_11_08_competing_goals_gate.md`. The real room with the new flags, the real
cochlea, the binder, the value gate and the motor map, wired by the functions of the training script.
No policy: the stand-in for the policy gives no movement and learns nothing, and with
`--tectal-orienting-idle still` the live runs do not use the policy's movement either. The tectum and
the cochlea are in training mode, the mode of the live runs. No weight is trained.

One step repeats the calls of the training loop. The cochlea hears the sound of the last room step
(no sound at the first step of an episode), the tectum sees the frame and runs the binder, the wired
gate chooses and the motor map gives the movement, the room steps, and the wired gate reads the
reward. The reward is the room reward. The intrinsic bonus of the training loop needs the policy
side and is not here, so check 3 tests the reward that the gate reads in a run.

RULE of the new check 2, written on 2026-11-14 before its first run (owner decisions of that day in
the rules file). The benefit arm with the salience floor of the gate, EPISODES episodes of STEPS
steps. For one room sequence:
  C1            light first in episodes 15 to 29, SHARE or more.
  C2            decoy first in episodes 45 to 59, SHARE or more.
  forced_first  the forced object is the first object in FORCED_FIRST or more of the 6 forced episodes.
  food_steps    the median of the gate's steps from the choice to the light's food, over the episodes
                15 to 29 in which the light was the first object, is FOOD_STEPS or less. An episode of
                these in which no food of the light closed the choice counts as above every limit.
                With no such episode the median has no value and the criterion fails.
  forced_tone   in each of the 6 forced episodes the gate record names the tone of the object that is
                present. This is scored from the gate record (its `choice`) against the band of that
                object's tone, and the forced schedule says which object is present.
STOP. Two counts must be 0 in each sequence: the forced episodes in which the gate record has two
candidates, and the free choices in which the floor removed a candidate. One of them above 0 stops
the check at that sequence, the later sequences do not run, and the verdict is STOPPED. A stop on a
pipeline sequence means that the held-out sequences do not run.
The check HOLDS when no sequence stops and all five criteria hold at 3 of 3 held-out sequences
(HELD_OUT). The sequences PIPELINE check the pipeline and give no verdict.

History. The first check 2 of 2026-11-14 used sequences 1311 to 1316, episodes of 60 steps and no
floor (this script at revision d897318). Its food_steps took the median over the episodes that the
light fed, so an episode in which the agent left the light dropped out and could not fail. `--rescore`
scores saved rows again by the rule above.

First object: within the first 60 steps of an episode, the first object closer than 60 room pixels,
the nearer one on a tie, by `first_object` of `scripts/analysis/probe_reversal.py`. No contact counts
as neither. In a forced episode the place of the absent object is counted by the same geometry.

Reported and not in the rule: the candidates that the floor removed in the forced episodes, the miss
rate over the free episodes with a choice, for each miss whether the agent reached the other object,
the episodes whose choice was still open at the episode end, and the step of each choice.

The band of each tone is measured before any sequence. A cochlea of its own hears each tone alone,
and the band is the largest band of its envelope. The two bands must differ.

Usage:
    python -m scripts.analysis.diagnose_competing_goals --sequences pipeline --output <json>
    python -m scripts.analysis.diagnose_competing_goals --sequences held-out --output <json>
    python -m scripts.analysis.diagnose_competing_goals --rescore <json of an earlier run>
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Optional, Tuple
from unittest import mock

import numpy as np
import torch

from scripts.analysis.probe_reversal import first_object

PIPELINE, HELD_OUT = (1317, 1318, 1319), (1320, 1321, 1322)
EPISODES, STEPS, FORCED = 60, 200, 6
SALIENCE_FLOOR = 0.15
STOPS = ("forced_with_two_candidates", "free_choices_with_a_removed_candidate")
WINDOW_BEFORE, WINDOW_AFTER = (15, 30), (45, 60)
SHARE, FORCED_FIRST, FOOD_STEPS = 0.8, 5, 50
CRITERIA = ("C1", "C2", "forced_first", "food_steps", "forced_tone")
OTHER = {"light": "decoy", "decoy": "light"}
# The flags of the value gate gate, the room flags of the design, the benefit arm and the floor.
GATE_FLAGS = (
    "--env dark_room --enable-audio --rssm-latent-mode continuous --capsule-workspace-source all_levels "
    "--existence-drive off --phi-control off --dark-room-audio binaural --dark-room-audio-channels 4 "
    "--dark-room-collision --dark-room-view agent_centered --vision-bid-reduction zscore "
    "--audio-salience surprise --ignition-rule tolerance --ignition-tolerance-sd 1.0 --bid-precision gain "
    "--dark-room-agent-mark ring --dark-room-agent-colour 217,119,87 --dark-room-decoy-sound "
    "--dark-room-decoy-level 0.5,3.0 --dark-room-decoy-visible --dark-room-start-between --dark-room-pulse "
    "--dark-room-sound-direction off --tectal-orienting value --tectal-orienting-idle still "
    "--map-binding rhythm --dark-room-forced 6 --dark-room-food-sizes 32,3 --dark-room-food-waits 40,0 "
    "--dark-room-food-change 30 --dark-room-food-sizes-after 4,3 --value-outcome hyperbolic "
    "--value-delay-rate 0.1 --value-salience-floor 0.15").split()


def forced_object(episode: int) -> Optional[str]:
    """The object that is present in a forced episode: the light in even ones, the decoy in odd ones."""
    if episode >= FORCED:
        return None
    return "light" if episode % 2 == 0 else "decoy"


def share(rows: List[dict], window: Tuple[int, int], target: str) -> float:
    """The share of the window's episodes in which `target` was the first object."""
    kept = [row for row in rows if window[0] <= row["episode"] < window[1]]
    if not kept:
        raise ValueError("no episode in the window %r" % (window,))
    return sum(row["first"] == target for row in kept) / len(kept)


def light_food_steps(row: dict) -> float:
    """The gate steps to the light's food in one episode, or infinity when no such food closed the choice."""
    fed_by_light = row["gate"] is not None and row["fed"] == "light"
    return float(row["gate"]["steps"]) if fed_by_light else math.inf


def median_light_food_steps(rows: List[dict], window: Tuple[int, int]) -> Optional[float]:
    """The median gate steps to the light's food over the window's episodes with the light first.

    An episode without the light's food counts as infinity. None when the light was first in none."""
    steps = [light_food_steps(row) for row in rows
             if window[0] <= row["episode"] < window[1] and row["first"] == "light"]
    return float(np.median(steps)) if steps else None


def light_first_without_food(rows: List[dict], window: Tuple[int, int]) -> List[int]:
    """The window's episodes with the light first in which no food of the light closed the choice."""
    return [row["episode"] for row in rows if window[0] <= row["episode"] < window[1]
            and row["first"] == "light" and math.isinf(light_food_steps(row))]


def free_choices_with_removed(rows: List[dict]) -> List[int]:
    """The free episodes in which the floor removed a candidate at the choice."""
    return [row["episode"] for row in rows
            if row["episode"] >= FORCED and row["gate"] is not None and row["gate"].get("removed")]


def forced_first(rows: List[dict]) -> int:
    """The forced episodes in which the present object was the first object."""
    return sum(row["first"] == forced_object(row["episode"]) for row in rows if row["episode"] < FORCED)


def forced_tones_named(rows: List[dict], tone_band: Dict[str, int]) -> int:
    """The forced episodes whose gate record chose the band of the present object's tone."""
    return sum(row["gate"] is not None and int(row["gate"]["choice"]) == tone_band[forced_object(row["episode"])]
               for row in rows if row["episode"] < FORCED)


def forced_with_two_candidates(rows: List[dict]) -> List[int]:
    """The forced episodes in which the gate had more than one candidate."""
    return [row["episode"] for row in rows
            if row["episode"] < FORCED and row["gate"] is not None and len(row["gate"]["salience"]) > 1]


def chosen_object(gate: dict, tone_band: Dict[str, int]) -> str:
    """The object whose tone the gate chose. A chosen band that is no tone raises."""
    for name, band in tone_band.items():
        if int(gate["choice"]) == band:
            return name
    raise ValueError("the gate chose band %r, which is the band of no tone %r" % (gate["choice"], tone_band))


def misses(rows: List[dict], tone_band: Dict[str, int]) -> dict:
    """Free episodes with a choice whose first object is not the chosen one, and where the agent went."""
    free = [row for row in rows if row["episode"] >= FORCED and row["gate"] is not None]
    missed = [{"episode": row["episode"], "chosen": chosen_object(row["gate"], tone_band), "first": row["first"]}
              for row in free if row["first"] != chosen_object(row["gate"], tone_band)]
    return {"choices": len(free), "misses": missed,
            "reached_the_other_object": sum(m["first"] == OTHER[m["chosen"]] for m in missed),
            "no_contact": sum(m["first"] is None for m in missed)}


def summarise(rows: List[dict], tone_band: Dict[str, int]) -> dict:
    """The numbers of the rule and of the report for one room sequence, and which criteria hold."""
    if len(rows) != EPISODES:
        raise ValueError("a sequence has %d episodes and not %d" % (len(rows), EPISODES))
    summary = {"light_first_15_29": share(rows, WINDOW_BEFORE, "light"),
               "decoy_first_45_59": share(rows, WINDOW_AFTER, "decoy"),
               "forced_first": forced_first(rows), "forced_tones_named": forced_tones_named(rows, tone_band),
               "forced_with_two_candidates": forced_with_two_candidates(rows),
               "free_choices_with_a_removed_candidate": free_choices_with_removed(rows),
               "light_first_without_light_food_15_29": light_first_without_food(rows, WINDOW_BEFORE)}
    median = median_light_food_steps(rows, WINDOW_BEFORE)
    # JSON has no infinity: a median above every limit is stored as the text "no food".
    summary["median_light_food_steps_15_29"] = "no food" if median is not None and math.isinf(median) else median
    summary["holds"] = {"C1": summary["light_first_15_29"] >= SHARE, "C2": summary["decoy_first_45_59"] >= SHARE,
                        "forced_first": summary["forced_first"] >= FORCED_FIRST,
                        "food_steps": median is not None and median <= FOOD_STEPS,
                        "forced_tone": summary["forced_tones_named"] == FORCED}
    return summary


def stopped(summary: dict) -> bool:
    """True when one of the two counts of the stop rule is above 0 in this sequence."""
    return any(len(summary[name]) > 0 for name in STOPS)


def verdict(summaries: List[dict]) -> str:
    """STOPPED when a sequence stopped. Else HOLDS when every criterion holds at each of exactly 3
    sequences, else FAILED."""
    if any(stopped(summary) for summary in summaries):
        return "STOPPED"
    if len(summaries) != 3:
        raise ValueError("the rule needs 3 sequences, got %d" % len(summaries))
    return "HOLDS" if all(s["holds"][name] for s in summaries for name in CRITERIA) else "FAILED"


class GateLog:
    """Pairs each closed gate record with the episode in which its choice was made.

    The gate closes a choice at the food step, at the end of its window, or at the first step of the
    next episode (a step without sound). `closed` is the number of closed records at the call."""

    def __init__(self) -> None:
        self.record_index: Dict[int, int] = {}
        self.fed: Dict[int, Optional[str]] = {}
        self.cut = set()
        self._seen, self._episode = 0, None

    def start_episode(self, episode: int, closed: int) -> None:
        self._take(self._episode, closed, cut=True)
        self._episode = episode

    def after_first_action(self, episode: int, closed: int) -> None:
        """The first action of an episode closes a choice of the episode before that is still open."""
        self._take(episode - 1, closed, cut=True)

    def end_step(self, episode: int, closed: int, fed: Optional[str]) -> None:
        if self._take(episode, closed, cut=False):
            self.fed[episode] = fed

    def finish(self, closed: int) -> None:
        self._take(self._episode, closed, cut=True)

    def _take(self, episode: Optional[int], closed: int, cut: bool) -> bool:
        if closed == self._seen:
            return False
        if closed != self._seen + 1 or episode is None or episode in self.record_index:
            raise RuntimeError("gate records and episodes do not pair at episode %r" % (episode,))
        self.record_index[episode], self._seen = self._seen, closed
        if cut:
            self.cut.add(episode)
        return True


class NoPolicy:
    """The stand-in for the policy: no movement, and its step learns nothing."""

    def select_action(self, *args, **kwargs):
        return np.zeros(2, dtype=np.float32), 0.0

    def step(self, *args, **kwargs):
        return {}


def gate_arguments():
    """The arguments of a gate run, parsed and checked by the training script's own parser."""
    from scripts.training import train_rlhf
    captured = {}

    def capture(parser, *args, **kwargs):
        captured["parser"] = parser
        raise RuntimeError("captured")

    with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
        try:
            train_rlhf.main()
        except RuntimeError as error:
            if str(error) != "captured":
                raise
    args = captured["parser"].parse_args(list(GATE_FLAGS) + ["--episodes", str(EPISODES), "--max-steps", str(STEPS)])
    train_rlhf._check_competing_goals_flags(captured["parser"], args)
    return args


def measure_tone_bands(config: dict) -> Dict[str, int]:
    """The cochlear band of the light's tone and of the decoy's tone, each heard alone. They must differ."""
    from models.audio.auditory_specialist import AuditorySpecialist
    from simulations.environments import audio_mixin
    from scripts.training.train_rlhf import audio_waveform_tensor
    rate = audio_mixin.DarkRoomAudioMixin._audio_sample_rate
    samples = int(audio_mixin.DarkRoomAudioMixin._audio_frame_duration * rate)
    bands = {}
    for name, hertz in (("light", audio_mixin.BINAURAL_TONE_HZ), ("decoy", audio_mixin.DECOY_TONE_HZ)):
        cochlea = AuditorySpecialist(config).to(config["device"])
        tone = audio_mixin.source_ears([40.0, 0.0], hertz, 4, rate, samples, directional=False)
        with torch.no_grad():
            for _ in range(5):
                cochlea(audio_waveform_tensor(tone.astype(np.float32), config["device"]))
        bands[name] = int(np.argmax(np.asarray(cochlea.last_band_envelope)))
    if bands["light"] == bands["decoy"]:
        raise RuntimeError("the two tones have the same band %r" % bands["light"])
    return bands


def build_pipeline(args, config: dict, seed: int):
    """The room, the cochlea, the tectum with the binder, and the wired gate and motor map of one sequence."""
    from models.audio.auditory_specialist import AuditorySpecialist
    from models.core.global_workspace import GlobalWorkspace
    from models.core.sensory_tectum import SensoryTectum
    from scripts.training import train_rlhf
    train_rlhf._set_global_seed(seed)
    tectum = SensoryTectum({key: config[key] for key in (
        "tectum_feature_dim", "tectum_grid_size", "workspace_dim", "rssm_latent_mode",
        "capsule_workspace_source", "vision_bid_reduction")}).to(config["device"])
    cochlea = AuditorySpecialist(config).to(config["device"])
    tectum.train(), cochlea.train()
    train_rlhf._wire_map_binding(tectum, GlobalWorkspace(config["workspace"]), None, args.map_binding,
                                 auditory_specialist=cochlea, seed=seed, bands_to_bind=2)
    core = NoPolicy()
    train_rlhf._wire_value_gate(core, tectum, seed, None, idle=args.tectal_orienting_idle,
                                learning_rate=args.value_learning_rate, outcome_rule=args.value_outcome,
                                delay_rate=args.value_delay_rate, salience_floor=args.value_salience_floor)
    room = train_rlhf._build_dark_room(args, None)
    train_rlhf.seed_environment(room, seed)
    room.seed_audio(seed)
    room.seed_decoy(seed)
    return room, cochlea, tectum, core


def sense_and_act(room_output, cochlea, tectum, core, config: dict) -> np.ndarray:
    """The calls of one step of the training loop before the room steps. Returns the movement."""
    from scripts.training.train_rlhf import audio_waveform_tensor, frame_to_tensor
    frame, facts = room_output
    sound = facts.get("audio_waveform")
    if sound is None:
        heard = torch.zeros(1, config["tectum_feature_dim"], 2, device=config["device"])
    else:
        cochlea(audio_waveform_tensor(sound, config["device"]))
        heard = cochlea.get_spatial_for_tectum()
    tectum(frame_to_tensor(frame, config["device"]), heard, action=None)
    action, _ = core.select_action(None)
    return action


def run_episode(episode: int, room, cochlea, tectum, core, config: dict, log: GateLog) -> dict:
    """One episode. Returns the first object, the food sizes of the room and the step of the choice."""
    gate = tectum.value_gate
    room_output = room.reset()
    log.start_episode(episode, len(gate.episodes))
    steps, choice_step, sizes, present = [], None, None, None
    for step in range(STEPS):
        action = sense_and_act(room_output, cochlea, tectum, core, config)
        if step == 0:
            log.after_first_action(episode, len(gate.episodes))
        if choice_step is None and gate.choice is not None:
            choice_step = step
        frame, reward, _, _, facts = room.step(action)
        core.step(raw_reward=reward)
        log.end_step(episode, len(gate.episodes), facts["_truth_food_object"])
        steps.append({"internals": {"info_after_step": facts}})
        sizes, present, room_output = facts["_truth_food_sizes"], facts["_truth_present"], (frame, facts)
    return {"episode": episode, "first": first_object(steps), "choice_step": choice_step,
            "food_sizes": sizes, "present": present}


def run_sequence(args, config: dict, seed: int) -> List[dict]:
    """The episode rows of one room sequence, each with its gate record."""
    room, cochlea, tectum, core = build_pipeline(args, config, seed)
    log, rows = GateLog(), []
    with torch.no_grad():
        for episode in range(EPISODES):
            rows.append(run_episode(episode, room, cochlea, tectum, core, config, log))
    gate = tectum.value_gate
    gate.new_episode()
    log.finish(len(gate.episodes))
    for row in rows:
        index = log.record_index.get(row["episode"])
        row["gate"] = None if index is None else gate.episodes[index]
        row["fed"] = log.fed.get(row["episode"])
        row["cut_by_episode_end"] = row["episode"] in log.cut
    return rows


def check_room_schedule(rows: List[dict]) -> None:
    """Raise unless the room ran the forced episodes and the size change of the design."""
    for row in rows:
        present = forced_object(row["episode"]) or "both"
        sizes = [32.0, 3.0] if row["episode"] < 30 else [4.0, 3.0]
        if row["present"] != present or row["food_sizes"] != sizes:
            raise RuntimeError("episode %d: the room had %r with sizes %r, and the design has %r with %r" % (
                row["episode"], row["present"], row["food_sizes"], present, sizes))


def report(seed: int, rows: List[dict], tone_band: Dict[str, int]) -> dict:
    """Print and return the numbers of one sequence."""
    summary = summarise(rows, tone_band)
    summary["sequence"], summary["misses"] = seed, misses(rows, tone_band)
    summary["no_choice"] = [row["episode"] for row in rows if row["gate"] is None]
    summary["cut_by_episode_end"] = [row["episode"] for row in rows if row["cut_by_episode_end"]]
    summary["choice_steps"] = sorted(set(row["choice_step"] for row in rows if row["choice_step"] is not None))
    summary["forced_gate_records"] = [
        {"episode": row["episode"], "present": row["present"], "first": row["first"],
         "choice": None if row["gate"] is None else row["gate"]["choice"],
         "salience": None if row["gate"] is None else row["gate"]["salience"],
         "removed": None if row["gate"] is None else row["gate"].get("removed")} for row in rows[:FORCED]]
    summary["values"] = {str(band): float(v) for band, v in rows_values(rows).items()}
    print("sequence %d" % seed)
    for key in ("light_first_15_29", "decoy_first_45_59", "forced_first", "forced_tones_named",
                "median_light_food_steps_15_29", "light_first_without_light_food_15_29",
                "forced_with_two_candidates", "free_choices_with_a_removed_candidate", "no_choice",
                "cut_by_episode_end", "choice_steps", "holds"):
        print("  %-32s %s" % (key, summary[key]))
    found = summary["misses"]
    print("  misses %d of %d choices; reached the other object %d; no contact %d" % (
        len(found["misses"]), found["choices"], found["reached_the_other_object"], found["no_contact"]))
    return summary


def rows_values(rows: List[dict]) -> Dict[int, float]:
    """The value of each band after its last closed choice, from the gate records."""
    values = {}
    for row in rows:
        if row["gate"] is not None:
            values[int(row["gate"]["choice"])] = row["gate"]["value_after"]
    return values


def rescore(path: str) -> None:
    """Print the numbers of the rule above for the saved rows of an earlier run. Nothing is run."""
    with open(path, encoding="utf-8") as handle:
        saved = json.load(handle)
    tone_band = {name: int(band) for name, band in saved["tone_band"].items()}
    for seed, rows in saved["rows"].items():
        summary = summarise(rows, tone_band)
        print("sequence %s: light first %.3f, decoy first %.3f, forced first %d, forced tones %d, median %s, "
              "light first without the light's food %s, holds %s" % (
                  seed, summary["light_first_15_29"], summary["decoy_first_45_59"], summary["forced_first"],
                  summary["forced_tones_named"], summary["median_light_food_steps_15_29"],
                  summary["light_first_without_light_food_15_29"], summary["holds"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sequences", choices=["pipeline", "held-out"],
                        help="pipeline: 1317 to 1319, no verdict. held-out: 1320 to 1322, with the verdict.")
    parser.add_argument("--output", help="Write the summaries and the episode rows as JSON here.")
    parser.add_argument("--rescore", help="Score the rows of this saved JSON again and run nothing.")
    options = parser.parse_args()
    if options.rescore:
        return rescore(options.rescore)
    if not options.sequences or not options.output:
        parser.error("--sequences and --output are needed for a run")
    from scripts.training import train_rlhf
    args = gate_arguments()
    config = train_rlhf.build_config(args)
    tone_band = measure_tone_bands(config)
    print("tone bands:", tone_band)
    seeds = PIPELINE if options.sequences == "pipeline" else HELD_OUT
    summaries, all_rows = [], {}
    for seed in seeds:
        rows = run_sequence(args, config, seed)
        check_room_schedule(rows)
        all_rows[str(seed)] = rows
        summaries.append(report(seed, rows, tone_band))
        if stopped(summaries[-1]):
            print("STOP at sequence %d: %s" % (seed, {name: summaries[-1][name] for name in STOPS}))
            break
    result = {"sequences": options.sequences, "tone_band": tone_band, "salience_floor": args.value_salience_floor,
              "steps": STEPS, "summaries": summaries, "rows": all_rows}
    if options.sequences == "held-out" or any(stopped(summary) for summary in summaries):
        result["verdict"] = verdict(summaries)
        print("VERDICT of check 2:", result["verdict"])
    else:
        print("Pipeline sequences. No stop. No verdict.")
    with open(options.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()

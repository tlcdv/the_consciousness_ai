"""Probe that re-checks RPT-1, GWT-1 and GWT-3 (scripts/analysis/probe_rubric_recheck.py).

Expected numbers are worked out by hand on constructed readings. The last class runs the real
training loop on untrained weights, so the replay is tested with the shapes and the calling
convention of the real caller.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core.bid_normalization import RunningZScoreBid
from models.core.semantic_pathway import SemanticPathway
from models.core.sensory_tectum import SensoryTectum
from models.self_model.embodiment_core import ProprioceptiveProcessor
from scripts.analysis.probe_rubric_recheck import (
    ABOVE,
    ABSENT,
    ARMS,
    BELOW,
    EVAL_ARMS,
    MODULES,
    UPSTREAM_TAPS,
    WARM_IN,
    ArmSpec,
    EpisodeRecord,
    ReplayEnv,
    arm_env,
    arm_tables,
    build_agent,
    changed_steps,
    claim_verdict,
    effect_table,
    exact_counts,
    exploration_std,
    gwt1_claims,
    gwt1_module_counts,
    gwt1_specialised,
    gwt3_claims,
    gwt3_module_holds,
    gwt3_policy_holds,
    history_rows,
    ignited_shares,
    jsonable,
    label_effect,
    lag_steps,
    load_weights,
    map_settings,
    policy_table,
    prewarm_statistics,
    random_same_norm,
    relative_change,
    remember_phi,
    replay_arm,
    rpt1_claims,
    rpt1_holds,
    schedule_steps,
    step_changes,
    upstream_steps,
    upstream_table,
    verdict_lines,
    weights_fingerprint,
)
from scripts.training.session_capture import modules_with_parameters
from scripts.training.session_recorder import _state_dicts
from scripts.training.train_rlhf import build_config


class TestRelativeChange:

    def test_is_the_distance_over_the_size_of_the_base(self):
        assert relative_change(np.array([3.0, 4.0]), np.array([3.0, 0.0])) == pytest.approx(4.0 / 5.0)

    def test_two_zero_vectors_do_not_differ(self):
        assert relative_change(np.zeros(3), np.zeros(3)) == 0.0

    def test_from_a_zero_base_to_a_vector_is_infinite(self):
        assert relative_change(np.zeros(2), np.array([0.0, 1.0])) == float("inf")


class TestLabelEffect:

    def test_at_or_below_the_tolerance_is_absent_whatever_the_floor(self):
        assert label_effect(1e-7, 0.5) == ABSENT
        assert label_effect(0.0, 0.0) == ABSENT

    def test_below_the_floor_is_below_noise(self):
        assert label_effect(0.1, 0.5) == BELOW

    def test_at_or_above_the_floor_is_above_noise(self):
        assert label_effect(0.5, 0.5) == ABOVE
        assert label_effect(0.01, 0.0) == ABOVE

    def test_refuses_a_number_that_is_not_a_number(self):
        with pytest.raises(ValueError):
            label_effect(float("nan"), 0.1)
        with pytest.raises(ValueError):
            label_effect(0.1, float("nan"))


class TestRandomSameNorm:

    def test_keeps_the_size_and_the_shape(self):
        tensor = torch.tensor([[3.0, 4.0, 0.0, 0.0]])
        moved = random_same_norm(tensor, seed=5)
        assert moved.shape == tensor.shape
        assert float(moved.norm()) == pytest.approx(5.0, abs=1e-5)

    def test_one_seed_gives_one_direction_and_another_seed_another(self):
        tensor = torch.ones(1, 8)
        assert torch.equal(random_same_norm(tensor, seed=1), random_same_norm(tensor, seed=1))
        assert not torch.equal(random_same_norm(tensor, seed=1), random_same_norm(tensor, seed=2))

    def test_draws_nothing_from_the_global_stream(self):
        before = torch.get_rng_state()
        random_same_norm(torch.ones(1, 8), seed=1)
        assert torch.equal(before, torch.get_rng_state())


class TestExplorationStd:

    def test_matches_the_formula_of_the_policy(self):
        # action_selection_core.py: 0.1 * (1 + max(0.1, 2 * arousal))
        assert exploration_std(0.0) == pytest.approx(0.11)
        assert exploration_std(0.5) == pytest.approx(0.2)


class TestRememberPhi:

    def test_a_repeated_input_gives_the_same_phi_and_the_same_error_without_a_second_call(self):
        class Metrics:
            last_phi_error = None
            calls = 0

            def calculate_phi(self, tpm, current_state, cm=None):
                self.calls += 1
                self.last_phi_error = "failed" if current_state[0] == 1 else None
                return 0.25 + current_state[0]
        metrics = Metrics()
        remember_phi(metrics)
        tpm = np.eye(2)
        first = metrics.calculate_phi(tpm, (0, 1), cm="unused")
        second = metrics.calculate_phi(tpm.copy(), (0, 1), cm="unused")
        other = metrics.calculate_phi(tpm, (1, 1), cm="unused")
        assert (first, second, other) == (0.25, 0.25, 1.25)
        assert metrics.calls == 2
        assert metrics.last_phi_error == "failed"
        metrics.calculate_phi(tpm, (0, 1), cm="unused")
        assert metrics.last_phi_error is None


class TestSchedules:

    def test_steps_start_at_the_first_step_and_keep_the_gap(self):
        assert schedule_steps(30, 10, 75) == [30, 40, 50, 60, 70]

    def test_lag_steps_leave_out_steps_past_the_end(self):
        assert lag_steps([10, 198], 1, 200) == [11, 199]
        assert lag_steps([10, 199], 1, 200) == [11]


class TestMapSettings:

    def test_the_value_gate_binds_two_bands_as_the_training_loop_does(self):
        argv = ["--map-binding", "rhythm", "--tectal-orienting", "value"]
        assert map_settings(argv) == ("rhythm", 2)

    def test_any_other_orienting_binds_one_band(self):
        assert map_settings(["--map-binding", "rhythm", "--tectal-orienting", "group"]) == ("rhythm", 1)
        assert map_settings(["--map-binding", "rhythm"]) == ("rhythm", 1)


class TestUpstreamTable:

    def _replays(self):
        length = 60
        base = {"winner": np.array(["vision"] * length)}
        steps = upstream_steps(length)

        def arm(size):
            found = {"tap_" + tap: np.full(length, np.nan) for tap in UPSTREAM_TAPS}
            for tap in UPSTREAM_TAPS:
                found["tap_" + tap][steps] = size
            return found
        return {"base": {38: base, 39: base}, "noise": {38: arm(0.5), 39: arm(0.5)},
                "history_swap": {38: arm(1.0), 39: arm(1.0)}, "frame_other": {38: arm(0.1), 39: arm(0.1)}}

    def test_each_tap_is_read_at_the_steps_of_its_lag_against_the_floor(self):
        table = upstream_table(self._replays(), 60)
        assert upstream_steps(60) == [40, 41, 42, 43, 45, 50, 59]
        assert table["history_swap, lag 0"]["h_state"] == {"effect": 1.0, "floor": 0.5, "label": ABOVE}
        assert table["frame_other"]["obs_map"] == {"effect": 0.1, "floor": 0.5, "label": BELOW}

    def test_a_reading_that_was_never_taken_stops_the_table(self):
        replays = self._replays()
        replays["history_swap"][38]["tap_h_state"][:] = np.nan
        replays["history_swap"][39]["tap_h_state"][:] = np.nan
        with pytest.raises(ValueError):
            upstream_table(replays, 60)


class TestReplayEnv:

    def _record(self):
        frames = np.arange(3 * 2 * 2 * 3, dtype=np.uint8).reshape(3, 2, 2, 3)
        return frames, {1: np.full((4, 8), 1.0, dtype=np.float32), 2: np.full((4, 8), 2.0, dtype=np.float32)}

    def test_reset_gives_the_first_frame_without_sound_as_the_room_does(self):
        frames, waves = self._record()
        env = ReplayEnv(frames, waves, [0.5, 0.25, 0.125], [{"battery": 0.9}] * 3)
        frame, facts = env.reset()
        assert np.array_equal(frame, frames[0])
        assert "audio_waveform" not in facts

    def test_step_gives_the_next_frame_the_reward_of_the_step_and_the_next_sound(self):
        frames, waves = self._record()
        env = ReplayEnv(frames, waves, [0.5, 0.25, 0.125], [{"battery": 0.9}] * 3)
        env.reset()
        frame, reward, terminated, truncated, facts = env.step(np.zeros(2))
        assert np.array_equal(frame, frames[1])
        assert reward == 0.5
        assert not terminated and not truncated
        assert facts["battery"] == 0.9
        assert np.array_equal(facts["audio_waveform"], waves[1])

    def test_ends_after_the_last_recorded_step_and_keeps_the_actions_it_received(self):
        frames, waves = self._record()
        env = ReplayEnv(frames, waves, [0.5, 0.25, 0.125], [{"battery": 0.9}] * 3)
        env.reset()
        ends = [env.step(np.array([float(i), 0.0]))[3] for i in range(3)]
        assert ends == [False, False, True]
        assert [float(a[0]) for a in env.received_actions] == [0.0, 1.0, 2.0]


def _record_with(mark: int) -> EpisodeRecord:
    frames = np.full((6, 2, 2, 3), mark, dtype=np.uint8)
    waves = {t: np.full((4, 8), float(mark + t), dtype=np.float32) for t in range(1, 6)}
    return EpisodeRecord(index=mark, frames=frames, waveforms=waves, rewards=[0.0] * 6,
                         facts=[{"battery": 0.9} for _ in range(6)], actions=[[0.0, 0.0]] * 6,
                         vectors={}, obs_map=None)


class TestArmEnv:

    def test_the_plain_arm_reads_its_own_episode(self):
        env = arm_env(ArmSpec(), _record_with(1), _record_with(2), [3])
        assert env.frames[0, 0, 0, 0] == 1
        assert env.waveforms[3][0, 0] == 4.0

    def test_frames_from_the_other_episode_leave_the_sound_alone(self):
        env = arm_env(ArmSpec(frames_from_other=True), _record_with(1), _record_with(2), [3])
        assert env.frames[0, 0, 0, 0] == 2
        assert env.waveforms[3][0, 0] == 4.0

    def test_sound_from_the_other_episode_leaves_the_frames_alone(self):
        env = arm_env(ArmSpec(waves_from_other=True), _record_with(1), _record_with(2), [3])
        assert env.frames[0, 0, 0, 0] == 1
        assert env.waveforms[3][0, 0] == 5.0

    def test_a_swap_before_the_schedule_changes_only_the_step_before_it(self):
        spec = ArmSpec(swap_wave_before_schedule=True)
        env = arm_env(spec, _record_with(1), _record_with(2), [3])
        assert env.waveforms[2][0, 0] == 4.0   # other episode, mark 2 + step 2
        assert env.waveforms[3][0, 0] == 4.0   # own episode, mark 1 + step 3
        assert env.waveforms[1][0, 0] == 2.0   # own episode, mark 1 + step 1

    def test_a_set_battery_replaces_the_recorded_battery_at_every_step(self):
        env = arm_env(ArmSpec(battery=0.2), _record_with(1), _record_with(2), [3])
        assert all(facts["battery"] == 0.2 for facts in env.facts)


class TestPrewarm:

    def _steps(self):
        return [
            {"input_source": {"audio": "none"}, "bound_bids": {"vision": 0.5, "audio": 0.1},
             "internals": {"kl_div_pre_tanh": 10.0, "audio_surprise_error": None}},
            {"input_source": {"audio": "sound"}, "bound_bids": {"vision": 0.2, "audio": 0.8},
             "internals": {"kl_div_pre_tanh": 20.0, "audio_surprise_error": None}},
            {"input_source": {"audio": "sound"}, "bound_bids": {"vision": 0.4, "audio": 0.3},
             "internals": {"kl_div_pre_tanh": 40.0, "audio_surprise_error": 0.5}},
        ]

    def test_history_rows_keep_the_error_only_of_steps_with_sound_and_a_value(self):
        rows = history_rows(self._steps())
        assert [row["kl"] for row in rows] == [10.0, 20.0, 40.0]
        assert [row["audio_error"] for row in rows] == [None, None, 0.5]
        assert [row["top_bound_bid"] for row in rows] == [0.5, 0.8, 0.4]

    def test_the_running_statistics_equal_a_direct_run_over_the_same_series(self):
        class Holder:
            pass
        agent = Holder()
        agent.tectum, agent.auditory, agent.auditory.surprise = Holder(), Holder(), Holder()
        agent.tectum.bid_normalizer = RunningZScoreBid()
        agent.auditory.surprise.normalizer = RunningZScoreBid(initial_var=0.0)
        agent.workspace = Holder()
        energies = []
        agent.workspace._update_tolerance = energies.append

        prewarm_statistics(agent, history_rows(self._steps()))

        direct = RunningZScoreBid()
        for kl in (10.0, 20.0, 40.0):
            direct.bid(kl)
        assert agent.tectum.bid_normalizer.mean == direct.mean
        assert agent.tectum.bid_normalizer.var == direct.var
        assert agent.auditory.surprise.normalizer.seen == 1
        assert agent.auditory.surprise.normalizer.mean == 0.5
        assert energies == [0.5, 0.8, 0.4]


def _readings(content_scale: float, bid: float, winner: str = "vision") -> dict:
    steps = 4
    zeros = np.zeros((steps, 2), dtype=np.float32)
    bids = np.tile(np.array([bid, 0.1, 0.1, 0.05, 0.0]), (steps, 1))
    return {"vision_content": np.full((steps, 2), 1.0) * np.array([1.0, content_scale]),
            "audio_content": np.full((steps, 2), 1.0), "semantic_content": zeros,
            "broadcast": np.full((steps, 2), 1.0), "gate": np.full((steps, 2), 1.0),
            "bid_raw": bids, "bid_modulated": bids, "bid_bound": bids, "bid_final": bids,
            "action_mean": zeros, "arousal": np.zeros(steps),
            "winner": np.array([winner] * steps), "ignited": np.ones(steps, dtype=bool)}


class TestStepChanges:

    def test_each_reading_is_compared_step_by_step(self):
        changes = step_changes(_readings(1.0, 0.5), _readings(0.0, 0.75, winner="audio"))
        # vision content: (1, 1) to (1, 0), distance 1, size sqrt 2
        assert changes["vision_content"] == pytest.approx([1.0 / np.sqrt(2.0)] * 4)
        assert changes["audio_content"] == pytest.approx([0.0] * 4)
        assert changes["semantic_content"] == pytest.approx([0.0] * 4)
        assert changes["bid_raw_vision"] == pytest.approx([0.25] * 4)
        assert changes["bid_final_memory"] == pytest.approx([0.0] * 4)
        assert changes["winner"] == pytest.approx([1.0] * 4)
        assert changes["ignited"] == pytest.approx([0.0] * 4)
        assert changes["broadcast_changed"] == pytest.approx([0.0] * 4)

    def test_every_module_has_a_change_at_every_bid_stage(self):
        changes = step_changes(_readings(1.0, 0.5), _readings(1.0, 0.5))
        for stage in ("raw", "modulated", "bound", "final"):
            for module in MODULES:
                assert "bid_%s_%s" % (stage, module) in changes


class TestEffectTable:

    def test_pools_episodes_takes_medians_and_labels(self):
        changes = {38: {"vision_content": np.array([0.0, 0.1, 0.2, 0.3])},
                   39: {"vision_content": np.array([0.4, 0.5, 0.6, 0.7])}}
        floors = {38: {"vision_content": np.array([0.05, 0.05, 0.05, 0.05])},
                  39: {"vision_content": np.array([0.05, 0.05, 0.05, 0.05])}}
        table = effect_table(changes, floors, {38: [1, 2], 39: [0]})
        # pooled effects 0.1, 0.2, 0.4 -> median 0.2. floor 0.05.
        assert table["vision_content"]["effect"] == pytest.approx(0.2)
        assert table["vision_content"]["floor"] == pytest.approx(0.05)
        assert table["vision_content"]["label"] == ABOVE
        assert table["vision_content"]["steps"] == 3

    def test_a_share_quantity_is_a_mean_and_gets_no_label(self):
        changes = {38: {"winner": np.array([1.0, 0.0, 0.0, 0.0])}}
        table = effect_table(changes, changes, {38: [0, 1, 2, 3]})
        assert table["winner"]["effect"] == pytest.approx(0.25)
        assert "label" not in table["winner"]

    def test_refuses_an_empty_set_of_steps(self):
        changes = {38: {"vision_content": np.array([0.1])}}
        with pytest.raises(ValueError):
            effect_table(changes, changes, {38: []})

    def test_the_policy_is_labelled_against_its_exploration_noise(self):
        changes = {38: {"action_mean": np.array([0.02, 0.02])}}
        arousal = {38: np.array([0.0, 0.0])}
        table = policy_table(changes, arousal, {38: [0, 1]})
        assert table["effect"] == pytest.approx(0.02)
        assert table["floor"] == pytest.approx(0.11)
        assert table["label"] == BELOW

    def test_ignited_shares_read_only_steps_where_the_base_ignited(self):
        length = WARM_IN + 4
        changed = np.zeros(length)
        changed[WARM_IN] = 1.0
        changes = {38: {"broadcast_changed": changed, "winner": np.zeros(length)}}
        ignited = np.zeros(length, dtype=bool)
        ignited[WARM_IN:WARM_IN + 2] = True
        shares = ignited_shares(changes, {38: ignited}, length)
        assert shares["broadcast_changed"] == pytest.approx(0.5)
        assert shares["winner"] == 0.0

    def test_ignited_shares_refuse_a_replay_that_never_ignited(self):
        length = WARM_IN + 2
        changes = {38: {"broadcast_changed": np.zeros(length), "winner": np.zeros(length)}}
        with pytest.raises(ValueError):
            ignited_shares(changes, {38: np.zeros(length, dtype=bool)}, length)


class TestWhatTheFeedbackRulesRead:
    """Facts of the code that the result document states, pinned on the rules themselves."""

    def test_the_vision_rule_reads_the_distance_and_not_the_direction(self):
        tectum = SensoryTectum({"workspace_dim": 4})
        tectum._last_content = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        first = {"tensor": torch.tensor([[1.0, 2.0, 0.0, 0.0]])}
        second = {"tensor": torch.tensor([[1.0, 0.0, 0.0, 2.0]])}
        assert tectum.receive_broadcast(first, 0.5) == tectum.receive_broadcast(second, 0.5)

    def test_the_vision_rule_writes_nothing_into_the_module(self):
        tectum = SensoryTectum({"workspace_dim": 4})
        own = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        tectum._last_content = own.clone()
        tectum.h_state, tectum.z_state = torch.ones(1, 2), torch.ones(1, 2)
        tectum.receive_broadcast({"tensor": torch.tensor([[1.0, 2.0, 0.0, 0.0]])}, 0.5)
        assert torch.equal(tectum._last_content, own)
        assert torch.equal(tectum.h_state, torch.ones(1, 2))
        assert torch.equal(tectum.z_state, torch.ones(1, 2))

    def test_the_semantic_rule_returns_the_bid_for_the_dict_that_the_workspace_sends(self):
        pathway = SemanticPathway(input_dim=8, workspace_dim=4)
        pathway._last_content = torch.ones(1, 4)
        assert pathway.receive_broadcast({"tensor": torch.zeros(1, 4), "source": "tectum"}, 0.5) == 0.5

    def test_the_body_rule_reads_keys_and_words_and_not_the_tensor(self):
        body = ProprioceptiveProcessor(raw_state_dim=4)
        first = {"tensor": torch.zeros(1, 4), "source": "tectum"}
        second = {"tensor": torch.ones(1, 4), "source": "tectum"}
        assert body.receive_broadcast(first, 0.5) == body.receive_broadcast(second, 0.5)


class TestVerdicts:

    def test_claim_verdict_passes_at_all_three_seeds(self):
        assert claim_verdict([True, True, True]) == "PASS"

    def test_claim_verdict_fails_at_two_or_more_seeds(self):
        assert claim_verdict([True, False, False]) == "FAILED"
        assert claim_verdict([False, False, False]) == "FAILED"

    def test_claim_verdict_is_undecided_at_one_failing_seed(self):
        assert claim_verdict([True, True, False]) == "UNDECIDED"

    def test_claim_verdict_needs_three_seeds(self):
        with pytest.raises(ValueError):
            claim_verdict([True, True])

    def test_rpt1_needs_both_the_carry_and_the_history_above_noise(self):
        assert rpt1_holds(ABOVE, ABOVE)
        assert not rpt1_holds(ABOVE, BELOW)
        assert not rpt1_holds(ABSENT, ABOVE)

    def test_gwt1_counts_a_module_when_its_content_or_its_bid_is_above_noise(self):
        labels = {"vision": [ABOVE, BELOW], "audio": [BELOW, ABOVE], "memory": [ABSENT],
                  "body": [ABSENT], "semantic": [ABSENT]}
        assert gwt1_module_counts(labels) == {"vision": True, "audio": True, "memory": False,
                                              "body": False, "semantic": False}

    def test_gwt1_specialised_needs_two_counted_modules_that_follow_their_own_input_more(self):
        counts = {"vision": True, "audio": True, "memory": False, "body": False, "semantic": False}
        assert gwt1_specialised(counts, own={"vision": 0.5, "audio": 0.5}, cross={"vision": 0.1, "audio": 0.0})
        assert not gwt1_specialised(counts, own={"vision": 0.1, "audio": 0.5}, cross={"vision": 0.2, "audio": 0.0})
        one_module = dict(counts, audio=False)
        assert not gwt1_specialised(one_module, own={"vision": 0.5}, cross={"vision": 0.1})

    def test_gwt3_module_holds_when_next_step_content_or_bid_is_above_noise_in_either_arm(self):
        assert gwt3_module_holds([ABSENT, ABSENT, ABOVE, ABSENT])
        assert not gwt3_module_holds([ABSENT, BELOW, ABSENT, BELOW])

    def test_gwt3_policy_holds_when_the_change_reaches_the_exploration_noise_in_either_arm(self):
        assert gwt3_policy_holds([BELOW, ABOVE])
        assert not gwt3_policy_holds([BELOW, BELOW])


SCENARIO_STEPS = 60
SCENARIO_EPISODES = (38, 39)


def _scenario_readings(episode: int) -> dict:
    rng = np.random.default_rng(episode)
    steps = SCENARIO_STEPS
    return {"vision_content": rng.normal(size=(steps, 8)), "audio_content": rng.normal(size=(steps, 8)),
            "semantic_content": np.zeros((steps, 8)), "broadcast": rng.normal(size=(steps, 8)),
            "gate": rng.normal(size=(steps, 5)), "bid_raw": rng.random((steps, 5)), "bid_modulated": rng.random((steps, 5)),
            "bid_bound": rng.random((steps, 5)), "bid_final": rng.random((steps, 5)),
            "action_mean": rng.normal(size=(steps, 2)) * 0.01, "arousal": np.zeros(steps),
            "winner": np.array(["vision"] * steps), "ignited": np.ones(steps, dtype=bool),
            "cycles": np.full(steps, 2), "injected": list(range(30, steps, 10)),
            "calls": {"semantic_forward": 0}, "memory_error": None, "kept_states": {}}


def _scenario_replays() -> dict:
    """Every arm a copy of the base, except `noise`, which moves the vision content a little."""
    base = {episode: _scenario_readings(episode) for episode in SCENARIO_EPISODES}
    replays = {name: {episode: {key: (item.copy() if isinstance(item, np.ndarray) else item)
                                for key, item in base[episode].items()}
                      for episode in SCENARIO_EPISODES} for name in ARMS}
    for episode in SCENARIO_EPISODES:
        replays["noise"][episode]["vision_content"] += 0.01
    return replays


def _shift(replays: dict, arm: str, reading: str, size: float, column: int = None) -> None:
    for episode in SCENARIO_EPISODES:
        if column is None:
            replays[arm][episode][reading] += size
        else:
            replays[arm][episode][reading][:, column] += size


def _by_seed(replays: dict, distinct: int = 1, calls: dict = None) -> dict:
    facts = {"bid_distinct": {module: distinct for module in MODULES}}
    return {seed: {"facts": facts, "train": {"tables": arm_tables(replays), "reproduction": {},
                                             "replica_largest_change": 0.0,
                                             "base_calls": {39: calls or {"semantic_forward": 0}}}}
            for seed in (801, 802, 803)}


class TestRulesOnConstructedReplays:

    def test_with_no_effect_anywhere_every_claim_fails_and_the_gate_control_voids_gwt3(self):
        verdicts = verdict_lines(_by_seed(_scenario_replays()))
        for claim in ("RPT-1 tectum", "RPT-1 audio", "RPT-1 settle cycles", "RPT-1 capsule feedback",
                      "GWT-1 indicator text", "GWT-1 row text"):
            assert verdicts[claim] == "FAILED", claim
        assert verdicts["GWT-3"] == "VOID"
        assert verdicts["V0"] is True
        assert verdicts["V1"] == {801: True, 802: True, 803: True}
        assert verdicts["V2"] == {801: False, 802: False, 803: False}
        assert json.dumps(jsonable(verdicts))

    def test_a_replay_that_does_not_match_the_record_voids_every_claim(self):
        by_seed = _by_seed(_scenario_replays())
        by_seed[802]["train"]["reproduction"] = {39: {"audio_content": 0.5, "tectum_content": 1.0}}
        verdicts = verdict_lines(by_seed)
        assert verdicts["V1"] == {801: True, 802: False, 803: True}
        for claim in ("RPT-1 tectum", "GWT-1 row text", "GWT-3"):
            assert verdicts[claim] == "VOID", claim

    def test_large_effects_in_the_arms_of_the_rules_make_every_claim_pass(self):
        replays = _scenario_replays()
        for arm in ("zero_h", "history_swap", "capsule_off", "settle_off", "frame_other", "bc_random", "bc_other"):
            _shift(replays, arm, "vision_content", 5.0)
        for arm in ("audio_history", "wave_other", "bc_random", "bc_other"):
            _shift(replays, arm, "audio_content", 5.0)
        _shift(replays, "frame_other", "bid_raw", 0.5, column=MODULES.index("memory"))
        _shift(replays, "bc_random", "bid_raw", 0.5, column=MODULES.index("memory"))
        for module in ("body", "semantic"):
            _shift(replays, "bc_random", "bid_raw", 0.5, column=MODULES.index(module))
        _shift(replays, "bc_random", "gate", 1.0)
        _shift(replays, "bc_other", "gate", 1.0)
        _shift(replays, "bc_random", "action_mean", 1.0)
        verdicts = verdict_lines(_by_seed(replays, distinct=2, calls={"semantic_forward": 200}))
        for claim in ("RPT-1 tectum", "RPT-1 audio", "RPT-1 settle cycles", "RPT-1 capsule feedback",
                      "GWT-1 indicator text", "GWT-1 row text", "GWT-3"):
            assert verdicts[claim] == "PASS", claim
        assert verdicts["V2"] == {801: True, 802: True, 803: True}

    def test_a_vision_effect_below_the_sampling_noise_does_not_hold(self):
        replays = _scenario_replays()
        for episode in SCENARIO_EPISODES:
            replays["noise"][episode]["vision_content"] += 5.0
        _shift(replays, "zero_h", "vision_content", 0.01)
        _shift(replays, "history_swap", "vision_content", 0.01)
        claims = rpt1_claims(arm_tables(replays))
        assert claims["carry_label"] == BELOW
        assert claims["history_label"] == BELOW
        assert claims["tectum"] is False
        assert claims["history_reach"] is None

    def test_gwt3_fails_when_one_counted_module_has_no_next_step_response(self):
        replays = _scenario_replays()
        _shift(replays, "frame_other", "vision_content", 5.0)
        _shift(replays, "wave_other", "audio_content", 5.0)
        _shift(replays, "bc_random", "vision_content", 5.0)
        _shift(replays, "bc_random", "action_mean", 1.0)
        tables = arm_tables(replays)
        counts = gwt1_claims(tables, {"bid_distinct": {module: 1 for module in MODULES}}, {})["counts"]
        claims = gwt3_claims(tables, counts)
        assert counts["vision"] and counts["audio"]
        assert claims["responses"]["vision"] is True
        assert claims["responses"]["audio"] is False
        assert claims["policy"] is True
        assert claims["holds"] is False

    def test_evaluation_mode_takes_its_floor_from_the_replica(self):
        replays = {name: found for name, found in _scenario_replays().items() if name in EVAL_ARMS}
        tables = arm_tables(replays)
        assert tables["zero_h"]["window"]["vision_content"]["floor"] == 0.0
        assert tables["replica"]["window"]["vision_content"]["label"] == ABSENT


class TestExactCounts:

    def _readings(self, folder):
        """Two episodes of 40 steps. Arm `bc_other` moves the gate and broadcast at step 30 of episode 38."""
        arrays = {}
        for episode in (38, 39):
            arrays["base__%d__broadcast" % episode] = np.zeros((40, 2))
            arrays["base__%d__gate" % episode] = np.zeros((40, 2))
            arrays["base__%d__winner" % episode] = np.array(["vision"] * 40)
            arrays["base__%d__ignited" % episode] = np.array([True] * 35 + [False] * 5)
            for name in ("vision_content", "audio_content", "semantic_content", "bid_raw"):
                arrays["base__%d__%s" % (episode, name)] = np.zeros((40, 2))
            for arm in ("bc_other", "settle_off"):
                for key in [k for k in arrays if k.startswith("base__%d__" % episode)]:
                    arrays[key.replace("base", arm, 1)] = arrays[key].copy()
            arrays["bc_other__%d__injected" % episode] = np.array([30, 34], dtype=np.int64)
        arrays["bc_other__38__broadcast"][30] = 1.0
        arrays["bc_other__38__gate"][30] = 1.0
        arrays["bc_other__39__winner"][34] = "audio"
        arrays["settle_off__38__broadcast"][32] = 1.0
        arrays["settle_off__38__broadcast"][36] = 1.0
        arrays["settle_off__39__audio_content"][3] = 1.0
        np.savez(os.path.join(folder, "readings_s1_train.npz"), **arrays)

    def test_every_count_is_worked_out_by_hand(self, tmp_path):
        self._readings(str(tmp_path))
        counts = exact_counts([str(tmp_path)])["s1_train"]
        assert counts["bc_other"]["injected"] == 4
        assert counts["bc_other"]["gate_moved"] == 1
        assert counts["bc_other"]["broadcast_moved"] == 1
        assert counts["bc_other"]["same_broadcast_other_winner"] == 1
        assert counts["bc_other"]["steps_compared_vision_content"] == 80
        assert counts["bc_other"]["steps_changed_vision_content"] == 0
        # Window steps 30 to 34 ignite, so step 32 counts and step 36 does not.
        assert counts["settle_off"][38]["ignited_steps"] == 5
        assert counts["settle_off"][38]["broadcast_moved"] == 1
        assert counts["settle_off"][39]["audio_content_moved"] == 1
        assert "bc_random" not in counts

    def test_changed_steps_has_no_tolerance(self):
        base = np.zeros((3, 2))
        other = base.copy()
        other[1, 1] = 1e-30
        assert changed_steps(base, other) == 1


def _run_config(steps: int) -> dict:
    args = argparse.Namespace(
        episodes=1, max_steps=steps, action_dim=2, lr=1e-3, render=False, env="dark_room",
        log_dir="runs/_test", seed=0, enable_audio=True, existence_drive="on",
        rssm_latent_mode="continuous", capsule_workspace_source="all_levels",
        vision_bid_reduction="zscore", audio_salience="surprise", bid_precision="gain",
        ignition_rule="tolerance", ignition_tolerance_sd=1.0, broadcast_merge="top_winner",
        map_binding="rhythm")
    config = build_config(args)
    config["device"] = "cpu"
    config["action_selection"]["device"] = "cpu"
    config["action_selection"]["env_continuous"] = True
    config["ablate_gate_diversity"] = True
    return config


STEPS = 7
PERTURBED_STEPS = [2, 4]


def _synthetic_record() -> EpisodeRecord:
    rng = np.random.default_rng(11)
    frames = rng.integers(0, 255, (STEPS, 224, 224, 3), dtype=np.uint8)
    waves = {t: rng.standard_normal((4, 1056)).astype(np.float32) * 0.1 for t in range(1, STEPS)}
    actions = [[0.25, -0.5]] * STEPS
    return EpisodeRecord(index=0, frames=frames, waveforms=waves, rewards=[-0.01] * STEPS,
                         facts=[{"battery": 0.9 - 0.001 * t} for t in range(STEPS)], actions=actions,
                         vectors={}, obs_map=None)


@pytest.fixture(scope="module")
def replays():
    config = _run_config(STEPS)
    record = _synthetic_record()

    def make_agent():
        return build_agent(config, "rhythm", seed=3, mode="train")

    found = {}
    for name in ("base", "replica", "noise", "zero_h", "settle_off", "bc_random"):
        found[name] = replay_arm(make_agent, ARMS[name], record, record, PERTURBED_STEPS)
    return found


@pytest.fixture(scope="module")
def upstream_replays():
    config = _run_config(STEPS)
    record = _synthetic_record()
    rng = np.random.default_rng(23)
    other = EpisodeRecord(index=1, frames=rng.integers(0, 255, (STEPS, 224, 224, 3), dtype=np.uint8),
                          waveforms=record.waveforms, rewards=record.rewards, facts=record.facts,
                          actions=record.actions, vectors={}, obs_map=None)

    def make_agent():
        return build_agent(config, "rhythm", seed=3, mode="train")

    steps = [2, 3]
    other_base = replay_arm(make_agent, ARMS["base"], other, record, [2])
    base = replay_arm(make_agent, ARMS["base"], record, other, [2], tap_steps=steps)
    reference = base["kept_taps"]
    return {"base": base,
            "replica": replay_arm(make_agent, ARMS["replica"], record, other, [2],
                                  tap_steps=steps, tap_reference=reference),
            "history_swap": replay_arm(make_agent, ARMS["history_swap"], record, other, [2],
                                       other_states=other_base["kept_states"], tap_steps=steps,
                                       tap_reference=reference)}


class TestUpstreamReadings:

    def test_the_base_keeps_each_reading_at_the_chosen_steps_only(self, upstream_replays):
        kept = upstream_replays["base"]["kept_taps"]
        assert sorted(kept) == [2, 3]
        assert kept[2]["obs_map"].size == 64 * 16 * 16
        assert kept[2]["h_state"].size == 64 * 16 * 16
        assert kept[2]["posterior"].size == 32 * 32 * 16 * 16

    def test_a_replica_reads_no_change_at_any_upstream_tap(self, upstream_replays):
        for tap in UPSTREAM_TAPS:
            assert upstream_replays["replica"]["tap_" + tap][2] == 0.0
            assert upstream_replays["replica"]["tap_" + tap][3] == 0.0

    def test_a_swapped_history_leaves_the_map_of_the_same_frame_and_moves_the_state(self, upstream_replays):
        swapped = upstream_replays["history_swap"]
        assert swapped["tap_obs_map"][2] == 0.0
        assert swapped["tap_h_state"][2] > 0.0
        assert swapped["tap_posterior"][2] > 0.0

    def test_steps_without_a_reading_hold_no_number(self, upstream_replays):
        assert np.isnan(upstream_replays["replica"]["tap_h_state"][0])


class TestRealLoop:
    """The training loop itself, on untrained weights, with the recorded action fed back."""

    def test_the_loop_ran_every_step_and_kept_every_reading(self, replays):
        assert replays["base"]["vision_content"].shape == (STEPS, 256)
        assert replays["base"]["audio_content"].shape == (STEPS, 256)
        assert replays["base"]["bid_raw"].shape == (STEPS, len(MODULES))
        assert replays["base"]["action_mean"].shape == (STEPS, 2)

    def test_the_environment_receives_the_recorded_action_and_not_the_policy_output(self, replays):
        assert all(list(action) == [0.25, -0.5] for action in replays["base"]["received_actions"])

    def test_two_replays_with_the_same_noise_are_identical(self, replays):
        changes = step_changes(replays["base"], replays["replica"])
        for name, change in changes.items():
            assert float(np.max(change)) == 0.0, name

    def test_another_noise_stream_changes_the_vision_content(self, replays):
        changes = step_changes(replays["base"], replays["noise"])
        assert float(np.median(changes["vision_content"])) > 0.0

    def test_a_zeroed_h_state_changes_the_vision_content_from_the_second_step_on(self, replays):
        changes = step_changes(replays["base"], replays["zero_h"])
        assert float(changes["vision_content"][0]) == 0.0
        assert float(changes["vision_content"][2]) > 0.0

    def test_the_arm_with_one_settle_cycle_uses_one_cycle(self, replays):
        assert set(replays["settle_off"]["cycles"].tolist()) == {1}

    def test_the_perturbation_reaches_a_reader_of_the_broadcast(self, replays):
        injected = replays["bc_random"]["injected"]
        assert injected, "no step ignited, so nothing was perturbed"
        changes = step_changes(replays["base"], replays["bc_random"])
        assert all(float(changes["gate"][t]) > 0.0 for t in injected)
        assert all(float(changes["broadcast"][t]) > 0.0 for t in injected)

    def test_the_default_loop_has_no_path_from_the_broadcast_to_the_next_module_output(self, replays):
        injected = replays["bc_random"]["injected"]
        assert injected, "no step ignited, so nothing was perturbed"
        changes = step_changes(replays["base"], replays["bc_random"])
        for t in lag_steps(injected, 1, STEPS):
            assert float(changes["vision_content"][t]) == 0.0
            assert float(changes["audio_content"][t]) == 0.0
            assert float(changes["bid_raw_vision"][t]) == 0.0
            assert float(changes["bid_raw_audio"][t]) == 0.0

    def test_only_vision_and_audio_receive_the_broadcast_in_the_loop(self, replays):
        calls = replays["base"]["calls"]
        assert calls["vision_feedback"] > 0
        assert calls["audio_feedback"] > 0
        assert calls["body_feedback"] == 0
        assert calls["semantic_feedback"] == 0
        assert calls["semantic_forward"] == 0

    def test_the_fingerprint_sees_a_changed_weight(self):
        agent = build_agent(_run_config(STEPS), "rhythm", seed=3, mode="train")
        before = weights_fingerprint(agent)
        with torch.no_grad():
            next(agent.tectum.parameters()).add_(1.0)
        assert weights_fingerprint(agent) != before

    def test_weights_saved_as_the_recorder_saves_them_load_into_a_new_agent(self):
        config = _run_config(STEPS)
        saved_agent = build_agent(config, "rhythm", seed=3, mode="train")
        saved_agent.tectum.h_state = torch.zeros(1, 64, 16, 16)
        saved_agent.tectum.z_state = torch.zeros(1, 32, 32, 16, 16)
        saved_agent.workspace.binding_system.current_phases = torch.ones(1, 5, 2)
        weights = _state_dicts(modules_with_parameters({
            "tectum": saved_agent.tectum, "gate": saved_agent.gate,
            "workspace_binding": saved_agent.workspace.binding_system, "rnd": saved_agent.rnd,
            "auditory_specialist": saved_agent.auditory, "action_core": saved_agent.action_core}))
        loaded_agent = build_agent(config, "rhythm", seed=4, mode="train")
        assert weights_fingerprint(loaded_agent) != weights_fingerprint(saved_agent)

        load_weights(loaded_agent, weights)

        assert weights_fingerprint(loaded_agent) == weights_fingerprint(saved_agent)
        binding = loaded_agent.workspace.binding_system
        assert torch.equal(binding.current_phases, torch.ones(1, 5, 2))
        assert binding.current_phases.device == next(binding.parameters()).device

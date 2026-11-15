"""The room options of the competing goals gate (simulations/environments/simple_visual_env.py).

`docs/decisions/2026_11_08_competing_goals_gate.md`, check 1. Food sizes, waits counted inside an
object's radius, a change of the sizes at one episode, and forced episodes with one object. Expected
rewards are the sizes given to the room. The digests of `tests/test_dark_room_test_bed.py` were
recorded before this change, so they show that the room with every new option off is unchanged.
"""
from __future__ import annotations

import os
import sys
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulations.environments import audio_mixin
from simulations.environments.simple_visual_env import LIGHT_COLOUR, SimpleVisualEnv
from tests.test_dark_room_test_bed import DIGEST_BEFORE, digest

ACTION = np.array([0.3, -0.7], dtype=np.float32)
STILL = np.zeros(2, dtype=np.float32)
LOW, HIGH = audio_mixin.BINAURAL_TONE_HZ, audio_mixin.DECOY_TONE_HZ
STEP_COST = -0.01
BIG, SMALL, DECOY_FOOD = 32.0, 4.0, 3.0


def make_room(view="agent_centered", seed=5, **options):
    np.random.seed(seed)
    env = SimpleVisualEnv(width=224, height=224, audio="binaural", audio_channels=4,
                          report_collision=True, view=view, decoy_sound=True, decoy_visible=True, **options)
    env.seed_audio(seed)
    env.seed_decoy(seed)
    return env


def stand_at(room, place) -> float:
    """The reward of one step with no movement at `place`."""
    room.agent_pos = np.array(place, dtype=np.float32)
    return room.step(STILL)[1]


def far_from_both(room) -> np.ndarray:
    """A corner of the room that is outside the radius of the light and of the decoy."""
    reach = room.light_radius + room.agent_radius
    for corner in ([0, 0], [room.width, 0], [0, room.height], [room.width, room.height]):
        gaps = [np.linalg.norm(np.array(corner) - place) for place in (room.light_pos, room.decoy_pos)]
        if min(gaps) >= reach:
            return np.array(corner, dtype=np.float32)
    raise AssertionError("no corner is outside both objects")


def heard_tones(room) -> list:
    """The frequencies of the tones that the room mixes at its next step."""
    with mock.patch.object(audio_mixin, "source_ears", wraps=audio_mixin.source_ears) as ears:
        room.step(STILL)
    return [call.args[1] for call in ears.call_args_list]


def episode_start(episode: int, view="full", **options):
    """The room and its first frame at the reset of `episode`, after one step in each earlier episode."""
    room = make_room(view=view, **options)
    frame, _ = room.reset()
    for _ in range(episode):
        room.step(ACTION)
        frame, _ = room.reset()
    return room, frame


class TestEveryOptionOff:
    @pytest.mark.parametrize("case", sorted(DIGEST_BEFORE, key=str))
    def test_the_output_is_the_output_before_the_change(self, case):
        view, pulse, food_switch = case
        room = make_room(view=view, pulse=pulse, food_switch=food_switch)
        assert digest(room) == DIGEST_BEFORE[case]

    def test_none_given_explicitly_is_the_default(self):
        room = make_room(food_sizes=None, food_waits=None, food_change=None, food_sizes_after=None, forced=None)
        assert digest(room) == DIGEST_BEFORE[("agent_centered", False, None)]

    def test_the_decoy_still_gives_no_food_and_no_new_truth_key_appears(self):
        room = make_room()
        room.reset()
        assert stand_at(room, room.decoy_pos) == STEP_COST
        assert stand_at(room, room.light_pos) == 1.0
        facts = room.step(STILL)[4]
        assert not [key for key in facts if key in ("_truth_food_sizes", "_truth_present", "_truth_food_object")]


class TestRules:
    @pytest.mark.parametrize("option", [{"food_sizes": (BIG, DECOY_FOOD)}, {"food_waits": (40, 0)},
                                        {"food_change": 1, "food_sizes_after": (SMALL, DECOY_FOOD)},
                                        {"forced": 6}])
    def test_each_option_needs_the_visible_decoy(self, option):
        np.random.seed(5)
        with pytest.raises(ValueError, match="decoy_visible"):
            SimpleVisualEnv(width=224, height=224, audio="binaural", audio_channels=4, decoy_sound=True, **option)

    @pytest.mark.parametrize("option", [{"food_sizes": (BIG, DECOY_FOOD)}, {"food_waits": (40, 0)},
                                        {"food_change": 1, "food_sizes_after": (SMALL, DECOY_FOOD)},
                                        {"forced": 6}])
    def test_no_option_works_with_the_food_switch(self, option):
        with pytest.raises(ValueError, match="food_switch"):
            make_room(food_switch=1, **option)

    def test_the_change_and_the_sizes_after_it_need_each_other(self):
        with pytest.raises(ValueError, match="food_sizes_after"):
            make_room(food_change=1)
        with pytest.raises(ValueError, match="food_change"):
            make_room(food_sizes_after=(SMALL, DECOY_FOOD))

    @pytest.mark.parametrize("sizes", [(BIG,), (BIG, DECOY_FOOD, 1.0), (BIG, 0.0), (-1.0, DECOY_FOOD)])
    def test_sizes_are_two_numbers_above_zero(self, sizes):
        with pytest.raises(ValueError, match="food_sizes"):
            make_room(food_sizes=sizes)

    @pytest.mark.parametrize("waits", [(40,), (40, 0, 0), (-1, 0), (1.5, 0)])
    def test_waits_are_two_whole_numbers_of_zero_or_more(self, waits):
        with pytest.raises(ValueError, match="food_waits"):
            make_room(food_waits=waits)

    @pytest.mark.parametrize("option", [{"forced": -1}, {"food_change": -1, "food_sizes_after": (SMALL, DECOY_FOOD)}])
    def test_an_episode_number_below_zero_is_refused(self, option):
        with pytest.raises(ValueError):
            make_room(**option)


class TestFoodSizes:
    def test_each_object_gives_its_own_size_and_outside_is_the_step_cost(self):
        room = make_room(food_sizes=(BIG, DECOY_FOOD))
        room.reset()
        assert stand_at(room, room.light_pos) == BIG
        assert stand_at(room, room.decoy_pos) == DECOY_FOOD
        assert stand_at(room, far_from_both(room)) == STEP_COST

    def test_the_food_comes_at_every_step_inside_as_the_present_food_does(self):
        room = make_room(food_sizes=(BIG, DECOY_FOOD))
        room.reset()
        assert [stand_at(room, room.decoy_pos) for _ in range(3)] == [DECOY_FOOD] * 3

    def test_the_sizes_change_the_reward_only(self):
        """Frames and sound are the same with and without the sizes; the digest of the test bed has the reward too."""
        plain = make_room()
        plain_frame, _ = plain.reset()
        plain_step = plain.step(ACTION)
        sized = make_room(food_sizes=(BIG, DECOY_FOOD))
        sized_frame, _ = sized.reset()
        sized_step = sized.step(ACTION)
        assert np.array_equal(plain_frame, sized_frame)
        assert np.array_equal(plain_step[0], sized_step[0])
        assert np.array_equal(plain_step[4]["audio_waveform"], sized_step[4]["audio_waveform"])

    def test_the_truth_keys_name_the_sizes_and_the_object_that_fed(self):
        room = make_room(food_sizes=(BIG, DECOY_FOOD))
        room.reset()
        room.agent_pos = room.decoy_pos.astype(np.float32).copy()
        facts = room.step(STILL)[4]
        assert facts["_truth_food_sizes"] == [BIG, DECOY_FOOD]
        assert facts["_truth_food_object"] == "decoy"
        room.agent_pos = far_from_both(room)
        assert room.step(STILL)[4]["_truth_food_object"] is None


class TestFoodWaits:
    def test_the_light_gives_the_step_cost_for_the_wait_and_then_its_food(self):
        room = make_room(food_sizes=(BIG, DECOY_FOOD), food_waits=(3, 0))
        room.reset()
        rewards = [stand_at(room, room.light_pos) for _ in range(5)]
        assert rewards == [STEP_COST, STEP_COST, STEP_COST, BIG, BIG]

    def test_an_object_with_no_wait_feeds_at_the_first_step_inside(self):
        room = make_room(food_sizes=(BIG, DECOY_FOOD), food_waits=(3, 0))
        room.reset()
        assert stand_at(room, room.decoy_pos) == DECOY_FOOD

    def test_the_steps_inside_are_counted_over_the_episode_and_not_in_a_row(self):
        room = make_room(food_sizes=(BIG, DECOY_FOOD), food_waits=(3, 0))
        room.reset()
        away = far_from_both(room)
        rewards = [stand_at(room, room.light_pos), stand_at(room, room.light_pos), stand_at(room, away),
                   stand_at(room, room.light_pos), stand_at(room, room.light_pos)]
        assert rewards == [STEP_COST, STEP_COST, STEP_COST, STEP_COST, BIG]

    def test_each_object_has_its_own_count(self):
        room = make_room(food_sizes=(BIG, DECOY_FOOD), food_waits=(2, 1))
        room.reset()
        rewards = [stand_at(room, room.decoy_pos), stand_at(room, room.light_pos), stand_at(room, room.decoy_pos),
                   stand_at(room, room.light_pos), stand_at(room, room.light_pos)]
        assert rewards == [STEP_COST, STEP_COST, DECOY_FOOD, STEP_COST, BIG]

    def test_the_count_starts_again_at_each_episode(self):
        room = make_room(food_sizes=(BIG, DECOY_FOOD), food_waits=(1, 0))
        room.reset()
        first = [stand_at(room, room.light_pos) for _ in range(2)]
        room.reset()
        second = [stand_at(room, room.light_pos) for _ in range(2)]
        assert first == [STEP_COST, BIG] and second == [STEP_COST, BIG]

    def test_waits_without_sizes_delay_the_present_food_of_the_light(self):
        room = make_room(food_waits=(1, 0))
        room.reset()
        assert [stand_at(room, room.light_pos) for _ in range(2)] == [STEP_COST, 1.0]
        assert stand_at(room, room.decoy_pos) == STEP_COST


class TestFoodChange:
    def test_from_the_change_episode_on_the_sizes_are_the_sizes_after(self):
        room = make_room(food_sizes=(BIG, DECOY_FOOD), food_change=1, food_sizes_after=(SMALL, DECOY_FOOD))
        room.reset()
        before = [stand_at(room, room.light_pos), stand_at(room, room.decoy_pos)]
        room.reset()
        after = [stand_at(room, room.light_pos), stand_at(room, room.decoy_pos)]
        room.reset()
        later = stand_at(room, room.light_pos)
        assert before == [BIG, DECOY_FOOD] and after == [SMALL, DECOY_FOOD] and later == SMALL

    def test_a_reset_with_no_step_after_it_does_not_count(self):
        """Counted as for food_switch: train_rlhf makes one reset before its first episode."""
        room = make_room(food_sizes=(BIG, DECOY_FOOD), food_change=1, food_sizes_after=(SMALL, DECOY_FOOD))
        room.reset()
        room.reset()
        assert stand_at(room, room.light_pos) == BIG
        room.reset()
        assert stand_at(room, room.light_pos) == SMALL

    def test_the_waits_stay_after_the_change(self):
        room = make_room(food_sizes=(BIG, DECOY_FOOD), food_waits=(2, 0), food_change=0,
                         food_sizes_after=(SMALL, DECOY_FOOD))
        room.reset()
        assert [stand_at(room, room.light_pos) for _ in range(3)] == [STEP_COST, STEP_COST, SMALL]

    def test_the_truth_key_names_the_sizes_of_the_episode(self):
        room = make_room(food_sizes=(BIG, DECOY_FOOD), food_change=1, food_sizes_after=(SMALL, DECOY_FOOD))
        room.reset()
        assert room.step(STILL)[4]["_truth_food_sizes"] == [BIG, DECOY_FOOD]
        room.reset()
        assert room.step(STILL)[4]["_truth_food_sizes"] == [SMALL, DECOY_FOOD]


class TestForcedEpisodes:
    OPTIONS = {"food_sizes": (BIG, DECOY_FOOD), "forced": 2}

    def test_an_even_forced_episode_has_the_light_only(self):
        room, _ = episode_start(0, **self.OPTIONS)
        assert room.step(STILL)[4]["_truth_present"] == "light"
        assert heard_tones(room) == [LOW]
        assert stand_at(room, room.decoy_pos) == STEP_COST
        assert stand_at(room, room.light_pos) == BIG

    def test_an_odd_forced_episode_has_the_decoy_only(self):
        room, _ = episode_start(1, **self.OPTIONS)
        assert room.step(STILL)[4]["_truth_present"] == "decoy"
        assert heard_tones(room) == [HIGH]
        assert stand_at(room, room.light_pos) == STEP_COST
        assert stand_at(room, room.decoy_pos) == DECOY_FOOD

    def test_after_the_forced_episodes_both_objects_are_present(self):
        room, _ = episode_start(2, **self.OPTIONS)
        assert room.step(STILL)[4]["_truth_present"] == "both"
        assert heard_tones(room) == [LOW, HIGH]
        assert stand_at(room, room.light_pos) == BIG and stand_at(room, room.decoy_pos) == DECOY_FOOD

    @pytest.mark.parametrize("episode, absent", [(0, "decoy_pos"), (1, "light_pos")])
    def test_the_absent_object_is_not_drawn_at_the_reset_or_after_a_step(self, episode, absent):
        _, plain = episode_start(episode)
        room, forced = episode_start(episode, **self.OPTIONS)
        changed = np.any(plain != forced, axis=-1)
        rows, columns = np.nonzero(changed)
        centre = np.array([columns.mean(), rows.mean()])
        assert changed.sum() > 0
        assert np.all(forced[changed] == 0) and np.all(plain[changed] == LIGHT_COLOUR)
        assert np.linalg.norm(centre - getattr(room, absent)) < room.light_radius
        after = room.step(STILL)[0]
        assert np.all(after[changed] == 0)

    @pytest.mark.parametrize("episode, present", [(0, "light_pos"), (1, "decoy_pos")])
    def test_the_absent_object_is_not_drawn_in_the_agent_centred_view(self, episode, present):
        """The agent starts midway, so the frame centre is the agent and each object fills one side."""
        _, plain = episode_start(episode, view="agent_centered", start_between=True)
        room, forced = episode_start(episode, view="agent_centered", start_between=True, **self.OPTIONS)
        lit = np.all(forced == LIGHT_COLOUR, axis=-1)
        lost = np.all(plain == LIGHT_COLOUR, axis=-1) & ~lit
        assert lit.sum() > 0 and lost.sum() > 0
        assert np.all(forced[lost] == 0)
        side = getattr(room, present) - room.agent_pos            # the direction of the present object
        rows, columns = np.nonzero(lit)
        offset = np.array([columns.mean(), rows.mean()]) - np.array([room.width, room.height]) / 2.0
        assert np.dot(offset, side) > 0
        rows, columns = np.nonzero(lost)
        offset = np.array([columns.mean(), rows.mean()]) - np.array([room.width, room.height]) / 2.0
        assert np.dot(offset, side) < 0
        after = room.step(STILL)[0]
        assert np.all(after[lost] == 0)

    @pytest.mark.parametrize("episode", [0, 1, 2])
    def test_the_places_are_the_places_without_forced_episodes(self, episode):
        plain, _ = episode_start(episode, start_between=True)
        places = [plain.light_pos.copy(), plain.decoy_pos.copy(), plain.agent_pos.copy()]
        room, _ = episode_start(episode, start_between=True, **self.OPTIONS)
        assert all(np.array_equal(a, b) for a, b in zip(places, (room.light_pos, room.decoy_pos, room.agent_pos)))

    def test_a_reset_with_no_step_after_it_does_not_count(self):
        room = make_room(**self.OPTIONS)
        room.reset()
        room.reset()
        assert room.step(STILL)[4]["_truth_present"] == "light"
        room.reset()
        assert room.step(STILL)[4]["_truth_present"] == "decoy"

    def test_the_pulse_rhythms_are_the_rhythms_without_forced_episodes(self):
        plain = make_room(pulse=True)
        plain.reset()
        rhythms = dict(plain.pulse)
        room = make_room(pulse=True, **self.OPTIONS)
        room.reset()
        assert room.pulse == rhythms

"""The two room options of the test bed for a relation of content.

docs/decisions/2026_11_02_content_relation_design.md asks for a decoy that vision can tell from the
light (`decoy_colour`) and for episodes in which the pairing of objects and tones is broken
(`tone_switch`). The digests below were recorded on the code before this change (revision 380223f),
on Python 3.12.10 and 3.8.3, so they show that the output with both options off is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulations.environments import audio_mixin
from simulations.environments.simple_visual_env import LIGHT_COLOUR, PULSE_VISUAL_GAIN, SimpleVisualEnv

ACTION = np.array([0.3, -0.7], dtype=np.float32)
DIGEST_BEFORE = {("agent_centered", False, None): "a4cd34ea22a2453f967e350246193cab",
                 ("full", False, None): "49cb28685d27d366c77709f3824bf846",
                 ("agent_centered", True, None): "5f6dd5249aeac687ecbd58ac53a9820b",
                 ("agent_centered", False, 1): "78f71b737e10609a9bfa1cab8e65fd00"}
DECOY_COLOUR = (200, 255, 255)
LOW, HIGH = audio_mixin.BINAURAL_TONE_HZ, audio_mixin.DECOY_TONE_HZ


def make_room(view="agent_centered", pulse=False, food_switch=None, seed=5, **options):
    np.random.seed(seed)
    env = SimpleVisualEnv(width=224, height=224, audio="binaural", audio_channels=4,
                          report_collision=True, view=view, decoy_sound=True, decoy_visible=True,
                          pulse=pulse, food_switch=food_switch, **options)
    env.seed_audio(seed)
    env.seed_decoy(seed)
    return env


def digest(env, episodes=2, steps=6):
    """md5 of the frames, the sound and the rewards of two short episodes."""
    record = hashlib.md5()
    for _ in range(episodes):
        frame, _ = env.reset()
        record.update(frame.tobytes())
        for _ in range(steps):
            frame, reward, _, _, step_facts = env.step(ACTION)
            record.update(frame.tobytes())
            record.update(step_facts["audio_waveform"].tobytes())
            record.update(np.float64(reward).tobytes())
    return record.hexdigest()


class TestBothOptionsOff:
    @pytest.mark.parametrize("case", sorted(DIGEST_BEFORE, key=str))
    def test_the_output_is_the_output_before_the_change(self, case):
        view, pulse, food_switch = case
        assert digest(make_room(view, pulse, food_switch)) == DIGEST_BEFORE[case]

    def test_none_given_explicitly_is_the_default(self):
        room = make_room(decoy_colour=None, tone_switch=None)
        assert digest(room) == DIGEST_BEFORE[("agent_centered", False, None)]


class TestDecoyColour:
    def test_only_the_decoy_pixels_change_and_they_take_its_colour(self):
        plain, _ = make_room(view="full").reset()
        coloured, _ = make_room(view="full", decoy_colour=DECOY_COLOUR).reset()
        changed = np.any(plain != coloured, axis=-1)
        assert changed.sum() > 0
        assert np.all(coloured[changed] == DECOY_COLOUR)
        assert np.all(plain[changed] == LIGHT_COLOUR)

    def test_the_sound_does_not_change(self):
        """Each room is reset right after it is made, because the layout comes from numpy's global stream."""
        plain = make_room()
        plain.reset()
        coloured = make_room(decoy_colour=DECOY_COLOUR)
        coloured.reset()
        assert np.array_equal(plain.step(ACTION)[4]["audio_waveform"],
                              coloured.step(ACTION)[4]["audio_waveform"])

    def test_the_pulse_scales_the_decoy_colour_as_it_scales_the_light(self):
        room = make_room(pulse=True, decoy_colour=DECOY_COLOUR)
        room.reset()
        room.step(ACTION)
        decoy_gain, light_gain = (room.pulse_gain(name, PULSE_VISUAL_GAIN) for name in ("decoy", "light"))
        assert room._object_colour("decoy") == tuple(int(round(c * decoy_gain)) for c in DECOY_COLOUR)
        assert room._object_colour("light") == tuple(int(round(c * light_gain)) for c in LIGHT_COLOUR)

    def test_the_colour_needs_a_visible_decoy(self):
        with pytest.raises(ValueError, match="decoy_colour needs decoy_visible"):
            SimpleVisualEnv(width=224, height=224, audio="binaural", audio_channels=4,
                            decoy_sound=True, decoy_colour=DECOY_COLOUR)

    def test_a_value_outside_0_to_255_is_refused(self):
        with pytest.raises(ValueError, match="decoy_colour must be three values"):
            make_room(decoy_colour=(0, 300, 0))


class TestToneSwitch:
    @staticmethod
    def tones(room):
        """The frequencies the room gives the light and then the decoy at its next step."""
        with mock.patch.object(audio_mixin, "source_ears", wraps=audio_mixin.source_ears) as ears:
            room.step(ACTION)
        return [call.args[1] for call in ears.call_args_list]

    def test_before_the_switch_the_light_has_the_lower_tone(self):
        room = make_room(tone_switch=2)
        room.reset()
        assert self.tones(room) == [LOW, HIGH]

    def test_from_the_switch_episode_on_the_tones_are_swapped(self):
        """Counted as for food_switch: a reset with no step after it does not count."""
        room = make_room(tone_switch=1)
        room.reset()
        room.reset()
        first = self.tones(room)
        room.reset()
        assert first == [LOW, HIGH]
        assert self.tones(room) == [HIGH, LOW]

    def test_the_truth_key_names_the_pairing_only_with_the_option(self):
        room, plain = make_room(tone_switch=0), make_room()
        room.reset()
        plain.reset()
        assert room.step(ACTION)[4]["_truth_tones_swapped"] is True
        assert "_truth_tones_swapped" not in plain.step(ACTION)[4]

    def test_the_pixels_do_not_change(self):
        plain = make_room()
        plain_start, _ = plain.reset()
        switched = make_room(tone_switch=0)
        switched_start, _ = switched.reset()
        assert np.array_equal(plain_start, switched_start)
        assert np.array_equal(plain.step(ACTION)[0], switched.step(ACTION)[0])

    def test_the_switch_needs_the_decoy_sound_and_a_count_of_0_or_more(self):
        with pytest.raises(ValueError, match="tone_switch needs decoy_sound"):
            SimpleVisualEnv(width=224, height=224, audio="binaural", audio_channels=4, tone_switch=0)
        with pytest.raises(ValueError, match="0 or more"):
            make_room(tone_switch=-1)


class TestTrainingFlags:
    @staticmethod
    def parser():
        from scripts.training import train_rlhf
        captured = {}

        def capture(self, *args, **kwargs):
            captured["parser"] = self
            raise RuntimeError("captured")

        with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
            with pytest.raises(RuntimeError):
                train_rlhf.main()
        return captured["parser"]

    def test_both_flags_are_off_by_default(self):
        args = self.parser().parse_args([])
        assert args.dark_room_decoy_colour is None
        assert args.dark_room_tone_switch is None

    def test_the_flags_take_a_colour_and_an_episode(self):
        args = self.parser().parse_args(["--dark-room-decoy-colour", "200,255,255",
                                         "--dark-room-tone-switch", "20"])
        assert args.dark_room_decoy_colour == "200,255,255"
        assert args.dark_room_tone_switch == 20

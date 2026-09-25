"""
Command line of scripts/training/train_rlhf.py: what its flags promise.

main() builds its parser inline, so these tests capture the parser at the moment
main() would parse sys.argv, then inspect or drive it without starting a run.

1. --enable-wm-predict trains a categorical KL (models/core/world_model_objective.py)
   on the RSSM prior and posterior logits. With --rssm-latent-mode continuous those
   slots hold Gaussian means, so the pair trained the wrong objective without a
   word. The pair is now a command line error.
2. Four help texts said things the code does not do. Each test below pins the
   corrected statement against the code it describes.
"""
from __future__ import annotations

import argparse
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.training import train_rlhf


class _ParserCaptured(Exception):
    pass


def _parser() -> argparse.ArgumentParser:
    captured = {}

    def capture(self, *args, **kwargs):
        captured["parser"] = self
        raise _ParserCaptured

    with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
        with pytest.raises(_ParserCaptured):
            train_rlhf.main()
    return captured["parser"]


def _help(flag: str) -> str:
    return next(action.help for action in _parser()._actions
                if flag in action.option_strings)


# --- 1. world model objective with the continuous latent -------------------------

def test_wm_predict_with_the_continuous_latent_is_rejected(capsys):
    parser = _parser()
    args = parser.parse_args(["--enable-wm-predict", "--rssm-latent-mode", "continuous"])
    with pytest.raises(SystemExit) as exit_info:
        train_rlhf._reject_incompatible_flags(parser, args)
    assert exit_info.value.code == 2
    message = capsys.readouterr().err
    assert "--enable-wm-predict" in message and "continuous" in message


@pytest.mark.parametrize("argv", [
    [],
    ["--enable-wm-predict"],
    ["--rssm-latent-mode", "continuous"],
    ["--enable-wm-predict", "--rssm-latent-mode", "discrete"],
])
def test_each_flag_on_its_own_is_accepted(argv):
    parser = _parser()
    train_rlhf._reject_incompatible_flags(parser, parser.parse_args(argv))


def test_main_stops_before_building_anything(monkeypatch):
    def fail_if_called(args):
        raise AssertionError("main() got past the flag check and built a config")

    monkeypatch.setattr(sys, "argv", ["train_rlhf", "--enable-wm-predict",
                                      "--rssm-latent-mode", "continuous"])
    monkeypatch.setattr(train_rlhf, "build_config", fail_if_called)
    with pytest.raises(SystemExit) as exit_info:
        train_rlhf.main()
    assert exit_info.value.code == 2


# --- 2. help texts -------------------------------------------------------------

def test_policy_input_help_drops_the_retracted_claim_and_names_every_choice():
    text = _help("--policy-input")
    # The 99% h_state decodability was a leakage artifact, corrected 2026-06-14
    # (see the tap comment in init_components).
    assert "99%" not in text
    assert "obsmem-conv" in text


def test_record_episodes_help_says_every_episode_gets_the_light_tier():
    # Pinned in tests/test_session_recorder.py
    # (test_every_episode_gets_steps_frames_and_vectors).
    text = _help("--record-episodes")
    assert "every episode" in text
    assert "light" in text


def test_difficulty_help_says_three_behaves_as_two():
    text = _help("--difficulty")
    assert "3 behaves as 2" in text
    assert "DMTS" in text


def test_dmts_difficulty_three_draws_the_same_distractors_as_two():
    """The help now states this. At most two features are ever shared."""
    from simulations.environments.dmts_env import DMTSEnv

    def distractors(overlap):
        env = DMTSEnv(distractor_overlap=overlap)
        env.reset(seed=7)
        return [tuple(env._make_distractor()[k] for k in ("shape", "color", "size"))
                for _ in range(30)]

    assert distractors(3) == distractors(2)
    assert distractors(2) != distractors(1)


def test_enable_audio_help_says_only_the_dark_room_emits_sound():
    text = _help("--enable-audio")
    assert "dark_room" in text
    assert "silent" in text

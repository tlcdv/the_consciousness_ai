"""Regression test: --seed must fix the environment's trial sequence.

Before this fix, train_rlhf never passed the seed to env.reset, and DMTSEnv and
WCSTEnv drew trials from an unseeded generator, so two runs with the same --seed
saw different stimuli. The help text said the reset was seeded.
"""
from __future__ import annotations

import numpy as np

from scripts.training.train_rlhf import seed_environment
from simulations.environments.dmts_env import DMTSEnv
from simulations.environments.simple_visual_env import SimpleVisualEnv
from simulations.environments.wcst_env import WCSTEnv


def _dmts_shapes(seed, trials=6):
    env = DMTSEnv(num_trials=trials)
    seed_environment(env, seed)
    _, info = env.reset()           # run_episode resets again without a seed
    shapes, done = [info["sample_shape"]], False
    while not done:
        action = 1 if info["phase"] == "choice" else 0
        _, _, done, _, info = env.step(action)
        if info["phase"] == "fixation" and info["sample_shape"] != shapes[-1]:
            shapes.append(info["sample_shape"])
    return shapes


def test_same_seed_gives_same_dmts_trials():
    assert _dmts_shapes(42) == _dmts_shapes(42)


def test_different_seeds_give_different_dmts_trials():
    # Six shapes over six trials; two seeds matching by chance is negligible.
    assert _dmts_shapes(42) != _dmts_shapes(43)


def test_same_seed_gives_same_wcst_state():
    a, b = WCSTEnv(), WCSTEnv()
    seed_environment(a, 7)
    seed_environment(b, 7)
    assert a.reset()[1] == b.reset()[1]


def test_no_seed_leaves_the_environment_alone():
    env = DMTSEnv()
    rng_before = env._rng
    assert seed_environment(env, None) is False
    assert env._rng is rng_before


# --- the dark room --------------------------------------------------------------
#
# SimpleVisualEnv places the agent and the light with np.random, numpy's global
# stream, and never reads the generator that reset(seed=) initialises. Runs repeat
# because train_rlhf seeds the global stream first. These pin both halves.

def _dark_room_layout(global_seed, env_seed):
    np.random.seed(global_seed)
    env = SimpleVisualEnv(width=224, height=224)
    seed_environment(env, env_seed)
    env.reset()                     # run_episode resets again without a seed
    return np.concatenate([env.agent_pos, env.light_pos])


def test_same_global_seed_gives_the_same_dark_room_layout():
    assert np.array_equal(_dark_room_layout(5, 5), _dark_room_layout(5, 5))


def test_reset_seed_alone_does_not_fix_the_dark_room_layout():
    """Tripwire. If the environment starts using its seeded generator, this fails,
    and the reset and seed_environment docstrings must change with it."""
    assert not np.array_equal(_dark_room_layout(1, 7), _dark_room_layout(2, 7))


def test_navigation_info_has_the_documented_keys():
    """The module docstring listed `goal_room`; the environment writes `goal_rooms`."""
    from simulations.environments import navigation_env
    from simulations.environments.navigation_env import NavigationEnv

    documented = next(line for line in navigation_env.__doc__.splitlines()
                      if line.startswith("Info dict:"))
    keys = {key.strip(" .") for key in documented.split(":", 1)[1].split(",")}
    _, info = NavigationEnv(width=224, height=224).reset(seed=0)
    assert set(info) == keys

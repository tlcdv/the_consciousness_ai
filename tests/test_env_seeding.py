"""Regression test: --seed must fix the environment's trial sequence.

Before this fix, train_rlhf never passed the seed to env.reset, and DMTSEnv and
WCSTEnv drew trials from an unseeded generator, so two runs with the same --seed
saw different stimuli. The help text said the reset was seeded.
"""
from __future__ import annotations

from scripts.training.train_rlhf import seed_environment
from simulations.environments.dmts_env import DMTSEnv
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

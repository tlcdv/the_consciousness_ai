"""
Every key build_config writes into config["workspace"] must be a key GlobalWorkspace reads.

build_config used to set "broadcast_threshold" to 0.6, a key GlobalWorkspace never
reads. The threshold in force was GlobalWorkspace's own default for "ignition_threshold",
also 0.6, so no run changed, but editing the dead key did nothing at all. This records
every key GlobalWorkspace reads across the four binding configurations training can build
(akorn or komplex, content binding on or off) and fails on a key that nobody reads.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core.global_workspace import GlobalWorkspace
from scripts.training.train_rlhf import build_config


class _ReadRecorder(dict):
    """A dict that remembers which keys were looked up."""

    def __init__(self, data):
        super().__init__(data)
        self.read = set()

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self.read.add(key)
        return super().__contains__(key)


def _args(**overrides) -> argparse.Namespace:
    args = argparse.Namespace(action_dim=2, lr=1e-3, episodes=1, max_steps=10)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_every_workspace_config_key_is_read():
    written, read = set(), set()
    for mechanism, content_binding in itertools.product(("akorn", "komplex"), (False, True)):
        config = build_config(_args(binding_mechanism=mechanism,
                                    enable_content_binding=content_binding))
        recorder = _ReadRecorder(config["workspace"])
        GlobalWorkspace(recorder)
        written |= set(recorder)
        read |= recorder.read
    assert written - read == set(), "keys build_config writes that GlobalWorkspace never reads"


def test_training_threshold_is_unchanged():
    """The key rename moves no number. The threshold in force stays 0.6."""
    workspace = GlobalWorkspace(build_config(_args())["workspace"])
    assert workspace.ignition_threshold == 0.6

"""The calculations of the content map probe (scripts/analysis/probe_content_map_sensitivity.py)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core.capsule_composition import squash
from scripts.analysis import probe_content_map_sensitivity as probe


class TestSquash:
    def test_a_short_vector_shrinks_by_the_square_of_its_length(self):
        """Closed form: length n becomes n^2 / (1 + n^2), the premise of the hypothesis."""
        x = torch.tensor([[3e-3, 4e-3]], dtype=torch.float64)
        assert squash(x).norm().item() == pytest.approx(2.5e-5 / (1 + 2.5e-5), rel=1e-3)


class TestLocateFall:
    def test_finds_the_stage_with_the_largest_fall(self):
        changes = {"obs_map": 0.5, "state_tensor": 0.1, "primary": 0.1, "flat": 0.1, "content": 1e-5}
        stage, factor = probe.locate_fall(changes)
        assert stage == "projection"
        assert factor == pytest.approx(1e4)


class TestOffsetShare:
    def test_a_fixed_bias_with_a_small_motion_in_one_direction(self):
        bias = np.zeros(8)
        bias[0] = 2.0
        wave = np.zeros((100, 8))
        wave[:, 1] = 0.01 * np.sin(np.arange(100))
        share = probe.offset_share(bias + wave, bias)
        assert share["pca_dims_95"] == 1
        assert share["cosine_mean_bias"] == pytest.approx(1.0, abs=1e-3)
        assert share["mean_minus_bias_over_mean"] < 1e-3
        assert share["centred_rms_norm"] < 0.02


class TestLabels:
    def test_the_right_object_is_the_one_with_the_larger_x(self):
        facts = {"_truth_light_offset": [-3.0, 4.0], "_truth_decoy_offset": [0.0, 2.0]}
        assert probe.right_object_direction(facts).tolist() == pytest.approx([0.0, 1.0])

    def test_swapped_labels_keep_the_length_and_take_labels_of_other_episodes(self):
        y = np.arange(12.0)
        groups = np.repeat([0, 1, 2], 4)
        swapped = probe.swapped_labels(y, groups, np.random.default_rng(0))
        assert swapped.shape == y.shape
        for episode in (0, 1, 2):
            assert not set(swapped[groups == episode]) & set(y[groups == episode])


class TestToneAccuracy:
    def test_reads_a_label_that_is_in_the_input(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, 200)
        x = np.stack([labels * 2.0 - 1.0 + rng.normal(0, 0.1, 200), rng.normal(0, 1, 200)], axis=1)
        assert probe.tone_accuracy(x, labels, np.repeat(np.arange(20), 10)) > 0.95

    def test_does_not_read_a_label_that_is_not_in_the_input(self):
        rng = np.random.default_rng(1)
        labels = rng.integers(0, 2, 400)
        accuracy = probe.tone_accuracy(rng.normal(0, 1, (400, 4)), labels, np.repeat(np.arange(40), 10))
        assert 0.35 < accuracy < 0.65

    def test_refuses_a_label_without_two_classes(self):
        with pytest.raises(ValueError):
            probe.tone_accuracy(np.zeros((10, 2)), np.zeros(10), np.arange(10))

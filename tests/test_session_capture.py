"""Tests for the functions that collect a training step's internal values.

They only read values the loop already holds. Expected values are closed forms.
"""

import unittest

import numpy as np
import torch

from scripts.training.session_capture import (
    LOSS_NAMES,
    collect_losses,
    gradient_norm,
    json_safe,
    modules_with_parameters,
    reduce_kl_map,
    tensor_or_none,
)


class TestCollectLosses(unittest.TestCase):

    def test_present_losses_become_floats_and_missing_ones_are_absent(self):
        snapshot = {"pred_loss": torch.tensor(0.25), "gate_loss": 0.5, "unrelated": 3}
        self.assertEqual(collect_losses(snapshot), {"pred_loss": 0.25, "gate_loss": 0.5})

    def test_every_listed_name_is_a_loss(self):
        self.assertTrue(all("loss" in name for name in LOSS_NAMES))


class TestJsonSafe(unittest.TestCase):

    def test_numbers_strings_and_nesting_survive(self):
        self.assertEqual(json_safe({"a": np.float32(0.5), "b": [np.int64(2), "x"], "c": None}),
                         {"a": 0.5, "b": [2, "x"], "c": None})

    def test_arrays_and_tensors_are_left_out(self):
        cleaned = json_safe({"wave": np.zeros(4), "t": torch.zeros(2), "keep": True})
        self.assertEqual(cleaned, {"keep": True})


class TestModulesWithParameters(unittest.TestCase):

    def test_a_module_is_kept_under_its_name(self):
        net = torch.nn.Linear(2, 2)
        self.assertEqual(modules_with_parameters({"tectum": net}), {"tectum": net})

    def test_modules_inside_a_plain_object_are_found(self):
        class Holder:
            def __init__(self):
                self.pfc = torch.nn.Linear(2, 2)
                self.rate = 0.1
        holder = Holder()
        self.assertEqual(modules_with_parameters({"action_core": holder}),
                         {"action_core.pfc": holder.pfc})

    def test_none_and_parameterless_objects_are_skipped(self):
        self.assertEqual(modules_with_parameters({"a": None, "b": torch.nn.ReLU()}), {})


class TestGradientNorm(unittest.TestCase):

    def test_closed_form(self):
        net = torch.nn.Linear(2, 1, bias=True)
        net.weight.grad = torch.tensor([[3.0, 0.0]])
        net.bias.grad = torch.tensor([4.0])
        self.assertAlmostEqual(gradient_norm(net), 5.0, places=6)

    def test_no_gradients_is_none(self):
        self.assertIsNone(gradient_norm(torch.nn.Linear(2, 1)))


class TestReductions(unittest.TestCase):

    def test_kl_map_is_summed_over_the_category_axis(self):
        kl_map = torch.ones(1, 3, 5, 2, 2)
        reduced = reduce_kl_map(kl_map)
        self.assertEqual(reduced.shape, (1, 3, 2, 2))
        self.assertTrue(np.all(reduced == 5.0))

    def test_tensor_or_none(self):
        self.assertIsNone(tensor_or_none(None))
        self.assertEqual(tensor_or_none(torch.tensor([1.5])).dtype, np.float32)


if __name__ == "__main__":
    unittest.main()

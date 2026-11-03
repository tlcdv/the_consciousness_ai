"""pyphi 1.2.0 on a current Python, and the stop when pyphi is missing.

pyphi 1.2.0 imports ABC names from `collections` that Python 3.10 removed. Before this
change the import failed there with an ImportError, `models/evaluation/iit_phi.py` set
`pyphi = None`, and the training loop ran on an unvalidated stand-in formula without a
word. Phi scales exploration, is a reward term and sets memory priority, so that run was
a different agent. Since 2026-11-03 the stop applies only with `--phi-control on`, the one mode
that computes phi; the default 'off' needs no pyphi (tests/test_phi_control.py).

The phi value of the basic example network, 2.3125, is the value in the pyphi
documentation for the state (1, 0, 0).
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.evaluation import iit_phi
from scripts.training import train_rlhf

needs_pyphi = pytest.mark.skipif(importlib.util.find_spec("pyphi") is None, reason="pyphi is not installed")


# --- the import ----------------------------------------------------------------------

@needs_pyphi
def test_an_installed_pyphi_is_imported_on_this_python():
    assert iit_phi.pyphi is not None


@needs_pyphi
def test_basic_network_gives_the_documented_phi(monkeypatch):
    pyphi = iit_phi.pyphi
    # pyphi starts worker processes by default, and they do not return under pytest on
    # Windows. IITMetrics sets the same two values before it computes.
    monkeypatch.setattr(pyphi.config, "PARALLEL_CUT_EVALUATION", False)
    monkeypatch.setattr(pyphi.config, "PROGRESS_BARS", False)
    network = pyphi.examples.basic_network()
    subsystem = pyphi.Subsystem(network, (1, 0, 0), range(network.size))
    assert float(pyphi.compute.phi(subsystem)) == pytest.approx(2.3125, abs=1e-6)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="the names still exist before Python 3.10")
def test_the_names_added_for_the_import_are_removed_again():
    left = [name for name in iit_phi.REMOVED_ABC_NAMES if hasattr(collections, name)]
    assert left == []


def test_the_name_list_holds_only_names_that_collections_abc_has():
    assert all(hasattr(collections.abc, name) for name in iit_phi.REMOVED_ABC_NAMES)


# --- the command line ----------------------------------------------------------------

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


def test_training_stops_when_pyphi_is_missing(capsys):
    parser = _parser()
    with pytest.raises(SystemExit) as exit_info:
        train_rlhf._require_pyphi(parser, parser.parse_args(["--phi-control", "on"]), None)
    assert exit_info.value.code == 2
    message = capsys.readouterr().err
    assert "pyphi" in message and "--phi-proxy" in message


def test_the_flag_allows_a_run_with_the_stand_in():
    parser = _parser()
    train_rlhf._require_pyphi(parser, parser.parse_args(["--phi-control", "on", "--phi-proxy"]), None)


def test_no_flag_is_needed_when_pyphi_is_present():
    parser = _parser()
    train_rlhf._require_pyphi(parser, parser.parse_args(["--phi-control", "on"]), object())


def test_the_flag_is_off_by_default_and_its_help_names_the_stand_in():
    parser = _parser()
    assert parser.parse_args([]).phi_proxy is False
    text = next(action.help for action in parser._actions if "--phi-proxy" in action.option_strings)
    assert "stand-in" in text and "pyphi" in text

"""
The workspace competition has never been recorded during a training run.

GWT-1 ("multiple specialized systems operating in parallel") is scored on five
specialists competing in the global workspace. Until 2026-08-17 no run logged the bids
or the winner, so the indicator rested on the architecture's intent rather than on a
measurement. Offline probes could not fill the gap: `_compute_broadcast` in
probe_perception_decodability.py substitutes the literals 0.1 and 0.05 for the `memory`
and `body` bids, so the live values had never been observed at all.

These tests pin the instrumentation, not the outcome. They assert that the five raw bids
and the winner reach the step CSV, and that the workspace records who won. They make no
claim about whether the competition is degenerate; that is what the run measures.
"""
import csv
import tempfile
from pathlib import Path

from models.core.global_workspace import WorkspaceState
from scripts.training.metrics_logger import ConsciousnessMetricsLogger, StepMetrics


BID_COLUMNS = ["bid_vision", "bid_audio", "bid_memory", "bid_body", "bid_semantic"]


def _log_one(**kwargs) -> dict:
    """Write a single step and read the row back as a dict."""
    with tempfile.TemporaryDirectory() as tmp:
        logger = ConsciousnessMetricsLogger(log_dir=tmp, use_tensorboard=False)
        base = dict(global_step=0, phi=0.0, sync_r=0.0, is_conscious=True,
                    reward=0.0, broadcast_mag=0.0)
        base.update(kwargs)
        logger.log_step(StepMetrics(**base))
        logger.close()
        csv_path = next(Path(tmp).rglob("*metrics.csv"))
        with open(csv_path, newline="") as fh:
            return next(iter(csv.DictReader(fh)))


# --- the columns must exist and carry the values --------------------------------

def test_all_five_bid_columns_are_written():
    row = _log_one()
    for col in BID_COLUMNS + ["bid_winner"]:
        assert col in row, f"{col} missing from the step CSV header"


def test_bid_values_round_trip():
    row = _log_one(bid_vision=1.0, bid_audio=0.25, bid_memory=0.6,
                   bid_body=0.15, bid_semantic=0.42)
    assert float(row["bid_vision"]) == 1.0
    assert float(row["bid_audio"]) == 0.25
    assert float(row["bid_memory"]) == 0.6
    assert float(row["bid_body"]) == 0.15
    assert float(row["bid_semantic"]) == 0.42


def test_winner_round_trips_as_a_name():
    assert _log_one(bid_winner="memory")["bid_winner"] == "memory"


def test_saturated_vision_bid_survives_formatting():
    """The measured bid is exactly 1.0 and must not be rounded into ambiguity.

    Nine decimal places is deliberate: the workspace-competition verdict rests on the
    bid reading exactly 1.000000000, and a format that hid trailing digits would make a
    saturated bid indistinguishable from a merely high one.
    """
    assert _log_one(bid_vision=1.0)["bid_vision"] == "1.000000000"


def test_default_bids_are_zero_not_missing():
    """An unset bid must be a number, so a run with a disabled module is readable."""
    row = _log_one()
    for col in BID_COLUMNS:
        assert float(row[col]) == 0.0
    assert row["bid_winner"] == ""


# --- the workspace must record who won ------------------------------------------

def test_workspace_state_defaults_to_no_winners():
    state = WorkspaceState(active_content={}, access_history=[],
                           broadcast_strength=0.0, competition_results={})
    assert state.winners == []


def test_winners_default_is_not_shared_between_instances():
    """A mutable default on a dataclass is a classic aliasing bug; pin it."""
    a = WorkspaceState(active_content={}, access_history=[],
                       broadcast_strength=0.0, competition_results={})
    b = WorkspaceState(active_content={}, access_history=[],
                       broadcast_strength=0.0, competition_results={})
    a.winners.append("vision")
    assert b.winners == []

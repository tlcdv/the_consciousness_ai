"""
The broadcast's DIRECTION has never been recorded during a training run.

`workspace_state` is the 3-tuple `(broadcast_mag, phi, sync_r)`, and the first of those
is `broadcast.norm()`. So the pipeline has always kept the broadcast's LENGTH and thrown
away where it points. An indirect estimate on 2026-08-17 put the cosine between the
current broadcast and its best recent match at 0.99999976 to 1.0
(docs/results/memory_retrieval_repair_2026_08.md), which would mean the conscious content
of this architecture points one way for a whole run. That estimate came from inverting a
bid formula and needs a direct measurement.

These tests pin the sidecar that makes the direct measurement possible. They assert
nothing about what it will show.
"""
import tempfile
from pathlib import Path

import numpy as np

from scripts.training.metrics_logger import ConsciousnessMetricsLogger, StepMetrics


def _base(**kw) -> dict:
    d = dict(global_step=0, phi=0.0, sync_r=0.0, is_conscious=True,
             reward=0.0, broadcast_mag=0.0)
    d.update(kw)
    return d


def test_sidecar_has_one_row_per_step_and_full_width():
    with tempfile.TemporaryDirectory() as tmp:
        log = ConsciousnessMetricsLogger(log_dir=tmp, use_tensorboard=False)
        for i in range(5):
            log.log_step(StepMetrics(**_base(
                global_step=i, broadcast_vector=np.arange(256, dtype=np.float32))))
        log.close()
        arr = np.load(Path(tmp) / "broadcast.npy")
    assert arr.shape == (5, 256)


def test_values_round_trip_exactly():
    """Direction is the point, so the values must not be rounded on the way out."""
    vec = np.linspace(-1.0, 1.0, 256).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        log = ConsciousnessMetricsLogger(log_dir=tmp, use_tensorboard=False)
        log.log_step(StepMetrics(**_base(broadcast_vector=vec)))
        log.close()
        arr = np.load(Path(tmp) / "broadcast.npy")
    assert np.array_equal(arr[0], vec)


def test_a_batched_broadcast_is_flattened():
    """train_rlhf passes shape [1, workspace_dim]; the sidecar stores one row."""
    with tempfile.TemporaryDirectory() as tmp:
        log = ConsciousnessMetricsLogger(log_dir=tmp, use_tensorboard=False)
        log.log_step(StepMetrics(**_base(
            broadcast_vector=np.zeros((1, 256), dtype=np.float32))))
        log.close()
        arr = np.load(Path(tmp) / "broadcast.npy")
    assert arr.shape == (1, 256)


def test_no_sidecar_when_no_broadcast_is_supplied():
    """Every caller that does not pass a broadcast must be unaffected."""
    with tempfile.TemporaryDirectory() as tmp:
        log = ConsciousnessMetricsLogger(log_dir=tmp, use_tensorboard=False)
        log.log_step(StepMetrics(**_base()))
        log.close()
        assert not (Path(tmp) / "broadcast.npy").exists()


def test_close_still_writes_the_step_csv_alongside_the_sidecar():
    with tempfile.TemporaryDirectory() as tmp:
        log = ConsciousnessMetricsLogger(log_dir=tmp, use_tensorboard=False)
        log.log_step(StepMetrics(**_base(
            broadcast_vector=np.ones(256, dtype=np.float32))))
        log.close()
        assert (Path(tmp) / "broadcast.npy").exists()
        assert list(Path(tmp).rglob("*metrics.csv")), "step CSV must still be written"


# --- the labels that make the sidecar decodable -------------------------------------

def _row(**kw) -> dict:
    import csv as _csv
    with tempfile.TemporaryDirectory() as tmp:
        log = ConsciousnessMetricsLogger(log_dir=tmp, use_tensorboard=False)
        log.log_step(StepMetrics(**_base(**kw)))
        log.close()
        path = next(Path(tmp).rglob("*metrics.csv"))
        with open(path, newline="") as fh:
            return next(iter(_csv.DictReader(fh)))


def test_env_labels_reach_the_step_csv():
    """Without these the broadcast matrix cannot be decoded against anything."""
    r = _row(env_phase="sample", env_trial=7, env_sample_shape="pentagon")
    assert r["env_phase"] == "sample"
    assert int(r["env_trial"]) == 7
    assert r["env_sample_shape"] == "pentagon"


def test_env_labels_default_to_empty_not_missing():
    r = _row()
    assert r["env_phase"] == ""
    assert int(r["env_trial"]) == -1
    assert r["env_sample_shape"] == ""

"""Tests for scripts/sessions/export_session.py.

The exporter turns one episode of one recorded run into the public bundle the
website's training sessions console replays. The tests build a tiny synthetic
run folder and pin the two invariants the public record depends on:

    nothing dated, timed or path-bearing survives the export, and
    the environment facts of step t come from the info_after_step of step t-1.

The fixture run mirrors the layout session_recorder.py writes, scaled down:
8-pixel frames, 6 steps, one episode.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from scripts.sessions import export_session as exporter

BUNDLE_FILES = ("frames.webm", "steps.json", "session.json",
                "ethics_manifest.json", "poster.jpg", "clip.mp4")


def _step_record(step: int) -> dict:
    """One steps.jsonl line, with the fields the slim table reads."""
    return {
        "step": step,
        "input_source": {"vision": "frame",
                         "audio": "none" if step == 0 else "sound",
                         "memory": "default", "body": "energy_counter",
                         "semantic": "none"},
        "raw_bids": {"vision": 0.12345678 + step, "audio": 0.5,
                     "memory": 0.1, "body": 0.05, "semantic": 0.0},
        "interoception": {"energy": 0.8009999999999998, "damage": 0.0,
                          "fatigue": 0.0},
        "winner": "vision" if step else "",
        "ignited": bool(step),
        "sync_r": 0.43194514513015747,
        "bound_bids": {"vision": 1.4955222010612488, "audio": 1.0,
                       "memory": 0.1592489417508034, "body": 0.07962443914074921,
                       "semantic": 0.0},
        "reward": -0.00995270418585278,
        "action": [-0.5, -0.5],
        "env_phase": "",
        "internals": {
            "kl_div_pre_tanh": 1524.22412109375,
            "learned_valence": {"values": {"vision": 0.0123456789},
                                "td_error": 0.005},
            "info_after_step": {
                "in_light": step == 3, "collision": False, "battery": 0.998,
                "distance_to_light": 115.54,
                "_truth_light_in_view": step % 2 == 1,
                "_truth_light_offset": [-31.8, -111.1]},
        },
        "seconds_inputs_to_end": 0.001,
    }


def write_vectors(episode: Path, records: list) -> None:
    """audio_spatial exists for every step but step 0, per the recorder contract."""
    steps = np.asarray([r["step"] for r in records[1:]], dtype=np.int32)
    rows = [np.asarray([[float(step) / 4.0 - 2.0, 0.25]]) for step in steps]
    np.savez_compressed(
        episode / "vectors.npz",
        audio_spatial=np.stack(rows),
        audio_spatial__steps=steps,
        tectum_content=np.zeros((len(records), 1, 4)),
        tectum_content__steps=np.arange(len(records), dtype=np.int32))


def write_run_facts(folder: Path) -> None:
    """session.json and ethics_manifest.json with every private field present."""
    run_facts = {
        "argv": ["python", "-m", "scripts.training.train_rlhf",
                 "--env", "dark_room", "--seed", "49",
                 "--log-dir", "runs/gate_b2_s49",
                 "--corpus", "C:\\Users\\zae\\corpus",
                 "--dark-room-view", "agent_centered"],
        "seed": 49, "env": "dark_room", "episodes": 10, "max_steps": 200,
        "existence_drive": "on", "framework_version": "1.0",
        "audio_seeded": True, "dark_room_audio": "binaural",
        "dark_room_audio_channels": 4, "dark_room_collision": True,
        "dark_room_view": "agent_centered", "dark_room_view_radius": 96}
    # The shape models/ethics/framework.py writes.
    manifest = {
        "framework_version": "1.0", "entry_point": "train_rlhf",
        "existence_drive": "on", "growth_stage": "L0",
        "rules_checked": ["E1", "E3"], "violations": [],
        "noxious_channels_present": ["battery"],
        "git_commit": "abc123", "git_tracked_changes": True,
        "utc_time": "2026-09-15T20:00:00+00:00"}
    folder.joinpath("session.json").write_text(
        json.dumps({"run": run_facts, "modules": {}, "recorded_episodes": [0],
                    "full_tensor_episodes": [0]}), encoding="utf-8")
    folder.joinpath("ethics_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")


def build_run(folder: Path, steps: int = 6, side: int = 8) -> Path:
    """A minimal run folder with one recorded episode, ep_0000."""
    episode = folder / "episodes" / "ep_0000"
    episode.mkdir(parents=True)
    frames = np.full((steps, side, side, 3), 7, dtype=np.uint8)
    np.savez_compressed(episode / "frames.npz", frames=frames)
    records = [_step_record(step) for step in range(steps)]
    lines = "".join(json.dumps(record) + "\n" for record in records)
    (episode / "steps.jsonl").write_text(lines, encoding="utf-8")
    write_vectors(episode, records)
    (episode / "meta.json").write_text(json.dumps(
        {"episode": 0, "steps": steps, "frames": steps, "complete": True,
         "tiers": ["light"], "skipped_tiers": []}), encoding="utf-8")
    write_run_facts(folder)
    return folder


# ---- the slim per-step table ---------------------------------------------

def test_environment_facts_pair_with_previous_step(tmp_path):
    run = build_run(tmp_path / "run")
    table = exporter.slim_table(run)
    # Step 4 reads the environment state after step 3, which the fixture puts in light.
    assert table[4]["in_light"] is True
    assert table[3]["in_light"] is False and table[2]["in_light"] is False
    assert table[1]["in_light"] is False


def test_step_zero_has_no_environment_facts(tmp_path):
    run = build_run(tmp_path / "run")
    table = exporter.slim_table(run)
    assert table[0]["in_light"] is None
    assert table[0]["collision"] is None
    assert table[0]["light_in_view"] is None


def test_light_in_view_carries_the_ground_truth_field(tmp_path):
    run = build_run(tmp_path / "run")
    table = exporter.slim_table(run)
    assert table[2]["light_in_view"] is True
    assert table[1]["light_in_view"] is False and table[3]["light_in_view"] is False


def test_wall_time_never_reaches_the_table(tmp_path):
    run = build_run(tmp_path / "run")
    table = exporter.slim_table(run)
    assert all("seconds_inputs_to_end" not in step for step in table)


def test_audio_spatial_aligns_by_steps(tmp_path):
    run = build_run(tmp_path / "run")
    table = exporter.slim_table(run)
    assert table[0]["audio_spatial"] is None
    assert table[2]["audio_spatial"] == [round(2 / 4.0 - 2, 4), 0.25]


def test_numbers_round_to_four_decimals(tmp_path):
    run = build_run(tmp_path / "run")
    table = exporter.slim_table(run)
    assert table[2]["raw_bids"]["vision"] == 2.1235
    assert table[2]["internals_kl_div_pre_tanh"] == 1524.2241
    assert table[2]["learned_valence"]["vision"] == 0.0123


def test_table_keeps_winner_silence_and_bids(tmp_path):
    run = build_run(tmp_path / "run")
    table = exporter.slim_table(run)
    assert table[0]["winner"] == "" and table[0]["ignited"] is False
    assert table[2]["winner"] == "vision" and table[2]["ignited"] is True
    assert table[2]["bound_bids"]["memory"] == 0.1592


# ---- the public run facts and manifest ------------------------------------

def test_public_argv_drops_log_dir_and_path_bearers(tmp_path):
    run = build_run(tmp_path / "run")
    session_record = exporter.read_json(run / "session.json")
    argv = exporter.public_argv(session_record["run"]["argv"])
    assert "--log-dir" not in argv and "runs/gate_b2_s49" not in argv
    assert not any("C:" in token for token in argv)
    assert "--dark-room-view" in argv and "agent_centered" in argv


def test_public_manifest_drops_private_fields(tmp_path):
    run = build_run(tmp_path / "run")
    manifest = exporter.public_manifest(exporter.read_json(run / "ethics_manifest.json"))
    for private in ("utc_time", "git_commit", "git_tracked_changes"):
        assert private not in manifest
    assert manifest["framework_version"] == "1.0"
    assert manifest["existence_drive"] == "on"


# ---- the leak scan ---------------------------------------------------------

def test_scan_raises_on_dates_times_and_epochs():
    for leak in ("recorded 2026-09-15", "at 20:00:00", "epoch 1760000000"):
        with pytest.raises(exporter.PublicLeakError):
            exporter.scan_text(leak, "steps.json")


def test_scan_raises_on_absolute_paths():
    for leak in ("crash in C:\\Users\\zae\\core.py", "log /Users/zae/x",
                 "path \\\\Users\\\\zae", "log /home/zae/x"):
        with pytest.raises(exporter.PublicLeakError):
            exporter.scan_text(leak, "session.json")


def test_scan_passes_on_clean_public_text():
    exporter.scan_text("seed 49, silence 0.314, rho -0.128", "steps.json")


def test_export_raises_when_a_public_field_carries_a_date(tmp_path):
    run = build_run(tmp_path / "run")
    episode = run / "episodes" / "ep_0000"
    lines = episode.joinpath("steps.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    records[2]["winner"] = "2026-09-15"
    text = "".join(json.dumps(record) + "\n" for record in records)
    episode.joinpath("steps.jsonl").write_text(text, encoding="utf-8")
    with pytest.raises(exporter.PublicLeakError):
        exporter.export_session(tmp_path / "run", 0, "dark-room-b2-seed49",
                                tmp_path / "site")


def test_export_raises_when_the_manifest_names_a_local_path(tmp_path):
    run = build_run(tmp_path / "run")
    manifest_path = run / "ethics_manifest.json"
    manifest = exporter.read_json(manifest_path)
    manifest["violations"] = ["E4 breach logged at C:\\Users\\zae\\tmp"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(exporter.PublicLeakError):
        exporter.export_session(run, 0, "dark-room-b2-seed49", tmp_path / "site")


# ---- the session id and the size guard -------------------------------------

def test_session_id_forbids_dates_and_odd_characters():
    exporter.validate_session_id("dark-room-b2-seed49-episode9")
    for bad in ("2026-09-15-gate", "dark room", "Seed49", "", "a/b"):
        with pytest.raises(ValueError):
            exporter.validate_session_id(bad)


def test_size_guard_refuses_a_bundle_over_the_limit(tmp_path):
    folder = tmp_path / "sessions"
    folder.mkdir()
    (folder / "heavy.bin").write_bytes(b"0" * 4096)
    with pytest.raises(exporter.SizeLimitError):
        exporter.check_sessions_size(tmp_path / "bundle", folder, limit_mb=0.001)


def test_size_total_does_not_count_the_copy_being_replaced(tmp_path):
    site = tmp_path / "site"
    (site / "dark-room-b2-seed49").mkdir(parents=True)
    (site / "dark-room-b2-seed49" / "old.bin").write_bytes(b"0" * 4096)
    bundle = tmp_path / "staging" / "dark-room-b2-seed49"
    bundle.mkdir(parents=True)
    (bundle / "new.bin").write_bytes(b"0" * 1000)
    total = exporter.check_sessions_size(bundle, site, limit_mb=1.0,
                                         replaced=site / "dark-room-b2-seed49")
    assert total == pytest.approx(1000 / 1e6)


def test_export_refuses_an_existing_session_id(tmp_path):
    run = build_run(tmp_path / "run")
    site = tmp_path / "site"
    (site / "dark-room-b2-seed49").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        exporter.export_session(run, 0, "dark-room-b2-seed49", site)


def test_export_refuses_when_the_site_folder_would_overflow(tmp_path):
    run = build_run(tmp_path / "run")
    site = tmp_path / "site"
    site.mkdir()
    (site / "heavy.bin").write_bytes(b"0" * 1024)
    with pytest.raises(exporter.SizeLimitError):
        exporter.export_session(run, 0, "dark-room-b2-seed49", site, limit_mb=0.001)


# ---- the exported bundle ----------------------------------------------------

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not on PATH")


@needs_ffmpeg
def test_export_writes_the_six_bundle_files(tmp_path):
    run = build_run(tmp_path / "run")
    site = tmp_path / "site"
    exporter.export_session(run, 0, "dark-room-b2-seed49", site)
    bundle = site / "dark-room-b2-seed49"
    assert sorted(p.name for p in bundle.iterdir()) == sorted(BUNDLE_FILES)
    for name in BUNDLE_FILES:
        assert (bundle / name).stat().st_size > 0, name


@needs_ffmpeg
def test_replace_swaps_the_bundle_instead_of_nesting_it(tmp_path):
    run = build_run(tmp_path / "run")
    site = tmp_path / "site"
    exporter.export_session(run, 0, "dark-room-b2-seed49", site)
    exporter.export_session(run, 0, "dark-room-b2-seed49", site, replace=True)
    bundle = site / "dark-room-b2-seed49"
    assert sorted(p.name for p in bundle.iterdir()) == sorted(BUNDLE_FILES)


@needs_ffmpeg
def test_exported_json_carries_no_private_fields(tmp_path):
    run = build_run(tmp_path / "run")
    site = tmp_path / "site"
    exporter.export_session(run, 0, "dark-room-b2-seed49", site)
    bundle = site / "dark-room-b2-seed49"
    steps_text = (bundle / "steps.json").read_text(encoding="utf-8")
    session_text = (bundle / "session.json").read_text(encoding="utf-8")
    for text in (steps_text, session_text):
        exporter.scan_text(text, "bundle")
    assert "seconds_inputs_to_end" not in steps_text
    assert "utc_time" not in session_text and "git_commit" not in session_text
    assert "C:" not in session_text
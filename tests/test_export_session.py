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
import wave
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.sessions import export_session as exporter

BUNDLE_FILES = ("frames.webm", "steps.json", "session.json",
                "ethics_manifest.json", "poster.jpg", "clip.mp4")

# Every test that runs the whole export writes media, so it needs ffmpeg, which
# the continuous integration image does not carry.
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not on PATH")


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
        "dark_room_view": "agent_centered", "dark_room_view_radius": 96,
        "agent_mark": "ring", "agent_colour": [217, 119, 87],
        "light_colour": [255, 255, 200], "wall_colour": [60, 60, 60]}
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


def test_public_run_record_carries_how_the_frames_were_drawn(tmp_path):
    run = build_run(tmp_path / "run")
    public = exporter.public_run_record(exporter.read_json(run / "session.json"))
    assert public["run"]["agent_mark"] == "ring"
    assert public["run"]["agent_colour"] == [217, 119, 87]
    assert public["run"]["light_colour"] == [255, 255, 200]
    assert public["run"]["wall_colour"] == [60, 60, 60]


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


@needs_ffmpeg
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


@needs_ffmpeg
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


@needs_ffmpeg
def test_export_refuses_when_the_site_folder_would_overflow(tmp_path):
    run = build_run(tmp_path / "run")
    site = tmp_path / "site"
    site.mkdir()
    (site / "heavy.bin").write_bytes(b"0" * 1024)
    with pytest.raises(exporter.SizeLimitError):
        exporter.export_session(run, 0, "dark-room-b2-seed49", site, limit_mb=0.001)


# ---- the exported bundle ----------------------------------------------------

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


def test_replay_keeps_full_recorded_chunk_and_duration():
    records = [_step_record(step) for step in range(3)]
    waves = np.full((2, 4, 1056), 0.3, dtype=np.float32)
    vectors = {"audio_waveform": waves, "audio_waveform__steps": np.array([2, 1])}
    samples, replay = exporter.prepare_replay(records, vectors, {"env": "dark_room"}, 3)
    assert samples.shape == (3168, 2)
    np.testing.assert_array_equal(samples[:1056], 0)
    np.testing.assert_allclose(samples[1056:], 0.3)
    assert replay["frame_rate"] == pytest.approx(16000 / 1056)
    assert 3 / replay["frame_rate"] == pytest.approx(0.198)
    assert replay["audio"] == {
        "available": True, "sample_rate": 16000, "samples_per_step": 1056,
        "source_channels": 4, "playback_channels": 2,
        "channel_mapping": "left=(L+U+D)/3; right=(R+U+D)/3"}


def replay_fixture():
    records = [_step_record(step) for step in range(3)]
    vectors = {"audio_waveform": np.full((2, 4, 1056), 0.3, dtype=np.float32),
               "audio_waveform__steps": np.array([1, 2], dtype=np.int32)}
    return records, vectors


@pytest.mark.parametrize("indices", [[1, 1], [-1, 2], [1, 3], [1.0, 2.0],
                                      [True, False], [[1, 2]], [1]])
def test_replay_rejects_invalid_audio_indices(indices):
    records, vectors = replay_fixture()
    vectors["audio_waveform__steps"] = np.asarray(indices)
    with pytest.raises(ValueError, match="indices"):
        exporter.prepare_replay(records, vectors, {"env": "dark_room"}, 3)


@pytest.mark.parametrize("steps", [[0, 0, 2], [0, 2, 1], [0, 1, 3],
                                   [0, 1.0, 2], [False, 1, 2]])
def test_replay_rejects_invalid_record_indices(steps):
    records, vectors = replay_fixture()
    for record, step in zip(records, steps):
        record["step"] = step
    with pytest.raises(ValueError, match="record steps"):
        exporter.prepare_replay(records, vectors, {"env": "dark_room"}, 3)


def test_replay_rejects_frame_count_mismatch():
    records, vectors = replay_fixture()
    with pytest.raises(ValueError, match="frame count"):
        exporter.prepare_replay(records, vectors, {"env": "dark_room"}, 4)


@pytest.mark.parametrize("source", ["sound", "silent", None, "unknown"])
def test_replay_rejects_missing_nonzero_step(source):
    records, vectors = replay_fixture()
    records[2]["input_source"]["audio"] = source
    vectors = {name: array[:1] for name, array in vectors.items()}
    with pytest.raises(ValueError, match="missing waveform.*2"):
        exporter.prepare_replay(records, vectors, {"env": "dark_room"}, 3)


@pytest.mark.parametrize("source", ["sound", "silent"])
def test_replay_rejects_missing_step_zero_unless_none(source):
    records, vectors = replay_fixture()
    records[0]["input_source"]["audio"] = source
    with pytest.raises(ValueError, match="missing waveform.*0"):
        exporter.prepare_replay(records, vectors, {"env": "dark_room"}, 3)


def test_replay_rejects_waveform_not_consumed():
    records, vectors = replay_fixture()
    records[1]["input_source"]["audio"] = "none"
    with pytest.raises(ValueError, match="input_source"):
        exporter.prepare_replay(records, vectors, {"env": "dark_room"}, 3)


@pytest.mark.parametrize("bad_sample", [np.nan, np.inf, -np.inf, 1.001, -1.001])
def test_replay_rejects_invalid_samples(bad_sample):
    records, vectors = replay_fixture()
    vectors["audio_waveform"][0, 0, 0] = bad_sample
    with pytest.raises(ValueError, match="samples"):
        exporter.prepare_replay(records, vectors, {"env": "dark_room"}, 3)


@pytest.mark.parametrize("shape", [(2, 3, 1056), (2, 4, 0), (2, 1, 4, 1056), (0, 4, 1056)])
def test_replay_rejects_unsupported_waveforms(shape):
    records, vectors = replay_fixture()
    vectors["audio_waveform"] = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError):
        exporter.prepare_replay(records, vectors, {"env": "dark_room"}, 3)


@pytest.mark.parametrize("keys", [{"audio_waveform": np.zeros((2, 1056))},
                                  {"audio_waveform__steps": np.array([1, 2])},
                                  {"audio_waveform__step1": np.zeros(1056)}])
def test_replay_rejects_partial_or_variable_shape_storage(keys):
    records, _ = replay_fixture()
    with pytest.raises(ValueError, match="waveform"):
        exporter.prepare_replay(records, keys, {"env": "dark_room"}, 3)


@pytest.mark.parametrize("channels", [1, 2, 4])
def test_replay_preserves_channel_mapping_and_index_order(channels):
    records, vectors = replay_fixture()
    waves = np.zeros((2, channels, 1056), dtype=np.float32)
    waves[0, 0] = 0.6
    waves[1, -1] = -0.3
    vectors.update(audio_waveform=waves, audio_waveform__steps=np.array([2, 1]))
    samples, replay = exporter.prepare_replay(records, vectors, {"env": "dark_room"}, 3)
    expected_first = {1: [-0.3], 2: [0, -0.3], 4: [-0.1, -0.1]}[channels]
    expected_last = {1: [0.6], 2: [0.6, 0], 4: [0.2, 0]}[channels]
    np.testing.assert_allclose(samples[1056], expected_first, atol=1e-7)
    np.testing.assert_allclose(samples[-1], expected_last, atol=1e-7)
    assert replay["audio"]["source_channels"] == channels
    assert replay["audio"]["playback_channels"] == min(channels, 2)


def build_audio_run(folder: Path) -> Path:
    run = build_run(folder)
    episode = exporter.episode_folder(run, 0)
    vectors = exporter.load_vectors(run, 0)
    phase = np.arange(1056) / 16000
    waves = np.stack([np.tile(0.3 * np.sin(2 * np.pi * step * 250 * phase), (4, 1))
                      for step in range(1, 6)]).astype(np.float32)
    vectors.update(audio_waveform=waves, audio_waveform__steps=np.arange(1, 6))
    np.savez_compressed(episode / "vectors.npz", **vectors)
    return run


def media_probe(path: Path, *options) -> dict:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", *options, "-of", "json", str(path)],
        capture_output=True, check=True)
    return json.loads(probe.stdout)


def mux_probe(path: Path) -> dict:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", path.name],
        capture_output=True, cwd=str(path.parent), check=True)
    streams = json.loads(probe.stdout.decode("utf-8"))["streams"]
    return {stream["codec_type"]: stream for stream in streams}


@needs_ffmpeg
def test_exported_session_json_carries_replay_metadata(tmp_path):
    run = build_run(tmp_path / "run")
    site = tmp_path / "site"
    exporter.export_session(run, 0, "dark-room-b2-seed49", site)
    replay = json.loads((site / "dark-room-b2-seed49" / "session.json").read_text(
        encoding="utf-8"))["replay"]
    assert replay == {
        "frame_rate": 30, "audio": {"available": False}}


@needs_ffmpeg
def test_exported_audio_streams_carry_the_replay_duration(tmp_path):
    run = build_audio_run(tmp_path / "run")
    bundle = tmp_path / "staged"
    exporter.write_bundle(bundle, run, 0)
    session = json.loads((bundle / "session.json").read_text(encoding="utf-8"))
    duration = 6 / session["replay"]["frame_rate"]
    for media in ("frames.webm", "clip.mp4"):
        streams = mux_probe(bundle / media)
        assert set(streams) == {"video", "audio"}, media
        probe = media_probe(bundle / media, "-show_format")
        tolerance = 1024 / 16000 + 0.001 if media.endswith("mp4") else 0.001
        assert float(probe["format"]["duration"]) == pytest.approx(duration, abs=tolerance)
        for stream in streams.values():
            clock = stream.get("duration", stream.get("tags", {}).get("DURATION"))
            seconds = sum(float(part) * 60 ** index
                          for index, part in enumerate(reversed(clock.split(":"))))
            assert seconds == pytest.approx(duration, abs=0.001)
        assert int(streams["audio"]["sample_rate"]) == 16000
        assert streams["audio"]["channels"] == 2


@pytest.mark.parametrize("available", [False, True])
def test_clip_panel_states_audio_notice(monkeypatch, available):
    records, vectors = replay_fixture()
    _, replay = exporter.prepare_replay(
        records, vectors if available else {}, {"env": "dark_room"}, 3)
    drawn = []
    monkeypatch.setattr(exporter.ImageDraw.ImageDraw, "text",
                        lambda self, xy, text, **kwargs: drawn.append((xy, text)))
    exporter.clip_composite(1, np.full((8, 8, 3), 7, np.uint8), records, 1.0,
                            exporter.panel_font(), replay)
    notice = " ".join(text for (x, y), text in drawn if 175 <= y <= 310)
    assert "not wall time" in notice
    if available:
        assert "Stereo mix" in notice and "lossy" in notice
        assert "(L+U+D)/3" in notice and "(R+U+D)/3" in notice
        assert "1056 samples/step" in notice and "16000 Hz" in notice
    else:
        assert "Audio unavailable" in notice and "silence" not in notice


@pytest.mark.parametrize("source", ["sound", "silent", "none"])
def test_slim_masks_spatial_direction_when_audio_source_is_none(source):
    records, vectors = replay_fixture()
    records[0]["input_source"]["audio"] = source
    vectors["audio_spatial"] = np.asarray([[0.9, -0.8], [0.5, 0.1]])
    vectors["audio_spatial__steps"] = np.array([0, 1])
    table = exporter.slim_records(records, vectors)
    assert table[0]["audio_spatial"] == (None if source == "none" else [0.9, -0.8])
    assert table[1]["audio_spatial"] == [0.5, 0.1]
    assert "action" not in table[0]


def test_legacy_bundle_without_waveform_reports_no_audio(tmp_path):
    run = build_run(tmp_path / "run")
    table = exporter.slim_table(run)
    samples, replay = exporter.prepare_replay(
        exporter.load_records(run, 0), exporter.load_vectors(run, 0),
        exporter.read_json(run / "session.json")["run"], len(table))
    assert samples is None
    assert replay["audio"]["available"] is False
    assert replay["frame_rate"] == 30


@needs_ffmpeg
def test_export_with_audio_streams_actual_recorded_waveform(tmp_path):
    run = build_run(tmp_path / "run")
    episode = run / "episodes" / "ep_0000"
    waves = np.zeros((5, 2, 1056), dtype=np.float32)
    waves[2, 0] = np.sin(np.linspace(0, 2 * np.pi * 40, 1056)).astype(np.float32)
    steps = np.arange(1, 6, dtype=np.int32)
    with np.load(episode / "vectors.npz") as packed:
        vectors = {name: packed[name] for name in packed.files}
    vectors["audio_waveform"] = waves
    vectors["audio_waveform__steps"] = steps
    np.savez_compressed(episode / "vectors.npz", **vectors)
    site = tmp_path / "site"
    exporter.export_session(run, 0, "dark-room-b2-seed49", site)
    session = json.loads((site / "dark-room-b2-seed49" / "session.json").read_text(
        encoding="utf-8"))
    assert session["replay"]["audio"]["available"] is True
    assert session["replay"]["audio"]["samples_per_step"] == 1056
    left = mux_probe(site / "dark-room-b2-seed49" / "frames.webm")["audio"]
    assert left["channels"] == 2 and int(left["sample_rate"]) == 16000


def test_replay_accepts_flat_mono_and_recorded_silence():
    records, vectors = replay_fixture()
    records[1]["input_source"]["audio"] = "silent"
    vectors["audio_waveform"] = np.zeros((2, 1056), dtype=np.float32)
    vectors["audio_waveform"][1] = np.linspace(-1, 1, 1056, dtype=np.float32)
    samples, replay = exporter.prepare_replay(records, vectors, {"env": "dark_room"}, 3)
    np.testing.assert_array_equal(samples[2112:, 0], vectors["audio_waveform"][1])
    np.testing.assert_array_equal(samples[:2112], 0)
    assert replay["audio"]["available"] is True


def test_replay_legacy_without_waveforms_has_explicit_no_audio():
    records, _ = replay_fixture()
    samples, replay = exporter.prepare_replay(records, {}, {"env": "unknown"}, 3)
    assert samples is None
    assert replay == {"frame_rate": 30, "audio": {"available": False}}


@pytest.mark.parametrize("facts", [{"env": "unknown"}, {},
                                    {"audio_sample_rate": "16000"},
                                    {"audio_sample_rate": True},
                                    {"audio_sample_rate": -1},
                                    {"audio_sample_rate": 16000.5}])
def test_replay_refuses_unproven_sample_rate(facts):
    records, vectors = replay_fixture()
    with pytest.raises(ValueError, match="sample rate"):
        exporter.prepare_replay(records, vectors, facts, 3)


def test_replay_accepts_explicit_source_sample_rate():
    records, vectors = replay_fixture()
    samples, replay = exporter.prepare_replay(
        records, vectors, {"env": "custom", "audio_sample_rate": 24000}, 3)
    assert replay["audio"]["sample_rate"] == 24000
    assert len(samples) / 24000 == pytest.approx(3 / replay["frame_rate"])


def test_replay_allows_explicit_none_gap_at_any_step():
    records, vectors = replay_fixture()
    records[1]["input_source"]["audio"] = "none"
    vectors = {name: array[1:] for name, array in vectors.items()}
    samples, replay = exporter.prepare_replay(records, vectors, {"env": "dark_room"}, 3)
    np.testing.assert_array_equal(samples[:2112], 0)
    np.testing.assert_allclose(samples[2112:], 0.3)
    assert replay["audio"]["available"] is True
    assert exporter.slim_records(records, vectors)[1]["input_source"]["audio"] == "none"


@pytest.mark.parametrize("shift", [-1, 1])
def test_replay_rejects_one_step_index_shift(shift):
    records, vectors = replay_fixture()
    vectors["audio_waveform__steps"] += shift
    with pytest.raises(ValueError):
        exporter.prepare_replay(records, vectors, {"env": "dark_room"}, 3)


def test_wav_roundtrip_preserves_all_four_channels_and_step_order(tmp_path):
    records = [_step_record(step) for step in range(5)]
    waves = np.zeros((4, 4, 1056), dtype=np.float32)
    for channel in range(4):
        waves[channel, channel] = np.linspace(-0.6, 0.6, 1056, dtype=np.float32)
    indices = np.array([4, 2, 1, 3])
    samples, replay = exporter.prepare_replay(records, {
        "audio_waveform": waves, "audio_waveform__steps": indices}, {"env": "dark_room"}, 5)
    target = tmp_path / "roundtrip.wav"
    exporter.write_wav(samples, target, replay["audio"]["sample_rate"])
    rate, restored = exporter.wavfile.read(target)
    assert rate == 16000 and restored.shape == (5280, 2)
    expected = np.zeros((5, 1056, 2), dtype=np.float32)
    for channel, step in enumerate(indices):
        for ear in ([channel] if channel < 2 else [0, 1]):
            expected[step, :, ear] = waves[channel, channel] / 3
    np.testing.assert_array_equal(restored, expected.reshape(-1, 2))


@needs_ffmpeg
@pytest.mark.parametrize("media", ["frames.webm", "clip.mp4"])
def test_video_frame_times_match_same_step_at_start_and_midpoint(tmp_path, media):
    run = build_audio_run(tmp_path / "run")
    bundle = tmp_path / "bundle"
    exporter.write_bundle(bundle, run, 0)
    probe = media_probe(bundle / media, "-select_streams", "v:0", "-show_frames")
    times = np.array([float(frame["best_effort_timestamp_time"]) for frame in probe["frames"]])
    assert len(times) == 6
    fps = exporter.read_json(bundle / "session.json")["replay"]["frame_rate"]
    np.testing.assert_allclose(times, np.arange(6) / fps, atol=0.001)
    np.testing.assert_array_equal(np.floor(times * fps + 0.02), np.arange(6))
    np.testing.assert_array_equal(np.floor((times + 0.5 / fps) * fps + 0.02), np.arange(6))
    assert sorted(path.name for path in bundle.iterdir()) == sorted(BUNDLE_FILES)


@needs_ffmpeg
@pytest.mark.parametrize("media", ["frames.webm", "clip.mp4"])
def test_mux_audio_preserves_chunk_pitch_and_alignment(tmp_path, media):
    run = build_audio_run(tmp_path / "run")
    bundle = tmp_path / "bundle"
    exporter.write_bundle(bundle, run, 0)
    target = tmp_path / "decoded.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(bundle / media),
                    "-vn", "-c:a", "pcm_s16le", str(target)], capture_output=True, check=True)
    with wave.open(str(target), "rb") as decoded:
        assert decoded.getframerate() == 16000 and decoded.getnchannels() == 2
        samples = np.frombuffer(decoded.readframes(decoded.getnframes()), dtype="<i2").reshape(-1, 2)
    assert len(samples) >= 6 * 1056
    assert len(samples) / 16000 == pytest.approx(6 * 1056 / 16000, abs=0.064)
    for step in range(1, 6):
        chunk = samples[step * 1056 + 128:(step + 1) * 1056 - 128, 0]
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        frequency = np.fft.rfftfreq(len(chunk), 1 / 16000)[spectrum.argmax()]
        assert frequency == pytest.approx(step * 250, abs=21)


def build_marker_run(folder):
    run = build_run(folder, side=32)
    episode = exporter.episode_folder(run, 0)
    frames = np.stack([np.full((32, 32, 3), 30 + step * 35, np.uint8)
                       for step in range(6)])
    np.savez_compressed(episode / "frames.npz", frames=frames)
    indices = np.array([4, 1, 5, 2, 3])
    phase = np.arange(1056) / 16000
    waves = np.stack([np.tile(step * 0.08 * np.sin(2 * np.pi * step * 500 * phase),
                             (4, 1)) for step in indices]).astype(np.float32)
    vectors = exporter.load_vectors(run, 0)
    vectors.update(audio_waveform=waves, audio_waveform__steps=indices)
    np.savez_compressed(episode / "vectors.npz", **vectors)
    return run


def assert_encoded_markers(path):
    streams = mux_probe(path)
    width, height = streams["video"]["width"], streams["video"]["height"]
    video = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path),
                                    "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    frames = np.frombuffer(video, np.uint8).reshape(-1, height, width, 3)
    audio = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path),
                                    "-vn", "-f", "f32le", "-acodec", "pcm_f32le", "-"])
    samples = np.frombuffer(audio, "<f4").reshape(-1, 2)
    assert len(frames) == 6
    for step, frame in enumerate(frames):
        marker = frame[height // 2 - 8:height // 2 + 8, 8:24].mean()
        assert marker == pytest.approx(30 + step * 35, abs=3)
        chunk = samples[step * 1056 + 192:(step + 1) * 1056 - 192]
        assert np.sqrt(np.mean(chunk ** 2)) == pytest.approx(step * 0.08 / np.sqrt(2), abs=0.015)
        if step:
            spectrum = np.abs(np.fft.rfft(chunk[:, 0] * np.hanning(len(chunk))))
            frequency = np.fft.rfftfreq(len(chunk), 1 / 16000)[spectrum.argmax()]
            assert frequency == pytest.approx(step * 500, abs=25)


@needs_ffmpeg
@pytest.mark.parametrize("media", ["frames.webm", "clip.mp4"])
@pytest.mark.parametrize("corruption", [None, "waveform_shift", "frame_reorder"])
def test_integrated_video_audio_markers_detect_alignment_errors(tmp_path, media, corruption):
    run = build_marker_run(tmp_path / "run")
    episode = exporter.episode_folder(run, 0)
    if corruption == "waveform_shift":
        vectors = exporter.load_vectors(run, 0)
        vectors["audio_waveform"] = np.roll(vectors["audio_waveform"], 1, axis=0)
        np.savez_compressed(episode / "vectors.npz", **vectors)
    if corruption == "frame_reorder":
        frames = exporter.load_frames(run, 0)
        frames[[2, 3]] = frames[[3, 2]]
        np.savez_compressed(episode / "frames.npz", frames=frames)
    bundle = tmp_path / "bundle"
    exporter.write_bundle(bundle, run, 0)
    if corruption:
        with pytest.raises(AssertionError):
            assert_encoded_markers(bundle / media)
    else:
        assert_encoded_markers(bundle / media)


@pytest.mark.parametrize("dtype", [complex, object, str])
def test_replay_rejects_nonreal_waveform_types(dtype):
    records, vectors = replay_fixture()
    vectors["audio_waveform"] = vectors["audio_waveform"].astype(dtype)
    with pytest.raises(ValueError, match="samples"):
        exporter.prepare_replay(records, vectors, {"env": "dark_room"}, 3)

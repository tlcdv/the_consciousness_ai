"""Export one recorded training episode as a public replay bundle.

The run folder written by scripts/training/session_recorder.py is the session
record. This script turns ONE episode of ONE run into the public bundle the
website's training sessions console replays:

    frames.webm           the episode frames, one video frame per step, VP9
    steps.json            a slim per-step table for the console
    session.json          public run facts (no utc_time, git_commit, paths)
    ethics_manifest.json  public manifest (same removals)
    poster.jpg            one frame on a console-coloured card, no EXIF
    clip.mp4              a short clip: frame left, instrument panels right

Two invariants hold over everything this script writes:

    nothing dated, timed, epoch-stamped or path-bearing survives, and
    step t's environment facts come from step t-1's info_after_step
    (the frame and sound the agent received at step t were produced by the
    environment state after step t-1; step 0 has none).

The bundle is staged in a temporary folder, scanned, and only then moved into
the website's sessions data folder, whose running total must stay under
SESSIONS_LIMIT_MB (200 MB).

The clip panels use DejaVu Sans Mono: PT Mono is not installed on this machine
as a system font. Same layout intent, stated here rather than hidden.

Run:
    python -m scripts.sessions.export_session --run runs/gate_b2_s49 \
        --episode 9 --session-id dark-room-b2-seed49-episode9 \
        --site-dir <the website's sessions data folder>
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

FRAME_RATE = 30
SLIM_DECIMALS = 4
SESSIONS_LIMIT_MB = 200.0
BUNDLE_FILES = ("frames.webm", "steps.json", "session.json",
                "ethics_manifest.json", "poster.jpg", "clip.mp4")
RUN_FACT_KEYS = ("argv", "env", "episodes", "max_steps", "seed",
                 "existence_drive", "framework_version", "audio_seeded",
                 "dark_room_audio", "dark_room_audio_channels",
                 "dark_room_collision", "dark_room_view", "dark_room_view_radius")
PRIVATE_MANIFEST_KEYS = ("utc_time", "git_commit", "git_tracked_changes")

DATE_SCAN = re.compile(r"\d{4}-\d{2}-\d{2}")
TIME_SCAN = re.compile(r"\d{2}:\d{2}:\d{2}")
ISO_SCAN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")
EPOCH_SCAN = re.compile(r"\b\d{10}\b")
PATH_SCANS = (re.compile(r"[A-Za-z]:[\\/]"), re.compile(r"/Users/"),
              re.compile(r"\\Users\\"), re.compile(r"/home/"))

POSTER_WIDTH, POSTER_HEIGHT = 1200, 630
CLIP_HEIGHT, CLIP_PANEL_WIDTH = 448, 520
INK_RGB = (11, 12, 14)
PANEL_RGB = (16, 18, 21)
TEXT_RGB = (240, 238, 230)
DIM_RGB = (156, 154, 148)
ACCENT_RGB = (217, 119, 87)
GOLD_RGB = (214, 171, 113)
LIVE_RGB = (111, 191, 115)
SILENCE_RGB = (40, 42, 46)


class PublicLeakError(ValueError):
    """A public bundle file carried a date, a time, an epoch or a path."""


class SizeLimitError(ValueError):
    """The export would push the website sessions folder over its limit."""


# ---- reading the run folder ------------------------------------------------

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def episode_folder(run: Path, episode: int) -> Path:
    folder = run / "episodes" / ("ep_%04d" % episode)
    if not folder.is_dir():
        raise FileNotFoundError("episode %d is not recorded in %s" % (episode, run))
    return folder


def load_records(run: Path, episode: int) -> list:
    lines = episode_folder(run, episode).joinpath("steps.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in lines.splitlines() if line.strip()]


def load_frames(run: Path, episode: int) -> np.ndarray:
    with np.load(episode_folder(run, episode) / "frames.npz") as packed:
        return packed["frames"]


def load_vectors(run: Path, episode: int) -> dict:
    with np.load(episode_folder(run, episode) / "vectors.npz") as packed:
        return {name: packed[name] for name in packed.files}


# ---- the slim per-step table ------------------------------------------------

def audio_directions(vectors: dict) -> dict:
    """Step number to [azimuth, elevation], aligned by the __steps index."""
    if "audio_spatial" not in vectors:
        return {}
    rows, steps = vectors["audio_spatial"], vectors["audio_spatial__steps"]
    return {int(step): [round(float(x), SLIM_DECIMALS) for x in rows[i].ravel()]
            for i, step in enumerate(steps)}


def slim_environment(previous: dict) -> dict:
    """Environment state AFTER the previous step, which produced this step's frame.

    The reward of step t comes from the environment step taken at t, so it is one
    step later than these fields: on the step the agent enters the light, reward is
    1.0 while in_light (the state that produced the frame) is still false.
    """
    if not previous:
        return {"in_light": None, "collision": None, "light_in_view": None}
    after = previous.get("internals", {}).get("info_after_step", {})
    return {"in_light": after.get("in_light"), "collision": after.get("collision"),
            "light_in_view": after.get("_truth_light_in_view")}


def slim_step(record: dict, previous: dict, direction) -> dict:
    internals = record.get("internals", {})
    step_entry = {
        "step": record["step"],
        "winner": record.get("winner", ""),
        "ignited": bool(record.get("ignited")),
        "raw_bids": record.get("raw_bids", {}),
        "bound_bids": record.get("bound_bids", {}),
        "input_source": record.get("input_source", {}),
        "sync_r": record.get("sync_r"),
        "reward": record.get("reward"),
        "interoception": record.get("interoception", {}),
        "audio_spatial": direction,
        "internals_kl_div_pre_tanh": internals.get("kl_div_pre_tanh"),
        "learned_valence": internals.get("learned_valence", {}).get("values", {}),
    }
    step_entry.update(slim_environment(previous))
    return step_entry


def round_public_numbers(node):
    if isinstance(node, float):
        return round(node, SLIM_DECIMALS)
    if node is None or isinstance(node, (bool, int, str)):
        return node
    if isinstance(node, dict):
        return {key: round_public_numbers(entry) for key, entry in node.items()}
    if isinstance(node, (list, tuple)):
        return [round_public_numbers(entry) for entry in node]
    raise TypeError("the slim table carries %r" % type(node))


def slim_records(records: list, vectors: dict) -> list:
    directions = audio_directions(vectors)
    table = []
    for index, record in enumerate(records):
        previous = records[index - 1] if index else None
        table.append(slim_step(record, previous, directions.get(record["step"])))
    return round_public_numbers(table)


def slim_table(run: Path, episode: int = 0) -> list:
    return slim_records(load_records(run, episode), load_vectors(run, episode))


# ---- the public run facts and manifest --------------------------------------

def carries_path(token: str) -> bool:
    return any(pattern.search(token) for pattern in PATH_SCANS)


def public_argv(argv: list) -> list:
    """Drop --log-dir, its value, and any token naming a local machine path."""
    cleaned, skip_next = [], False
    for token in argv:
        if skip_next:
            skip_next = False
        elif token == "--log-dir" or token.startswith("--log-dir="):
            skip_next = token == "--log-dir"
        elif carries_path(token):
            if cleaned and cleaned[-1].startswith("--"):
                cleaned.pop()
        else:
            cleaned.append(token)
    return cleaned


def public_run_record(session_record: dict) -> dict:
    run_facts = session_record.get("run", {})
    public_run = {key: run_facts[key] for key in RUN_FACT_KEYS if key in run_facts}
    public_run["argv"] = public_argv(run_facts.get("argv", []))
    return {"run": public_run, "modules": session_record.get("modules", {}),
            "recorded_episodes": session_record.get("recorded_episodes", [])}


def public_manifest(manifest: dict) -> dict:
    return {key: content for key, content in manifest.items()
            if key not in PRIVATE_MANIFEST_KEYS}


# ---- the leak scan -----------------------------------------------------------

def scan_text(text: str, source: str) -> None:
    for pattern in (DATE_SCAN, TIME_SCAN, EPOCH_SCAN) + PATH_SCANS:
        hit = pattern.search(text)
        if hit:
            raise PublicLeakError("%s carries %r (pattern %s)"
                                  % (source, hit.group(0), pattern.pattern))


def scan_media(path: Path) -> None:
    """Container metadata: dates, ISO datetimes, epochs and paths.

    A bare HH:MM:SS is skipped here on purpose: every webm and mp4 reports its
    own duration in that shape, which is not a recording date.
    """
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_format",
                            "-show_streams", "-of", "json", path.name],
                           capture_output=True, cwd=str(path.parent))
    if probe.returncode:
        raise RuntimeError("ffprobe failed for %s" % path)
    for pattern in (DATE_SCAN, ISO_SCAN, EPOCH_SCAN) + PATH_SCANS:
        hit = pattern.search(probe.stdout.decode("utf-8"))
        if hit:
            raise PublicLeakError("%s metadata carries %r (pattern %s)"
                                  % (path.name, hit.group(0), pattern.pattern))


def scan_poster(path: Path) -> None:
    tags = dict(Image.open(path).getexif())
    if tags:
        scan_text(str(tags), path.name + " EXIF")


def scan_bundle(bundle_dir: Path) -> None:
    for path in sorted(bundle_dir.iterdir()):
        if path.suffix == ".json":
            scan_text(path.read_text(encoding="utf-8"), path.name)
        elif path.suffix in (".webm", ".mp4"):
            scan_media(path)
        elif path.suffix == ".jpg":
            scan_poster(path)


# ---- the size guard -----------------------------------------------------------

def folder_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def check_sessions_size(bundle_dir: Path, site_dir: Path,
                        limit_mb: float = SESSIONS_LIMIT_MB,
                        replaced: Path = None) -> float:
    """Total after the export; a bundle being replaced is not counted twice."""
    bundle_mb = folder_bytes(bundle_dir) / 1e6 if bundle_dir.exists() else 0.0
    existing = []
    if site_dir.is_dir():
        existing = [entry for entry in site_dir.iterdir()
                    if entry not in (bundle_dir, replaced)]
    total_mb = bundle_mb + sum(folder_bytes(entry) for entry in existing) / 1e6
    print("bundle %.1f MB, sessions folder total %.1f MB of %.0f MB limit"
          % (bundle_mb, total_mb, limit_mb))
    if total_mb > limit_mb:
        raise SizeLimitError(
            "the export would push the sessions data folder to %.1f MB, over the "
            "%.0f MB limit. Move the folder to object storage before exporting more."
            % (total_mb, limit_mb))
    return total_mb


# ---- media encoding -----------------------------------------------------------

def dump_frames(frames: np.ndarray, folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(np.asarray(frame)).save(folder / (stem + "_%04d.png" % index))


def encode_video(frames_dir: Path, target: Path, frame_stem: str,
                 codec_args: list) -> None:
    command = (["ffmpeg", "-y", "-framerate", str(FRAME_RATE),
                "-i", str(frames_dir / (frame_stem + "_%04d.png")),
                "-map_metadata", "-1"] + codec_args + [str(target)])
    completed = subprocess.run(command, capture_output=True)
    if completed.returncode or not target.exists():
        tail = completed.stderr.decode("utf-8", "replace")[-400:]
        raise RuntimeError("ffmpeg failed for %s: %s" % (target.name, tail))


def write_webm(frames: np.ndarray, target: Path) -> None:
    with tempfile.TemporaryDirectory() as staging:
        folder = Path(staging)
        dump_frames(frames, folder, "frame")
        codec = ["-c:v", "libvpx-vp9", "-crf", "34", "-b:v", "0",
                 "-pix_fmt", "yuv420p"]
        encode_video(folder, target, "frame", codec)


def poster_canvas(frames: np.ndarray) -> Image.Image:
    middle = np.asarray(frames[len(frames) // 2])
    height, width = middle.shape[:2]
    scale = max(1, min(POSTER_HEIGHT // height, POSTER_WIDTH // width))
    frame_image = Image.fromarray(middle).resize(
        (width * scale, height * scale), Image.NEAREST)
    canvas = Image.new("RGB", (POSTER_WIDTH, POSTER_HEIGHT), INK_RGB)
    canvas.paste(frame_image, ((POSTER_WIDTH - frame_image.width) // 2,
                               (POSTER_HEIGHT - frame_image.height) // 2))
    return canvas


def write_poster(frames: np.ndarray, target: Path) -> None:
    poster_canvas(frames).save(target, "JPEG", quality=88)


# ---- the clip -----------------------------------------------------------------

def panel_font(size: int = 14) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans Mono"), size)


def module_palette() -> dict:
    return {"vision": ACCENT_RGB, "audio": GOLD_RGB, "memory": LIVE_RGB,
            "body": DIM_RGB, "semantic": TEXT_RGB}


def draw_bid_bars(draw: ImageDraw.ImageDraw, record: dict, origin: tuple,
                  max_bid: float, font) -> None:
    left, top = origin
    for row, module in enumerate(("vision", "audio", "memory", "body", "semantic")):
        y = top + row * 26
        draw.text((left, y - 2), module[:4], fill=DIM_RGB, font=font)
        raw = float(record.get("raw_bids", {}).get(module) or 0.0)
        bound = float(record.get("bound_bids", {}).get(module) or 0.0)
        span = CLIP_PANEL_WIDTH - 90
        draw.rectangle([left + 42, y + 2, left + 42 + span, y + 16],
                       outline=PANEL_RGB, width=1)
        draw.rectangle([left + 42, y + 2, left + 42 + span * min(1, raw / max_bid), y + 16],
                       outline=DIM_RGB, width=1)
        draw.rectangle([left + 42, y + 2, left + 42 + span * min(1, bound / max_bid), y + 16],
                       fill=module_palette()[module])


def draw_winner_strip(draw: ImageDraw.ImageDraw, records: list, index: int,
                      origin: tuple) -> None:
    left, top = origin
    cell_width = CLIP_PANEL_WIDTH / len(records)
    for position, record in enumerate(records):
        module = record.get("winner") or ""
        colour = module_palette().get(module, SILENCE_RGB)
        x = left + position * cell_width
        draw.rectangle([x, top, x + cell_width, top + 14], fill=colour)
    cursor = left + index * cell_width
    draw.rectangle([cursor - 1, top - 3, cursor + 1, top + 17], fill=TEXT_RGB)


def draw_trace(draw: ImageDraw.ImageDraw, series: list, index: int,
               rect: tuple, colour) -> None:
    left, top, right, bottom = rect
    draw.rectangle(rect, outline=PANEL_RGB, width=1)
    low, high = min(series), max(series)
    span = (high - low) or 1.0
    points = []
    for position, number in enumerate(series):
        x = left + (right - left) * position / max(1, len(series) - 1)
        y = bottom - (bottom - top - 2) * (number - low) / span - 1
        points.append((x, y))
    draw.line(points, fill=colour, width=1)
    if points:
        draw.ellipse([points[index][0] - 2, points[index][1] - 2,
                      points[index][0] + 2, points[index][1] + 2], fill=TEXT_RGB)


def clip_composite(index: int, frame: np.ndarray, records: list, max_bid: float,
                   font) -> Image.Image:
    frame_side = frame.shape[0] * 2
    canvas = Image.new("RGB", (frame_side + CLIP_PANEL_WIDTH, CLIP_HEIGHT), INK_RGB)
    scaled = Image.fromarray(np.asarray(frame)).resize(
        (frame_side, frame_side), Image.NEAREST)
    canvas.paste(scaled, (0, (CLIP_HEIGHT - frame_side) // 2))
    draw = ImageDraw.Draw(canvas)
    record = records[index]
    left = frame_side + 14
    winner = record.get("winner") or "silence"
    draw.text((left, 8), "STEP %04d/%04d  WINNER %s%s"
              % (index, len(records) - 1, winner.upper(),
                 "" if record.get("ignited") else "  SILENT"),
              fill=ACCENT_RGB if record.get("ignited") else DIM_RGB, font=font)
    draw_bid_bars(draw, record, (left, 34), max_bid, font)
    strip_top = CLIP_HEIGHT - 92
    draw.text((left, strip_top - 16), "WINNER STRIP ACROSS THE EPISODE",
              fill=DIM_RGB, font=font)
    draw_winner_strip(draw, records, index, (left, strip_top))
    draw_trace(draw, [float(r.get("sync_r") or 0.0) for r in records], index,
               (left, strip_top + 26, left + CLIP_PANEL_WIDTH - 8, strip_top + 48),
               GOLD_RGB)
    draw_trace(draw, [float(r.get("reward") or 0.0) for r in records], index,
               (left, strip_top + 54, left + CLIP_PANEL_WIDTH - 8, strip_top + 76),
               LIVE_RGB)
    return canvas


def write_clip(records: list, frames: np.ndarray, target: Path) -> None:
    font = panel_font()
    bids = [float(record.get("raw_bids", {}).get(module) or 0.0)
            for record in records for module in ("vision", "audio", "memory",
                                                 "body", "semantic")]
    max_bid = max(bids) or 1.0
    with tempfile.TemporaryDirectory() as staging:
        folder = Path(staging)
        dump_frames([clip_composite(i, frame, records, max_bid, font)
                     for i, frame in enumerate(frames)], folder, "clip")
        codec = ["-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart"]
        encode_video(folder, target, "clip", codec)


# ---- assembly -----------------------------------------------------------

def validate_session_id(session_id: str) -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]+", session_id):
        raise ValueError("session ids are lowercase words joined by dashes: %r" % session_id)
    if DATE_SCAN.search(session_id):
        raise ValueError("session ids must not carry a date: %r" % session_id)


def write_json(path: Path, content, compact: bool = False) -> None:
    if compact:
        text = json.dumps(content, separators=(",", ":"))
    else:
        text = json.dumps(content, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def write_bundle(bundle_dir: Path, run: Path, episode: int) -> None:
    bundle_dir.mkdir(parents=True)
    records = load_records(run, episode)
    frames = load_frames(run, episode)
    vectors = load_vectors(run, episode)
    write_json(bundle_dir / "steps.json",
               slim_records(records, vectors), compact=True)
    write_json(bundle_dir / "session.json",
               public_run_record(read_json(run / "session.json")))
    write_json(bundle_dir / "ethics_manifest.json",
               public_manifest(read_json(run / "ethics_manifest.json")))
    write_webm(frames, bundle_dir / "frames.webm")
    write_poster(frames, bundle_dir / "poster.jpg")
    write_clip(records, frames, bundle_dir / "clip.mp4")


def export_session(run: Path, episode: int, session_id: str, site_dir: Path,
                   limit_mb: float = SESSIONS_LIMIT_MB, replace: bool = False) -> Path:
    """Stage, scan and size-check the bundle, then move it into the site folder.

    An existing bundle with the same id is refused unless replace is true. Without
    this check shutil.move would put the new bundle INSIDE the old folder.
    """
    validate_session_id(session_id)
    target = site_dir / session_id
    if target.exists() and not replace:
        raise FileExistsError("session %r already exists in %s; pass replace=True "
                              "(--replace) to overwrite it" % (session_id, site_dir))
    with tempfile.TemporaryDirectory() as staging:
        bundle_dir = Path(staging) / session_id
        write_bundle(bundle_dir, run, episode)
        scan_bundle(bundle_dir)
        check_sessions_size(bundle_dir, site_dir, limit_mb, replaced=target)
        site_dir.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(bundle_dir), str(target))
    return target


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path, help="the run folder")
    parser.add_argument("--episode", required=True, type=int, help="episode index")
    parser.add_argument("--session-id", required=True,
                        help="public id, lowercase words and dashes, no dates")
    parser.add_argument("--site-dir", required=True, type=Path,
                        help="the website's sessions data folder")
    parser.add_argument("--limit-mb", type=float, default=SESSIONS_LIMIT_MB,
                        help="sessions folder limit the export must respect")
    parser.add_argument("--replace", action="store_true",
                        help="overwrite an existing bundle with the same session id")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    bundle_dir = export_session(args.run, args.episode, args.session_id, args.site_dir,
                                args.limit_mb, replace=args.replace)
    for path in sorted(bundle_dir.iterdir()):
        print("%-22s %8.1f KB" % (path.name, path.stat().st_size / 1024))
    return 0


if __name__ == "__main__":
    main()
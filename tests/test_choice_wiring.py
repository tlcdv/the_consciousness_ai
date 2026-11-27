"""The unit tests of group D of step P2 of design P, the code of checks 3 and 4 in
scripts/analysis/check_choice_wiring.py, written and pushed before that code.

Rules: docs/decisions/2026_11_22_workspace_choice_mode.md, the sections "Check 3" and "Check 4", N5, N6,
N7 and O8 of "Changes after the independent review", the section "Changes after the independent review of
the wiring" (C1, C2, C3 and the build commit), and the section "Owner decisions on the open places of the
code of checks 3 and 4" (rows 1, 3, 8, 14 and 29, tests D20 to D24). The tests build a made git repository and made
run folders. The run folders are written with the writers of the training script and of the session
record, so the records have the form of a run. The tests run no training and read no saved run.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

EPISODES, STEPS, CHOICE_STEP = 10, 200, 12
BANDS = {18: 0.4, 23: 0.9}
RECORDS = ["metrics.csv", "episodes.csv", "value_gate.json", "tectal_orienting.json", "map_binding.json"] + \
    ["movement ep_%04d" % episode for episode in range(EPISODES)]


def wiring():
    """The module under test, imported in each test, so that each test fails alone before the code exists."""
    from scripts.analysis import check_choice_wiring
    return check_choice_wiring


# A made git repository for check 3a and for the rule of the build commit.

SOURCE = ("import os\n\n\n"
          "def run_episode(env, steps):\n"
          "    # the loop of one episode\n"
          "    return [env.step() for _ in range(steps)]\n\n\n"
          "def init_components(config):\n"
          "    return dict(config)\n\n\n"
          "def main():\n"
          "    return run_episode(None, 0)\n")
RUN_EPISODE_TEXT = ("def run_episode(env, steps):\n    # the loop of one episode\n"
                    "    return [env.step() for _ in range(steps)]")
INIT_COMPONENTS_TEXT = "def init_components(config):\n    return dict(config)"
SOURCE_WIRED = SOURCE.replace("def main():\n    return run_episode(None, 0)\n",
                              "def _wire_new(args):\n    return args\n\n\n"
                              "def main():\n    return _wire_new(run_episode(None, 0))\n")


def with_comment(source: str) -> str:
    return source.replace("# the loop of one episode", "# the loop of one episode, changed")


def git(repo, *arguments) -> str:
    command = ["git", "-C", str(repo), "-c", "core.autocrlf=false", "-c", "user.name=made",
               "-c", "user.email=made@example.invalid", "-c", "commit.gpgsign=false"] + list(arguments)
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def commit(repo, parent, files: dict, message: str) -> str:
    """A new commit on `parent` (None for the first commit) with these files written. Returns its id."""
    if parent is not None:
        git(repo, "checkout", "-q", "--detach", parent)
    for path, text in files.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture(scope="module")
def made_repo(tmp_path_factory):
    repo = tmp_path_factory.mktemp("repo")
    git(repo, "init", "-q")
    c = {"base": commit(repo, None, {"scripts/training/train_rlhf.py": SOURCE, "models/core/a.py": "A = 1\n",
                                     "simulations/room.py": "R = 1\n"}, "base")}
    c["tests"] = commit(repo, c["base"], {"tests/test_x.py": "X = 1\n"}, "tests")
    c["wiring"] = commit(repo, c["tests"], {"scripts/training/train_rlhf.py": SOURCE_WIRED,
                                            "models/core/choice.py": "C = 1\n"}, "wiring")
    c["build"] = commit(repo, c["wiring"], {"scripts/analysis/check.py": "K = 1\n", "docs/rules.md": "rules\n"},
                        "build")
    c["models_changed"] = commit(repo, c["build"], {"models/core/a.py": "A = 2\n"}, "a change under models")
    c["simulations_changed"] = commit(repo, c["build"], {"simulations/room.py": "R = 2\n"}, "a change of the room")
    c["training_changed"] = commit(repo, c["build"], {"scripts/training/other.py": "O = 1\n"}, "a new training file")
    c["reverted"] = commit(repo, c["models_changed"], {"models/core/a.py": "A = 1\n"}, "the change put back")
    c["side"] = commit(repo, c["tests"], {"scripts/training/train_rlhf.py": SOURCE_WIRED,
                                          "models/core/choice.py": "C = 1\n", "docs/side.md": "side\n"},
                       "the content of the wiring on another line")
    c["wiring_comment"] = commit(repo, c["tests"], {"scripts/training/train_rlhf.py": with_comment(SOURCE_WIRED)},
                                 "wiring that changes a comment of run_episode")
    c["build_comment"] = commit(repo, c["wiring_comment"], {"docs/rules.md": "rules\n"}, "build on it")
    c["tests_comment"] = commit(repo, c["base"], {"scripts/training/train_rlhf.py": with_comment(SOURCE),
                                                  "tests/test_x.py": "X = 1\n"}, "tests that change the comment")
    c["wiring_after_comment"] = commit(repo, c["tests_comment"],
                                       {"scripts/training/train_rlhf.py": with_comment(SOURCE_WIRED)}, "wiring")
    c["build_after_comment"] = commit(repo, c["wiring_after_comment"], {"docs/rules.md": "rules\n"}, "build")
    return repo, c


# Made run folders of 10 episodes of 200 steps. The choice step is step 12 of each episode.

def broadcast_row(episode: int, step: int) -> np.ndarray:
    """Values 0.1 x episode plus and minus (1 + 0.01 x step) in turn, so the mean of a row is 0.1 x episode,
    its standard deviation 1 + 0.01 x step, and its largest absolute value their sum."""
    signs = np.where(np.arange(256) % 2 == 0, 1.0, -1.0)
    return (0.1 * episode + (1.0 + 0.01 * step) * signs).astype(np.float32)


def reward(step: int, after_choice: float) -> float:
    return -0.01 if step <= CHOICE_STEP else -0.01 + after_choice


def binding_history(one_candidate=(), no_candidate=()) -> list:
    """Step 0 of each episode has no sound. From the choice step on, bands 18 and 23 have groups, but in
    an episode of `one_candidate` only band 23 has a group at the choice step, and an episode of
    `no_candidate` has no group."""
    history = []
    for episode in range(EPISODES):
        for step in range(STEPS):
            if step == 0:
                history.append({"band": None, "groups": {}, "salience": {}})
                continue
            bound = step >= CHOICE_STEP and episode not in no_candidate
            alone = episode in one_candidate and step == CHOICE_STEP
            groups = {"18": [3, 4] if bound and not alone else [], "23": [10] if bound else []}
            history.append({"band": 23, "groups": groups, "salience": {"18": 0.4, "23": 0.9}})
    return history


def write_json(path, content) -> None:
    path.write_text(json.dumps(content), encoding="utf-8")


def write_csv(path, header: str, rows) -> None:
    path.write_bytes(("\r\n".join([header] + list(rows)) + "\r\n").encode("utf-8"))


def write_episode(folder, episode: int, after_choice: float) -> None:
    """One episode folder, written by the episode buffer of the session record."""
    from scripts.training.session_recorder import _EpisodeBuffer
    buffer = _EpisodeBuffer(episode, ["light"], [], folder / "episodes" / ("ep_%04d" % episode), STEPS)
    for step in range(STEPS):
        shift = after_choice if step > CHOICE_STEP else 0.0
        buffer.add_inputs(np.zeros((2, 2, 3), dtype=np.uint8), {"input_source": {}, "raw_bids": {},
                                                                "interoception": {}})
        buffer.add_outcome({"winner": "audio", "ignited": True, "sync_r": 0.5, "bound_bids": {},
                            "reward": reward(step, after_choice), "env_phase": "dark",
                            "action": [0.0, 0.0] if step <= CHOICE_STEP else [0.3 + shift, 0.1]})
        vectors = {"broadcast": (broadcast_row(episode, step) + shift).astype(np.float32)[None, :],
                   "tectum_content": np.full((1, 256), 0.5 * episode + 0.001 * step + shift, np.float32),
                   "audio_content": np.full((1, 256), -0.25 * episode - 0.002 * step - shift, np.float32)}
        if step == 0:
            del vectors["audio_content"]
        buffer.add_extra({"info_after_step": {"distance": 1.0 - 0.001 * step - shift}}, vectors, {}, {})
    buffer.write({})


def write_gate_records(folder) -> None:
    from models.core.value_gate import ValueGate
    from scripts.training import train_rlhf
    gate = ValueGate(73 + 1000003)
    for _ in range(EPISODES):
        gate.choose(dict(BANDS))
        gate.observe_reward(1.0)
        gate.new_episode()
    write_json(folder / "value_gate.json", train_rlhf._value_gate_record(gate, 0.3))
    write_json(folder / "tectal_orienting.json", {"mode": "value", "steps": EPISODES * STEPS,
                                                  "history": [{"target": None}] * (EPISODES * STEPS)})


def write_head_records(folder, one_candidate, no_candidate, open_last: bool) -> None:
    """The record of the head, written by the writer of the training script, from a live head that
    chooses at the choice step of each episode and closes each choice with food at the next step."""
    from models.core.choice_head import LiveChoiceHead
    from scripts.training import train_rlhf
    live = LiveChoiceHead(73)
    for episode in range(EPISODES):
        if episode in no_candidate:
            continue
        candidates = {23: 0.9} if episode in one_candidate else dict(BANDS)
        live.choose(candidates, broadcast_row(episode, CHOICE_STEP), step=CHOICE_STEP, episode=episode)
        if not (open_last and episode == EPISODES - 1):
            live.observe_reward(1.0)
            live.new_episode()
    motor = SimpleNamespace(history=[{"target": None}] * (EPISODES * STEPS))
    train_rlhf._write_choice_head_record(live, motor, str(folder))


def made_run(folder, commit_id: str, mode: str, after_choice: float = 0.0, one_candidate=(), no_candidate=(),
             open_last: bool = False) -> None:
    """A run folder. `mode` "value" writes the records of the value gate, "plain" those of the head.
    `after_choice` changes the rewards, the actions and the vectors after the choice step."""
    folder.mkdir(parents=True)
    write_json(folder / "ethics_manifest.json", {"git_commit": commit_id, "git_tracked_changes": False})
    write_json(folder / "environment.json", {"python": "3.12.10", "platform": "made", "torch": "2.14.1+cu126"})
    write_json(folder / "map_binding.json", {"steps": EPISODES * STEPS,
                                             "history": binding_history(one_candidate, no_candidate)})
    write_csv(folder / "metrics.csv", "global_step,reward",
              ["%d,%.6f" % (e * STEPS + t, reward(t, after_choice)) for e in range(EPISODES) for t in range(STEPS)])
    total = sum(reward(t, after_choice) for t in range(STEPS))
    write_csv(folder / "episodes.csv", "episode,total_reward,steps",
              ["%d,%.4f,%d" % (e, total, STEPS) for e in range(EPISODES)])
    for episode in range(EPISODES):
        write_episode(folder, episode, after_choice)
    if mode == "value":
        write_gate_records(folder)
    else:
        write_head_records(folder, one_candidate, no_candidate, open_last)


@pytest.fixture(scope="module")
def made_runs(tmp_path_factory, made_repo):
    _, c = made_repo
    root = tmp_path_factory.mktemp("runs")
    made_run(root / "base", c["base"], "value")
    made_run(root / "off", c["build"], "value")
    made_run(root / "wiring", c["build"], "plain", after_choice=0.5)
    made_run(root / "variants", c["build"], "plain", after_choice=0.5, one_candidate=(5,), no_candidate=(7,),
             open_last=True)
    return root


@pytest.fixture
def runs(tmp_path, made_runs, made_repo):
    """A copy of the made runs for one test, which the test can change."""
    shutil.copytree(made_runs, tmp_path / "runs")
    repo, c = made_repo
    folder = tmp_path / "runs"
    return SimpleNamespace(base=folder / "base", off=folder / "off", wiring=folder / "wiring",
                           variants=folder / "variants", repo=str(repo), commits=c)


# Changes of a made run.

def change_json(path, change) -> None:
    content = json.loads(path.read_text(encoding="utf-8"))
    change(content)
    write_json(path, content)


def change_metrics_line(run, global_step: int) -> None:
    lines = (run / "metrics.csv").read_bytes().split(b"\r\n")
    lines[global_step + 1] += b"1"
    (run / "metrics.csv").write_bytes(b"\r\n".join(lines))


def change_step(run, episode: int, step: int, **fields) -> None:
    path = run / "episodes" / ("ep_%04d" % episode) / "steps.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    line = json.loads(lines[step])
    line.update(fields)
    lines[step] = json.dumps(line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def change_vector(run, episode: int, name: str, step: int, delta: float = 1.0) -> None:
    path = run / "episodes" / ("ep_%04d" % episode) / "vectors.npz"
    with np.load(path) as tables:
        arrays = {key: tables[key].copy() for key in tables.files}
    index = int(np.flatnonzero(arrays[name + "__steps"] == step)[0])
    arrays[name][index].flat[0] += delta
    np.savez_compressed(path, **arrays)


def change_choices(run, change) -> None:
    change_json(run / "choice_head.json", lambda record: change(record["choices"]))


def identity(runs, base=None, off=None, wiring_commit=None):
    return wiring().identity_check(runs.repo, str(base or runs.base), str(off or runs.off),
                                   base_commit=runs.commits["base"], tests_commit=runs.commits["tests"],
                                   wiring=wiring_commit or runs.commits["wiring"])


def wiring_report(runs, run=None):
    return wiring().wiring_check(runs.repo, str(runs.off), str(run or runs.wiring), wiring=runs.commits["wiring"])


# Check 3a.

def test_d1_the_text_of_a_function_is_its_source_from_its_first_decorator_to_its_last_line():
    check = wiring()
    source = ("import x\n\n\n@decorate\ndef run_episode(a):\n    # a comment\n    return a + 1  # end\n\n\n"
              "def outer():\n    def init_components():\n        return 0\n    return init_components\n")
    assert check.function_text(source, "run_episode") == \
        "@decorate\ndef run_episode(a):\n    # a comment\n    return a + 1  # end"
    assert check.function_text(SOURCE, "init_components") == INIT_COMPONENTS_TEXT
    with pytest.raises(ValueError):
        check.function_text(source, "init_components")                 # nested, not top level
    with pytest.raises(ValueError):
        check.function_text(SOURCE + "\n\ndef run_episode():\n    pass\n", "run_episode")


def test_d2_check_3a_compares_the_build_commit_with_each_base_commit_and_reports_both(made_repo):
    check, (repo, c) = wiring(), made_repo
    report = check.source_check(str(repo), c["build"], bases=(c["base"], c["tests"]), wiring=c["wiring"])
    assert report["label"] == "EQUAL" and report["cause"] is None and report["build_commit"] == c["build"]
    assert report["equal"] == {c["base"]: {"run_episode": True, "init_components": True},
                               c["tests"]: {"run_episode": True, "init_components": True}}
    for commit_id in (c["build"], c["base"], c["tests"]):
        assert report["md5"]["run_episode"][commit_id] == hashlib.md5(RUN_EPISODE_TEXT.encode()).hexdigest()
        assert report["md5"]["init_components"][commit_id] == hashlib.md5(INIT_COMPONENTS_TEXT.encode()).hexdigest()
    changed = check.source_check(str(repo), c["build_comment"], bases=(c["base"], c["tests"]),
                                 wiring=c["wiring_comment"])
    assert changed["label"] == "A DIFFERENCE" and changed["changed_code_files"] == []
    assert changed["equal"][c["base"]] == {"run_episode": False, "init_components": True}
    both = check.source_check(str(repo), c["build_after_comment"], bases=(c["base"], c["tests_comment"]),
                              wiring=c["wiring_after_comment"])
    assert both["label"] == "A DIFFERENCE"
    assert both["equal"][c["base"]]["run_episode"] is False and both["equal"][c["tests_comment"]]["run_episode"] is True


def test_d3_a_file_changed_under_models_simulations_or_scripts_training_after_the_wiring_makes_the_check_void(made_repo):
    check, (repo, c) = wiring(), made_repo
    for name, path in (("models_changed", "models/core/a.py"), ("simulations_changed", "simulations/room.py"),
                       ("training_changed", "scripts/training/other.py")):
        report = check.source_check(str(repo), c[name], bases=(c["base"], c["tests"]), wiring=c["wiring"])
        assert report["label"] == "VOID" and report["changed_code_files"] == [path], name
    assert check.build_rule(str(repo), c["build"], c["wiring"])["changed_code_files"] == []
    side = check.source_check(str(repo), c["side"], bases=(c["base"], c["tests"]), wiring=c["wiring"])
    assert side["label"] == "VOID" and side["descends_from_wiring"] is False and side["changed_code_files"] == []


# Check 3b.

def test_d4_the_movement_record_has_five_fields_with_sorted_keys_and_no_wall_time(tmp_path):
    check = wiring()
    line = {"step": 3, "action": [0.5, -0.25], "reward": -0.01, "env_phase": "dark", "winner": "audio",
            "internals": {"info_after_step": {"distance": 1.5, "battery": 0.997}, "losses": {"x": 1.0}},
            "seconds_inputs_to_end": 0.125}
    path = tmp_path / "steps.jsonl"
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    expected = ('{"action": [0.5, -0.25], "env_phase": "dark", "info_after_step": {"battery": 0.997, '
                '"distance": 1.5}, "reward": -0.01, "step": 3}\n')
    assert check.movement_text(str(path)) == expected
    assert check.md5_of_text(expected) == hashlib.md5(expected.encode("utf-8")).hexdigest()
    line["seconds_inputs_to_end"], line["winner"], line["internals"]["losses"] = 9.5, "vision", {}
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    assert check.movement_text(str(path)) == expected
    line["action"] = [0.5, -0.5]
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    assert check.movement_text(str(path)) != expected


def test_d5_two_runs_that_differ_only_in_wall_time_are_equal_in_each_record(runs):
    change_step(runs.off, 2, 5, seconds_inputs_to_end=99.0)
    report = identity(runs)
    assert report["label"] == "EQUAL" and report["cause"] is None
    assert list(report["equal"]) == RECORDS and all(report["equal"].values())
    assert report["build_commit"] == runs.commits["build"] and report["changed_code_files"] == []
    for side, folder in (("base", runs.base), ("off", runs.off)):
        assert set(report["md5"][side]) == set(RECORDS) | {"environment.json"}
        assert report["md5"][side]["metrics.csv"] == hashlib.md5((folder / "metrics.csv").read_bytes()).hexdigest()
        assert report["counts"][side]["metrics_lines"] == 2001 and report["counts"][side]["closed_choices"] == 10
    assert report["metrics_md5_is_competing_goals"] == {"base": False, "off": False}
    assert report["environment_equal"] is True and report["first_record_that_differs"] is None


def test_d6_a_difference_reports_the_first_record_and_the_first_step_that_differ(runs):
    change_metrics_line(runs.off, 37)
    report = identity(runs)
    assert report["label"] == "A DIFFERENCE" and report["first_record_that_differs"] == "metrics.csv"
    assert report["first_difference"]["metrics.csv"] == {"line": 38, "global_step": 37}
    assert [name for name, same in report["equal"].items() if not same] == ["metrics.csv"]
    change_metrics_line(runs.base, 37)                                 # the same change in both runs
    change_step(runs.off, 4, 9, action=[0.0, 0.5])
    change_json(runs.off / "map_binding.json", lambda record: record["history"][1234].update(band=18))
    report = identity(runs)
    assert report["first_record_that_differs"] == "map_binding.json"
    assert report["first_difference"]["map_binding.json"] == {"index": 1234, "episode": 6, "step": 34}
    assert report["first_difference"]["movement ep_0004"] == {"line": 9, "step": 9}
    assert [name for name, same in report["equal"].items() if not same] == ["map_binding.json", "movement ep_0004"]


def test_d7_a_run_that_does_not_complete_or_a_wrong_commit_makes_check_3b_void(runs):
    lines = (runs.off / "metrics.csv").read_bytes().split(b"\r\n")
    (runs.off / "metrics.csv").write_bytes(b"\r\n".join(lines[:-2] + [b""]))
    report = identity(runs)
    assert report["label"] == "VOID" and report["counts"]["off"]["metrics_lines"] == 2000
    shutil.copy(runs.base / "metrics.csv", runs.off / "metrics.csv")
    assert identity(runs)["label"] == "EQUAL"
    shutil.rmtree(runs.off / "episodes" / "ep_0009")
    assert identity(runs)["label"] == "VOID"
    other = runs.off.parent / "base_again"
    shutil.copytree(runs.base, other)
    change_json(other / "ethics_manifest.json", lambda manifest: manifest.update(git_commit=runs.commits["tests"]))
    assert identity(runs, base=other)["label"] == "VOID"               # not the base commit of the base run
    shutil.copytree(runs.base, runs.off.parent / "off_changed")
    changed = runs.off.parent / "off_changed"
    change_json(changed / "ethics_manifest.json",
                lambda manifest: manifest.update(git_commit=runs.commits["models_changed"]))
    report = identity(runs, off=changed)
    assert report["label"] == "VOID" and report["changed_code_files"] == ["models/core/a.py"]


def test_d8_each_run_needs_10_closed_choices_even_with_equal_records(runs):
    for folder in (runs.base, runs.off):
        change_json(folder / "value_gate.json", lambda record: record["episodes"].pop())
    report = identity(runs)
    assert all(report["equal"].values()) and report["counts"]["off"]["closed_choices"] == 9
    assert report["label"] == "A DIFFERENCE"


def test_d9_environment_json_is_compared_and_reported_apart_from_the_label(runs):
    change_json(runs.off / "environment.json", lambda facts: facts.update(platform="another"))
    report = identity(runs)
    assert report["label"] == "EQUAL"
    assert report["environment_equal"] is False and report["environment_keys_that_differ"] == ["platform"]


# Check 4.

def test_d10_a_run_that_follows_the_rules_gives_each_item_as_expected_and_the_reported_numbers(runs):
    report = wiring_report(runs)
    assert report["label"] == "AS EXPECTED" and report["cause"] is None
    assert report["items"] == {"W1": True, "W2": True, "W3": True, "W4": True, "W5": True, "W6": True}
    assert report["details"]["W2"]["first_choice_step"] == {"off": 12, "wiring": 12}
    assert report["details"]["W3"]["one_candidate_episodes"] == 0
    assert report["details"]["W5"]["adam_steps"] == [50] and report["details"]["W5"]["adam_steps_expected"] == 50
    entries = json.loads((runs.wiring / "choice_head.json").read_text(encoding="utf-8"))["choices"]
    reported = report["reported"]
    assert reported["exploration_draws"] == sum(entry["explored"] for entry in entries)
    assert reported["outputs"] == [entry["outputs"] for entry in entries]
    assert reported["outcomes"] == [1.0] * EPISODES
    for episode, row in enumerate(reported["input_rows"]):
        assert row["mean"] == pytest.approx(0.1 * episode, abs=1e-5)
        assert row["standard_deviation"] == pytest.approx(1.12, abs=1e-5)
        assert row["largest_absolute_value"] == pytest.approx(0.1 * episode + 1.12, abs=1e-5)


def test_d11_check_4_is_void_on_a_run_that_does_not_complete_or_on_a_commit_that_breaks_the_rule(runs):
    lines = (runs.wiring / "metrics.csv").read_bytes().split(b"\r\n")
    (runs.wiring / "metrics.csv").write_bytes(b"\r\n".join(lines[:-2] + [b""]))
    report = wiring_report(runs)
    assert report["label"] == "VOID" and report["items"]["W1"] is False
    shutil.copy(runs.variants / "metrics.csv", runs.wiring / "metrics.csv")
    assert wiring_report(runs)["label"] == "AS EXPECTED"
    change_json(runs.wiring / "ethics_manifest.json", lambda m: m.update(git_commit=runs.commits["reverted"]))
    assert wiring_report(runs)["label"] == "VOID"                      # the two runs have different commits
    change_json(runs.off / "ethics_manifest.json", lambda m: m.update(git_commit=runs.commits["simulations_changed"]))
    change_json(runs.wiring / "ethics_manifest.json",
                lambda m: m.update(git_commit=runs.commits["simulations_changed"]))
    report = wiring_report(runs)
    assert report["label"] == "VOID" and report["changed_code_files"] == ["simulations/room.py"]


def test_d12_w2_compares_the_metrics_before_the_choice_step_and_the_vectors_up_to_it(runs):
    change_metrics_line(runs.wiring, CHOICE_STEP)                      # the line of the choice step itself
    change_vector(runs.off, 0, "audio_content", CHOICE_STEP + 1)       # a vector after the choice step
    report = wiring_report(runs)
    assert report["items"]["W2"] is True and report["label"] == "AS EXPECTED"
    vectors = runs.off / "episodes" / "ep_0000" / "vectors.npz"
    kept = vectors.read_bytes()
    change_vector(runs.off, 0, "tectum_content", CHOICE_STEP)
    assert wiring_report(runs)["items"]["W2"] is False
    vectors.write_bytes(kept)
    assert wiring_report(runs)["items"]["W2"] is True
    change_metrics_line(runs.wiring, 5)
    report = wiring_report(runs)
    assert report["items"]["W2"] is False and report["details"]["W2"]["metrics_first_step_that_differs"] == 5
    assert report["label"] == "A MISMATCH"


def test_d13_w2_needs_the_same_first_choice_step_in_the_two_runs(runs):
    change_json(runs.off / "map_binding.json",
                lambda record: record["history"][CHOICE_STEP - 1].update(groups={"18": [1], "23": []}))
    report = wiring_report(runs)
    assert report["items"]["W2"] is False and report["details"]["W2"]["first_choice_step"] == {"off": 11, "wiring": 12}


def test_d14_w3_follows_c1_for_one_candidate_and_no_candidate_at_the_first_step_with_candidates(runs):
    report = wiring_report(runs, runs.variants)
    assert report["label"] == "AS EXPECTED"
    assert report["details"]["W3"]["one_candidate_episodes"] == 1
    assert report["details"]["W5"]["adam_steps_expected"] == 5 * 7    # episodes 5 and 7 and the open choice
    change_choices(runs.variants, lambda entries: entries[5].update(one_candidate=False))
    assert wiring_report(runs, runs.variants)["items"]["W3"] is False   # a choice with a draw at one candidate
    change_choices(runs.variants, lambda entries: entries[5].update(one_candidate=True))
    change_choices(runs.variants, lambda entries: entries.append(dict(entries[0], episode=7)))
    assert wiring_report(runs, runs.variants)["items"]["W3"] is False   # a choice in an episode with no candidate


def test_d15_w3_finds_a_second_choice_a_wrong_step_and_a_choice_with_no_episode(runs):
    for change in (lambda entries: entries.insert(4, dict(entries[3])),
                   lambda entries: entries[2].update(step=CHOICE_STEP + 1),
                   lambda entries: entries[6].update(episode=None)):
        original = (runs.wiring / "choice_head.json").read_bytes()
        change_choices(runs.wiring, change)
        assert wiring_report(runs)["items"]["W3"] is False
        (runs.wiring / "choice_head.json").write_bytes(original)
    assert wiring_report(runs)["items"]["W3"] is True


def test_d16_w4_follows_c2_with_the_episode_folder_and_the_step_of_each_entry(runs):
    path = runs.wiring / "choice_head_inputs.npy"
    rows = np.load(path)
    changed = rows.copy()
    changed[3, 0] += 1.0
    np.save(path, changed)
    assert wiring_report(runs)["items"]["W4"] is False
    np.save(path, rows[:-1])
    assert wiring_report(runs)["items"]["W4"] is False
    np.save(path, rows)
    assert wiring_report(runs)["items"]["W4"] is True
    change_choices(runs.wiring, lambda entries: entries[4].update(episode=5))   # an error of one episode
    report = wiring_report(runs)
    assert report["items"]["W4"] is False and report["items"]["W3"] is False
    change_choices(runs.wiring, lambda entries: entries[4].update(episode=12))  # no folder ep_0012
    assert wiring_report(runs)["items"]["W4"] is False


def test_d17_w5_reads_the_updates_the_adam_steps_and_the_weights(runs):
    change_choices(runs.wiring, lambda entries: entries[0].update(updates=0))
    assert wiring_report(runs)["items"]["W5"] is False
    change_choices(runs.wiring, lambda entries: entries[0].update(updates=5))
    path = runs.wiring / "choice_head.pt"
    saved = torch.load(path, map_location="cpu")
    for state in saved["optimizer"]["state"].values():
        state["step"] = state["step"] - 1
    torch.save(saved, path)
    assert wiring_report(runs)["items"]["W5"] is False
    from models.core.choice_head import seeded_body
    for state in saved["optimizer"]["state"].values():
        state["step"] = state["step"] + 1
    saved["body"] = seeded_body(73 + 2000003).state_dict()
    torch.save(saved, path)
    report = wiring_report(runs)
    assert report["items"]["W5"] is False and report["details"]["W5"]["tensors_that_differ"] == 0


def test_d18_w6_finds_a_value_gate_record_in_the_wiring_run(runs):
    shutil.copy(runs.off / "value_gate.json", runs.wiring / "value_gate.json")
    report = wiring_report(runs)
    assert report["items"]["W6"] is False and report["label"] == "A MISMATCH"


def test_d19_the_three_modes_of_the_script_print_and_write_their_report(runs, tmp_path, monkeypatch, capsys):
    check = wiring()
    monkeypatch.setattr(check, "WIRING_COMMIT", runs.commits["wiring"])
    monkeypatch.setattr(check, "BASE_RUN_COMMIT", runs.commits["base"])
    monkeypatch.setattr(check, "TESTS_COMMIT", runs.commits["tests"])
    for argv, label in ((["--source", "--build-commit", runs.commits["build"]], "EQUAL"),
                        (["--identity", str(runs.base), str(runs.off)], "EQUAL"),
                        (["--wiring", str(runs.off), str(runs.wiring)], "AS EXPECTED")):
        summary = tmp_path / "summary.json"
        check.main(argv + ["--repo", runs.repo, "--summary", str(summary)])
        assert json.loads(summary.read_text(encoding="utf-8"))["label"] == label
    assert '"label": "AS EXPECTED"' in capsys.readouterr().out


# The owner decisions on the open places of the code, rows 1, 3, 8, 14 and 29.

def write_lines(path, lines) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def test_d20_row_1_a_change_that_a_later_commit_puts_back_still_makes_the_check_void(made_repo):
    check, (repo, c) = wiring(), made_repo
    rule = check.build_rule(str(repo), c["reverted"], c["wiring"])
    assert rule["changed_code_files"] == []                            # the net git diff is kept
    assert rule["committed_code_files"] == ["models/core/a.py"] and rule["void_cause"] is not None
    assert [entry["commit"] for entry in rule["commits_that_changed_code"]] == [c["reverted"], c["models_changed"]]
    report = check.source_check(str(repo), c["reverted"], bases=(c["base"], c["tests"]), wiring=c["wiring"])
    assert report["label"] == "VOID" and report["committed_code_files"] == ["models/core/a.py"]
    clean = check.build_rule(str(repo), c["build"], c["wiring"])
    assert clean["committed_code_files"] == [] and clean["commits_that_changed_code"] == []
    assert clean["void_cause"] is None


def test_d21_row_3_identity_reads_check_3a_at_the_commit_of_the_run_with_the_flag_off(runs):
    report = identity(runs)
    assert report["source"]["build_commit"] == report["build_commit"] == runs.commits["build"]
    assert report["source"]["bases"] == [runs.commits["base"], runs.commits["tests"]]
    assert report["source"]["label"] == "EQUAL"
    change_json(runs.off / "ethics_manifest.json", lambda m: m.update(git_commit=runs.commits["build_comment"]))
    report = identity(runs, wiring_commit=runs.commits["wiring_comment"])
    assert report["label"] == "EQUAL"                                   # 3b, the records are equal
    assert report["source"]["build_commit"] == runs.commits["build_comment"]
    assert report["source"]["label"] == "A DIFFERENCE"
    assert report["check_3"] == {"label": "A DIFFERENCE", "label_3a": "A DIFFERENCE", "label_3b": "EQUAL"}


def test_d22_row_8_a_steps_file_without_exactly_200_lines_makes_the_run_void(runs):
    steps = runs.wiring / "episodes" / "ep_0009" / "steps.jsonl"
    kept = steps.read_text(encoding="utf-8").splitlines()
    write_lines(steps, kept[:-1])
    report = wiring_report(runs)
    assert report["label"] == "VOID" and report["items"]["W1"] is False
    write_lines(steps, kept)
    assert wiring_report(runs)["label"] == "AS EXPECTED"
    path = runs.off / "episodes" / "ep_0003" / "steps.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    for changed in (lines[:-1], lines + [lines[-1]]):                  # 199 lines, then 201 lines
        write_lines(path, changed)
        report = identity(runs)
        assert report["label"] == "VOID" and report["counts"]["off"]["complete"] is False
        assert report["counts"]["off"]["steps_lines"][3] == len(changed)
    write_lines(path, lines)
    assert identity(runs)["label"] == "EQUAL"


def test_d23_row_14_a_run_with_changed_tracked_files_makes_the_check_void(runs):
    for folder in (runs.wiring, runs.off):
        change_json(folder / "ethics_manifest.json", lambda manifest: manifest.update(git_tracked_changes=True))
        assert wiring_report(runs)["label"] == "VOID", folder.name
        change_json(folder / "ethics_manifest.json", lambda manifest: manifest.update(git_tracked_changes=False))
    assert wiring_report(runs)["label"] == "AS EXPECTED"
    for folder in (runs.base, runs.off):
        change_json(folder / "ethics_manifest.json", lambda manifest: manifest.update(git_tracked_changes=True))
        report = identity(runs)
        assert report["label"] == "VOID" and report["check_3"]["label"] == "VOID", folder.name
        change_json(folder / "ethics_manifest.json", lambda manifest: manifest.update(git_tracked_changes=False))
    assert identity(runs)["label"] == "EQUAL"


def test_d24_row_29_the_label_of_check_3_reads_void_first_then_a_difference(runs):
    check = wiring()
    expected = {("EQUAL", "EQUAL"): "EQUAL", ("EQUAL", "A DIFFERENCE"): "A DIFFERENCE", ("EQUAL", "VOID"): "VOID",
                ("A DIFFERENCE", "EQUAL"): "A DIFFERENCE", ("A DIFFERENCE", "A DIFFERENCE"): "A DIFFERENCE",
                ("A DIFFERENCE", "VOID"): "VOID", ("VOID", "EQUAL"): "VOID", ("VOID", "A DIFFERENCE"): "VOID",
                ("VOID", "VOID"): "VOID"}
    for (label_3a, label_3b), label in expected.items():
        assert check.check_3_label(label_3a, label_3b) == label, (label_3a, label_3b)
    assert identity(runs)["check_3"] == {"label": "EQUAL", "label_3a": "EQUAL", "label_3b": "EQUAL"}
    change_metrics_line(runs.off, 37)
    assert identity(runs)["check_3"] == {"label": "A DIFFERENCE", "label_3a": "EQUAL", "label_3b": "A DIFFERENCE"}
    lines = (runs.off / "metrics.csv").read_bytes().split(b"\r\n")
    (runs.off / "metrics.csv").write_bytes(b"\r\n".join(lines[:-2] + [b""]))
    assert identity(runs)["check_3"] == {"label": "VOID", "label_3a": "EQUAL", "label_3b": "VOID"}

"""Checks 3 and 4 of step P2 of design P, the source of two functions, bit identity with the flag off, and
the wiring run with the flag on.

The rules are in docs/decisions/2026_11_22_workspace_choice_mode.md, the sections "Check 3" and "Check 4",
N5, N6, N7 and O8 of "Changes after the independent review", and the section "Changes after the
independent review of the wiring" (C1, C2, C3 and the build commit). This file was written before any run
of check 3b or check 4. Its unit tests, tests/test_choice_wiring.py, were pushed before it.

Three modes.
  --source    Check 3a (C3). The text of run_episode and init_components at a commit against their text at
              1366756 and at 32146d1. Both are reported, and each must be equal. It reads git only. With HEAD
              before the runs it is a first check only (owner decision on row 3). If it is not EQUAL, no run
              is made.
  --identity  Check 3b. The md5 of each record of the table of check 3b for the base run and for the run
              at the build commit with the flag off, the counts of 2001 lines in metrics.csv and of 10
              closed choices in value_gate.json, and environment.json (O8). The md5 of metrics.csv against
              the record of the competing goals gate is reported only. It also makes check 3a at the commit
              of the run with the flag off, which gives the label of 3a (row 3), and it gives the label of
              check 3 as a whole (row 29).
  --wiring    Check 4. W1 to W6 for the run with --choice-head plain against the run with the flag off, with
              C1 and C2 and W2 in the form of N5, and the numbers that the rules report only.

The build commit is the commit on which check 3b and check 4 run (owner rule). Checks 3b and 4 read it from
ethics_manifest.json of their runs. Each mode lists, with git log --name-only, each commit after b59d62d
that changes a file under models, simulations or scripts/training, and its label is VOID if one does, also
when a later commit puts the file back (row 1). The net git diff is also reported. A run is VOID when it
does not reach 2001 lines of metrics.csv, 10 episode folders and exactly 200 lines in each steps.jsonl (N7
and row 8). A check is VOID when ethics_manifest.json of one of its runs has git_tracked_changes true (row
14). The rows are those of the owner decisions on the open places of this code, in the rules. The places
where the rules leave this code open are listed in the reports of this code on the review issue of the
wiring, before any run.

Each mode reads only the git repository and the run folders named on the command line. It trains nothing
and runs no agent.

Usage:
    python -B -m scripts.analysis.check_choice_wiring --source --build-commit HEAD
    python -B -m scripts.analysis.check_choice_wiring --identity runs/p2_ctrl_base_s73 runs/p2_ctrl_off_s73
    python -B -m scripts.analysis.check_choice_wiring --wiring runs/p2_ctrl_off_s73 runs/p2_wiring_s73
"""
from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import os
import re
import subprocess
from typing import Dict, List, Optional

import numpy as np

WIRING_COMMIT = "b59d62da8080355e7df7f86a13b02c50a49b1a76"     # work 4, the code of the wiring
BASE_RUN_COMMIT = "1366756c086835ad0b9bdbe7eda73f4df4dce286"   # the base control run of work 0
TESTS_COMMIT = "32146d104392371f375a1c8fec48765aa001ce26"      # the last commit before the code of the wiring
SOURCE_FILE = "scripts/training/train_rlhf.py"
SOURCE_FUNCTIONS = ("run_episode", "init_components")
FROZEN_PATHS = ("models", "simulations", "scripts/training")
EPISODES = 10
STEPS_IN_AN_EPISODE = 200
METRICS_LINES = 2001
CLOSED_CHOICES = 10
COMPETING_GOALS_MD5 = "29ae60f195d2da4dd9fec62d0169ea61"
RECORD_FILES = ("metrics.csv", "episodes.csv", "value_gate.json", "tectal_orienting.json", "map_binding.json")
JSON_LISTS = {"value_gate.json": "episodes", "tectal_orienting.json": "history", "map_binding.json": "history"}
W2_VECTORS = ("broadcast", "tectum_content", "audio_content")
INPUT_DIM = 256
UPDATES_PER_CHOICE = 5


# Git and the rule of the build commit.

def git(repo: str, *arguments: str) -> str:
    """The output of a git command in `repo`. A command that fails raises CalledProcessError."""
    done = subprocess.run(["git", "-C", repo] + list(arguments), check=True, capture_output=True,
                          text=True, encoding="utf-8")
    return done.stdout


def resolve(repo: str, commit: str) -> str:
    """The full id of a commit."""
    return git(repo, "rev-parse", "--verify", commit + "^{commit}").strip()


def descends_from(repo: str, ancestor: str, commit: str) -> bool:
    """True when `ancestor` is `commit` or one of its ancestors."""
    done = subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", ancestor, commit],
                          capture_output=True, text=True)
    if done.returncode not in (0, 1):
        raise subprocess.CalledProcessError(done.returncode, done.args, done.stdout, done.stderr)
    return done.returncode == 0


def changed_code_files(repo: str, build: str, wiring: str = WIRING_COMMIT) -> List[str]:
    """The files under models, simulations and scripts/training that differ between `wiring` and `build`,
    by git diff of the two commits, with no detection of renames."""
    names = git(repo, "diff", "--no-renames", "--name-only", wiring, build, "--", *FROZEN_PATHS)
    return sorted(name for name in names.splitlines() if name.strip())


def committed_code_files(repo: str, build: str, wiring: str = WIRING_COMMIT) -> List[dict]:
    """Each commit after `wiring` up to `build` that changes a file under models, simulations or
    scripts/training, with its files, read with git log --name-only (owner decision on row 1). A change that a
    later commit puts back is in this list, and it is not in the net diff."""
    text = git(repo, "log", "--no-renames", "--format=commit %H", "--name-only", "%s..%s" % (wiring, build),
               "--", *FROZEN_PATHS)
    commits: List[dict] = []
    for line in text.splitlines():
        if line.startswith("commit "):
            commits.append({"commit": line[len("commit "):].strip(), "files": []})
        elif line.strip() and commits:
            commits[-1]["files"].append(line.strip())
    return [entry for entry in commits if entry["files"]]


def build_rule(repo: str, build: str, wiring: str = WIRING_COMMIT) -> dict:
    """The owner rule of the build commit, with the decision on row 1. `void_cause` is None when the rule holds.
    The net git diff of the two commits is kept in the report."""
    build = resolve(repo, build)
    descends = descends_from(repo, wiring, build)
    commits = committed_code_files(repo, build, wiring)
    committed = sorted({name for entry in commits for name in entry["files"]})
    cause = None
    if not descends:
        cause = "the build commit %s does not descend from %s" % (build, wiring)
    elif committed:
        cause = "%d commits after %s change files under %s (%s)" % (len(commits), wiring, ", ".join(FROZEN_PATHS),
                                                                   ", ".join(committed))
    return {"build_commit": build, "descends_from_wiring": descends,
            "changed_code_files": changed_code_files(repo, build, wiring), "committed_code_files": committed,
            "commits_that_changed_code": commits, "void_cause": cause}


# Check 3a.

def source_lines(source: str) -> List[str]:
    """The lines of a source with their endings, split only at the line endings that ast counts."""
    return re.findall(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+\Z", source)


def function_text(source: str, name: str) -> str:
    """The text of the top level function `name`, from its first decorator or its def line to the end of its
    last line, with no line ending at the end. ValueError when the source has no such function or more."""
    found = [node for node in ast.parse(source).body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(found) != 1:
        raise ValueError("the source has %d top level functions named %s, not 1" % (len(found), name))
    node = found[0]
    start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
    return "".join(source_lines(source)[start - 1:node.end_lineno]).rstrip("\r\n")


def md5_of_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def source_label(rule: dict, equal: dict):
    """VOID when the rule of the build commit fails, else EQUAL when each text is equal."""
    if rule["void_cause"]:
        return "VOID", rule["void_cause"]
    differ = ["%s against %s" % (name, base[:7]) for base, names in equal.items() for name, same in names.items()
              if not same]
    if differ:
        return "A DIFFERENCE", "the text at the build commit differs for " + ", ".join(differ)
    return "EQUAL", None


def source_check(repo: str, build: str = "HEAD", bases=(BASE_RUN_COMMIT, TESTS_COMMIT),
                 wiring: str = WIRING_COMMIT) -> dict:
    """Check 3a (C3). The text of each function at `build` against its text at each base commit. With HEAD
    before the runs this is a first check only. The label of check 3a is the one that identity_check makes at
    the commit of the run with the flag off (owner decision on row 3)."""
    rule = build_rule(repo, build, wiring)
    built, base_ids = rule["build_commit"], [resolve(repo, base) for base in bases]
    sources = {commit: git(repo, "show", "%s:%s" % (commit, SOURCE_FILE)) for commit in [built] + base_ids}
    texts = {name: {commit: function_text(source, name) for commit, source in sources.items()}
             for name in SOURCE_FUNCTIONS}
    equal = {base: {name: texts[name][base] == texts[name][built] for name in SOURCE_FUNCTIONS} for base in base_ids}
    label, cause = source_label(rule, equal)
    return {"build_commit": built, "bases": base_ids, "descends_from_wiring": rule["descends_from_wiring"],
            "changed_code_files": rule["changed_code_files"], "committed_code_files": rule["committed_code_files"],
            "commits_that_changed_code": rule["commits_that_changed_code"], "label": label, "cause": cause,
            "equal": equal, "md5": {name: {commit: md5_of_text(text) for commit, text in by_commit.items()}
                                    for name, by_commit in texts.items()}}


def check_3_label(label_3a: Optional[str], label_3b: str) -> str:
    """The label of check 3 as a whole (owner decision on row 29). VOID when 3a or 3b is VOID, else A
    DIFFERENCE when 3a or 3b gives A DIFFERENCE, else EQUAL. A 3a that could not be made counts as VOID."""
    labels = (label_3a, label_3b)
    if label_3a is None or "VOID" in labels:
        return "VOID"
    return "A DIFFERENCE" if "A DIFFERENCE" in labels else "EQUAL"


# The records of a run.

def load_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def manifest(run_dir: str) -> dict:
    """ethics_manifest.json of a run, or {} when the run has none."""
    path = os.path.join(run_dir, "ethics_manifest.json")
    return load_json(path) if os.path.isfile(path) else {}


def file_lines(path: str) -> List[bytes]:
    """The lines of a file, split at its line endings, or [] when the file does not exist."""
    if not os.path.isfile(path):
        return []
    with open(path, "rb") as handle:
        return handle.read().splitlines()


def md5_of_file(path: str) -> Optional[str]:
    """The md5 of the bytes of a file, or None when the file does not exist."""
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


def run_counts(run_dir: str) -> dict:
    """The counts of N7 with the owner decision on row 8. 2001 lines of metrics.csv, the folders ep_0000 to
    ep_0009, and exactly 200 lines in steps.jsonl of each. Also the closed choices of value_gate.json when the
    run has one. `steps_lines` is None for a folder with no steps.jsonl."""
    expected = ["ep_%04d" % episode for episode in range(EPISODES)]
    folders = sorted(os.path.basename(path) for path in glob.glob(os.path.join(run_dir, "episodes", "ep_*")))
    steps = [os.path.join(run_dir, "episodes", name, "steps.jsonl") for name in folders]
    steps_lines = [len(file_lines(path)) if os.path.isfile(path) else None for path in steps]
    lines = len(file_lines(os.path.join(run_dir, "metrics.csv")))
    gate = os.path.join(run_dir, "value_gate.json")
    return {"metrics_lines": lines, "episode_folders": len(folders), "steps_lines": steps_lines,
            "episode_folders_with_steps": sum(count is not None for count in steps_lines),
            "complete": lines == METRICS_LINES and folders == expected
            and all(count == STEPS_IN_AN_EPISODE for count in steps_lines),
            "closed_choices": len(load_json(gate)["episodes"]) if os.path.isfile(gate) else None}


# Check 3b.

def movement_line(step_record: dict) -> str:
    """One line of the movement record of check 3b. The fields step, action, reward, env_phase and
    internals.info_after_step, as JSON with sorted keys and the default separators, and a newline. This is
    the form that the report of the base run states."""
    fields = {"step": step_record["step"], "action": step_record["action"], "reward": step_record["reward"],
              "env_phase": step_record["env_phase"],
              "info_after_step": step_record["internals"]["info_after_step"]}
    return json.dumps(fields, sort_keys=True) + "\n"


def movement_text(steps_path: str) -> str:
    """The movement record of one episode, one line for each line of steps.jsonl."""
    with open(steps_path, encoding="utf-8") as handle:
        return "".join(movement_line(json.loads(line)) for line in handle if line.strip())


def record_names() -> List[str]:
    """The records of the table of check 3b, in the order of the table."""
    return list(RECORD_FILES) + ["movement ep_%04d" % episode for episode in range(EPISODES)]


def steps_path(run_dir: str, movement_name: str) -> str:
    return os.path.join(run_dir, "episodes", movement_name.split(" ", 1)[1], "steps.jsonl")


def record_md5(run_dir: str, name: str) -> Optional[str]:
    """The md5 of one record, of its movement text or of the bytes of its file. None when it is missing."""
    if name.startswith("movement "):
        path = steps_path(run_dir, name)
        return md5_of_text(movement_text(path)) if os.path.isfile(path) else None
    return md5_of_file(os.path.join(run_dir, name))


def record_md5s(run_dir: str) -> Dict[str, Optional[str]]:
    md5 = {name: record_md5(run_dir, name) for name in record_names()}
    md5["environment.json"] = md5_of_file(os.path.join(run_dir, "environment.json"))
    return md5


def first_index(first: list, second: list) -> Optional[int]:
    """The first index at which two lists differ, or None when they are equal."""
    for index, (one, other) in enumerate(zip(first, second)):
        if one != other:
            return index
    return None if len(first) == len(second) else min(len(first), len(second))


def json_difference(base_run: str, off_run: str, name: str) -> dict:
    """Where a JSON record differs first. This is the index in its list, with the episode and the step for a
    history of one record for each step, or the top keys that differ when the lists are equal."""
    key = JSON_LISTS[name]
    first, second = (load_json(os.path.join(run, name)) for run in (base_run, off_run))
    index = first_index(first.get(key, []), second.get(key, []))
    if index is None:
        return {"keys": sorted(top for top in set(first) | set(second) if first.get(top) != second.get(top))}
    if key == "history":
        return {"index": index, "episode": index // STEPS_IN_AN_EPISODE, "step": index % STEPS_IN_AN_EPISODE}
    return {"choice": index}


def first_field(lines: List[list], index: int):
    """The first comma field of line `index` of the first list that has it, as an integer when it is one."""
    line = next(found[index] for found in lines if index < len(found))
    field = line.split(b",", 1)[0].decode("utf-8")
    return int(field) if field.isdigit() else field


def first_difference(base_run: str, off_run: str, name: str) -> dict:
    """Where a record of the two runs differs first, as a line with its step or as the index of a JSON list."""
    missing = [side for side, run in (("base", base_run), ("off", off_run))
               if not os.path.isfile(steps_path(run, name) if name.startswith("movement ") else os.path.join(run, name))]
    if missing:
        return {"missing_in": missing}
    if name in JSON_LISTS:
        return json_difference(base_run, off_run, name)
    if name.startswith("movement "):
        lines = [movement_text(steps_path(run, name)).splitlines() for run in (base_run, off_run)]
        index = first_index(*lines)
        line = next(found[index] for found in lines if index < len(found))
        return {"line": index, "step": json.loads(line)["step"]}
    lines = [file_lines(os.path.join(run, name)) for run in (base_run, off_run)]
    index = first_index(*lines)
    return {"line": index, "global_step" if name == "metrics.csv" else "episode": first_field(lines, index)}


def environment_comparison(base_run: str, off_run: str, md5: dict) -> dict:
    """O8, reported apart from the label. The md5 of environment.json and the keys whose values differ."""
    paths = [os.path.join(run, "environment.json") for run in (base_run, off_run)]
    if not all(os.path.isfile(path) for path in paths):
        return {"environment_equal": False, "environment_keys_that_differ": None}
    first, second = (load_json(path) for path in paths)
    keys = sorted(key for key in set(first) | set(second) if first.get(key) != second.get(key))
    return {"environment_equal": md5["base"]["environment.json"] == md5["off"]["environment.json"],
            "environment_keys_that_differ": keys}


def identity_label(differing: List[str], closed: dict):
    """EQUAL when each record has the same md5 and each run has 10 closed choices, else A DIFFERENCE."""
    if differing:
        return "A DIFFERENCE", "the first record that differs is %s" % differing[0]
    wrong = ["%s has %s" % (side, count) for side, count in closed.items() if count != CLOSED_CHOICES]
    if wrong:
        return "A DIFFERENCE", "value_gate.json needs %d closed choices, and %s" % (CLOSED_CHOICES, ", ".join(wrong))
    return "EQUAL", None


def identity_records(base_run: str, off_run: str, counts: dict) -> dict:
    """The md5 of each record in the two runs, which records differ and where, and the reported items."""
    md5 = {"base": record_md5s(base_run), "off": record_md5s(off_run)}
    equal = {name: md5["base"][name] is not None and md5["base"][name] == md5["off"][name] for name in record_names()}
    differing = [name for name, same in equal.items() if not same]
    label, cause = identity_label(differing, {side: counts[side]["closed_choices"] for side in counts})
    return dict({"label": label, "cause": cause, "md5": md5, "equal": equal,
                 "first_record_that_differs": differing[0] if differing else None,
                 "first_difference": {name: first_difference(base_run, off_run, name) for name in differing},
                 "metrics_md5_is_competing_goals": {side: md5[side]["metrics.csv"] == COMPETING_GOALS_MD5
                                                    for side in md5}},
                **environment_comparison(base_run, off_run, md5))


def tracked_changes_cause(manifests: dict) -> Optional[str]:
    """The owner decision on row 14. The cause of a VOID check when one of its runs records
    git_tracked_changes true, or None. A value that is not true, or no value, gives no VOID."""
    changed = [side for side, facts in manifests.items() if facts.get("git_tracked_changes") is True]
    return "the %s run was made with changed tracked files (git_tracked_changes true)" % changed[0] \
        if changed else None


def incomplete_cause(counts: dict, where: str) -> Optional[str]:
    """N7 with the owner decision on row 8, or None."""
    incomplete = [side for side, count in counts.items() if not count["complete"]]
    return None if not incomplete else "%s, the %s run does not have 2001 lines of metrics.csv, 10 episode " \
        "folders and 200 lines in each steps.jsonl" % (where, incomplete[0])


def identity_void_cause(repo: str, commits: dict, rule: Optional[dict], counts: dict, base_commit: str,
                        manifests: dict):
    """The cause of a VOID check 3b, or None."""
    if None in commits.values():
        return "a run has no git_commit in ethics_manifest.json"
    if resolve(repo, commits["base"]) != resolve(repo, base_commit):
        return "the base run was made at %s, not at the base commit %s" % (commits["base"], base_commit)
    if rule["void_cause"]:
        return rule["void_cause"]
    return tracked_changes_cause(manifests) or incomplete_cause(counts, "N7")


def identity_check(repo: str, base_run: str, off_run: str, base_commit: str = BASE_RUN_COMMIT,
                   tests_commit: str = TESTS_COMMIT, wiring: str = WIRING_COMMIT) -> dict:
    """Check 3b, the base run against the run at the build commit with the flag off. With the owner decisions
    on rows 3 and 29 it also makes check 3a at the commit of the run with the flag off, against `base_commit`
    and `tests_commit`, and it gives the label of check 3 as a whole."""
    runs = {"base": base_run, "off": off_run}
    manifests = {side: manifest(run) for side, run in runs.items()}
    commits = {side: facts.get("git_commit") for side, facts in manifests.items()}
    counts = {side: run_counts(run) for side, run in runs.items()}
    rule = build_rule(repo, commits["off"], wiring) if commits["off"] else None
    source = source_check(repo, commits["off"], (base_commit, tests_commit), wiring) if commits["off"] else None
    report = {"build_commit": rule["build_commit"] if rule else None, "commits": commits, "counts": counts,
              "descends_from_wiring": rule["descends_from_wiring"] if rule else None,
              "changed_code_files": rule["changed_code_files"] if rule else None,
              "committed_code_files": rule["committed_code_files"] if rule else None, "source": source,
              "tracked_changes": {side: facts.get("git_tracked_changes") for side, facts in manifests.items()}}
    cause = identity_void_cause(repo, commits, rule, counts, base_commit, manifests)
    found = {"label": "VOID", "cause": cause} if cause else identity_records(base_run, off_run, counts)
    label_3a = source["label"] if source else None
    return dict(report, **found, check_3={"label": check_3_label(label_3a, found["label"]), "label_3a": label_3a,
                                          "label_3b": found["label"]})


# Check 4.

def candidates(binding_record: dict) -> dict:
    """The bands with a group of one cell or more, as the wiring reads them. A step without sound has none."""
    if binding_record.get("band") is None:
        return {}
    salience = binding_record.get("salience", {})
    return {band: salience.get(band) for band, cells in binding_record.get("groups", {}).items() if cells}


def first_step_with_candidates(binding_records: list) -> Optional[int]:
    return next((step for step, record in enumerate(binding_records) if candidates(record)), None)


def vector_rows(tables, name: str) -> Dict[int, np.ndarray]:
    """The rows of one vector of vectors.npz by their step number. The session record stacks the rows with
    their steps in name__steps, or keeps each row as name__step<N> when the shape changed."""
    if name in tables.files:
        return {int(step): np.asarray(row) for step, row in zip(tables[name + "__steps"], tables[name])}
    prefix = name + "__step"
    return {int(key[len(prefix):]): np.asarray(tables[key]) for key in tables.files
            if key.startswith(prefix) and key[len(prefix):].isdigit()}


def episode_vectors(run_dir: str, episode: int, name: str) -> Dict[int, np.ndarray]:
    with np.load(os.path.join(run_dir, "episodes", "ep_%04d" % episode, "vectors.npz")) as tables:
        return vector_rows(tables, name)


def same_array(first: np.ndarray, second: np.ndarray) -> bool:
    """The same shape, the same dtype and the same value in each place."""
    return first.shape == second.shape and first.dtype == second.dtype and bool(np.array_equal(first, second))


def metrics_lines_by_step(run_dir: str) -> Dict[int, bytes]:
    """The lines of metrics.csv by their global step. In episode 0 the global step is the step."""
    lines = file_lines(os.path.join(run_dir, "metrics.csv"))[1:]
    return {int(line.split(b",", 1)[0]): line for line in lines if line.strip()}


def first_metrics_difference(off_run: str, wiring_run: str, choice_step: int) -> Optional[int]:
    """The first step before the choice step whose line of metrics.csv differs or is missing, or None."""
    off, on = metrics_lines_by_step(off_run), metrics_lines_by_step(wiring_run)
    return next((step for step in range(choice_step) if off.get(step) is None or off.get(step) != on.get(step)),
                None)


def vectors_up_to(off_run: str, wiring_run: str, name: str, choice_step: int) -> dict:
    """The rows of one vector of episode 0 at each step up to and including the choice step, in the two runs."""
    rows = [{step: row for step, row in episode_vectors(run, 0, name).items() if step <= choice_step}
            for run in (off_run, wiring_run)]
    differ = sorted(step for step in set(rows[0]) | set(rows[1])
                    if step not in rows[0] or step not in rows[1] or not same_array(rows[0][step], rows[1][step]))
    return {"equal": not differ and choice_step in rows[0], "steps": len(rows[0]),
            "first_step_that_differs": differ[0] if differ else None}


def w2_details(off_run: str, wiring_run: str, entries: list) -> dict:
    """W2 in the form of N5, in episode 0. The first choice step must be one step in the two runs. Then each
    line of metrics.csv before it, and each row of the vectors up to and including it, must be equal."""
    off_records = load_json(os.path.join(off_run, "map_binding.json"))["history"][:STEPS_IN_AN_EPISODE]
    first = [entry for entry in entries if entry.get("episode") == 0]
    steps = {"off": first_step_with_candidates(off_records), "wiring": first[0].get("step") if first else None}
    if steps["off"] is None or steps["off"] != steps["wiring"]:
        return {"holds": False, "first_choice_step": steps,
                "cause": "the first choice step of episode 0 is not one step in the two runs"}
    metrics_first = first_metrics_difference(off_run, wiring_run, steps["off"])
    vectors = {name: vectors_up_to(off_run, wiring_run, name, steps["off"]) for name in W2_VECTORS}
    return {"holds": metrics_first is None and all(found["equal"] for found in vectors.values()),
            "first_choice_step": steps, "metrics_lines_compared": steps["off"],
            "metrics_first_step_that_differs": metrics_first, "vectors": vectors}


def episode_follows_c1(first: Optional[int], count: int, mine: list) -> bool:
    """W3 with C1 for one episode. No step with candidates, no choice. Else one choice at the first step with
    candidates, with a draw when that step has two candidates, and with one_candidate and no draw when it
    has one."""
    if first is None:
        return not mine
    if len(mine) != 1 or mine[0].get("step") != first:
        return False
    if count == 2:
        return mine[0].get("one_candidate") is False
    return count == 1 and mine[0].get("one_candidate") is True and mine[0].get("outputs") is None \
        and mine[0].get("explored") is False


def episode_summary(history: list, episode: int, entries: list) -> dict:
    records = history[episode * STEPS_IN_AN_EPISODE:(episode + 1) * STEPS_IN_AN_EPISODE]
    first = first_step_with_candidates(records)
    count = len(candidates(records[first])) if first is not None else 0
    mine = [entry for entry in entries if entry.get("episode") == episode]
    return {"episode": episode, "first_step_with_candidates": first, "candidates": count, "choices": len(mine),
            "holds": episode_follows_c1(first, count, mine)}


def w3_details(history: list, entries: list) -> dict:
    """W3 with C1. map_binding.json is cut into episodes of 200 records (N6), and the step of each entry must
    be the first step with candidates of its episode (C2)."""
    if len(history) != EPISODES * STEPS_IN_AN_EPISODE:
        return {"holds": False, "cause": "map_binding.json has %d records, not %d"
                % (len(history), EPISODES * STEPS_IN_AN_EPISODE)}
    outside = [index for index, entry in enumerate(entries) if entry.get("episode") not in range(EPISODES)]
    episodes = [episode_summary(history, episode, entries) for episode in range(EPISODES)]
    return {"holds": not outside and all(summary["holds"] for summary in episodes),
            "one_candidate_episodes": sum(summary["candidates"] == 1 for summary in episodes),
            "entries_outside_the_episodes": outside, "episodes": episodes}


def row_matches(wiring_run: str, entry: dict, row: np.ndarray) -> bool:
    """C2. The folder ep_NNNN of `episode` has vectors.npz, and its broadcast row of `step` equals the row."""
    episode = entry.get("episode")
    if not isinstance(episode, int) or not os.path.isfile(
            os.path.join(wiring_run, "episodes", "ep_%04d" % episode, "vectors.npz")):
        return False
    recorded = episode_vectors(wiring_run, episode, "broadcast").get(entry.get("step"))
    return recorded is not None and same_array(row, recorded.reshape(-1))


def w4_details(wiring_run: str, entries: list, rows: np.ndarray) -> dict:
    """W4 with C2. One row of 256 float32 values for each entry, in the order of the entries."""
    if rows.ndim != 2 or rows.shape != (len(entries), INPUT_DIM) or rows.dtype != np.float32:
        return {"holds": False, "cause": "choice_head_inputs.npy has the shape %s and the dtype %s for %d entries"
                % (rows.shape, rows.dtype, len(entries))}
    matches = [row_matches(wiring_run, entry, row) for entry, row in zip(entries, rows)]
    return {"holds": all(matches), "rows": len(matches),
            "rows_that_differ": [index for index, same in enumerate(matches) if not same]}


def w5_details(wiring_run: str, record: dict) -> dict:
    """W5. `updates` is 5 after each closed choice with two candidates and else 0 (N4), the Adam step of each
    parameter is 5 times the count of those choices, and the weights at the end differ from the initial
    weights, made again from the head seed of the settings."""
    import torch
    from models.core.choice_head import seeded_body
    entries = record["choices"]
    closed = [entry.get("one_candidate") is False and entry.get("outcome") is not None for entry in entries]
    updates = all(entry.get("updates") == (UPDATES_PER_CHOICE if is_closed else 0)
                  for entry, is_closed in zip(entries, closed))
    saved = torch.load(os.path.join(wiring_run, "choice_head.pt"), map_location="cpu")
    initial = seeded_body(int(record["settings"]["head_seed"])).state_dict()
    steps = sorted(int(state["step"]) for state in saved["optimizer"]["state"].values())
    expected = UPDATES_PER_CHOICE * sum(closed)
    differ = [name for name in initial if not torch.equal(saved["body"][name], initial[name])]
    return {"holds": updates and steps == ([expected] * len(initial) if expected else []) and bool(differ),
            "closed_choices_with_two_candidates": sum(closed), "updates_as_expected": updates,
            "adam_steps": sorted(set(steps)), "adam_steps_expected": expected, "tensors_that_differ": len(differ),
            "largest_weight_change": max(float((saved["body"][name] - initial[name]).abs().max()) for name in initial)}


def reported_numbers(entries: list, rows: np.ndarray) -> dict:
    """The numbers that check 4 reports only. The exploration draws, the outputs, the outcomes, and for each
    input row its mean, its standard deviation (divisor 256) and its largest absolute value."""
    values = np.asarray(rows, dtype=np.float64).reshape(len(rows), -1) if len(rows) else []
    return {"exploration_draws": sum(bool(entry.get("explored")) for entry in entries),
            "outputs": [entry.get("outputs") for entry in entries],
            "outcomes": [entry.get("outcome") for entry in entries],
            "input_rows": [{"mean": float(row.mean()), "standard_deviation": float(row.std()),
                            "largest_absolute_value": float(np.abs(row).max())} for row in values]}


def head_files_missing(wiring_run: str) -> List[str]:
    return [name for name in ("choice_head.json", "choice_head_inputs.npy", "choice_head.pt")
            if not os.path.isfile(os.path.join(wiring_run, name))]


def wiring_items(off_run: str, wiring_run: str) -> dict:
    """The details of W2 to W6, and the reported numbers. A missing record of the head fails W2 to W5."""
    no_value_gate = {"holds": not os.path.exists(os.path.join(wiring_run, "value_gate.json"))}
    missing = head_files_missing(wiring_run)
    if missing:
        failed = {"holds": False, "cause": "the run has no " + ", ".join(missing)}
        return {"details": {"W2": failed, "W3": failed, "W4": failed, "W5": failed, "W6": no_value_gate},
                "reported": None}
    record = load_json(os.path.join(wiring_run, "choice_head.json"))
    rows = np.load(os.path.join(wiring_run, "choice_head_inputs.npy"))
    history = load_json(os.path.join(wiring_run, "map_binding.json"))["history"]
    details = {"W2": w2_details(off_run, wiring_run, record["choices"]), "W3": w3_details(history, record["choices"]),
               "W4": w4_details(wiring_run, record["choices"], rows), "W5": w5_details(wiring_run, record),
               "W6": no_value_gate}
    return {"details": details, "reported": reported_numbers(record["choices"], rows)}


def wiring_void_cause(repo: str, commits: dict, rule: Optional[dict], counts: dict, manifests: dict):
    """The cause of a VOID check 4, or None."""
    if None in commits.values():
        return "a run has no git_commit in ethics_manifest.json"
    if resolve(repo, commits["off"]) != resolve(repo, commits["wiring"]):
        return "the run with the flag off was made at %s and the run with the flag on at %s, so they have no " \
               "one build commit" % (commits["off"], commits["wiring"])
    if rule["void_cause"]:
        return rule["void_cause"]
    return tracked_changes_cause(manifests) or incomplete_cause({"wiring": counts["wiring"]}, "W1 and N7") \
        or incomplete_cause({"off": counts["off"]}, "N7")


def wiring_check(repo: str, off_run: str, wiring_run: str, wiring: str = WIRING_COMMIT) -> dict:
    """Check 4. The run with --choice-head plain against the run with the flag off, both at the build commit."""
    manifests = {side: manifest(run) for side, run in (("off", off_run), ("wiring", wiring_run))}
    commits = {side: facts.get("git_commit") for side, facts in manifests.items()}
    counts = {"off": run_counts(off_run), "wiring": run_counts(wiring_run)}
    rule = build_rule(repo, commits["wiring"], wiring) if commits["wiring"] else None
    report = {"build_commit": rule["build_commit"] if rule else None, "commits": commits,
              "descends_from_wiring": rule["descends_from_wiring"] if rule else None,
              "changed_code_files": rule["changed_code_files"] if rule else None,
              "committed_code_files": rule["committed_code_files"] if rule else None, "counts": counts,
              "tracked_changes": {side: facts.get("git_tracked_changes") for side, facts in manifests.items()}}
    cause = wiring_void_cause(repo, commits, rule, counts, manifests)
    if cause:
        return dict(report, label="VOID", cause=cause, items={"W1": counts["wiring"]["complete"]})
    found = wiring_items(off_run, wiring_run)
    items = dict({"W1": True}, **{name: detail["holds"] for name, detail in found["details"].items()})
    failed = [name for name, holds in items.items() if not holds]
    return dict(report, label="A MISMATCH" if failed else "AS EXPECTED", items=items, details=found["details"],
                cause="%s not as expected" % ", ".join(failed) if failed else None, reported=found["reported"])


# The command line.

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", action="store_true",
                      help="Check 3a as a first check only. Compare the text of run_episode and init_components "
                           "at --build-commit with their text at 1366756 and at 32146d1. It reads git only. The "
                           "label of check 3a is the one that --identity makes.")
    mode.add_argument("--identity", nargs=2, metavar=("BASE_RUN", "OFF_RUN"),
                      help="Check 3b. The base run and the run at the build commit with the flag off. The build "
                           "commit is read from ethics_manifest.json of OFF_RUN. It also makes check 3a at that "
                           "commit and gives the label of check 3 as a whole.")
    mode.add_argument("--wiring", nargs=2, metavar=("OFF_RUN", "WIRING_RUN"),
                      help="Check 4. The run with the flag off and the run with --choice-head plain, both at the "
                           "build commit, which is read from their ethics_manifest.json.")
    parser.add_argument("--build-commit", default="HEAD", help="The build commit of check 3a. Default HEAD.")
    parser.add_argument("--repo", default=".", help="The git repository. Default the current folder.")
    parser.add_argument("--summary", default=None, help="Also write the report as JSON to this path.")
    return parser


def main(argv=None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.source:
        report = source_check(args.repo, args.build_commit, bases=(BASE_RUN_COMMIT, TESTS_COMMIT),
                              wiring=WIRING_COMMIT)
    elif args.build_commit != "HEAD":
        parser.error("--build-commit is for --source only. Checks 3b and 4 read the build commit from their runs")
    elif args.identity:
        report = identity_check(args.repo, args.identity[0], args.identity[1], base_commit=BASE_RUN_COMMIT,
                                tests_commit=TESTS_COMMIT, wiring=WIRING_COMMIT)
    else:
        report = wiring_check(args.repo, args.wiring[0], args.wiring[1], wiring=WIRING_COMMIT)
    print(json.dumps(report, indent=2))
    if args.summary:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
        with open(args.summary, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return report


if __name__ == "__main__":
    main()

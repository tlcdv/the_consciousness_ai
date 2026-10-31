"""The training commands in the public documents run as written.

A run that changes weights stops without a declared existence drive (rule E1,
models/ethics/framework.py). The README commands lacked the flag from 2026-09-15 to 2026-10-31, and
the simulation guide used a flag that does not exist.
"""
from __future__ import annotations

import os
import re
from typing import List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCUMENTS = ("README.md", os.path.join("docs", "simulation_guide.md"))
TRAINER = os.path.join("scripts", "training", "train_rlhf.py")
COMMAND = re.compile(r"python (?:-m scripts\.training\.train_rlhf|scripts/training/train_rlhf\.py)\b")


def documented_commands(document: str) -> List[str]:
    """Each command of the document that runs the trainer, with its continuation lines joined."""
    with open(os.path.join(ROOT, document), encoding="utf-8") as handle:
        text = handle.read().replace("\\\n", " ")
    return [line.strip() for line in text.splitlines() if COMMAND.search(line)]


def test_the_documents_hold_commands():
    assert len(documented_commands("README.md")) >= 8
    assert len(documented_commands(os.path.join("docs", "simulation_guide.md"))) >= 1


def test_each_command_declares_the_existence_drive():
    for document in DOCUMENTS:
        for command in documented_commands(document):
            assert re.search(r"--existence-drive (on|off)\b", command), (document, command)


def test_each_flag_of_a_command_exists_in_the_trainer():
    with open(os.path.join(ROOT, TRAINER), encoding="utf-8") as handle:
        trainer = handle.read()
    for document in DOCUMENTS:
        for command in documented_commands(document):
            for flag in re.findall(r"(--[A-Za-z0-9_-]+)", command):
                assert '"%s"' % flag in trainer, (document, command, flag)

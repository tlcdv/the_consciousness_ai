"""Tests for the ethics framework preconditions and the run manifest.

The framework text is `docs/ethics_framework.md`. These tests pin the two
PRECONDITION rules that code enforces before a run's first step:

    E1  every run that changes weights declares the existence drive
    E3  any growth stage past L0 runs with the existence drive removed

and the rule that the document version and the code version are the same.
"""

import json
import re
import tempfile
import unittest
from pathlib import Path

from models.ethics.framework import (
    FRAMEWORK_VERSION,
    EthicsViolation,
    RunDeclaration,
    check_preconditions,
    enforce_preconditions,
    noxious_channels_present,
    resolve_drive_declaration,
    write_ethics_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_DOC = REPO_ROOT / "docs" / "ethics_framework.md"


class TestE1DriveDeclaration(unittest.TestCase):

    def test_missing_declaration_violates_e1(self):
        declaration = RunDeclaration(entry_point="train_rlhf", existence_drive=None)
        self.assertEqual(check_preconditions(declaration), ("E1",))

    def test_unknown_value_violates_e1(self):
        declaration = RunDeclaration(entry_point="train_rlhf", existence_drive="maybe")
        self.assertEqual(check_preconditions(declaration), ("E1",))

    def test_each_valid_value_passes_at_l0(self):
        for drive in ("on", "off", "absent"):
            declaration = RunDeclaration(entry_point="train_rlhf", existence_drive=drive)
            self.assertEqual(check_preconditions(declaration), ())

    def test_enforce_raises_and_names_the_rule(self):
        declaration = RunDeclaration(entry_point="train_rlhf", existence_drive=None)
        with self.assertRaises(EthicsViolation) as caught:
            enforce_preconditions(declaration)
        self.assertIn("E1", str(caught.exception))
        self.assertIn("--existence-drive", str(caught.exception))

    def test_enforce_passes_silently_when_valid(self):
        declaration = RunDeclaration(entry_point="train_rlhf", existence_drive="on")
        self.assertIsNone(enforce_preconditions(declaration))


class TestE3GrowthRequiresDriveRemoved(unittest.TestCase):

    def test_growth_with_drive_on_violates_e3(self):
        declaration = RunDeclaration(
            entry_point="plastic_substrate", existence_drive="on", growth_stage="L1",
        )
        self.assertEqual(check_preconditions(declaration), ("E3",))

    def test_growth_with_drive_off_passes(self):
        declaration = RunDeclaration(
            entry_point="plastic_substrate", existence_drive="off", growth_stage="L1",
        )
        self.assertEqual(check_preconditions(declaration), ())

    def test_growth_on_a_substrate_with_no_drive_passes(self):
        declaration = RunDeclaration(
            entry_point="plastic_substrate", existence_drive="absent", growth_stage="L1",
        )
        self.assertEqual(check_preconditions(declaration), ())

    def test_growth_without_declaration_violates_e1_and_e3(self):
        declaration = RunDeclaration(
            entry_point="plastic_substrate", existence_drive=None, growth_stage="L1",
        )
        self.assertEqual(check_preconditions(declaration), ("E1", "E3"))

    def test_enforce_names_e3(self):
        declaration = RunDeclaration(
            entry_point="plastic_substrate", existence_drive="on", growth_stage="L1",
        )
        with self.assertRaises(EthicsViolation) as caught:
            enforce_preconditions(declaration)
        self.assertIn("E3", str(caught.exception))


class TestManifest(unittest.TestCase):

    def test_manifest_round_trips(self):
        declaration = RunDeclaration(entry_point="train_rlhf", existence_drive="off")
        with tempfile.TemporaryDirectory() as log_dir:
            path = write_ethics_manifest(log_dir, declaration, noxious_channels=("battery",))
            manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(Path(path).name, "ethics_manifest.json")
        self.assertEqual(manifest["framework_version"], FRAMEWORK_VERSION)
        self.assertEqual(manifest["entry_point"], "train_rlhf")
        self.assertEqual(manifest["existence_drive"], "off")
        self.assertEqual(manifest["growth_stage"], "L0")
        self.assertEqual(manifest["rules_checked"], ["E1", "E3"])
        self.assertEqual(manifest["violations"], [])
        self.assertEqual(manifest["noxious_channels_present"], ["battery"])
        self.assertIn("utc_time", manifest)
        self.assertIn("git_commit", manifest)
        self.assertIn(manifest["git_tracked_changes"], (True, False, None))

    def test_manifest_refuses_a_violating_declaration(self):
        declaration = RunDeclaration(entry_point="train_rlhf", existence_drive=None)
        with tempfile.TemporaryDirectory() as log_dir:
            with self.assertRaises(EthicsViolation):
                write_ethics_manifest(log_dir, declaration)
            self.assertFalse((Path(log_dir) / "ethics_manifest.json").exists())

    def test_manifest_creates_missing_log_dir(self):
        declaration = RunDeclaration(entry_point="train_rlhf", existence_drive="on")
        with tempfile.TemporaryDirectory() as root:
            log_dir = Path(root) / "new" / "run"
            path = write_ethics_manifest(str(log_dir), declaration)
            self.assertTrue(Path(path).exists())


class TestDriveResolution(unittest.TestCase):
    """--existence-drive is the declaration; --ablate-existence-bias is an alias for off."""

    def test_no_flag_resolves_to_undeclared(self):
        self.assertIsNone(resolve_drive_declaration(None, ablate_flag=False))

    def test_ablate_flag_alone_resolves_to_off(self):
        self.assertEqual(resolve_drive_declaration(None, ablate_flag=True), "off")

    def test_explicit_values_pass_through(self):
        self.assertEqual(resolve_drive_declaration("on", ablate_flag=False), "on")
        self.assertEqual(resolve_drive_declaration("off", ablate_flag=False), "off")
        self.assertEqual(resolve_drive_declaration("off", ablate_flag=True), "off")

    def test_contradiction_raises(self):
        with self.assertRaises(EthicsViolation) as caught:
            resolve_drive_declaration("on", ablate_flag=True)
        self.assertIn("--ablate-existence-bias", str(caught.exception))


class TestNoxiousChannels(unittest.TestCase):

    def test_battery_environments_list_battery_with_drive_on(self):
        self.assertEqual(noxious_channels_present("dark_room", "on"), ("battery",))
        self.assertEqual(noxious_channels_present("navigation", "on"), ("battery",))

    def test_drive_off_cuts_the_battery(self):
        self.assertEqual(noxious_channels_present("dark_room", "off"), ())

    def test_environments_without_battery_list_nothing(self):
        self.assertEqual(noxious_channels_present("dmts", "on"), ())
        self.assertEqual(noxious_channels_present("wcst", "on"), ())


class TestDocumentAndCodeVersionsMatch(unittest.TestCase):

    def test_framework_doc_version_equals_code_version(self):
        text = FRAMEWORK_DOC.read_text(encoding="utf-8")
        found = re.search(r"\*\*Framework version: ([0-9]+\.[0-9]+)\*\*", text)
        self.assertIsNotNone(found, "version line missing from docs/ethics_framework.md")
        self.assertEqual(found.group(1), FRAMEWORK_VERSION)


if __name__ == "__main__":
    unittest.main()

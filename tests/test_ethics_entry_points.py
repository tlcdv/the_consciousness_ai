"""Every in-scope entry point checks the ethics framework before it trains.

The scope table in docs/ethics_framework.md lists the scripts that change
weights. Scripts built on train_rlhf's init_components are covered by the check
inside init_components (tests/test_existence_drive_cuts.py). The scripts below
build their own models, so each must call the framework itself, and each may
declare the drive "absent" only because it builds no part of the existence
drive. Both facts are pinned here from the source text.
"""

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ABSENT_DRIVE_ENTRY_POINTS = (
    "scripts/training/train_baseline_dqn.py",
    "scripts/analysis/diagnose_phi_zero_v2.py",
    "scripts/analysis/diagnose_phi_zero_v3.py",
    "scripts/analysis/diagnose_phi_zero_v4.py",
)

EXISTENCE_DRIVE_MODULES = (
    "affective_modulator",
    "reward_shaping",
    "self_representation_core",
    "train_rlhf",
)


def _source(relative_path):
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


class TestAbsentDriveEntryPoints(unittest.TestCase):

    def test_each_declares_absent_through_the_framework(self):
        for path in ABSENT_DRIVE_ENTRY_POINTS:
            with self.subTest(path=path):
                text = _source(path)
                self.assertIn("from models.ethics.framework import", text)
                self.assertIn('existence_drive="absent"', text)

    def test_none_imports_a_part_of_the_existence_drive(self):
        for path in ABSENT_DRIVE_ENTRY_POINTS:
            import_lines = [line for line in _source(path).splitlines()
                            if line.lstrip().startswith(("import ", "from "))]
            for module in EXISTENCE_DRIVE_MODULES:
                with self.subTest(path=path, module=module):
                    self.assertFalse([line for line in import_lines if module in line])


class TestShellLaunchers(unittest.TestCase):

    def test_ablation_campaign_declares_the_drive(self):
        text = _source("scripts/training/_run_ablation_campaign.sh")
        self.assertIn("--existence-drive on", text)


if __name__ == "__main__":
    unittest.main()

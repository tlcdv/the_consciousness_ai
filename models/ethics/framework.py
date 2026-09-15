"""Ethics framework preconditions, checked before any run that changes weights.

The canonical text is docs/ethics_framework.md. This module enforces the two
PRECONDITION rules that code can check before a run's first step:

    E1  every run that changes weights declares the existence drive
        ("on", "off", or "absent" for a model with no interoceptive drive)
    E3  any growth stage past L0 runs with the existence drive removed

The other rules (E2, E4, E6, E7, E8) are REVIEW rules. E5 is planned for a governed
plasticity stage, which is not in this repository yet. FRAMEWORK_VERSION must equal the
version line of the document; a test fails if they differ. A revision changes both in one
commit.

Nothing here measures or claims consciousness.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Tuple

FRAMEWORK_VERSION = "1.0"
DRIVE_DECLARATIONS = ("on", "off", "absent")
DRIVE_REMOVED = ("off", "absent")
RULES_CHECKED = ("E1", "E3")
MANIFEST_NAME = "ethics_manifest.json"
# Environments whose step output meets the E6 definition of a noxious channel
# (docs/ethics_framework.md). Both emit a battery that drains every step.
NOXIOUS_CHANNELS_BY_ENV = {"dark_room": ("battery",), "navigation": ("battery",)}

_FIX_BY_RULE = {
    "E1": "declare the existence drive with --existence-drive on or "
          "--existence-drive off (on reproduces every run made before the framework)",
    "E3": "a growth stage past L0 needs the existence drive off",
}


class EthicsViolation(RuntimeError):
    """A run declaration breaks a PRECONDITION rule of the ethics framework."""


@dataclass(frozen=True)
class RunDeclaration:
    """What a run states about itself before its first step."""

    entry_point: str
    existence_drive: Optional[str]
    growth_stage: str = "L0"


def check_preconditions(declaration: RunDeclaration) -> Tuple[str, ...]:
    """Return the IDs of the PRECONDITION rules the declaration violates."""
    violated = []
    if declaration.existence_drive not in DRIVE_DECLARATIONS:
        violated.append("E1")
    grows = declaration.growth_stage != "L0"
    if grows and declaration.existence_drive not in DRIVE_REMOVED:
        violated.append("E3")
    return tuple(violated)


def enforce_preconditions(declaration: RunDeclaration) -> None:
    """Raise EthicsViolation naming every violated rule and its fix."""
    violated = check_preconditions(declaration)
    if not violated:
        return
    lines = ["%s: %s" % (rule, _FIX_BY_RULE[rule]) for rule in violated]
    raise EthicsViolation(
        "Ethics framework %s refuses to start %s. %s. See docs/ethics_framework.md."
        % (FRAMEWORK_VERSION, declaration.entry_point, "; ".join(lines))
    )


def resolve_drive_declaration(existence_drive: Optional[str], ablate_flag: bool) -> Optional[str]:
    """Combine --existence-drive with its alias --ablate-existence-bias (= off).

    Returns None when neither was given, so E1 fails later with its own message.
    """
    if ablate_flag and existence_drive == "on":
        raise EthicsViolation(
            "--existence-drive on contradicts --ablate-existence-bias, which means "
            "--existence-drive off. Give one of them."
        )
    if ablate_flag:
        return "off"
    return existence_drive


def noxious_channels_present(env_name: str, existence_drive: Optional[str]) -> Tuple[str, ...]:
    """Noxious channels (rule E6) that reach the agent in this run.

    The battery reaches the agent only with the drive on; with the drive off the
    training loop does not copy it into the self-model's energy.
    """
    if existence_drive != "on":
        return ()
    return NOXIOUS_CHANNELS_BY_ENV.get(env_name, ())


def write_ethics_manifest(
    log_dir: str,
    declaration: RunDeclaration,
    noxious_channels: Sequence[str] = (),
) -> str:
    """Enforce the preconditions, then write the manifest into the run folder."""
    enforce_preconditions(declaration)
    folder = Path(log_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / MANIFEST_NAME
    manifest = _manifest_fields(declaration, noxious_channels)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return str(path)


def _manifest_fields(declaration: RunDeclaration, noxious_channels: Sequence[str]) -> dict:
    return {
        "framework_version": FRAMEWORK_VERSION,
        "entry_point": declaration.entry_point,
        "existence_drive": declaration.existence_drive,
        "growth_stage": declaration.growth_stage,
        "rules_checked": list(RULES_CHECKED),
        "violations": [],
        "noxious_channels_present": list(noxious_channels),
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "git_tracked_changes": _has_tracked_changes(),
        "utc_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def git_output(arguments) -> Optional[str]:
    """Output of a git command, or None when git is unavailable. Never a made-up value."""
    try:
        completed = subprocess.run(
            # UTF-8 explicitly: on Windows the default code page fails on non-ASCII
            # text in a diff, and the reader thread error surfaces as IndexError.
            ["git"] + list(arguments), capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            cwd=Path(__file__).resolve().parents[2], timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _has_tracked_changes() -> Optional[bool]:
    """True when tracked files differ from the commit, so the hash alone cannot reproduce the run."""
    status = git_output(["status", "--porcelain", "--untracked-files=no"])
    return None if status is None else bool(status)

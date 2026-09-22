"""
Isolation guard for the experimental substrate package (models/thermodynamic/).

WHY THIS FILE EXISTS. The substrate work is experimental and must never change a
production run. The training entry point, the action-selection core, and every file
under models/core/ must not import models.thermodynamic, directly or through any
chain of in-repo imports. A shell grep misses indirect imports; this test walks the
import graph with the ast module and follows every in-repo module it reaches.

It also pins the lazy-loading contract: importing models.thermodynamic must not load
its submodules or any optional simulator library.

If this test fails, remove the import. Do not add an exemption here.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Set

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = "models.thermodynamic"
OPTIONAL_LIBRARIES = ("snntorch", "spikingjelly", "brian2", "torchdiffeq", "thrml", "torx", "jax", "nir", "nirtorch")

sys.path.insert(0, str(REPO_ROOT))


def _module_file(module: str, root: Path) -> Optional[Path]:
    base = root.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative(node: ast.ImportFrom, current: str, is_package: bool) -> str:
    package = current.split(".") if is_package else current.split(".")[:-1]
    base = package[: len(package) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _imported_modules(path: Path, root: Path) -> Iterator[str]:
    current = _module_name(path, root)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = _resolve_relative(node, current, path.name == "__init__.py")
            yield module
            yield from (f"{module}.{alias.name}" for alias in node.names)


def find_forbidden_imports(entry_points: List[Path], root: Path) -> List[str]:
    """Return 'importer -> module' for every reachable import of FORBIDDEN."""
    seen: Set[Path] = set()
    stack = list(entry_points)
    hits: List[str] = []
    while stack:
        path = stack.pop()
        if path in seen:
            continue
        seen.add(path)
        for module in _imported_modules(path, root):
            if module == FORBIDDEN or module.startswith(FORBIDDEN + "."):
                hits.append(f"{path.relative_to(root)} -> {module}")
            target = _module_file(module, root)
            if target is not None:
                stack.append(target)
    return hits


def _production_entry_points() -> List[Path]:
    fixed = [
        REPO_ROOT / "scripts" / "training" / "train_rlhf.py",
        REPO_ROOT / "models" / "self_model" / "action_selection_core.py",
    ]
    return fixed + sorted((REPO_ROOT / "models" / "core").glob("*.py"))


class TestProductionDoesNotImportSubstratePackage:
    def test_entry_points_exist(self):
        missing = [str(p) for p in _production_entry_points() if not p.is_file()]
        assert missing == []

    def test_no_production_path_reaches_models_thermodynamic(self):
        assert find_forbidden_imports(_production_entry_points(), REPO_ROOT) == []


class TestGuardDetectsPlantedImports:
    """The guard itself must work, or the test above proves nothing."""

    def _plant(self, tmp_path: Path, source: str) -> List[str]:
        (tmp_path / "models" / "thermodynamic").mkdir(parents=True)
        (tmp_path / "models" / "__init__.py").write_text("")
        (tmp_path / "models" / "thermodynamic" / "__init__.py").write_text("")
        (tmp_path / "models" / "helper.py").write_text(source)
        entry = tmp_path / "entry.py"
        entry.write_text("from models import helper\n")
        return find_forbidden_imports([entry], tmp_path)

    def test_direct_import_is_found_through_an_intermediate_module(self, tmp_path):
        hits = self._plant(tmp_path, "import models.thermodynamic.p_bit_emulator\n")
        assert hits == ["models/helper.py -> models.thermodynamic.p_bit_emulator".replace("/", os.sep)]

    def test_from_package_import_is_found(self, tmp_path):
        hits = self._plant(tmp_path, "from models import thermodynamic\n")
        assert any(hit.endswith("-> models.thermodynamic") for hit in hits)

    def test_relative_import_is_found(self, tmp_path):
        hits = self._plant(tmp_path, "from .thermodynamic import energy_minimization\n")
        assert any("models.thermodynamic" in hit for hit in hits)


class TestLazyLoading:
    def test_package_import_loads_no_submodule_and_no_optional_library(self):
        probe = (
            "import sys; import models.thermodynamic; "
            "names = [m for m in sys.modules if m.startswith('models.thermodynamic.') "
            f"or m.split('.')[0] in {OPTIONAL_LIBRARIES!r}]; print(sorted(names))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True
        )
        assert completed.stdout.strip() == "[]"

    def test_attribute_access_loads_the_submodule(self):
        import models.thermodynamic as package

        assert package.BlockGibbsSampler.__module__ == "models.thermodynamic.p_bit_emulator"

    def test_unknown_attribute_raises(self):
        import models.thermodynamic as package

        with pytest.raises(AttributeError):
            package.not_a_name

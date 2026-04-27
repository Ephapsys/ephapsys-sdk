"""Pytest setup: load ephapsys.a2a and ephapsys.journal under their canonical
package names so relative imports inside them resolve, without paying the cost
of importing ephapsys.__init__ (which would load the whole heavy agent module).
"""
import importlib.util
import sys
import types
from pathlib import Path


_EPHAPSYS_DIR = Path(__file__).parent / "ephapsys"


def _ensure_package() -> None:
    if "ephapsys" in sys.modules:
        return
    pkg = types.ModuleType("ephapsys")
    pkg.__path__ = [str(_EPHAPSYS_DIR)]  # type: ignore[attr-defined]
    sys.modules["ephapsys"] = pkg


def _load_submodule(name: str, filename: str):
    full = f"ephapsys.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, _EPHAPSYS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_package()
journal_mod = _load_submodule("journal", "journal.py")
a2a = _load_submodule("a2a", "a2a.py")

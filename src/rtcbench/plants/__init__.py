"""The MIT reference plant pack, and the registry that builds one from a task file.

Everything here is a pure-Python, pure-numpy reimplementation from published equations, so
that the scored suite runs on any laptop with no GPU, no Windows, no COM and no proprietary
engine. Higher-fidelity backends (FMI, AcaysiaRT, DWSIM) plug in through the same
:class:`~rtcbench.plant.Plant` protocol but live outside this package — see DESIGN.md §1.

**Adding a plant is adding one file.** The registry discovers modules rather than listing
them, so a new plant never touches a shared file and several can be authored in parallel
without colliding. A plant module declares two names::

    PLANT_ID = "four_tank"

    def build(cfg: Mapping[str, Any]) -> Plant:
        ...

Discovery is lazy and failure-tolerant: a module that raises on import disables only itself
and reports why on use, rather than taking the whole pack down with it. One half-finished
contribution should not stop the suite from running.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable, Mapping

from ..plant import Plant

_REGISTRY: dict[str, Callable[[Mapping[str, Any]], Plant]] | None = None
_BROKEN: dict[str, str] = {}


def _discover() -> dict[str, Callable[[Mapping[str, Any]], Plant]]:
    found: dict[str, Callable[[Mapping[str, Any]], Plant]] = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f".{info.name}", __name__)
        except Exception as exc:  # a broken contribution disables itself, nothing else
            _BROKEN[info.name] = f"{type(exc).__name__}: {exc}"
            continue
        plant_id = getattr(module, "PLANT_ID", None)
        builder = getattr(module, "build", None)
        if not plant_id or not callable(builder):
            continue
        if plant_id in found:
            raise RuntimeError(
                f"two plant modules both claim PLANT_ID {plant_id!r}; ids must be unique"
            )
        found[plant_id] = builder
    return found


def registry() -> dict[str, Callable[[Mapping[str, Any]], Plant]]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _discover()
    return dict(_REGISTRY)


def list_plants() -> list[str]:
    return sorted(registry())


def build_plant(cfg: Mapping[str, Any]) -> Plant:
    """Build a plant from a task file's ``plant:`` block."""
    kwargs = dict(cfg)
    kind = kwargs.pop("kind", None)
    if kind is None:
        raise ValueError("plant config needs a 'kind'")
    known = registry()
    if kind not in known:
        hint = ""
        if kind in _BROKEN:
            hint = f" (its module failed to import: {_BROKEN[kind]})"
        raise ValueError(
            f"unknown plant kind {kind!r}{hint}; registered: {sorted(known)}"
        )
    return known[kind](kwargs)


# Re-exported for convenience and for the tests that predate discovery.
from .four_tank import P_MINUS, P_PLUS, FourTank, FourTankParams  # noqa: E402

__all__ = [
    "FourTank",
    "FourTankParams",
    "P_MINUS",
    "P_PLUS",
    "build_plant",
    "list_plants",
    "registry",
]

"""The MIT reference plant pack, and the registry that builds one from a task file.

Everything here is a pure-Python, pure-numpy reimplementation from published equations, so
that the scored suite runs on any laptop with no GPU, no Windows, no COM and no proprietary
engine. Higher-fidelity backends (FMI, AcaysiaRT, DWSIM) plug in through the same
:class:`~rtcbench.plant.Plant` protocol but live outside this package — see DESIGN.md §1.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..plant import Plant
from .four_tank import P_MINUS, P_PLUS, FourTank, FourTankParams

_PRESETS: dict[str, dict[str, Any]] = {
    "four_tank": {"P_minus": P_MINUS, "P_plus": P_PLUS},
}


def _build_four_tank(cfg: Mapping[str, Any]) -> Plant:
    kwargs = dict(cfg)
    preset = kwargs.pop("preset", "P_minus")
    if preset not in _PRESETS["four_tank"]:
        raise ValueError(
            f"four_tank preset must be one of {sorted(_PRESETS['four_tank'])}, got {preset!r}"
        )
    params = _PRESETS["four_tank"][preset]
    if overrides := kwargs.pop("params", None):
        params = FourTankParams(**{**params.__dict__, **overrides})
    return FourTank(params, **kwargs)


REGISTRY: dict[str, Callable[[Mapping[str, Any]], Plant]] = {
    "four_tank": _build_four_tank,
}


def build_plant(cfg: Mapping[str, Any]) -> Plant:
    """Build a plant from a task file's ``plant:`` block."""
    kwargs = dict(cfg)
    kind = kwargs.pop("kind", None)
    if kind is None:
        raise ValueError("plant config needs a 'kind'")
    if kind not in REGISTRY:
        raise ValueError(f"unknown plant kind {kind!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[kind](kwargs)


__all__ = ["FourTank", "FourTankParams", "P_MINUS", "P_PLUS", "REGISTRY", "build_plant"]

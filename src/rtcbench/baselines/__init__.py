"""The two anchors, plus the loading rules for a submitted controller.

`hold` and `pid` are not decoration — they *are* the units the leaderboard is denominated
in (see :mod:`rtcbench.score`). That makes the tuning of :class:`MultiLoopPID` a public,
reviewable artifact rather than an implementation detail: if the reference is weak, every
score in the project is inflated, and anyone should be able to open the task file, see the
gains, and argue with them.
"""

from __future__ import annotations

from .decoupler import DecoupledPID
from .hold import Hold
from .pid import MultiLoopPID, PIDLoop

__all__ = ["DecoupledPID", "Hold", "MultiLoopPID", "PIDLoop",
           "build_reference", "BUILTINS"]

BUILTINS = {"hold": Hold, "pid": MultiLoopPID, "decoupled_pid": DecoupledPID}


def build_reference(brief, config):
    """Build the task's reference anchor from its ``reference:`` block."""
    cfg = dict(config or {})
    kind = cfg.pop("kind", "pid")
    if kind not in BUILTINS:
        raise ValueError(f"unknown reference controller {kind!r}; have {sorted(BUILTINS)}")
    if kind == "hold":
        return Hold(brief)
    if kind == "decoupled_pid":
        return DecoupledPID(brief, **cfg)
    return MultiLoopPID(brief, **cfg)

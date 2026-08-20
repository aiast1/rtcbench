"""The reference anchor: decentralized PI(D), tuned properly.

This is the controller that defines 1.0 on the leaderboard, so it is written the way an
industrial loop actually is, not the way a textbook first introduces one:

* **Derivative on measurement**, so a setpoint step does not produce a derivative spike.
* **Conditional integration** for anti-windup — the integrator stops accumulating in the
  direction that would push a saturated output further into its limit. Windup is the single
  most common reason a hand-written PID looks bad against a plant with real valve limits,
  and a reference that suffered from it would inflate every score in the project.
* **Bumpless start** from the commissioned duty point, so t=0 is not a step disturbance.
* **Quality-aware**: on a bad reading the loop holds its output rather than integrating
  against a stale number.

Pairing is explicit and per-task. It has to be: on the quadruple-tank's non-minimum-phase
setting the diagonal pairing is the *wrong* one, and a reference that always paired
diagonally would quietly turn the hardest task in the pack into a task where beating the
anchor is trivial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..controller import TaskBrief


@dataclass
class PIDLoop:
    """One SISO loop: measurement ``y_index`` -> actuator ``u_index``."""

    y_index: int
    u_index: int
    kp: float
    ti: float = float("inf")
    """Integral time in seconds. ``inf`` disables integral action."""

    td: float = 0.0
    """Derivative time in seconds."""

    beta: float = 1.0
    """Setpoint weight on the proportional term. Below 1.0 softens setpoint kicks without
    touching disturbance rejection."""


class MultiLoopPID:
    """A bank of independent PID loops — the standard decentralized industrial control."""

    def __init__(
        self,
        brief: TaskBrief,
        *,
        loops: Sequence[dict] | None = None,
        pairing: Sequence[int] | None = None,
        kp: Sequence[float] | float = 1.0,
        ti: Sequence[float] | float = float("inf"),
        td: Sequence[float] | float = 0.0,
        beta: Sequence[float] | float = 1.0,
    ) -> None:
        self._brief = brief
        self._ts = brief.sample_time
        self._lo = np.array([c.lo for c in brief.actuators], dtype=float)
        self._hi = np.array([c.hi for c in brief.actuators], dtype=float)
        self._bias = np.asarray(brief.initial_u, dtype=float)
        if self._bias.size != brief.n_u:
            self._bias = self._lo.copy()

        if loops is not None:
            self._loops = [PIDLoop(**spec) for spec in loops]
        else:
            controlled = list(brief.controlled)
            pair = list(pairing) if pairing is not None else list(range(len(controlled)))
            if len(pair) != len(controlled):
                raise ValueError(
                    f"pairing has {len(pair)} entries for {len(controlled)} controlled channels"
                )
            g = _spread(kp, len(controlled), "kp")
            i = _spread(ti, len(controlled), "ti")
            d = _spread(td, len(controlled), "td")
            b = _spread(beta, len(controlled), "beta")
            self._loops = [
                PIDLoop(y_index=y, u_index=u, kp=g[n], ti=i[n], td=d[n], beta=b[n])
                for n, (y, u) in enumerate(zip(controlled, pair))
            ]

        self._integral = np.zeros(len(self._loops))
        self._y_prev = np.full(len(self._loops), np.nan)
        self._u = self._bias.copy()

    def reset(self) -> None:
        self._integral[:] = 0.0
        self._y_prev[:] = np.nan
        self._u = self._bias.copy()

    def step(
        self,
        t: float,
        y: NDArray[np.float64],
        r: NDArray[np.float64],
        quality: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        u = self._u.copy()

        for n, loop in enumerate(self._loops):
            yi = float(y[loop.y_index])
            ri = float(r[loop.y_index])
            if not np.isfinite(ri):
                continue
            if not bool(quality[loop.y_index]):
                # Bad reading: hold this loop's output. Integrating against a held-over
                # value is how a dropout turns into a slow ramp away from setpoint.
                continue

            error = ri - yi
            proportional = loop.kp * (loop.beta * ri - yi)

            derivative = 0.0
            if loop.td > 0.0 and np.isfinite(self._y_prev[n]):
                derivative = -loop.kp * loop.td * (yi - self._y_prev[n]) / self._ts

            candidate = self._bias[loop.u_index] + proportional + self._integral[n] + derivative
            lo, hi = self._lo[loop.u_index], self._hi[loop.u_index]

            if np.isfinite(loop.ti) and loop.ti > 0.0:
                increment = loop.kp * (self._ts / loop.ti) * error
                saturated_high = candidate >= hi and increment > 0.0
                saturated_low = candidate <= lo and increment < 0.0
                if not (saturated_high or saturated_low):
                    self._integral[n] += increment
                    candidate += increment

            u[loop.u_index] = float(np.clip(candidate, lo, hi))
            self._y_prev[n] = yi

        self._u = u
        return u.copy()


def _spread(value: Sequence[float] | float, n: int, name: str) -> list[float]:
    if np.isscalar(value):
        return [float(value)] * n  # type: ignore[arg-type]
    seq = [float(v) for v in value]  # type: ignore[union-attr]
    if len(seq) != n:
        raise ValueError(f"{name}: expected 1 or {n} values, got {len(seq)}")
    return seq

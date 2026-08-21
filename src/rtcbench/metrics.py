"""Closed-loop metrics, all normalized by engineering span.

Normalizing by span is what lets a level loop in cm and a temperature loop in K land in the
same aggregate without one of them dominating purely because of its units. Every quantity
here is a time average, so it is also insensitive to scenario length — a controller cannot
improve its number by asking for a shorter run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Cost:
    """The decomposed cost of one scenario. Kept decomposed on purpose: an aggregate that
    cannot be broken back into tracking, effort and violations hides *why* a controller
    lost, and a benchmark that cannot say why is not diagnostic."""

    iae: float
    """Time-averaged fractional tracking error, dimensionless."""

    tv: float
    """Time-averaged fractional actuator movement, per second."""

    violations: int
    """Number of control periods with at least one hard-constraint violation."""

    overruns: int
    """Number of periods where the controller blew its compute budget."""

    total: float
    """``w_e * iae + w_u * tv``. Lower is better. Not the score — see :mod:`rtcbench.score`."""

    duty_exceeded: bool = False
    """True if actuator travel exceeded the task's published duty limit.

    Kept distinct from ``violations`` on purpose. Both gate the scenario, but a safety-
    envelope breach and an actuator worn out by chattering are different failures, and a
    scoring record that conflates them cannot tell you which one happened.

    Defaulted, and last, so that records written before duty limits existed still load."""


def integral_absolute_error(
    y: NDArray[np.float64],
    r: NDArray[np.float64],
    controlled: Sequence[int],
    spans: NDArray[np.float64],
    dt: float,
) -> float:
    """Span-normalized IAE, averaged over controlled channels and over time.

    ``y`` and ``r`` are ``[n_steps, n_y]``. Setpoints for unscored channels are ``nan`` and
    are ignored rather than propagating.
    """
    if not len(controlled):
        return 0.0
    idx = np.asarray(controlled, dtype=int)
    err = np.abs(r[:, idx] - y[:, idx]) / spans[idx]
    horizon = max(y.shape[0] * dt, dt)
    return float(np.nansum(err) * dt / (len(idx) * horizon))


def total_variation(u: NDArray[np.float64], spans: NDArray[np.float64], dt: float) -> float:
    """Span-normalized total variation of the control signal, averaged over actuators.

    This is the term that separates a controller that settles from one that holds setpoint
    by chattering the valve — which in a real plant is the difference between an actuator
    that lasts ten years and one that lasts ten months.
    """
    if u.shape[0] < 2:
        return 0.0
    moves = np.abs(np.diff(u, axis=0)) / spans
    horizon = max(u.shape[0] * dt, dt)
    return float(np.sum(moves) / (u.shape[1] * horizon))


def scenario_cost(
    y: NDArray[np.float64],
    r: NDArray[np.float64],
    u: NDArray[np.float64],
    controlled: Sequence[int],
    y_spans: NDArray[np.float64],
    u_spans: NDArray[np.float64],
    dt: float,
    *,
    w_error: float = 1.0,
    w_effort: float = 0.5,
    violations: int = 0,
    overruns: int = 0,
    max_total_variation: float | None = None,
) -> Cost:
    iae = integral_absolute_error(y, r, controlled, y_spans, dt)
    tv = total_variation(u, u_spans, dt)
    return Cost(
        iae=iae,
        tv=tv,
        violations=violations,
        overruns=overruns,
        duty_exceeded=max_total_variation is not None and tv > max_total_variation,
        total=w_error * iae + w_effort * tv,
    )


def cvar(scores: Sequence[float], alpha: float = 0.10) -> float:
    """Mean of the worst ``alpha`` fraction of scores.

    The headline aggregate. Ranking on the mean rewards a controller that is excellent on
    most parameter draws and unsafe on a few, which is precisely the controller nobody
    wants commissioned. At least one scenario always counts, so this is well-defined for
    small ensembles.
    """
    arr = np.sort(np.asarray(list(scores), dtype=float))
    if arr.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(alpha * arr.size)))
    return float(arr[:k].mean())

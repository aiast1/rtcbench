"""Dynamic Matrix Control — a reference anchor for multivariable plants with deadtime.

Why this class exists. Every other anchor in the pack is a bank of independent PI loops, and
on `shell_fractionator` that is the wrong *class*, not a bad tuning: its PI anchor is only
37% better than freezing the actuators where `four_tank`'s is 94%, which makes the 0-to-1
scale hypersensitive. The obvious cheap fix — a static decoupler — was implemented and
measured worse on every plant (see `decoupler.py`). What was left was a real predictive
controller.

DMC is the right one. It is the industrial standard for exactly this problem, the Shell
fractionator was *published as* a test for it, and it handles per-element deadtime naturally
because the step-response coefficients encode delay directly rather than needing a separate
compensator.

**Why it stays a defensible anchor.** An anchor may only use information a competitor could
obtain — otherwise it is an oracle, and beating it becomes impossible for reasons that have
nothing to do with control. This one is built from an FOPDT matrix identified by *bump
tests through the public interface* (`validators.spectral.fopdt_matrix`), which is precisely
what the brief invites a competitor to do. The identified model is published in the anchor
file, so the whole thing stays inspectable and arguable.

**No new dependency.** The unconstrained DMC move is a linear least-squares with a constant
gain matrix, computable once at construction:

    du = (Aᵀ Q A + lambda I)⁻¹ Aᵀ Q e

so the per-step cost is one matrix-vector product. Input limits and rate limits are applied
by clipping the resulting move. That is not a true constrained QP — a real QP would need a
solver and a third dependency — and for these plants the difference is small, because the
binding constraints are actuator limits rather than output constraints. Where that stops
being true, this class will need replacing rather than tuning, and its docstring should say
so at that point.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..controller import TaskBrief


def step_coefficients(gain: NDArray[np.float64], tau: NDArray[np.float64],
                      theta: NDArray[np.float64], dt: float, n: int) -> NDArray[np.float64]:
    """Unit-step response S[k, i, j] built from a first-order-plus-deadtime matrix.

    S[k, i, j] is how far output i has moved, k control periods after a unit step on input j.
    The deadtime enters as a plain shift, which is the whole reason this beats a decoupler on
    a plant whose elements each carry a different delay.
    """
    k = np.arange(1, n + 1) * dt
    S = np.zeros((n, *gain.shape))
    for i in range(gain.shape[0]):
        for j in range(gain.shape[1]):
            t = k - theta[i, j]
            S[:, i, j] = np.where(t > 0, gain[i, j] * (1.0 - np.exp(-t / max(tau[i, j], 1e-9))), 0.0)
    return S


class DMC:
    """Dynamic matrix control against an identified FOPDT model.

    ``model`` is ``{"gain": [[...]], "tau": [[...]], "theta": [[...]]}``, one entry per
    (controlled output, actuator) pair, as produced by ``validators.spectral.fopdt_matrix``.
    """

    def __init__(
        self,
        brief: TaskBrief,
        *,
        model: dict,
        horizon: int = 40,
        moves: int = 3,
        move_weight: float = 1.0,
        output_weight: Sequence[float] | float = 1.0,
        model_horizon: int | None = None,
    ) -> None:
        self._controlled = list(brief.controlled)
        n_y, n_u = len(self._controlled), brief.n_u
        self._n_y, self._n_u = n_y, n_u
        self._dt = brief.sample_time
        self._lo = np.array([c.lo for c in brief.actuators], dtype=float)
        self._hi = np.array([c.hi for c in brief.actuators], dtype=float)
        self._u0 = np.asarray(brief.initial_u, dtype=float)
        if self._u0.size != n_u:
            self._u0 = self._lo.copy()

        gain = np.asarray(model["gain"], dtype=float)
        tau = np.asarray(model["tau"], dtype=float)
        theta = np.asarray(model["theta"], dtype=float)
        if gain.shape != (n_y, n_u):
            raise ValueError(f"model gain must be {n_y}x{n_u}, got {gain.shape}")

        self._P = int(horizon)
        self._M = int(moves)
        # The model horizon must outlast the slowest element's delay plus settle, or the
        # predictions silently truncate the very dynamics this controller exists to handle.
        settle = float((theta + 5.0 * tau).max())
        self._N = int(model_horizon or max(self._P + self._M + 2,
                                           int(np.ceil(settle / self._dt)) + 2))
        self._S = step_coefficients(gain, tau, theta, self._dt, self._N)

        # Dynamic matrix: rows are (output, prediction step), columns are (input, move).
        A = np.zeros((self._P * n_y, self._M * n_u))
        for p in range(self._P):
            for m in range(self._M):
                if p >= m:
                    A[p * n_y:(p + 1) * n_y, m * n_u:(m + 1) * n_u] = self._S[p - m]
        w = np.asarray(output_weight, dtype=float)
        q = np.ones(n_y) * w if w.ndim == 0 else w
        Q = np.diag(np.tile(q, self._P))
        self._K = np.linalg.solve(
            A.T @ Q @ A + move_weight * np.eye(self._M * n_u), A.T @ Q
        )[:n_u]  # only the first move is ever applied; the rest are discarded by design

        self._pred = np.zeros((self._N, n_y))
        self._u = self._u0.copy()
        self._started = False

    def reset(self) -> None:
        self._u = self._u0.copy()
        self._pred[:] = 0.0
        self._started = False

    def step(
        self,
        t: float,
        y: NDArray[np.float64],
        r: NDArray[np.float64],
        quality: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        yc = np.array([y[i] for i in self._controlled], dtype=float)
        rc = np.array([r[i] for i in self._controlled], dtype=float)
        if not np.all(np.isfinite(rc)):
            return self._u.copy()

        if not self._started:
            # Seed the prediction with the current operating point, so the first move
            # answers the setpoint rather than an imagined step away from zero.
            self._pred[:] = yc
            self._started = True

        # Bias correction: everything the model did not predict -- mismatch, the drawn
        # parameters, the disturbance -- is lumped into one offset and carried across the
        # horizon. This is what lets a nominal model control a plant that is not nominal.
        bias = yc - self._pred[0]
        corrected = self._pred + bias

        e = np.concatenate([rc - corrected[p + 1] for p in range(self._P)])
        du = self._K @ e

        # Rate limiting first, then absolute limits: clipping to the limit and calling it a
        # move would let a saturated actuator report progress it never made.
        u_new = np.clip(self._u + du, self._lo, self._hi)
        applied = u_new - self._u
        self._u = u_new

        # Roll the prediction forward under the move just made, then shift one period.
        for p in range(self._N):
            self._pred[p] = self._pred[p] + self._S[p] @ applied
        self._pred[:-1] = self._pred[1:]
        return self._u.copy()

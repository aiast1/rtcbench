"""A static decoupler + PI bank — **measured, and it does not beat a tuned PI bank.**

Read the result before reaching for this. It was written to fix a specific problem and does
not fix it; it is kept because a refuted approach with its evidence recorded is worth more
than a deleted one.

**The hypothesis.** Every other anchor in the pack is a bank of loops that cannot see each
other. On `shell_fractionator` that anchor is only 37% better than freezing the actuators
where `four_tank`'s is 94%, and a narrow anchor gap makes the 0-to-1 scale hypersensitive. It
looked like the wrong controller *class*, not a bad tuning — so: identify the plant's
steady-state gain matrix, invert it, control in decoupled coordinates, and let each PI see
the approximately-independent loop it already assumes.

**The measurement.** Worse on every plant tried, including with equal tuning effort:

    plant             cond(G)   tuned PI    tuned decoupler
    shell               51      0.0928       0.1146   (-23%)
    column_a           273      0.0786       0.0890   (-13%)
    four_tank_nmp        4.3    0.0380       0.0425   (-12%)

Column A is the strongest case for decoupling that exists in this pack — condition number
273, textbook ill-conditioning — and per-loop coordinate descent on both controllers still
puts the decoupler 13% behind.

**Why, most likely.** The decoupler inverts a *nominal* gain matrix, and every task in this
suite draws its plant parameters per scenario. Inverting a matrix you do not have exactly is
fragile: the inverse amplifies precisely the directions where the plant responds weakly, so a
small error in `G` becomes a large error in the actuator move. A detuned decentralized PI
never forms that inverse and never pays that price. Static decoupling also does nothing about
the dynamics that actually make these plants hard — Shell's difficulty is per-element
deadtime, and inverting a DC gain mixes signals that arrive at different times.

This is the second independent result pointing the same way. The blind tier found that
withholding the nominal model changes scores by a median of -0.028; this finds that *using*
the nominal model to invert the plant makes things worse. Under real parameter mismatch,
model information is not buying what one would expect.

**Where it might still earn its place:** a plant with strong interaction, little deadtime,
and small mismatch. Nothing in the pack currently fits. A dynamic decoupler would perform
better and would not be a defensible anchor anyway — it needs the full model the mismatch and
blind tiers deliberately withhold, and an anchor may not use information a competitor cannot
have.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..controller import TaskBrief
from .pid import MultiLoopPID


class DecoupledPID:
    """Static decoupler + PI bank. Same tuning surface as :class:`MultiLoopPID`.

    ``gain_matrix`` is the plant's steady-state gain from actuators to controlled outputs,
    row per controlled channel, column per actuator. It is measured once by the task author
    (``validators.spectral.dc_gain`` does it through the public interface) and published in
    the anchor file alongside the gains, so the whole anchor stays inspectable.
    """

    def __init__(
        self,
        brief: TaskBrief,
        *,
        gain_matrix: Sequence[Sequence[float]],
        kp: Sequence[float] | float = 1.0,
        ti: Sequence[float] | float = float("inf"),
        td: Sequence[float] | float = 0.0,
        beta: Sequence[float] | float = 1.0,
        condition_cap: float = 50.0,
    ) -> None:
        self._brief = brief
        self._controlled = list(brief.controlled)
        self._lo = np.array([c.lo for c in brief.actuators], dtype=float)
        self._hi = np.array([c.hi for c in brief.actuators], dtype=float)
        self._bias = np.asarray(brief.initial_u, dtype=float)
        if self._bias.size != brief.n_u:
            self._bias = self._lo.copy()

        G = np.asarray(gain_matrix, dtype=float)
        if G.shape != (len(self._controlled), brief.n_u):
            raise ValueError(
                f"gain_matrix must be {len(self._controlled)}x{brief.n_u}, got {G.shape}"
            )
        self._G = G

        # A raw inverse of an ill-conditioned G is a machine for amplifying measurement
        # noise into full-scale actuator swings -- on Column A, cond(G) = 273. Truncating
        # the small singular values keeps the useful directions and drops the ones where the
        # plant barely responds and the inverse would demand enormous moves.
        u_, s_, vt_ = np.linalg.svd(G, full_matrices=False)
        keep = s_ >= (s_.max() / condition_cap) if s_.max() > 0 else s_ > 0
        s_inv = np.where(keep, 1.0 / np.where(s_ > 0, s_, 1.0), 0.0)
        self._Ginv = vt_.T @ np.diag(s_inv) @ u_.T
        self._directions_dropped = int((~keep).sum())

        # The PI bank now runs in decoupled coordinates: one virtual loop per controlled
        # channel, each with unit gain, pairing straight through.
        n = len(self._controlled)
        virtual = TaskBrief(
            task_id=brief.task_id,
            tier=brief.tier,
            measurements=brief.measurements,
            actuators=tuple(brief.actuators[: n]) if brief.n_u >= n else brief.actuators,
            controlled=tuple(self._controlled),
            sample_time=brief.sample_time,
            horizon=brief.horizon,
            initial_u=tuple([0.0] * n),
            constraints=(),
            description=brief.description,
        )
        self._inner = MultiLoopPID(
            virtual, pairing=list(range(n)), kp=kp, ti=ti, td=td, beta=beta
        )
        # The virtual loops are unbounded; saturation is handled on the real actuators after
        # the decoupler, where the limits actually live.
        self._inner._lo = np.full(n, -np.inf)
        self._inner._hi = np.full(n, np.inf)
        self._inner._bias = np.zeros(n)
        self._inner.reset()

    def reset(self) -> None:
        self._inner.reset()

    def step(
        self,
        t: float,
        y: NDArray[np.float64],
        r: NDArray[np.float64],
        quality: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        v = np.asarray(self._inner.step(t, y, r, quality), dtype=float)
        return np.clip(self._bias + self._Ginv @ v, self._lo, self._hi)

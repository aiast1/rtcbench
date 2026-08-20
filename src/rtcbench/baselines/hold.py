"""The zero anchor: put the loop in manual and walk away.

Every score in RTCbench is measured from here. It is a real control strategy — for a stable
self-regulating process it is what happens when the DCS block is left in MAN — and it is
often not terrible, which is exactly why it makes an honest zero.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..controller import TaskBrief


class Hold:
    """Freeze the actuators at the commissioned duty point for the whole scenario."""

    def __init__(self, brief: TaskBrief) -> None:
        self._u = np.asarray(brief.initial_u, dtype=float)
        if self._u.size != brief.n_u:
            self._u = np.array([c.lo for c in brief.actuators], dtype=float)

    def reset(self) -> None:
        return None

    def step(
        self,
        t: float,
        y: NDArray[np.float64],
        r: NDArray[np.float64],
        quality: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        return self._u.copy()

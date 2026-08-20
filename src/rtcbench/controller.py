"""The controller boundary — what a submission implements, and what it is told.

A submission is a sealed artifact: a module exposing a ``Controller`` class. The harness
constructs it once with a :class:`TaskBrief`, then calls :meth:`Controller.step` every
control period. Nothing else crosses the boundary.

:class:`TaskBrief` is the *operating manual*, not the model. It is what a control engineer
gets on day one of a commissioning job: a tag list, units, engineering ranges, actuator
limits, the scan rate, the safety envelope, and prose about what the unit does. On the
blind tier that is the whole of it — deriving dynamics from it is the task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .plant import Channel, Constraint


@dataclass(frozen=True)
class TaskBrief:
    """Everything a controller is allowed to know before it sees a single measurement."""

    task_id: str
    tier: str
    """``nominal`` | ``mismatch`` | ``blind`` — see DESIGN.md section 4."""

    measurements: tuple[Channel, ...]
    actuators: tuple[Channel, ...]
    controlled: tuple[int, ...]
    """Indices into :attr:`measurements` that carry a setpoint and are scored for tracking."""

    sample_time: float
    horizon: float
    """Scenario duration in seconds, so a controller can size its own schedules."""

    initial_u: tuple[float, ...] = ()
    """Where the actuators sit at t=0. A real loop is put in service bumplessly from the
    operator's current output; a controller that ignores this and starts from zero will
    kick the plant on its first scan and pay for it in the effort term."""

    constraints: tuple[Constraint, ...] = ()
    """The published safety envelope. Stated in true-signal terms; on the blind tier the
    controller is told the limits exist and what they mean without being told the model
    that produces them."""

    description: str = ""
    """Prose operating notes: what the unit is, what the actuators physically do, known
    interactions, anything a commissioning engineer would be handed."""

    model_hint: Mapping[str, Any] = field(default_factory=dict)
    """Tier-dependent. Empty on ``blind``. On ``mismatch`` it carries the *nominal*
    parameters — deliberately not the ones the plant is running. On ``nominal`` it carries
    the true ones."""

    @property
    def n_y(self) -> int:
        return len(self.measurements)

    @property
    def n_u(self) -> int:
        return len(self.actuators)

    def tags(self) -> dict[str, int]:
        """Tag -> measurement index, so a controller can address channels by name rather
        than by a positional convention it has to guess."""
        return {c.tag: i for i, c in enumerate(self.measurements)}


@runtime_checkable
class Controller(Protocol):
    """What a submission implements.

    Contract notes that decide runs:

    * :meth:`step` must return within the task's step budget. Overrunning is not a crash —
      the harness holds the previous output and logs a scan overrun, the way a DCS does.
      Enough overruns fail the run.
    * :meth:`step` may be called with ``quality`` all-``False`` for a channel. The value is
      still populated with the last good reading.
    * The same instance is reused across a scenario, and :meth:`reset` is called between
      scenarios. Carrying integral state across a reset is a bug the harness will not
      catch for you, but the scores will.
    """

    def __init__(self, brief: TaskBrief) -> None:
        ...

    def reset(self) -> None:
        """Clear internal state. Called once before each scenario."""
        ...

    def step(
        self,
        t: float,
        y: NDArray[np.float64],
        r: NDArray[np.float64],
        quality: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Return the control vector for this period.

        ``r`` holds setpoints for every measurement; only the entries named by
        :attr:`TaskBrief.controlled` are scored, and the rest are ``nan``.
        """
        ...

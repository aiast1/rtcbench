"""The plant boundary — the only thing RTCbench core knows about a simulation engine.

Everything downstream (harness, metrics, scoring) is written against these types alone, so
an AcaysiaRT model, an FMU, a DWSIM flowsheet and a physical rig on OPC UA are all the
same kind of object here.

The load-bearing decision is what :class:`Observation` does *not* carry. A controller sees
measured values and their quality flags; it never sees the state vector. Hand out state and
the benchmark quietly stops being about control.

Ground truth still has to go somewhere — you cannot judge a safety violation from a sensor
the controller might have driven into saturation — so it goes to :meth:`Plant.audit`, which
the harness calls and the controller cannot reach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Channel:
    """One measured or manipulated variable, described the way a P&ID would describe it."""

    tag: str
    """Instrument tag, e.g. ``LT-101``. This is the name the controller sees."""

    name: str
    """Human description, e.g. ``tank 1 level``."""

    unit: str
    """Engineering unit, e.g. ``cm``, ``degC``, ``%``."""

    lo: float
    """Bottom of the engineering range (a sensor's 4 mA, a valve's shut position)."""

    hi: float
    """Top of the engineering range."""

    @property
    def span(self) -> float:
        """Engineering span. Used to normalize error and effort so that a level loop in cm
        and a temperature loop in K contribute comparably to a score."""
        return self.hi - self.lo

    def clip(self, value: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(value, self.lo, self.hi)


@dataclass(frozen=True)
class Constraint:
    """A hard safety limit, evaluated by the harness against *true* state, never sensors.

    Violating one zeroes the scenario. This is a gate rather than a penalty term on
    purpose: real plants do not trade an overflow against a bit of integral error.
    """

    name: str
    signal: str
    """Name of a signal in the plant's true state vector (see :attr:`PlantSpec.state_names`)."""

    lo: float = -np.inf
    hi: float = np.inf

    def violated(self, value: float) -> bool:
        return bool(value < self.lo or value > self.hi)


@dataclass(frozen=True)
class PlantSpec:
    """Static description of a plant — the part a controller is allowed to know for free."""

    plant_id: str
    measurements: tuple[Channel, ...]
    actuators: tuple[Channel, ...]
    sample_time: float
    """Control period T_s in seconds. The harness calls the controller exactly this often."""

    state_names: tuple[str, ...] = ()
    """Names of the true state vector's elements. Harness/records only — never given to a
    controller, and only used to resolve :class:`Constraint` signals."""

    constraints: tuple[Constraint, ...] = ()

    initial_u: tuple[float, ...] = ()
    """The commissioned duty point — where the actuators sit when the scenario starts.

    Real loops are put in service from wherever the operator left the output, not from zero,
    and a rate-limited actuator starting at zero against a plant already at steady state
    would produce a startup transient that belongs to the harness rather than to the
    process. Empty falls back to the bottom of each actuator's range.
    """

    description: str = ""

    @property
    def n_y(self) -> int:
        return len(self.measurements)

    @property
    def n_u(self) -> int:
        return len(self.actuators)

    def actuator_lo(self) -> NDArray[np.float64]:
        return np.array([c.lo for c in self.actuators], dtype=float)

    def actuator_hi(self) -> NDArray[np.float64]:
        return np.array([c.hi for c in self.actuators], dtype=float)

    def initial_actuation(self) -> NDArray[np.float64]:
        if not self.initial_u:
            return self.actuator_lo()
        return np.clip(np.asarray(self.initial_u, dtype=float), self.actuator_lo(), self.actuator_hi())


@dataclass(frozen=True)
class Observation:
    """What the controller receives each control period. Deliberately thin."""

    t: float
    """Seconds since the scenario started."""

    y: NDArray[np.float64]
    """Measured values in engineering units, one per :attr:`PlantSpec.measurements`."""

    quality: NDArray[np.bool_]
    """Per-channel validity. ``False`` means the reading is stale or bad — a dropped
    sample, a failed transmitter. The value is still populated (with the last good
    reading, as a DCS would) so a naive controller keeps running, but a good one notices."""

    def __post_init__(self) -> None:
        if self.y.shape != self.quality.shape:
            raise ValueError(f"y {self.y.shape} and quality {self.quality.shape} disagree")


@dataclass(frozen=True)
class Audit:
    """Ground truth. Harness-only — never crosses the boundary to a controller."""

    t: float
    x: NDArray[np.float64]
    """True state vector, aligned with :attr:`PlantSpec.state_names`."""

    violations: tuple[str, ...] = ()
    """Names of hard constraints currently violated."""

    notes: Mapping[str, float] = field(default_factory=dict)
    """Free-form engine diagnostics for the run record (solver residuals, etc.)."""


@runtime_checkable
class Plant(Protocol):
    """A dynamic process that can be marched forward one control period at a time.

    Implementations must be *deterministic given the seed*: same seed, same control
    sequence, same trajectory, bit for bit on the same build. Everything about
    reproducibility here rests on that.
    """

    spec: PlantSpec

    def reset(self, seed: int) -> Observation:
        """Start a scenario. Draws whatever randomness the plant owns (parameter mismatch,
        initial condition) from ``seed`` and returns the first measurement."""
        ...

    def step(self, u: NDArray[np.float64]) -> Observation:
        """Advance exactly one control period under held control ``u``."""
        ...

    def audit(self) -> Audit:
        """Ground truth at the current time, for the harness and the run record."""
        ...

    def set_params(self, overrides: Mapping[str, float]) -> None:
        """Apply a parameter change mid-run. This is how disturbances are injected: a
        fouling coefficient drifts, an outlet area opens up into a leak. Raise
        :class:`NotImplementedError` if the backend cannot do it."""
        ...

    def close(self) -> None:
        ...


def as_array(u: Sequence[float] | NDArray[np.float64], n: int, what: str) -> NDArray[np.float64]:
    """Coerce a controller's output into a well-formed control vector, or say why not.

    Controllers here are written by language models, so this runs on every step and the
    message matters more than the check: a scalar returned for a 2x2 plant should read as
    a shape error, not as a broadcasting surprise three modules later.
    """
    arr = np.asarray(u, dtype=float).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"{what}: expected {n} values, got shape {np.shape(u)}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what}: non-finite value {arr!r}")
    return arr

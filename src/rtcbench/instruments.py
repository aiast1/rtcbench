"""The instrument layer — noise, deadtime, saturation, stiction, dropout.

Kept strictly separate from the physics. A plant module in :mod:`rtcbench.plants` is ideal
physics with perfect sensors and perfect actuators; :class:`InstrumentedPlant` wraps one and
adds everything that makes a real loop hard. Two payoffs: plant models stay short enough to
audit against the paper they came from, and every stressor is written once and applies to
every plant in the pack.

Randomness here is drawn from a stream derived separately from the plant's own, so changing
a plant's parameter draw does not reshuffle its measurement noise. Scenarios stay comparable
across the mismatch ensemble.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .plant import Audit, Observation, Plant, PlantSpec, as_array


def _per_channel(value: float | Sequence[float] | None, n: int, name: str) -> NDArray[np.float64]:
    """Broadcast a scalar or per-channel sequence to length ``n``."""
    if value is None:
        return np.zeros(n, dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full(n, float(arr[0]))
    if arr.size != n:
        raise ValueError(f"{name}: expected 1 or {n} values, got {arr.size}")
    return arr


@dataclass(frozen=True)
class SensorConfig:
    """Measurement-path imperfections, in engineering units of each channel."""

    noise_std: float | Sequence[float] | None = None
    bias: float | Sequence[float] | None = None
    deadtime_samples: int | Sequence[int] = 0
    """Transport delay in whole control periods. The classic destroyer of aggressive gains."""

    quantization: float | Sequence[float] | None = None
    """Resolution of the reading, e.g. a 12-bit transmitter over its span."""

    dropout_prob: float | Sequence[float] | None = None
    """Per-sample probability the reading fails. The last good value is held and the
    channel's quality flag goes ``False``."""


@dataclass(frozen=True)
class ActuatorConfig:
    """Manipulated-path imperfections. Saturation is always on — it is physics, not a stressor."""

    rate_limit: float | Sequence[float] | None = None
    """Maximum change per second, in engineering units. ``None`` means unlimited."""

    deadband: float | Sequence[float] | None = None
    """Stiction: the valve does not move until the request differs from its current
    position by more than this. Quietly responsible for a large share of real-world
    oscillating loops."""


@dataclass(frozen=True)
class InstrumentConfig:
    sensors: SensorConfig = field(default_factory=SensorConfig)
    actuators: ActuatorConfig = field(default_factory=ActuatorConfig)


class InstrumentedPlant:
    """Wraps an ideal :class:`~rtcbench.plant.Plant` in a realistic instrument layer.

    Satisfies the same protocol, so the harness cannot tell the difference — and neither
    can the controller, which is the point.
    """

    def __init__(self, inner: Plant, config: InstrumentConfig | None = None) -> None:
        self._inner = inner
        self._cfg = config or InstrumentConfig()
        self.spec: PlantSpec = inner.spec

        n_y, n_u = self.spec.n_y, self.spec.n_u
        s, a = self._cfg.sensors, self._cfg.actuators

        self._noise = _per_channel(s.noise_std, n_y, "noise_std")
        self._bias = _per_channel(s.bias, n_y, "bias")
        self._quant = _per_channel(s.quantization, n_y, "quantization")
        self._dropout = _per_channel(s.dropout_prob, n_y, "dropout_prob")

        dt_raw = np.asarray(s.deadtime_samples, dtype=int).reshape(-1)
        if dt_raw.size == 1:
            dt_raw = np.full(n_y, int(dt_raw[0]))
        if dt_raw.size != n_y:
            raise ValueError(f"deadtime_samples: expected 1 or {n_y} values, got {dt_raw.size}")
        if np.any(dt_raw < 0):
            raise ValueError("deadtime_samples must be non-negative")
        self._deadtime = dt_raw

        rl = a.rate_limit
        self._rate_limit = None if rl is None else _per_channel(rl, n_u, "rate_limit")
        self._deadband = _per_channel(a.deadband, n_u, "deadband")

        self._u_lo = self.spec.actuator_lo()
        self._u_hi = self.spec.actuator_hi()

        self._rng: np.random.Generator = np.random.default_rng(0)
        self._delay: list[deque[float]] = []
        self._last_good = np.zeros(n_y, dtype=float)
        self._u_actual = np.zeros(n_u, dtype=float)
        self._u_commanded = np.zeros(n_u, dtype=float)

    # -- protocol ---------------------------------------------------------------

    def reset(self, seed: int) -> Observation:
        obs = self._inner.reset(seed)
        # A stream of its own: re-drawing plant parameters must not reshuffle sensor noise.
        self._rng = np.random.default_rng((seed, 0xC0FFEE))
        self._u_actual = self.spec.initial_actuation()
        self._u_commanded = self._u_actual.copy()
        self._last_good = obs.y.copy()
        # Prime each delay line with the initial reading, so deadtime shows up as a lag
        # rather than as a spurious startup transient from zeros.
        self._delay = [
            deque([float(obs.y[i])] * (int(self._deadtime[i]) + 1), maxlen=int(self._deadtime[i]) + 1)
            for i in range(self.spec.n_y)
        ]
        return self._measure(obs)

    def step(self, u: NDArray[np.float64]) -> Observation:
        u_req = as_array(u, self.spec.n_u, "controller output")
        self._u_commanded = u_req.copy()
        self._u_actual = self._actuate(u_req)
        obs = self._inner.step(self._u_actual)
        return self._measure(obs)

    def audit(self) -> Audit:
        return self._inner.audit()

    def set_params(self, overrides: Mapping[str, float]) -> None:
        self._inner.set_params(overrides)

    def close(self) -> None:
        self._inner.close()

    # -- the actual instrument behaviour ----------------------------------------

    @property
    def u_actual(self) -> NDArray[np.float64]:
        """Where the valves really went. Recorded so a run's trend plot can show commanded
        and actual side by side — which is how stiction is diagnosed in the field."""
        return self._u_actual.copy()

    def _actuate(self, u_req: NDArray[np.float64]) -> NDArray[np.float64]:
        target = np.clip(u_req, self._u_lo, self._u_hi)

        if np.any(self._deadband > 0):
            moved = np.abs(target - self._u_actual) > self._deadband
            target = np.where(moved, target, self._u_actual)

        if self._rate_limit is not None:
            max_step = self._rate_limit * self.spec.sample_time
            delta = np.clip(target - self._u_actual, -max_step, max_step)
            target = self._u_actual + delta

        return np.clip(target, self._u_lo, self._u_hi)

    def _measure(self, obs: Observation) -> Observation:
        y = obs.y.astype(float, copy=True)

        y = y + self._bias
        if np.any(self._noise > 0):
            y = y + self._rng.normal(0.0, 1.0, size=y.shape) * self._noise

        for i in range(self.spec.n_y):
            self._delay[i].append(float(y[i]))
        y = np.array([self._delay[i][0] for i in range(self.spec.n_y)], dtype=float)

        quant = self._quant > 0
        if np.any(quant):
            y = np.where(quant, np.round(y / np.where(quant, self._quant, 1.0)) * self._quant, y)

        quality = np.ones(self.spec.n_y, dtype=bool)
        if np.any(self._dropout > 0):
            failed = self._rng.random(self.spec.n_y) < self._dropout
            if np.any(failed):
                # Hold last good value, as a DCS would, and flag it. A controller that
                # ignores quality still runs; one that reads it can stop integrating.
                y = np.where(failed, self._last_good, y)
                quality = ~failed

        self._last_good = np.where(quality, y, self._last_good)
        return Observation(t=obs.t, y=y, quality=quality)

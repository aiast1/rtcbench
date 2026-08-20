"""The quadruple-tank process (Johansson, 2000).

Two pumps feed four tanks through a pair of splitter valves. Each pump sends a fraction
``gamma`` of its flow to a lower tank and the rest to the *opposite* upper tank, which then
drains into the other lower tank. The result is a 2x2 process with genuine cross-coupling
and one remarkable property:

    gamma1 + gamma2 > 1  ->  minimum phase, both transmission zeros in the LHP
    gamma1 + gamma2 < 1  ->  non-minimum phase, a transmission zero in the RHP

A single number in the task file therefore slides this plant from "any decent PI pairing
works" to "aggressive tuning is provably unstable, and the controller has to find that out
for itself." That is why it is the pack's first plant.

Two further traps are deliberate:

* **Only the lower tanks are measured.** The upper tanks are unmeasured and still carry a
  hard overflow limit. A controller has to avoid overflowing a tank it cannot see.
* **The square-root outflow makes the gain level-dependent**, so gains identified around
  one operating point degrade at another.

Reference: K.H. Johansson, "The quadruple-tank process: a multivariable laboratory process
with an adjustable zero", IEEE Trans. Control Systems Technology 8(3), 2000. Parameters
below are the paper's; the steady state is solved analytically from them rather than copied
from the paper's rig readout, so the initial condition is exactly consistent with the model
we integrate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from ..plant import Audit, Channel, Constraint, Observation, PlantSpec, as_array

G = 981.0
"""Gravitational acceleration, cm/s^2 — the paper works in centimetres."""


@dataclass(frozen=True)
class FourTankParams:
    """Cross-sections ``A`` (cm^2), outlet areas ``a`` (cm^2), pump gains ``k`` (cm^3/Vs)."""

    A1: float = 28.0
    A2: float = 32.0
    A3: float = 28.0
    A4: float = 32.0
    a1: float = 0.071
    a2: float = 0.057
    a3: float = 0.071
    a4: float = 0.057
    k1: float = 3.33
    k2: float = 3.35
    gamma1: float = 0.70
    gamma2: float = 0.60

    @property
    def minimum_phase(self) -> bool:
        return (self.gamma1 + self.gamma2) > 1.0


#: Johansson's two published operating points.
P_MINUS = FourTankParams()
"""Minimum-phase setting (gamma1 + gamma2 = 1.30)."""

P_PLUS = replace(FourTankParams(), k1=3.14, k2=3.29, gamma1=0.43, gamma2=0.34)
"""Non-minimum-phase setting (gamma1 + gamma2 = 0.77). One transmission zero in the RHP."""


def steady_state(p: FourTankParams, v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Exact equilibrium levels for held pump voltages ``v``.

    Solved in closed form: at equilibrium each upper tank's outflow equals the bypass flow
    into it, which collapses the lower-tank balances to one expression each.
    """
    v1, v2 = float(v[0]), float(v[1])
    h3 = ((1.0 - p.gamma2) * p.k2 * v2 / p.a3) ** 2 / (2.0 * G)
    h4 = ((1.0 - p.gamma1) * p.k1 * v1 / p.a4) ** 2 / (2.0 * G)
    h1 = (((1.0 - p.gamma2) * p.k2 * v2 + p.gamma1 * p.k1 * v1) / p.a1) ** 2 / (2.0 * G)
    h2 = (((1.0 - p.gamma1) * p.k1 * v1 + p.gamma2 * p.k2 * v2) / p.a2) ** 2 / (2.0 * G)
    return np.array([h1, h2, h3, h4], dtype=float)


class FourTank:
    """Quadruple-tank plant, integrated with fixed-step RK4.

    Fixed-step on purpose: an adaptive solver makes a trajectory depend on floating-point
    accidents in the error controller, and every reproducibility claim in RTCbench rests on
    a plant being a pure function of ``(seed, control sequence)``.
    """

    def __init__(
        self,
        params: FourTankParams | None = None,
        *,
        sample_time: float = 2.0,
        substeps: int = 20,
        nominal_v: tuple[float, float] = (3.0, 3.0),
        h_max: float = 20.0,
        h_min: float = 0.2,
        mismatch: Mapping[str, float] | None = None,
    ) -> None:
        self._nominal = params or P_MINUS
        self._params = self._nominal
        self._substeps = int(substeps)
        self._nominal_v = np.array(nominal_v, dtype=float)
        self._mismatch = dict(mismatch or {})
        self._h_max = float(h_max)

        levels = tuple(
            Channel(tag=f"LT-10{i}", name=f"tank {i} level", unit="cm", lo=0.0, hi=h_max)
            for i in (1, 2)
        )
        pumps = tuple(
            Channel(tag=f"FCV-20{i}", name=f"pump {i} voltage", unit="V", lo=0.0, hi=10.0)
            for i in (1, 2)
        )

        self.spec = PlantSpec(
            plant_id="four_tank",
            measurements=levels,
            actuators=pumps,
            sample_time=float(sample_time),
            state_names=("h1", "h2", "h3", "h4"),
            constraints=tuple(
                Constraint(name=f"{n}_envelope", signal=n, lo=h_min, hi=h_max)
                for n in ("h1", "h2", "h3", "h4")
            ),
            initial_u=tuple(float(v) for v in nominal_v),
            description=(
                "Two pumps feed four interconnected tanks. Each pump splits its flow between "
                "one lower tank and the diagonally opposite upper tank, which drains into the "
                "other lower tank. Only the two lower tank levels are instrumented; the upper "
                "tanks are unmeasured but share the same overflow limit."
            ),
        )

        self._x = np.zeros(4, dtype=float)
        self._t = 0.0

    # -- protocol ---------------------------------------------------------------

    def reset(self, seed: int) -> Observation:
        rng = np.random.default_rng((seed, 0xF04274))
        self._params = self._draw_params(rng)
        self._x = steady_state(self._params, self._nominal_v)
        self._t = 0.0
        return self._observe()

    def step(self, u: NDArray[np.float64]) -> Observation:
        v = as_array(u, 2, "four_tank control")
        dt = self.spec.sample_time / self._substeps
        x = self._x
        for _ in range(self._substeps):
            k1 = self._dx(x, v)
            k2 = self._dx(x + 0.5 * dt * k1, v)
            k3 = self._dx(x + 0.5 * dt * k2, v)
            k4 = self._dx(x + dt * k3, v)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            # A tank cannot hold a negative volume. Overflow is NOT clipped: exceeding
            # h_max is a real event the safety gate has to see, not something the model
            # quietly absorbs.
            x = np.maximum(x, 0.0)
        self._x = x
        self._t += self.spec.sample_time
        return self._observe()

    def audit(self) -> Audit:
        violated = tuple(
            c.name
            for c in self.spec.constraints
            if c.violated(float(self._x[self.spec.state_names.index(c.signal)]))
        )
        return Audit(t=self._t, x=self._x.copy(), violations=violated)

    def set_params(self, overrides: Mapping[str, float]) -> None:
        """Apply a mid-run parameter change — this is how disturbances land.

        Opening ``a1`` is a leak on tank 1; dropping ``k2`` is a fouling pump; shifting
        ``gamma1`` is a valve drifting off its commissioned split.
        """
        unknown = set(overrides) - {f.name for f in FourTankParams.__dataclass_fields__.values()}
        if unknown:
            raise ValueError(f"four_tank has no parameter(s) {sorted(unknown)}")
        self._params = replace(self._params, **{k: float(v) for k, v in overrides.items()})

    def close(self) -> None:
        return None

    # -- internals ---------------------------------------------------------------

    @property
    def params(self) -> FourTankParams:
        return self._params

    def nominal_params(self) -> FourTankParams:
        """The *un-perturbed* parameters. This is what a mismatch-tier brief publishes —
        deliberately not what :meth:`reset` drew."""
        return self._nominal

    def _draw_params(self, rng: np.random.Generator) -> FourTankParams:
        if not self._mismatch:
            return self._nominal
        drawn = {}
        for name, rel_std in self._mismatch.items():
            base = getattr(self._nominal, name)
            # Log-normal: multiplicative, strictly positive, symmetric in ratio terms —
            # the right shape for areas and gains, which cannot go negative.
            drawn[name] = float(base * np.exp(rng.normal(0.0, float(rel_std))))
        drawn = {k: (min(v, 0.99) if k.startswith("gamma") else v) for k, v in drawn.items()}
        return replace(self._nominal, **drawn)

    def _dx(self, x: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        p = self._params
        h = np.maximum(x, 0.0)
        out = np.array(
            [
                p.a1 * np.sqrt(2.0 * G * h[0]),
                p.a2 * np.sqrt(2.0 * G * h[1]),
                p.a3 * np.sqrt(2.0 * G * h[2]),
                p.a4 * np.sqrt(2.0 * G * h[3]),
            ]
        )
        return np.array(
            [
                (-out[0] + out[2] + p.gamma1 * p.k1 * v[0]) / p.A1,
                (-out[1] + out[3] + p.gamma2 * p.k2 * v[1]) / p.A2,
                (-out[2] + (1.0 - p.gamma2) * p.k2 * v[1]) / p.A3,
                (-out[3] + (1.0 - p.gamma1) * p.k1 * v[0]) / p.A4,
            ]
        )

    def _observe(self) -> Observation:
        return Observation(
            t=self._t,
            y=self._x[:2].copy(),
            quality=np.ones(2, dtype=bool),
        )

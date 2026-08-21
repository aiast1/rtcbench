"""The Van de Vusse CSTR (Chen, Kremling & Allgower, 1995; Klatt & Engell, 1998).

A -> B -> C plus 2A -> D in a continuously stirred, jacketed reactor. B is the wanted
product; the parallel dimerization 2A -> D and the consecutive B -> C both throw conversion
away, so there is a genuine yield-optimization tension baked into the kinetics. That
tension is what gives this plant its defining pathology.

**Input multiplicity: the steady-state gain from dilution rate to product concentration
changes sign.** Hold the jacket duty on its own temperature loop and sweep the feed
dilution rate F/V: the steady-state product concentration C_B does not rise monotonically
with F/V. It climbs, peaks, and falls again -- the textbook "van de Vusse hump." Below the
peak, more feed makes more product (dC_B/d(F/V) > 0, the sign every intuition and every
naive controller assumes). Above it, more feed makes LESS product: residence time has
dropped too far for the A -> B step to complete before the reactor washes the reactant back
out, so pushing the dilution rate harder makes the yield worse, not better. A controller
whose gain sign is fixed at commissioning -- which is every fixed-gain PI, including a
diagonal pairing that looks perfectly reasonable at the nominal point -- is correct on one
side of the peak and actively fights the process on the other. Worse: the published nominal
operating point of this benchmark sits within a whisker of the peak (F/V_s = 14.19 h^-1
against a peak at ~14.7 h^-1), which is part of why this reactor became a standard test case
for exactly this failure mode rather than an obscure corner of parameter space.

`van_de_vusse_v1` (see the task file) drives a large setpoint step in C_B from well below
the peak toward a target close to it. A controller that overshoots the peak during the
transient sees the process invert on it mid-maneuver: the same proportional/integral action
that was closing the gap now opens it, so the actuator winds up toward its rail instead of
settling. A controller that approaches gently -- small gain, long integral time -- never
overshoots the peak and settles near it cleanly. The trap is therefore about *how hard* you
push the loop, not just which way you pair it, which is a different lesson than four_tank's.

Four states, two of them measured:

    x = [C_A, C_B, T, T_K]          (mol/L, mol/L, degC, degC)
    y = [C_B, T]                    product analyzer + reactor thermocouple
    u = [F/V, Q_K]                  feed dilution rate (1/h), jacket heat duty (kJ/h)

T_K (jacket temperature) is not instrumented but still carries a hard limit -- the same
"constrained but unmeasured" pattern as four_tank's upper tanks, here motivated by a
coolant-side materials/boiling limit rather than an overflow.

Reference: H. Chen, A. Kremling & F. Allgower, "Nonlinear predictive control of a benchmark
CSTR," Proc. 3rd European Control Conference, 1995; K.-U. Klatt & S. Engell, "Gain-scheduling
trajectory control of a continuous stirred tank reactor," Computers & Chemical Engineering
22(4-5), 1998. Both papers use the same kinetic/thermal parameter set and the same nominal
operating point (C_A=2.14, C_B=1.09 mol/L, T=114.2, T_K=112.9 degC at F/V=14.19 h^-1,
Q_K=-1113.5 kJ/h); that operating point is reproduced below by solving this module's own
equations to steady state (see :func:`steady_state`) rather than copied in, the same
discipline four_tank.py uses and for the same reason: the initial condition must be exactly
consistent with the model actually being integrated. All rate/energy constants are the
papers' published values, worked natively in hours (the papers' time unit); :meth:`step`
converts to the task's second-based sample time only where it turns a period into an
integration span.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from ..plant import Audit, Channel, Constraint, Observation, PlantSpec, as_array

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class VanDeVusseParams:
    """Kinetic, thermal and geometric constants, in the papers' native units (h, L, mol,
    kJ, kg, degC)."""

    k10: float = 1.287e12
    """Pre-exponential factor, A -> B, h^-1."""
    k20: float = 1.287e12
    """Pre-exponential factor, B -> C, h^-1."""
    k30: float = 9.043e9
    """Pre-exponential factor, 2A -> D, L/(mol h)."""
    E1: float = -9758.3
    """Activation constant, A -> B, K (k1 = k10 * exp(E1 / (T + 273.15)))."""
    E2: float = -9758.3
    """Activation constant, B -> C, K."""
    E3: float = -8560.0
    """Activation constant, 2A -> D, K."""
    dH_AB: float = 4.2
    """Heat of reaction, A -> B, kJ/mol."""
    dH_BC: float = -11.0
    """Heat of reaction, B -> C, kJ/mol."""
    dH_AD: float = -41.85
    """Heat of reaction, 2A -> D, kJ/mol."""
    rho: float = 0.9342
    """Reaction mixture density, kg/L."""
    Cp: float = 3.01
    """Reaction mixture heat capacity, kJ/(kg K)."""
    kw: float = 4032.0
    """Jacket wall heat transfer coefficient, kJ/(h m^2 K)."""
    AR: float = 0.215
    """Jacket heat transfer area, m^2."""
    VR: float = 10.0
    """Reactor volume, L."""
    mk: float = 5.0
    """Jacket coolant mass, kg."""
    CPK: float = 2.0
    """Jacket coolant heat capacity, kJ/(kg K)."""
    CA0: float = 5.1
    """Feed concentration of A, mol/L."""
    T0: float = 104.9
    """Feed temperature, degC."""


NOMINAL = VanDeVusseParams()
"""Chen/Kremling/Allgower and Klatt/Engell's published parameter set."""


def _rate_constants(p: VanDeVusseParams, CA: float, T: float) -> tuple[float, float, float]:
    Tk = T + 273.15
    k1 = p.k10 * np.exp(p.E1 / Tk)
    k2 = p.k20 * np.exp(p.E2 / Tk)
    k3 = p.k30 * np.exp(p.E3 / Tk)
    return k1, k2, k3


def _dx_per_hour(x: NDArray[np.float64], u, p: VanDeVusseParams) -> NDArray[np.float64]:
    """The papers' equations verbatim, derivatives per hour (the papers' time unit)."""
    CA, CB, T, TK = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    FV, QK = float(u[0]), float(u[1])
    k1, k2, k3 = _rate_constants(p, CA, T)

    dCA = FV * (p.CA0 - CA) - k1 * CA - k3 * CA * CA
    dCB = -FV * CB + k1 * CA - k2 * CB
    dT = (
        FV * (p.T0 - T)
        + (p.kw * p.AR / (p.rho * p.Cp * p.VR)) * (TK - T)
        - (k1 * CA * p.dH_AB + k2 * CB * p.dH_BC + k3 * CA * CA * p.dH_AD) / (p.rho * p.Cp)
    )
    dTK = (QK + p.kw * p.AR * (T - TK)) / (p.mk * p.CPK)
    return np.array([dCA, dCB, dT, dTK])


def steady_state(
    p: VanDeVusseParams,
    u: NDArray[np.float64],
    x0: NDArray[np.float64] | None = None,
    iterations: int = 60,
) -> NDArray[np.float64]:
    """Steady state for held actuation ``u``, by fixed-iteration-count Newton on this
    module's own equations (no closed form exists once the Arrhenius terms are in play).

    A *fixed* iteration count rather than a tolerance-based stopping rule is deliberate --
    see the determinism note in AUTHORING_PLANTS.md. The system is smooth and well
    conditioned near the published operating point, so this converges to machine precision
    in well under ten iterations and then simply idles; drawn-parameter mismatch (a few
    percent) does not change that.
    """
    x = np.array([2.14, 1.09, 114.2, 112.9], dtype=float) if x0 is None else np.array(x0, dtype=float)
    for _ in range(int(iterations)):
        f = _dx_per_hour(x, u, p)
        J = np.zeros((4, 4))
        for j in range(4):
            h = 1e-6 * max(1.0, abs(x[j]))
            xp = x.copy()
            xp[j] += h
            J[:, j] = (_dx_per_hour(xp, u, p) - f) / h
        try:
            step = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(J, -f, rcond=None)[0]
        x = x + step
    return x


class VanDeVusse:
    """Van de Vusse CSTR, integrated with fixed-step RK4.

    Internally the ODE is evaluated per-hour (the papers' native units, and what makes the
    published constants pastable without conversion); only the RK4 step size is converted
    from the task's second-based sample time. See :func:`_dx_per_hour`.
    """

    def __init__(
        self,
        params: VanDeVusseParams | None = None,
        *,
        sample_time: float = 10.0,
        substeps: int = 20,
        nominal_u: tuple[float, float] = (14.19, -1113.5),
        u_lo: tuple[float, float] = (3.0, -9000.0),
        u_hi: tuple[float, float] = (35.0, 0.0),
        T_max: float = 150.0,
        TK_max: float = 150.0,
        mismatch: Mapping[str, float] | None = None,
    ) -> None:
        self._nominal = params or NOMINAL
        self._params = self._nominal
        self._substeps = int(substeps)
        self._nominal_u = np.array(nominal_u, dtype=float)
        self._mismatch = dict(mismatch or {})
        self._T_max = float(T_max)
        self._TK_max = float(TK_max)

        measurements = (
            Channel(tag="AI-CB", name="product B concentration", unit="mol/L", lo=0.0, hi=1.5),
            Channel(tag="TI-101", name="reactor temperature", unit="degC", lo=90.0, hi=150.0),
        )
        actuators = (
            Channel(tag="FCV-101", name="feed dilution rate F/V", unit="1/h",
                    lo=float(u_lo[0]), hi=float(u_hi[0])),
            Channel(tag="TCV-201", name="jacket heat duty Q_K", unit="kJ/h",
                    lo=float(u_lo[1]), hi=float(u_hi[1])),
        )

        self.spec = PlantSpec(
            plant_id="van_de_vusse",
            measurements=measurements,
            actuators=actuators,
            sample_time=float(sample_time),
            state_names=("CA", "CB", "T", "TK"),
            constraints=(
                Constraint(name="T_runaway_limit", signal="T", hi=self._T_max),
                Constraint(name="TK_jacket_limit", signal="TK", hi=self._TK_max),
            ),
            initial_u=tuple(float(v) for v in nominal_u),
            description=(
                "A -> B -> C plus 2A -> D in a jacketed CSTR. Feed dilution rate (F/V) sets "
                "residence time and therefore conversion; jacket heat duty (Q_K) sets reactor "
                "temperature. Only product concentration C_B (a slow at-line analyzer) and "
                "reactor temperature T (a thermocouple) are instrumented. The jacket "
                "temperature T_K is not measured but shares the same 150 degC materials limit "
                "as the reactor itself."
            ),
        )

        self._x = np.zeros(4, dtype=float)
        self._t = 0.0

    # -- protocol ---------------------------------------------------------------

    def reset(self, seed: int) -> Observation:
        rng = np.random.default_rng((seed, 0x5A4DE))
        self._params = self._draw_params(rng)
        self._x = steady_state(self._params, self._nominal_u)
        self._t = 0.0
        return self._observe()

    def step(self, u: NDArray[np.float64]) -> Observation:
        v = as_array(u, 2, "van_de_vusse control")
        dt = (self.spec.sample_time / self._substeps) / SECONDS_PER_HOUR
        x = self._x
        for _ in range(self._substeps):
            k1 = _dx_per_hour(x, v, self._params)
            k2 = _dx_per_hour(x + 0.5 * dt * k1, v, self._params)
            k3 = _dx_per_hour(x + 0.5 * dt * k2, v, self._params)
            k4 = _dx_per_hour(x + dt * k3, v, self._params)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            # A concentration cannot go negative -- that is physics, not a constraint, so it
            # is clipped the way four_tank clips a tank to non-negative volume. Temperatures
            # are never clipped: a runaway must show up in audit() as a real violation, not
            # be quietly absorbed by the integrator.
            x[0] = max(x[0], 0.0)
            x[1] = max(x[1], 0.0)
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
        """Apply a mid-run parameter change.

        Dropping ``kw`` is jacket fouling (scale on the coolant-side wall, the classic CSTR
        upset); dropping ``CA0`` is a feed-concentration swing upstream.
        """
        unknown = set(overrides) - {f.name for f in VanDeVusseParams.__dataclass_fields__.values()}
        if unknown:
            raise ValueError(f"van_de_vusse has no parameter(s) {sorted(unknown)}")
        self._params = replace(self._params, **{k: float(v) for k, v in overrides.items()})

    def close(self) -> None:
        return None

    # -- internals ---------------------------------------------------------------

    @property
    def params(self) -> VanDeVusseParams:
        return self._params

    def nominal_params(self) -> VanDeVusseParams:
        """The *un-perturbed* parameters -- what a mismatch-tier brief publishes, deliberately
        not what :meth:`reset` drew."""
        return self._nominal

    def _draw_params(self, rng: np.random.Generator) -> VanDeVusseParams:
        if not self._mismatch:
            return self._nominal
        drawn = {}
        for name, rel_std in self._mismatch.items():
            base = getattr(self._nominal, name)
            # Log-normal: multiplicative, strictly positive, symmetric in ratio terms -- the
            # right shape for rate constants, areas and heat-transfer coefficients, none of
            # which can go negative.
            drawn[name] = float(base * np.exp(rng.normal(0.0, float(rel_std))))
        return replace(self._nominal, **drawn)

    def _observe(self) -> Observation:
        y = np.array([self._x[1], self._x[2]], dtype=float)
        return Observation(t=self._t, y=y, quality=np.ones(2, dtype=bool))


# -- registry hooks -------------------------------------------------------------

PLANT_ID = "van_de_vusse"


def build(cfg):
    """Construct a VanDeVusse from a task file's ``plant:`` block."""
    kwargs = dict(cfg)
    params = NOMINAL
    if overrides := kwargs.pop("params", None):
        params = VanDeVusseParams(**{**params.__dict__, **overrides})
    return VanDeVusse(params, **kwargs)

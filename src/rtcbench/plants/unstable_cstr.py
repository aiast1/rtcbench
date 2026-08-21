"""A jacketed exothermic CSTR held on the *unstable* branch of its ignition curve.

An irreversible first-order reaction ``A -> B`` runs in a continuous stirred tank with a
cooling jacket. The reaction releases heat, the reaction rate rises exponentially with
temperature, and the jacket removes heat only in proportion to a temperature difference.
Heat generation therefore beats heat removal over part of the temperature range, and the
steady-state solution turns S-shaped: for a coolant supply near 300 K this reactor has
*three* equilibria — a cold quenched one near 324 K, a hot ignited one near 370 K, and one
in between, at 350 K, which is the one the plant is actually operated at because it is the
only one that makes the product on spec.

The middle equilibrium is a **saddle**. Linearized about it the pair of eigenvalues is
roughly ``+2.83`` and ``-0.45`` per minute: one of them is in the right half plane, so the
operating point is open-loop unstable with a doubling time of about 15 seconds. The hot
equilibrium is no refuge either — at this coolant temperature it is an unstable focus
(``1.36 +/- 1.54j``), so the *only* attractor the reactor has on its own is the cold one.

**The pathology this plant exists to exercise is open-loop instability.** Doing nothing does
not hold position. Put the loop in manual and the reactor leaves inside two minutes, from a
disturbance no larger than the commissioning offset of the temperature element — either
sliding straight down into the quench, or igniting through a ~100 K excursion and *then*
quenching, because the reaction burns off its inventory of A faster than the feed replaces
it. Every other plant in the pack can, in principle, be survived by a controller that does
very little. This one cannot: feedback is not an optimization here, it is the only thing
that keeps the unit on its operating point.

Two further traps come free with that geometry and are deliberate:

* **The steady-state gain has the opposite sign to the dynamic gain.** Raising the coolant
  supply temperature immediately raises the reactor temperature (``dT/dt`` picks up
  ``+2.09 K/min`` per K of coolant), but the *middle root itself moves down* as the coolant
  gets hotter, because the middle and cold roots march toward each other and annihilate at
  ignition. So ``G(s) = 2.09 (s + 2) / ((s - 2.83)(s + 0.45))`` — DC gain ``-3.25``, high
  frequency gain ``+2.09``. A controller that identifies the sign from a steady-state or
  low-frequency test gets it backwards, and backwards on an unstable plant is not a slow
  loop, it is a runaway. To hold a *higher* reactor temperature you must run *colder*
  coolant.
* **Detuning destabilizes.** Because the plant has a RHP pole, there is a *minimum* gain as
  well as a maximum: with ``ti = 60 s`` the loop needs ``kp > ~1.7`` merely to be stable, so
  the usual safe move of backing off the gain when a loop looks lively makes things worse.

Composition is not instrumented. There is no analyser on this reactor — only the reactor
thermocouple and a readback of what the tempered-water skid actually delivered — so the
conversion, which is what the operating point is really about, has to be inferred from
temperature or not at all.

Two hard trips bracket the reactor thermocouple: a high trip at 470 K (the vessel's design
temperature, where the relief device lifts) and a low trip at 300 K (a quench cold enough to
drop product out of solution onto the coils). Both are wide, and deliberately so. This
reactor's adiabatic temperature rise is 209 K, half of which is stored in the unconverted A
sitting in the vessel at any moment, so an *unattended* ignition is worth about 100 K on its
own and peaks near 447 K. Placing the high trip below that would mean "put the loop in
manual" was a safety event rather than a lost batch, which is both untrue of a vessel rated
for it and fatal to the scoring — the hold anchor would gate on every scenario and there
would be no zero to measure from. Set where they are, the trips still fire, and they fire
for the right reason:

* a controller that *backs off its gain* (``kp <= 1.0`` at ``ti = 60 s``) does not merely
  track badly, it destabilizes, and the resulting excursions reach 480-495 K;
* a controller with the sign taken from the DC gain drives the coolant to its cold limit and
  trips the reactor low, at 290 K;
* a controller that first quenches the reactor and then heats it has re-run the classic
  accumulation accident: nearly a full 1 mol/L of A now goes off at once.

What does *not* trip is the pair of things the reactor does by itself. That is the line
between "this controller lost the plant" (expensive, and it shows up in the cost) and "this
controller endangered the plant" (gated).

Reference: the non-isothermal CSTR of D.E. Seborg, T.F. Edgar, D.A. Mellichamp and F.J.
Doyle, *Process Dynamics and Control*, Wiley — the ``q = 100 L/min``, ``V = 100 L``,
``E/R = 8750 K``, ``k0 = 7.2e10 /min``, ``UA = 5e4 J/(min K)`` parameter set, which is the
one that yields the multiple steady states discussed there and the (0.5 mol/L, 350 K)
unstable point under a 300 K coolant supply. The same reactor and the same class of
unstable-middle-branch operating point is the one used by G.A. Hicks and W.H. Ray,
"Approximation methods for optimal control synthesis", Can. J. Chem. Eng. 49(4), 1971, in
its dimensionless form; this module uses Seborg's dimensional parameters because a
benchmark task has to be written in engineering units an operator would recognise.

Everything below is reimplemented from those published balance equations. The operating
point is *solved* from the parameters rather than copied out of the text, so the initial
condition is exactly consistent with the model that is integrated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from ..plant import Audit, Channel, Constraint, Observation, PlantSpec, as_array

SECONDS_PER_MINUTE = 60.0
"""The published parameter set is per *minute*; the harness works in seconds.

The parameters are stored exactly as they appear in the source and converted once, at the
point of use, rather than being silently pre-divided. A reader checking this module against
the book should not have to undo an arithmetic step first.
"""


@dataclass(frozen=True)
class CSTRParams:
    """Seborg's non-isothermal CSTR parameter set, in the source's own units."""

    q: float = 100.0
    """Feed (and effluent) volumetric flow, L/min."""

    V: float = 100.0
    """Reactor liquid volume, L. Constant — the level loop is someone else's problem."""

    rho: float = 1000.0
    """Liquid density, g/L."""

    Cp: float = 0.239
    """Liquid heat capacity, J/(g K)."""

    dH: float = -5.0e4
    """Heat of reaction, J/mol. Negative because the reaction is exothermic."""

    EoverR: float = 8750.0
    """Activation temperature E/R, K."""

    k0: float = 7.2e10
    """Arrhenius pre-exponential factor, 1/min."""

    UA: float = 5.0e4
    """Jacket heat transfer coefficient x area, J/(min K)."""

    Caf: float = 1.0
    """Feed concentration of A, mol/L."""

    Tf: float = 350.0
    """Feed temperature, K."""

    @property
    def theta(self) -> float:
        """Feed-side space velocity ``q/V``, 1/min."""
        return self.q / self.V

    @property
    def adiabatic_rise(self) -> float:
        """``(-dH)/(rho Cp)``, K per mol/L consumed — the full adiabatic temperature rise
        for complete conversion of a 1 mol/L feed. About 209 K for this reactor, which is
        why it is capable of running away in the first place."""
        return (-self.dH) / (self.rho * self.Cp)

    @property
    def jacket_gain(self) -> float:
        """``UA/(V rho Cp)``, 1/min — how fast the jacket pulls the contents toward the
        coolant temperature."""
        return self.UA / (self.V * self.rho * self.Cp)


#: The published parameter set. Under a 300 K coolant supply this reactor has three
#: steady states and the middle one is (0.5 mol/L, 350 K).
SEBORG = CSTRParams()


def reaction_rate(p: CSTRParams, T: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """Arrhenius rate constant, 1/min. ``T`` is clipped to a physically meaningful floor
    first: ``exp(-E/RT)`` at a negative temperature is ``exp(+huge)``, i.e. an overflow to
    ``inf`` that would poison every downstream number rather than reporting a problem."""
    safe = np.clip(np.asarray(T, dtype=float), 1.0, None)
    return p.k0 * np.exp(-p.EoverR / safe)


def concentration_at(p: CSTRParams, T: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """Concentration of A in steady state at reactor temperature ``T``.

    The mass balance is linear in ``Ca`` at fixed ``T``, so this is exact and turns the
    two-state equilibrium problem into one scalar equation in ``T``.
    """
    k = reaction_rate(p, T)
    return p.theta * p.Caf / (p.theta + k)


def energy_residual(p: CSTRParams, T: float | NDArray[np.float64], Tc: float) -> NDArray[np.float64]:
    """``dT/dt`` at the mass-balance-consistent concentration. Its roots are the equilibria.

    Heat generated by the reaction minus heat removed by the feed and the jacket. Where this
    crosses zero going *upward* the equilibrium is thermally unstable: a nudge up generates
    more heat than it removes.
    """
    Ca = concentration_at(p, T)
    Tarr = np.asarray(T, dtype=float)
    return (
        p.theta * (p.Tf - Tarr)
        + p.adiabatic_rise * reaction_rate(p, Tarr) * Ca
        + p.jacket_gain * (Tc - Tarr)
    )


def steady_states(
    p: CSTRParams,
    Tc: float,
    *,
    search: tuple[float, float] = (250.0, 500.0),
    grid: int = 4001,
    refine: int = 80,
) -> list[tuple[float, float]]:
    """Every ``(Ca, T)`` equilibrium for a held coolant supply temperature ``Tc``.

    Deterministic by construction: a fixed scan of :func:`energy_residual` for sign changes,
    then a fixed number of bisections on each bracket. No adaptive root finder and no scipy
    — the same reason the integrator is fixed-step RK4. Bisection rather than Newton because
    the residual's derivative changes sign three times across the S-curve and Newton's basin
    of attraction near the middle root is exactly where it is worst behaved.
    """
    lo, hi = float(search[0]), float(search[1])
    Ts = np.linspace(lo, hi, int(grid))
    f = energy_residual(p, Ts, Tc)

    roots: list[tuple[float, float]] = []
    for i in range(len(Ts) - 1):
        a, b, fa, fb = Ts[i], Ts[i + 1], float(f[i]), float(f[i + 1])
        if fa == 0.0:
            roots.append((float(concentration_at(p, a)), float(a)))
            continue
        if fa * fb >= 0.0:
            continue
        for _ in range(int(refine)):
            m = 0.5 * (a + b)
            fm = float(energy_residual(p, m, Tc))
            if fa * fm <= 0.0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        T = 0.5 * (a + b)
        roots.append((float(concentration_at(p, T)), float(T)))
    return roots


def unstable_steady_state(p: CSTRParams, Tc: float, **kw) -> tuple[float, float]:
    """The middle (open-loop unstable) root of the ignition curve.

    Raises if the reactor does not actually exhibit multiplicity at this coolant
    temperature. That is not a numerical edge case to paper over — it means the operating
    point this plant is built around does not exist for the given parameters, and starting a
    scenario somewhere that is merely *near* an equilibrium would make the whole task a
    fiction.
    """
    roots = steady_states(p, Tc, **kw)
    if len(roots) != 3:
        raise ValueError(
            f"expected three steady states at Tc={Tc} K (an S-curve), found {len(roots)}: "
            f"{[round(T, 2) for _, T in roots]}. There is no unstable middle branch here."
        )
    return roots[1]


def jacobian(p: CSTRParams, Ca: float, T: float) -> NDArray[np.float64]:
    """Linearization ``d(dx/dt)/dx`` about ``(Ca, T)``, per minute.

    Provided so the instability claim in this module's docstring can be *checked* rather
    than believed. The harness never calls it.
    """
    k = float(reaction_rate(p, T))
    dk_dT = k * p.EoverR / (T * T)
    return np.array(
        [
            [-p.theta - k, -Ca * dk_dT],
            [p.adiabatic_rise * k, -p.theta + p.adiabatic_rise * Ca * dk_dT - p.jacket_gain],
        ],
        dtype=float,
    )


class UnstableCSTR:
    """Jacketed exothermic CSTR on its unstable middle branch, fixed-step RK4.

    Fixed-step on purpose, as everywhere in this pack: an adaptive solver makes the
    trajectory depend on floating-point accidents inside its error controller, and every
    reproducibility claim in RTCbench rests on a plant being a pure function of
    ``(seed, control sequence)``. On an unstable plant that matters more than usual —
    trajectories here diverge exponentially, so a solver that is merely *nearly*
    deterministic is not deterministic at all after ninety seconds.
    """

    def __init__(
        self,
        params: CSTRParams | None = None,
        *,
        sample_time: float = 3.0,
        substeps: int = 30,
        nominal_tc: float = 300.0,
        coolant_range: tuple[float, float] = (270.0, 340.0),
        sensor_range: tuple[float, float] = (280.0, 480.0),
        T_trip_lo: float = 300.0,
        T_trip_hi: float = 470.0,
        commissioning_T: float = 0.6,
        commissioning_Ca: float = 0.005,
        T_physical: tuple[float, float] = (200.0, 700.0),
        Ca_physical_hi: float = 10.0,
        mismatch: Mapping[str, float] | None = None,
    ) -> None:
        self._nominal = params or SEBORG
        self._params = self._nominal
        self._substeps = int(substeps)
        self._nominal_tc = float(nominal_tc)
        self._mismatch = dict(mismatch or {})
        self._commissioning_T = float(commissioning_T)
        self._commissioning_Ca = float(commissioning_Ca)
        self._T_floor, self._T_ceiling = (float(v) for v in T_physical)
        self._Ca_ceiling = float(Ca_physical_hi)

        # The commissioned duty point, solved from the NOMINAL data sheet rather than from
        # whatever this scenario's draw turns out to be. That is what an operator actually
        # has: the reactor is lined out on the number in the file, and the real unit is
        # somewhere else. On a stable plant the difference decays; here it grows.
        self._x_op = np.array(unstable_steady_state(self._nominal, self._nominal_tc), dtype=float)

        tc_lo, tc_hi = (float(v) for v in coolant_range)
        y_lo, y_hi = (float(v) for v in sensor_range)
        if not (self._T_floor < T_trip_lo < T_trip_hi < self._T_ceiling):
            raise ValueError(
                "the physical clamp must lie strictly outside the trip envelope, or a "
                "violation would be clipped away before audit() could report it"
            )

        self.spec = PlantSpec(
            plant_id="unstable_cstr",
            measurements=(
                Channel(tag="TI-101", name="reactor temperature", unit="K", lo=y_lo, hi=y_hi),
                Channel(
                    tag="TI-102",
                    name="jacket supply temperature",
                    unit="K",
                    lo=tc_lo,
                    hi=tc_hi,
                ),
            ),
            actuators=(
                Channel(
                    tag="TIC-201",
                    name="coolant supply temperature",
                    unit="K",
                    lo=tc_lo,
                    hi=tc_hi,
                ),
            ),
            sample_time=float(sample_time),
            state_names=("Ca", "T"),
            constraints=(
                Constraint(name="thermal_runaway", signal="T", hi=float(T_trip_hi)),
                Constraint(name="quench_trip", signal="T", lo=float(T_trip_lo)),
            ),
            initial_u=(self._nominal_tc,),
            description=(
                "Jacketed CSTR running one exothermic first-order reaction, held at the "
                "middle steady state of its ignition curve. The coolant supply temperature "
                "is the only handle. The reactor thermocouple and a readback of the "
                "delivered coolant temperature are the only measurements — there is no "
                "composition analyser. The operating point is open-loop unstable: with the "
                "loop in manual the reactor ignites or quenches within about two minutes."
            ),
        )

        self._x = self._x_op.copy()
        self._u = np.array([self._nominal_tc], dtype=float)
        self._t = 0.0

    # -- protocol ---------------------------------------------------------------

    def reset(self, seed: int) -> Observation:
        rng = np.random.default_rng((seed, 0xC57A11))
        self._params = self._draw_params(rng)
        # Commissioning scatter: the unit is lined out *near* its design point, never
        # exactly on it. On a stable plant this is cosmetic. Here it is the whole story —
        # it is the seed of the divergence, and its sign decides whether this scenario's
        # open-loop reactor ignites or quenches.
        self._x = self._x_op + np.array(
            [
                rng.normal(0.0, self._commissioning_Ca),
                rng.normal(0.0, self._commissioning_T),
            ]
        )
        self._x = self._clamp(self._x)
        self._u = np.array([self._nominal_tc], dtype=float)
        self._t = 0.0
        return self._observe()

    def step(self, u: NDArray[np.float64]) -> Observation:
        self._u = as_array(u, 1, "unstable_cstr control")
        dt = self.spec.sample_time / self._substeps
        x = self._x
        for _ in range(self._substeps):
            k1 = self._dx(x, self._u)
            k2 = self._dx(x + 0.5 * dt * k1, self._u)
            k3 = self._dx(x + 0.5 * dt * k2, self._u)
            k4 = self._dx(x + dt * k3, self._u)
            x = self._clamp(x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
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

        Raising ``Tf`` is an upstream preheater upset arriving as a hot feed slug; dropping
        ``UA`` is the cooling jacket fouling; raising ``Caf`` is a richer feed. All three
        push an already-unstable reactor toward ignition.
        """
        unknown = set(overrides) - set(CSTRParams.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unstable_cstr has no parameter(s) {sorted(unknown)}")
        self._params = replace(self._params, **{k: float(v) for k, v in overrides.items()})

    def close(self) -> None:
        return None

    # -- internals ---------------------------------------------------------------

    @property
    def params(self) -> CSTRParams:
        return self._params

    @property
    def operating_point(self) -> NDArray[np.float64]:
        """The commissioned ``(Ca, T)`` — the unstable root of the nominal data sheet."""
        return self._x_op.copy()

    def nominal_params(self) -> CSTRParams:
        """The *un-perturbed* parameters. This is what a mismatch-tier brief publishes —
        deliberately not what :meth:`reset` drew."""
        return self._nominal

    def _draw_params(self, rng: np.random.Generator) -> CSTRParams:
        if not self._mismatch:
            return self._nominal
        drawn = {}
        for name, rel_std in self._mismatch.items():
            base = getattr(self._nominal, name)
            # Log-normal: multiplicative, strictly positive, symmetric in ratio terms. The
            # right shape for a rate constant, a heat transfer coefficient and a flow, none
            # of which can change sign. ``dH`` is negative, so the draw multiplies its
            # magnitude and the sign rides along unchanged.
            drawn[name] = float(base * np.exp(rng.normal(0.0, float(rel_std))))
        return replace(self._nominal, **drawn)

    def _clamp(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Hold the state inside the physically meaningful box.

        This exists because the plant is unstable and a bad controller can drive it hard:
        without it, an excursion feeds ``exp(-E/RT)`` a temperature the model was never
        written for and the run comes back as ``inf`` or ``nan``, which does not score as a
        very bad controller — it scores as nothing at all.

        The box is deliberately far outside the trip envelope (200-700 K against trips at
        300 and 470 K, and concentration merely kept non-negative). Clamping *physics* is
        allowed; clamping away the constraint violation itself is not, and a plant whose
        clamp tightened onto its own trips would silently make every runaway unreportable.

        Bounding the state also bounds the derivatives, which is what actually keeps the
        arithmetic finite: an excursion past ~500 K makes the mass balance stiff enough that
        a fixed-step RK4 stage overshoots, and without a box the next Arrhenius evaluation
        would be handed a negative temperature and return ``inf``.
        """
        return np.array(
            [
                min(max(float(x[0]), 0.0), self._Ca_ceiling),
                min(max(float(x[1]), self._T_floor), self._T_ceiling),
            ],
            dtype=float,
        )

    def _dx(self, x: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Mass and energy balance, converted from the source's per-minute units."""
        p = self._params
        Ca = min(max(float(x[0]), 0.0), self._Ca_ceiling)
        T = min(max(float(x[1]), self._T_floor), self._T_ceiling)
        Tc = float(u[0])

        rate = float(reaction_rate(p, T)) * Ca
        dCa = p.theta * (p.Caf - Ca) - rate
        dT = p.theta * (p.Tf - T) + p.adiabatic_rise * rate + p.jacket_gain * (Tc - T)
        return np.array([dCa, dT], dtype=float) / SECONDS_PER_MINUTE

    def _observe(self) -> Observation:
        # Reactor temperature, plus a readback of the coolant the skid actually delivered.
        # Composition is NOT measured: there is no analyser on this unit, and inferring
        # conversion from the thermocouple is part of the job.
        return Observation(
            t=self._t,
            y=np.array([self._x[1], self._u[0]], dtype=float),
            quality=np.ones(2, dtype=bool),
        )


# -- registry hooks -------------------------------------------------------------

PLANT_ID = "unstable_cstr"


def build(cfg):
    """Construct an UnstableCSTR from a task file's ``plant:`` block."""
    kwargs = dict(cfg)
    params = SEBORG
    if overrides := kwargs.pop("params", None):
        params = CSTRParams(**{**SEBORG.__dict__, **overrides})
    for key in ("coolant_range", "sensor_range", "T_physical"):
        if key in kwargs:
            kwargs[key] = tuple(float(v) for v in kwargs[key])
    return UnstableCSTR(params, **kwargs)

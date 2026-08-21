"""pH neutralization in a stirred tank (McAvoy; Gustafsson & Waller; Hall & Seborg).

An acid stream and a small carbonate buffer stream run continuously into a stirred tank.
A caustic stream is the only thing the controller can move. The effluent leaves through a
standpipe, so the tank sets its own level, and the pH of that effluent is the controlled
variable.

**The pathology: process gain that varies by more than an order of magnitude across the
operating range.** Near the equivalence point a tenth of a millilitre per second of caustic
swings the pH by a quarter of a unit; two pH units away on the carbonate buffer shelf the
same move barely registers, and out at pH 10 the loop is nearly deaf. With this pack's
parameters the static gain runs from 0.065 pH/(mL/s) at pH 2.7 to 2.94 pH/(mL/s) at pH 8.2
— a factor of 45 — and the task's setpoint schedule deliberately visits both ends. A single
fixed-gain PI cannot be simultaneously stable near neutrality and responsive away from it:
tune it on the shelf and it oscillates violently through the equivalence point; tune it at
the equivalence point and it takes the whole horizon to cross the shelf. Gain scheduling on
the titration curve, or any controller that inverts the curve to work in reaction-invariant
coordinates instead of pH, beats the reference by a wide margin. Nothing else does.

The titration curve is *not* tabulated here. The state is the pair of reaction invariants
of Gustafsson & Waller — a charge-related invariant ``Wa`` and total carbonate ``Wb`` —
which mix linearly, and the pH is recovered each period by solving the electroneutrality
relation

    Wa + 10^(pH-14) - 10^(-pH)
       + Wb * (1 + 2*10^(pH-pK2)) / (1 + 10^(pK1-pH) + 10^(pH-pK2)) = 0

for pH. So the S-curve, its steepness, and the way the carbonate buffer flattens the shelf
at pH ~6.4 (that is pK1) all *emerge* from equilibrium chemistry. Change the buffer
concentration in the task file and the curve changes shape the way the real unit's would.

That split — linear mixing dynamics, all the nonlinearity in the output map — is also why
this model is cheap. The ODEs do not involve pH at all, so the equilibrium solve happens
once per measurement rather than once per integration stage.

**The solve is a bracketed bisection with a fixed iteration count**, not a library root
finder. The residual above is strictly increasing in pH (raising pH raises [OH-], lowers
[H+], and monotonically shifts carbonate from H2CO3 through HCO3- to CO3(2-)), so
``[-2, 16]`` is a guaranteed bracket for any physically reachable feed and 64 halvings
reach double precision. An adaptive solver would make the trajectory depend on its own
error controller, which would break the reproducibility claim the whole benchmark rests on.

Reference: the model and its parameter set are those of M.A. Henson and D.E. Seborg,
"Adaptive nonlinear control of a pH neutralization process", IEEE Trans. Control Systems
Technology 2(3), 1994, 169-182, which reimplements the UCSB rig of R.C. Hall and D.E.
Seborg, "Modelling and self-tuning control of a multivariable pH neutralization process"
(ACC 1989); the reaction-invariant formulation is T.K. Gustafsson and K.V. Waller,
"Dynamic modeling and reaction invariant control of pH", Chem. Eng. Sci. 38(3), 1983, and
the strong-acid/strong-base CSTR original is T.J. McAvoy, E. Lowenthal and H. Argue,
"Dynamics of pH in controlled stirred tank reactor", I&EC Process Des. Dev. 11(1), 1972.

Parameters used, all from Henson & Seborg Table 1: feeds q1 = 16.6 mL/s of 0.003 M HNO3,
q2 = 0.55 mL/s of 0.03 M NaHCO3, and a manipulated q3 of 0.003 M NaOH + 5e-5 M NaHCO3;
tank cross-section A1 = 207 cm^2; effluent q4 = Cv4 (h + z)^n with Cv4 = 4.59, n = 0.607,
z = 11.5 cm. Their published operating point is q3 = 15.6 mL/s, h = 14.0 cm, pH = 7.0, and
:func:`steady_state` reproduces it to better than 0.03 pH — the steady state here is solved
from the model rather than copied off the rig, so the initial condition is exactly
consistent with what is integrated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from ..plant import Audit, Channel, Constraint, Observation, PlantSpec, as_array


@dataclass(frozen=True)
class PHParams:
    """Henson & Seborg's parameter set.

    Concentrations are in mol/L of the two reaction invariants, not of the species: for a
    feed stream, ``Wa = [H+] - [OH-] - [HCO3-] - 2[CO3(2-)]`` and ``Wb`` is total carbonate.
    A strong acid contributes its concentration to ``Wa``; a strong base contributes minus
    its concentration; NaHCO3 contributes minus its concentration to ``Wa`` and plus it to
    ``Wb``. Hence 0.003 M HNO3 -> Wa1 = +3e-3, 0.03 M NaHCO3 -> (Wa2, Wb2) = (-0.03, 0.03),
    and 0.003 M NaOH with 5e-5 M NaHCO3 -> (Wa3, Wb3) = (-3.05e-3, 5e-5).
    """

    A1: float = 207.0
    """Tank cross-sectional area, cm^2."""

    Cv4: float = 4.59
    """Effluent valve coefficient, mL/(s cm^n). Fouling this is a real disturbance."""

    n_exp: float = 0.607
    """Effluent exponent. Not 0.5: the rig drains through a standpipe, not an orifice."""

    z: float = 11.5
    """Standpipe offset, cm. The tank still drains at h = 0, which is why level is
    self-regulating and why a shut base valve empties the tank rather than parking it."""

    q1: float = 16.6
    """Acid feed, mL/s. The load. Not manipulated and not measured."""

    q2: float = 0.55
    """Buffer feed, mL/s."""

    Wa1: float = 3.0e-3
    Wb1: float = 0.0
    Wa2: float = -3.0e-2
    Wb2: float = 3.0e-2
    Wa3: float = -3.05e-3
    Wb3: float = 5.0e-5

    pK1: float = 6.35
    """First carbonate dissociation. This is what puts the flat shelf at pH ~6.4 — the
    reason the loop goes deaf either side of neutrality instead of just at the extremes."""

    pK2: float = 10.25


HENSON_SEBORG = PHParams()
"""The published parameter set. Nominal operating point q3 = 15.6 mL/s -> pH 7.0, h 14 cm."""

_BRACKET = (-2.0, 16.0)
"""Guaranteed bracket on pH. The residual is < 0 at -2 (dominated by -10^2) and > 0 at 16
(dominated by +10^2) for any |Wa| below 100 M, which every physically meaningful feed is."""

_BISECTIONS = 64
"""Fixed, unconditional halvings. 18 / 2^64 is far below the double-precision resolution of
a number near 7, so the loop is converged; the count is fixed rather than tolerance-driven
so the work — and therefore the result — is bit-identical on every call."""


def ph_residual(ph: float, wa: float, wb: float, pK1: float, pK2: float) -> float:
    """Electroneutrality residual. Zero at the equilibrium pH, strictly increasing in pH.

    The carbonate factor is ``(alpha1 + 2*alpha2)``, the mean negative charge carried per
    mole of total carbonate, written over a common denominator.
    """
    carbonate = (1.0 + 2.0 * 10.0 ** (ph - pK2)) / (
        1.0 + 10.0 ** (pK1 - ph) + 10.0 ** (ph - pK2)
    )
    return wa + 10.0 ** (ph - 14.0) - 10.0 ** (-ph) + wb * carbonate


def ph_from_invariants(wa: float, wb: float, pK1: float = 6.35, pK2: float = 10.25) -> float:
    """Solve the electroneutrality relation for pH by bracketed bisection.

    Deterministic by construction: a fixed bracket, a fixed number of halvings, no
    convergence test that could take a different branch on a different machine.
    """
    lo, hi = _BRACKET
    for _ in range(_BISECTIONS):
        mid = 0.5 * (lo + hi)
        if ph_residual(mid, wa, wb, pK1, pK2) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def steady_state(p: PHParams, q3: float) -> NDArray[np.float64]:
    """Exact equilibrium ``(Wa, Wb, h)`` for a held base flow.

    Closed form: the invariants mix linearly, so at equilibrium each is the flow-weighted
    average of the feeds, and the level is wherever the standpipe passes the total flow.
    """
    q = p.q1 + p.q2 + float(q3)
    wa = (p.q1 * p.Wa1 + p.q2 * p.Wa2 + float(q3) * p.Wa3) / q
    wb = (p.q1 * p.Wb1 + p.q2 * p.Wb2 + float(q3) * p.Wb3) / q
    h = (q / p.Cv4) ** (1.0 / p.n_exp) - p.z
    return np.array([wa, wb, h], dtype=float)


def steady_ph(p: PHParams, q3: float) -> float:
    """Steady-state effluent pH for a held base flow — one point on the titration curve."""
    wa, wb, _ = steady_state(p, q3)
    return ph_from_invariants(float(wa), float(wb), p.pK1, p.pK2)


def titration_gain(p: PHParams, q3: float, delta: float = 1e-3) -> float:
    """Static process gain dpH/dq3 at a base flow, pH per (mL/s).

    Central difference on the analytic steady state, so it measures the plant's own gain
    rather than something a simulation transient happened to produce. This is the number
    the whole task is about: it is not a constant, it is not close to constant, and a
    controller that assumes it is will fail at one end of the schedule or the other.
    """
    return (steady_ph(p, q3 + delta) - steady_ph(p, q3 - delta)) / (2.0 * delta)


class PHNeutralization:
    """pH neutralization CSTR, integrated with fixed-step RK4 on the reaction invariants.

    State is ``(Wa, Wb, h)``; pH is algebraic and is recomputed from the invariants
    whenever it is needed. Fixed-step for the same reason as every other plant in the
    pack: an adaptive integrator makes a trajectory depend on floating-point accidents in
    its error controller, and every reproducibility claim here rests on a plant being a
    pure function of ``(seed, control sequence)``.
    """

    def __init__(
        self,
        params: PHParams | None = None,
        *,
        sample_time: float = 5.0,
        substeps: int = 10,
        nominal_q3: float = 14.22,
        q3_max: float = 30.0,
        h_max: float = 30.0,
        h_min: float = 5.0,
        ph_min: float = 4.0,
        ph_max: float = 10.5,
        mismatch: Mapping[str, float] | None = None,
    ) -> None:
        self._nominal = params or HENSON_SEBORG
        self._params = self._nominal
        self._substeps = int(substeps)
        self._nominal_q3 = float(nominal_q3)
        self._mismatch = dict(mismatch or {})

        measurements = (
            Channel(tag="AIT-101", name="effluent pH", unit="pH", lo=0.0, hi=14.0),
            Channel(tag="LIT-102", name="tank level", unit="cm", lo=0.0, hi=35.0),
        )
        actuators = (
            Channel(tag="FCV-103", name="base (NaOH) flow", unit="mL/s", lo=0.0, hi=q3_max),
        )

        self.spec = PlantSpec(
            plant_id="ph_neutralization",
            measurements=measurements,
            actuators=actuators,
            sample_time=float(sample_time),
            # pH is algebraic, not integrated, but it is carried in the audit vector so the
            # discharge consent can be gated on TRUE pH rather than on an analyzer reading
            # a controller might have driven into saturation.
            state_names=("Wa", "Wb", "h", "pH"),
            constraints=(
                Constraint(name="level_envelope", signal="h", lo=h_min, hi=h_max),
                Constraint(name="discharge_consent", signal="pH", lo=ph_min, hi=ph_max),
            ),
            initial_u=(float(nominal_q3),),
            description=(
                "Continuous neutralization of a nitric acid stream with sodium hydroxide in "
                "a stirred tank, with a small sodium bicarbonate buffer stream. Only the "
                "base flow can be manipulated; the acid and buffer feeds are uncontrolled "
                "load. Effluent pH and tank level are instrumented. The tank drains through "
                "a standpipe, so level is self-regulating but follows the base flow: the "
                "more caustic is added, the higher the tank sits."
            ),
        )

        self._h_floor = 0.5
        """Lower clamp on the mixing volume in the invariant balances, cm.

        Purely a numerical guard on a division by the tank volume. It cannot be used to
        escape a violation: it only engages at a tenth of the low-level trip, which the
        audit has already reported by then."""

        self._x = np.zeros(3, dtype=float)
        self._t = 0.0

    # -- protocol ---------------------------------------------------------------

    def reset(self, seed: int) -> Observation:
        rng = np.random.default_rng((seed, 0x7048C0))
        self._params = self._draw_params(rng)
        self._x = steady_state(self._params, self._nominal_q3)
        self._t = 0.0
        return self._observe()

    def step(self, u: NDArray[np.float64]) -> Observation:
        q3 = float(as_array(u, 1, "ph_neutralization control")[0])
        dt = self.spec.sample_time / self._substeps
        x = self._x
        for _ in range(self._substeps):
            k1 = self._dx(x, q3)
            k2 = self._dx(x + 0.5 * dt * k1, q3)
            k3 = self._dx(x + 0.5 * dt * k2, q3)
            k4 = self._dx(x + dt * k3, q3)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            # A tank cannot hold a negative volume. Nothing else is clipped -- in
            # particular a level above h_max and a pH outside the consent band are real
            # events the safety gate has to see, not something the model quietly absorbs.
            x[2] = max(x[2], 0.0)
        self._x = x
        self._t += self.spec.sample_time
        return self._observe()

    def audit(self) -> Audit:
        x = self._full_state()
        value = dict(zip(self.spec.state_names, (float(v) for v in x)))
        violated = tuple(c.name for c in self.spec.constraints if c.violated(value[c.signal]))
        return Audit(t=self._t, x=x, violations=violated)

    def set_params(self, overrides: Mapping[str, float]) -> None:
        """Apply a mid-run parameter change — this is how disturbances land.

        Raising ``q1`` or ``Wa1`` is an acid load surge from upstream; raising ``Wb2`` is a
        buffer dosing pump that has been turned up, which flattens the titration curve out
        from under a controller tuned on the sharp one; dropping ``Cv4`` is a fouling
        effluent standpipe.
        """
        unknown = set(overrides) - set(PHParams.__dataclass_fields__)
        if unknown:
            raise ValueError(f"ph_neutralization has no parameter(s) {sorted(unknown)}")
        self._params = replace(self._params, **{k: float(v) for k, v in overrides.items()})

    def close(self) -> None:
        return None

    # -- internals ---------------------------------------------------------------

    @property
    def params(self) -> PHParams:
        return self._params

    def nominal_params(self) -> PHParams:
        """The *un-perturbed* parameters. This is what a mismatch-tier brief publishes —
        deliberately not what :meth:`reset` drew."""
        return self._nominal

    @property
    def ph(self) -> float:
        """True effluent pH right now, from the current invariants."""
        p = self._params
        return ph_from_invariants(float(self._x[0]), float(self._x[1]), p.pK1, p.pK2)

    def _draw_params(self, rng: np.random.Generator) -> PHParams:
        if not self._mismatch:
            return self._nominal
        drawn = {}
        for name, rel_std in self._mismatch.items():
            base = getattr(self._nominal, name)
            # Log-normal: multiplicative, sign-preserving, symmetric in ratio terms. The
            # right shape for flows and concentrations, and it keeps the two negative
            # invariants (Wa2, Wa3 -- sodium-bearing streams) negative.
            drawn[name] = float(base * np.exp(rng.normal(0.0, float(rel_std))))
        return replace(self._nominal, **drawn)

    def _dx(self, x: NDArray[np.float64], q3: float) -> NDArray[np.float64]:
        p = self._params
        volume = p.A1 * max(float(x[2]), self._h_floor)
        # d(V*W)/dt = sum(qi*Wi) - q4*W and dV/dt = sum(qi) - q4 together collapse to a
        # flow-weighted pull toward each feed, with no q4 term left: the effluent carries
        # the tank's own composition, so it cannot change it.
        d_wa = (
            p.q1 * (p.Wa1 - x[0]) + p.q2 * (p.Wa2 - x[0]) + q3 * (p.Wa3 - x[0])
        ) / volume
        d_wb = (
            p.q1 * (p.Wb1 - x[1]) + p.q2 * (p.Wb2 - x[1]) + q3 * (p.Wb3 - x[1])
        ) / volume
        q4 = p.Cv4 * math.pow(max(float(x[2]), 0.0) + p.z, p.n_exp)
        d_h = (p.q1 + p.q2 + q3 - q4) / p.A1
        return np.array([d_wa, d_wb, d_h], dtype=float)

    def _full_state(self) -> NDArray[np.float64]:
        return np.array([self._x[0], self._x[1], self._x[2], self.ph], dtype=float)

    def _observe(self) -> Observation:
        return Observation(
            t=self._t,
            y=np.array([self.ph, float(self._x[2])], dtype=float),
            quality=np.ones(2, dtype=bool),
        )


# -- registry hooks -------------------------------------------------------------

PLANT_ID = "ph_neutralization"


def build(cfg):
    """Construct a PHNeutralization from a task file's ``plant:`` block."""
    kwargs = dict(cfg)
    params = HENSON_SEBORG
    if overrides := kwargs.pop("params", None):
        params = replace(params, **{k: float(v) for k, v in overrides.items()})
    return PHNeutralization(params, **kwargs)

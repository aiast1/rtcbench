"""Skogestad's "Column A" — a 41-stage binary distillation column, LV configuration.

The pack's ill-conditioned plant. Two products are controlled with two flows, and the 2x2
process between them is very nearly singular in one direction:

    G(0) = [[ 87.5  -86.2 ]      sigma = 1.97, 0.0136      cond(G) = 145
            [108.5 -109.8]]      RGA lambda_11 = 35.9

from ``(L, V)`` to ``(yD, xB)``, in units of 0.01 mole fraction per kmol/min — the scaling
Skogestad publishes it in, and within half a percent of his published numbers. (The bottoms
analyser here reads purity ``1 - xB``, which flips the sign of the second row and leaves the
singular values and the RGA untouched, both being invariant to that scaling.) The two
columns of that matrix are almost parallel, and everything hard about this plant follows
from it:

* **The reachable directions are wildly unequal.** Making both products purer — reflux and
  boilup up *together*, the separation direction — moves the compositions by 0.005 for a
  0.4 kmol/min move in each flow. Making the split change — reflux and boilup moving
  *apart* — moves them by the same 0.005 for a 0.005 kmol/min move. Two orders of magnitude
  between the two directions, on the same plant, at the same operating point.
* **So a small input error is a large composition error.** Nudging reflux alone by
  0.01 kmol/min (0.4% of its flow, less than a decent flow controller's own error) more
  than doubles the bottoms impurity: xB goes 0.010 -> 0.024. A controller that inverts the
  plant to get the sluggish direction back amplifies exactly this, which is why the
  inverse-based controller is the classic wrong answer here.
* **And the two loops fight.** RGA lambda_11 = 35.9: closing the bottoms loop cuts the top
  loop's gain by a factor of 36. Gains identified one loop at a time are wrong by that
  factor, and the pairing that looks natural on a single-loop step test is a trap.
* **Which is why the flow controllers' own error matters so much.** Each flow loop here has
  a span error (:attr:`ColumnAParams.gain_L`), the uncertainty this column is classically
  studied against. A controller that inverts the plant asks for two large moves whose
  *difference* is the small quantity it actually wants; a few percent of error between the
  two flows delivers a different difference, amplified back out by the large gain. This is
  the published reason inverse-based control fails on ill-conditioned plants, and it is in
  the model rather than in the prose.
* **The directions have different speeds too.** The high-gain split direction is the slow
  one (the column's dominant time constant is 194 min); the low-gain separation direction
  settles in about ten. A controller that tunes for the time constant it sees on a step test
  gets the other direction's dynamics wrong as well as its gain.

Everything else is deliberately benign — no inverse response, no instability, no deadtime
worth the name in the process itself. If a submission does badly here it is because of
directionality, which is the point.

Model
-----
41 stages numbered from the bottom: stage 1 is the reboiler, stages 2-40 are trays, stage 41
is a total condenser. Feed enters on stage 21 as saturated liquid. Binary mixture, constant
relative volatility, constant molar overflow, constant liquid holdup on every stage, so the
state is one liquid mole fraction per stage and nothing else — 41 states, no flow dynamics.
Vapour-liquid equilibrium is ``y = a x / (1 + (a-1) x)``.

Reference: S. Skogestad, "Dynamics and control of distillation columns: A tutorial
introduction", Trans. IChemE 75(A), 1997, and S. Skogestad & I. Postlethwaite,
*Multivariable Feedback Control*, 2nd ed., section 12.4 / the distillation case studies. The
parameter set is the published "column A": 41 stages, feed stage 21, alpha = 1.5, zF = 0.5,
saturated liquid feed F = 1 kmol/min, holdup 0.5 kmol on every stage including the reboiler
and the condenser. At the nominal duty point L = 2.70629, V = 3.20629 kmol/min this model
reproduces the published operating point yD = 0.99, xB = 0.01 with D = B = 0.5 kmol/min, and
the gain matrix, RGA and 194 min dominant time constant quoted above — all recomputed from
the equations here rather than copied, so the initial condition is exactly consistent with
what is integrated. The steady state is solved by Newton rather than by long integration for
the same reason.

Units
-----
**Time is in minutes throughout this plant**, because the column's flows are kmol/min and
its dynamics are hours long: ``sample_time = 1.0`` is a one-minute control period and a
task's ``horizon: 400`` is 400 minutes. Rate limits in a task file are therefore per minute,
and reported total variation is per minute. The compute budget (``step_seconds``) is
wall-clock and unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from ..plant import Audit, Channel, Constraint, Observation, PlantSpec, as_array

N_STAGES = 41
"""Stage count: 39 trays plus the reboiler (stage 1) and the total condenser (stage 41)."""

FEED_STAGE = 21
"""1-based stage the feed enters on."""

NOMINAL_LV = (2.70629, 3.20629)
"""Skogestad's nominal duty point, kmol/min: reflux L and boilup V. Gives yD = 0.99,
xB = 0.01, D = B = 0.5."""


@dataclass(frozen=True)
class ColumnAParams:
    """Column A's published parameter set. Flows in kmol/min, holdups in kmol."""

    alpha: float = 1.5
    """Relative volatility, constant over the column."""

    F: float = 1.0
    """Feed rate."""

    zF: float = 0.5
    """Feed composition, mole fraction light."""

    qF: float = 1.0
    """Feed liquid fraction. 1.0 is saturated liquid, which is the published case."""

    M_tray: float = 0.5
    """Liquid holdup on each tray. Sets the tray time constant, not the steady state."""

    M_reboiler: float = 0.5
    M_condenser: float = 0.5

    gain_L: float = 1.0
    """Span error of the reflux flow controller: a *commanded change* of 1 kmol/min away
    from the duty point delivers ``gain_L`` kmol/min.

    Not chemistry — instrument calibration, and it lives here so a scenario can draw it.
    This is the uncertainty the distillation literature actually runs this column against
    (Skogestad & Postlethwaite use +/-20% on each input independently), and on an
    ill-conditioned plant it is the difference between an academic example and the real
    problem: a controller that inverts G to reach the sluggish direction commands a large,
    finely balanced pair of moves, and a few percent of error between the two flows turns
    that balance into exactly the large split change it was trying not to make. Applied to
    the *deviation* from the duty point rather than to the flow itself, because a meter is
    calibrated where it runs: the column still starts where the operator left it.
    """

    gain_V: float = 1.0
    """Span error of the boilup flow controller. See :attr:`gain_L`."""


P_COLUMN_A = ColumnAParams()
"""The published parameter set."""


# -- physics ---------------------------------------------------------------------


def _flows(
    p: ColumnAParams, u: NDArray[np.float64], nt: int, nf: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    """Internal liquid/vapour flows leaving each stage, plus the net products.

    Constant molar overflow: the internal flows are algebraic in the two manipulated flows,
    so there is no flow dynamics and no holdup state. ``L[0]`` (liquid off the reboiler,
    which leaves as bottoms) and ``V[-1]`` (vapour off the total condenser, which does not
    exist) are never read.
    """
    reflux, boilup = float(u[0]), float(u[1])
    liquid = np.empty(nt, dtype=float)
    vapour = np.empty(nt, dtype=float)
    # Below and including the feed stage the liquid carries the feed's liquid fraction.
    liquid[:nf] = reflux + p.qF * p.F
    liquid[nf:] = reflux
    vapour[:nf] = boilup
    vapour[nf:] = boilup + (1.0 - p.qF) * p.F
    distillate = float(vapour[nt - 2] - liquid[nt - 1])
    bottoms = float(liquid[1] - vapour[0])
    return liquid, vapour, distillate, bottoms


def _derivative(
    x: NDArray[np.float64],
    liquid: NDArray[np.float64],
    vapour: NDArray[np.float64],
    distillate: float,
    bottoms: float,
    p: ColumnAParams,
    nf: int,
    m_inv: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Component balance on every stage at once, per minute.

    Vectorized deliberately. This is the pack's largest plant and the harness runs it
    thousands of times per scoring pass; a Python loop over 41 trays inside four RK4 stages
    is 160 interpreter trips per substep where three numpy expressions will do.
    """
    y = p.alpha * x / (1.0 + (p.alpha - 1.0) * x)
    d = np.empty_like(x)
    # Trays: liquid down from above, vapour up from below, minus what leaves this stage.
    d[1:-1] = (
        liquid[2:] * x[2:]
        - liquid[1:-1] * x[1:-1]
        + vapour[:-2] * y[:-2]
        - vapour[1:-1] * y[1:-1]
    )
    # Reboiler: liquid down from tray 2, minus the vapour it boils and the bottoms it draws.
    d[0] = liquid[1] * x[1] - vapour[0] * y[0] - bottoms * x[0]
    # Total condenser: all vapour from the top tray condenses, leaving as reflux + distillate.
    d[-1] = vapour[-2] * y[-2] - (liquid[-1] + distillate) * x[-1]
    d[nf - 1] += p.F * p.zF
    return d * m_inv


def _holdup_inverse(p: ColumnAParams, nt: int) -> NDArray[np.float64]:
    m = np.full(nt, p.M_tray, dtype=float)
    m[0] = p.M_reboiler
    m[-1] = p.M_condenser
    return 1.0 / m


def steady_state(
    p: ColumnAParams,
    u: NDArray[np.float64] | tuple[float, float],
    *,
    nt: int = N_STAGES,
    nf: int = FEED_STAGE,
    guess: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Equilibrium composition profile for held flows ``u``, by damped Newton.

    The column's slowest mode is 194 minutes, so "integrate until it stops moving" would
    cost thousands of steps and still leave a residual that differs from draw to draw. The
    balance equations are tridiagonal in ``x`` and their Jacobian is analytic, so Newton
    lands on the equilibrium in a handful of iterations and every scenario starts *exactly*
    stationary instead of nearly so.
    """
    liquid, vapour, distillate, bottoms = _flows(p, np.asarray(u, dtype=float), nt, nf)
    m_inv = _holdup_inverse(p, nt)
    x = np.linspace(0.01, 0.99, nt) if guess is None else np.clip(guess, 1e-12, 1.0 - 1e-12)

    def residual(v: NDArray[np.float64]) -> NDArray[np.float64]:
        return _derivative(v, liquid, vapour, distillate, bottoms, p, nf, m_inv)

    f = residual(x)
    for _ in range(100):
        worst = float(np.max(np.abs(f)))
        if worst < 1e-13:
            return x
        j = _jacobian(x, liquid, vapour, distillate, p, m_inv)
        try:
            step = np.linalg.solve(j, -f)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - would mean a broken draw
            raise RuntimeError(f"column_a steady state: singular Jacobian ({exc})") from exc
        scale, improved = 1.0, False
        while scale > 1e-6:
            trial = np.clip(x + scale * step, 1e-12, 1.0 - 1e-12)
            f_trial = residual(trial)
            if np.max(np.abs(f_trial)) < worst:
                improved = True
                break
            scale *= 0.5
        if not improved:
            # The line search has run out of room. On a nearly-perfect separation the top
            # and bottom compositions sit against the 1e-12 floor and the residual cannot
            # be driven to 1e-13 in double precision — but 1e-8 kmol/min of imbalance is a
            # composition drift of 2e-6 over the whole 400 minute scenario, which is a
            # thousand times below the analyser's resolution. Accept it and say so; only
            # raise when the point is genuinely not an equilibrium (asking for more product
            # than the feed supplies, say, which has no steady state at all).
            if worst < 1e-8:
                return x
            break
        x, f = trial, f_trial
    raise RuntimeError(
        f"column_a steady state did not converge (residual {np.max(np.abs(f)):.2e}); "
        f"parameters {p} have no equilibrium at u={tuple(np.asarray(u, dtype=float))} — "
        f"check that both net products are positive (D = V - L, B = L + qF*F - V)"
    )


def _jacobian(
    x: NDArray[np.float64],
    liquid: NDArray[np.float64],
    vapour: NDArray[np.float64],
    distillate: float,
    p: ColumnAParams,
    m_inv: NDArray[np.float64],
) -> NDArray[np.float64]:
    """d(dx/dt)/dx. Tridiagonal: a stage only sees its immediate neighbours."""
    dy = p.alpha / (1.0 + (p.alpha - 1.0) * x) ** 2
    nt = x.size
    j = np.zeros((nt, nt), dtype=float)
    idx = np.arange(1, nt - 1)
    j[idx, idx + 1] = liquid[2:]
    j[idx, idx] = -liquid[1:-1] - vapour[1:-1] * dy[1:-1]
    j[idx, idx - 1] = vapour[:-2] * dy[:-2]
    j[0, 1] = liquid[1]
    j[0, 0] = -vapour[0] * dy[0] - (liquid[1] - vapour[0])
    j[-1, -2] = vapour[-2] * dy[-2]
    j[-1, -1] = -(liquid[-1] + distillate)
    return j * m_inv[:, None]


class ColumnA:
    """Column A, integrated with fixed-step RK4.

    Fixed-step on purpose (see :mod:`rtcbench.plants.four_tank`): an adaptive solver makes
    the trajectory depend on its own error controller, and every reproducibility claim in
    RTCbench rests on a plant being a pure function of ``(seed, control sequence)``.

    The substep count is set by stability, not taste. The fastest eigenvalue of the tray
    balances is about -33 min^-1 (a 0.5 kmol holdup against ~3 kmol/min of traffic), and
    RK4 needs ``|lambda| h < 2.78``, i.e. ``h < 0.086 min``. The default 20 substeps of a
    one-minute period gives ``h = 0.05 min``, a factor 1.7 inside the limit.
    """

    def __init__(
        self,
        params: ColumnAParams | None = None,
        *,
        sample_time: float = 1.0,
        substeps: int = 20,
        n_stages: int = N_STAGES,
        feed_stage: int = FEED_STAGE,
        nominal_lv: tuple[float, float] = NOMINAL_LV,
        reflux_range: tuple[float, float] = (1.5, 4.5),
        boilup_range: tuple[float, float] = (2.0, 5.0),
        yd_min: float = 0.85,
        xb_max: float = 0.15,
        min_product: float = 0.05,
        mismatch: Mapping[str, float] | None = None,
    ) -> None:
        if not 2 <= feed_stage <= n_stages - 1:
            raise ValueError(f"feed_stage must lie strictly inside the column, got {feed_stage}")
        self._nominal = params or P_COLUMN_A
        self._params = self._nominal
        self._substeps = int(substeps)
        self._nt = int(n_stages)
        self._nf = int(feed_stage)
        self._nominal_lv = np.array(nominal_lv, dtype=float)
        self._mismatch = dict(mismatch or {})
        self._m_inv = _holdup_inverse(self._params, self._nt)

        analysers = (
            Channel(
                tag="AT-101",
                name="distillate purity (light key)",
                unit="mole fraction",
                lo=0.90,
                hi=1.00,
            ),
            # Reported as the heavy-key PURITY, 1 - xB, which is both the industry
            # convention for a bottoms product and the only sign convention under which the
            # natural pairing is direct-acting: dxB/dV is negative, so an analyser reading
            # light-key impurity would need a reverse-acting bottoms loop, and a controller
            # that got the sign wrong would drive the column to a wall rather than merely
            # perform badly. Directionality is what this plant is here to test; sign is not.
            Channel(
                tag="AT-102",
                name="bottoms purity (heavy key)",
                unit="mole fraction",
                lo=0.90,
                hi=1.00,
            ),
        )
        flows = (
            Channel(tag="FC-201", name="reflux flow L", unit="kmol/min",
                    lo=reflux_range[0], hi=reflux_range[1]),
            Channel(tag="FC-202", name="reboiler boilup V", unit="kmol/min",
                    lo=boilup_range[0], hi=boilup_range[1]),
        )

        # Ground truth is 41 compositions plus the two net product rates. The product rates
        # are algebraic in the inputs rather than states, and they are in here for one
        # reason: D = V - L and B = L + qF*F - V are what a controller silently destroys if
        # it lets the two flows converge. A column making no distillate is not off-spec, it
        # is not operating, and the harness can only gate on something it can see in the
        # audit vector.
        stages = ("xB",) + tuple(f"x{i}" for i in range(2, self._nt)) + ("yD",)
        self._state_names = stages + ("D", "B")

        self.spec = PlantSpec(
            plant_id="column_a",
            measurements=analysers,
            actuators=flows,
            sample_time=float(sample_time),
            state_names=self._state_names,
            constraints=(
                # One-sided, like an overflow: purer than spec is never a hazard, and a
                # two-sided band would gate scenarios for being *too good*.
                Constraint(name="distillate_offspec", signal="yD", lo=float(yd_min)),
                Constraint(name="bottoms_offspec", signal="xB", hi=float(xb_max)),
                Constraint(name="distillate_flow", signal="D", lo=float(min_product)),
                Constraint(name="bottoms_flow", signal="B", lo=float(min_product)),
            ),
            initial_u=(float(nominal_lv[0]), float(nominal_lv[1])),
            description=(
                "A 41-stage binary distillation column: reboiler, 39 trays, total condenser, "
                "feed on stage 21. Reflux flow and reboiler boilup are manipulated; both "
                "products are analysed for their own purity — light key overhead, heavy key "
                "in the bottoms. The 39 tray compositions are not measured, and neither "
                "product draw is measured. Time is in minutes."
            ),
        )

        self._x = np.zeros(self._nt, dtype=float)
        self._t = 0.0
        self._u = self._nominal_lv.copy()

    # -- protocol ---------------------------------------------------------------

    def reset(self, seed: int) -> Observation:
        rng = np.random.default_rng((seed, 0xC01A41))
        self._params = self._draw_params(rng)
        self._m_inv = _holdup_inverse(self._params, self._nt)
        self._u = self._nominal_lv.copy()
        # Start at the equilibrium of the *drawn* column, not of the nominal one. The plant
        # is therefore stationary at t=0 but generally off setpoint: a mismatched column
        # sitting at the commissioned flows is not making the commissioned purity, and
        # closing that gap is the first thing the controller has to do.
        self._x = steady_state(
            self._params, self._delivered(self._u), nt=self._nt, nf=self._nf
        )
        self._t = 0.0
        return self._observe()

    def step(self, u: NDArray[np.float64]) -> Observation:
        self._u = as_array(u, 2, "column_a control")
        liquid, vapour, distillate, bottoms = _flows(
            self._params, self._delivered(self._u), self._nt, self._nf
        )
        p, nf, m_inv = self._params, self._nf, self._m_inv
        dt = self.spec.sample_time / self._substeps
        x = self._x
        for _ in range(self._substeps):
            k1 = _derivative(x, liquid, vapour, distillate, bottoms, p, nf, m_inv)
            k2 = _derivative(x + 0.5 * dt * k1, liquid, vapour, distillate, bottoms, p, nf, m_inv)
            k3 = _derivative(x + 0.5 * dt * k2, liquid, vapour, distillate, bottoms, p, nf, m_inv)
            k4 = _derivative(x + dt * k3, liquid, vapour, distillate, bottoms, p, nf, m_inv)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            # A mole fraction is in [0, 1] by definition — this is the same kind of clip as
            # "a tank cannot hold negative volume", not a way to hide a violation: both
            # composition limits sit well inside the range.
            x = np.clip(x, 0.0, 1.0)
        self._x = x
        self._t += self.spec.sample_time
        return self._observe()

    def audit(self) -> Audit:
        x = self._augmented_state()
        violated = tuple(
            c.name
            for c in self.spec.constraints
            if c.violated(float(x[self._state_names.index(c.signal)]))
        )
        return Audit(t=self._t, x=x, violations=violated)

    def set_params(self, overrides: Mapping[str, float]) -> None:
        """Apply a parameter change mid-run — this is how disturbances land.

        ``F`` is a throughput change, ``zF`` a feed-composition upset (both classic column
        disturbances and both published as the ones Column A is studied against), ``alpha``
        a change in what is being separated, ``M_tray`` a hydraulic change that moves the
        dynamics without moving the steady state.
        """
        unknown = set(overrides) - set(ColumnAParams.__dataclass_fields__)
        if unknown:
            raise ValueError(f"column_a has no parameter(s) {sorted(unknown)}")
        self._params = replace(self._params, **{k: float(v) for k, v in overrides.items()})
        self._m_inv = _holdup_inverse(self._params, self._nt)

    def close(self) -> None:
        return None

    # -- internals ---------------------------------------------------------------

    @property
    def params(self) -> ColumnAParams:
        return self._params

    def nominal_params(self) -> ColumnAParams:
        """The *un-perturbed* parameters — what a mismatch-tier brief publishes, and
        deliberately not what :meth:`reset` drew."""
        return self._nominal

    def _draw_params(self, rng: np.random.Generator) -> ColumnAParams:
        if not self._mismatch:
            return self._nominal
        drawn = {}
        for name, rel_std in self._mismatch.items():
            if name not in ColumnAParams.__dataclass_fields__:
                raise ValueError(f"column_a has no parameter {name!r} to mismatch")
            base = getattr(self._nominal, name)
            # Log-normal: multiplicative, strictly positive, symmetric in ratio terms —
            # right for volatilities, flows and holdups, none of which can go negative.
            drawn[name] = float(base * np.exp(rng.normal(0.0, float(rel_std))))
        if "zF" in drawn:
            drawn["zF"] = min(drawn["zF"], 0.99)
        if "qF" in drawn:
            drawn["qF"] = min(drawn["qF"], 1.0)
        return replace(self._nominal, **drawn)

    def _delivered(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Flows the column actually sees, given the flow controllers' span errors.

        Calibrated at the duty point, so ``_delivered(nominal_lv) == nominal_lv`` exactly
        and a scenario still starts stationary however the gains were drawn.
        """
        p = self._params
        return self._nominal_lv + np.array([p.gain_L, p.gain_V]) * (u - self._nominal_lv)

    def _augmented_state(self) -> NDArray[np.float64]:
        _, _, distillate, bottoms = _flows(
            self._params, self._delivered(self._u), self._nt, self._nf
        )
        return np.concatenate([self._x, [distillate, bottoms]])

    def _observe(self) -> Observation:
        # Two analysers on a 41-state column: the purity of each product. The 39 tray
        # compositions that determine how the column will respond are not measured. Both
        # readings are purities — light key overhead, heavy key in the bottoms — so the
        # bottoms analyser reads 1 - xB.
        return Observation(
            t=self._t,
            y=np.array([self._x[-1], 1.0 - self._x[0]], dtype=float),
            quality=np.ones(2, dtype=bool),
        )


# -- registry hooks -------------------------------------------------------------

PLANT_ID = "column_a"


def build(cfg):
    """Construct a ColumnA from a task file's ``plant:`` block."""
    kwargs = dict(cfg)
    params = P_COLUMN_A
    if overrides := kwargs.pop("params", None):
        params = ColumnAParams(**{**params.__dict__, **overrides})
    for key in ("nominal_lv", "reflux_range", "boilup_range"):
        if key in kwargs:
            kwargs[key] = tuple(float(v) for v in kwargs[key])
    return ColumnA(params, **kwargs)

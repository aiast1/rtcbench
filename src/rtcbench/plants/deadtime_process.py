"""A heated transport line with flow-dependent deadtime (FOPDT, deadtime-dominant).

A trace-heated pipe carries process fluid from a supply header, through an electrically
heated section, into a long transport line, to a downstream unit where its temperature is
what matters. The heater has its own (fast) thermal lag; the transport line itself has no
dynamics beyond pure convection -- whatever temperature enters one end reappears, unchanged,
at the other, after a transit time

    theta(t) = V_line / q(t)

where ``V_line`` is the line's fixed physical volume and ``q(t)`` is the current volumetric
throughput. The result is a first-order-plus-deadtime process, G(s) = K*exp(-theta*s) /
(tau*s + 1), with theta held well above tau across this task's whole flow range (theta/tau
runs from about 3 at the highest scheduled flow to about 17 at the lowest) -- the "deadtime
dominant" regime discussed as a distinct tuning problem in Seborg, Edgar, Mellichamp & Doyle,
*Process Dynamics and Control* (transportation lag as line volume over flow rate is their
standard example), and in Astrom & Hagglund, *PID Controllers: Theory, Design, and Tuning*,
ch. 4, on how badly a Smith predictor degrades once its internal deadtime estimate is wrong.
Parameters below are chosen to be representative of that textbook regime, not lifted from one
paper's rig.

**The trap is that theta is not constant.** Throughput is set by an upstream pump/valve this
loop does not own -- a production-rate change, not a fault -- and every time it moves, the
transport delay moves with it. A PI (or a Smith predictor) commissioned against one theta and
then left alone is, after the next rate change, running against the wrong one: a fixed,
aggressively-tuned loop that looks excellent at theta=40s can ring into a limit cycle at
theta=200s, and a fixed Smith predictor's internal delay model desyncs from the real one, so
the "prediction" it feeds back stops predicting anything. Only a controller that tracks theta,
rather than one that identifies it once and bakes it in, does well across the whole scenario.

A second trap, structural rather than temporal: only the *downstream* temperature is
instrumented (TT at the far end of the line). The heater's own outlet -- immediately after
the heated section, before the long unmeasured transport run -- carries no sensor, but it
still shares the line's hard thermal-degradation limit. Because of the deadtime, a controller
correcting on what it can see is reacting to an error that is already theta seconds stale; lean
on the heater hard enough to close that error quickly and the unmeasured near-field
temperature can sail past its limit long before the (correct) correction ever shows up at the
sensor -- the same "constrained but not instrumented" shape as the four-tank plant's upper
tanks, here driven by time rather than by geometry.

Deadtime is modelled as genuine plug-flow transport, not a lookup table keyed on the current
theta. The heater's outlet temperature is written every substep into a running
(cumulative-volume, temperature) history; the line's outlet reads back the value at
cumulative volume ``V_now - V_line``, linearly interpolated between the two bracketing
history samples. That makes the effective delay exactly ``V_line / q`` whenever q has been
steady for at least one transit time, it is correct straight through a flow change (a parcel
already inside the line has to travel the remaining *volume*, not some remaining *time*, so
indexing by accumulated volume rather than elapsed time is what makes the transition honest),
and it varies continuously with q rather than snapping between whole control periods the way
a naive ``round(theta / Ts)`` sample-shift buffer would. It also degrades gracefully at the
edges a formula would not: q -> 0 simply freezes the outlet (nothing moves), rather than
sending theta to infinity through a division.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from ..plant import Audit, Channel, Constraint, Observation, PlantSpec, as_array


@dataclass(frozen=True)
class DeadtimeProcessParams:
    """Physical parameters of the heated line. ``q`` is throughput, not a fixture: real
    plants change it constantly, which is the whole point of this task."""

    V_line: float = 240.0
    """Transport line volume, L. Fixed geometry -- but see the module docstring: the task's
    mismatch draw still treats it as imperfectly known, the way a commissioning volume
    calculated from an as-built drawing usually is."""

    q: float = 3.0
    """Volumetric throughput, L/s. Mutable via :meth:`DeadtimeProcess.set_params` -- this is
    how a rate change lands, exactly as a leak lands on ``four_tank`` via ``a1``."""

    tau_h: float = 12.0
    """Heater thermal time constant, s. Fixed and comparatively fast; theta/tau_h is what
    makes this plant deadtime-dominant rather than merely laggy."""

    K_h: float = 0.8
    """Heater gain, degC per % duty."""

    T_supply: float = 15.0
    """Upstream supply temperature, degC, ahead of the heater."""


def steady_state(p: DeadtimeProcessParams, u: float) -> float:
    """Closed-form equilibrium for a held duty ``u``.

    Both the heater outlet and the line outlet settle to the same value: transport delay
    only postpones a signal, and a constant signal delayed is still that constant signal.
    """
    return p.T_supply + p.K_h * float(u)


class DeadtimeProcess:
    """A trace-heated transport line, integrated with fixed-step RK4 plus a volume-indexed
    transport buffer.

    The heater's thermal lag is a genuine ODE state and is marched with RK4 like every other
    plant in the pack. The transport line is not an ODE in any low-dimensional state -- it is
    a hyperbolic transport equation -- so it is not shoehorned into RK4; instead its buffer is
    advanced once per the same fixed substep, deterministically, with no adaptive step-size
    logic anywhere in either path. Same seed, same control sequence, same trajectory.
    """

    def __init__(
        self,
        params: DeadtimeProcessParams | None = None,
        *,
        sample_time: float = 5.0,
        substeps: int = 10,
        nominal_u: float = 50.0,
        T_in_max: float = 90.0,
        T_out_lo: float = 0.0,
        T_out_hi: float = 90.0,
        y_lo: float = 0.0,
        y_hi: float = 100.0,
        mismatch: Mapping[str, float] | None = None,
    ) -> None:
        self._nominal = params or DeadtimeProcessParams()
        self._params = self._nominal
        self._substeps = int(substeps)
        self._mismatch = dict(mismatch or {})

        self.spec = PlantSpec(
            plant_id="deadtime_process",
            measurements=(
                Channel(
                    tag="TT-401", name="line outlet temperature", unit="degC",
                    lo=float(y_lo), hi=float(y_hi),
                ),
            ),
            actuators=(
                Channel(tag="TIC-401", name="trace heater duty", unit="%", lo=0.0, hi=100.0),
            ),
            sample_time=float(sample_time),
            state_names=("T_in", "T_out"),
            constraints=(
                # T_in: the heater's own outlet. Never measured -- see the module docstring.
                Constraint(name="T_in_envelope", signal="T_in", lo=-np.inf, hi=float(T_in_max)),
                Constraint(name="T_out_envelope", signal="T_out", lo=float(T_out_lo), hi=float(T_out_hi)),
            ),
            initial_u=(float(nominal_u),),
            description=(
                "A trace-heated transport line. TIC-401 sets the heater duty at the pipe "
                "inlet; TT-401, at the far end of a long, purely-convective transport run, "
                "is the only instrumented temperature. Throughput is set by an upstream pump "
                "this loop does not own, so the transport delay is not fixed -- it is line "
                "volume divided by current flow, and flow changes during operation. The "
                "heater's own outlet is not instrumented but shares the line's thermal "
                "degradation limit."
            ),
        )

        self._T1 = 0.0
        self._T_out = 0.0
        self._cum_vol = 0.0
        self._buffer: list[tuple[float, float]] = []
        self._t = 0.0

    # -- protocol ---------------------------------------------------------------

    def reset(self, seed: int) -> Observation:
        rng = np.random.default_rng((seed, 0xDEADF10))
        self._params = self._draw_params(rng)
        u0 = float(self.spec.initial_actuation()[0])
        T0 = steady_state(self._params, u0)
        self._T1 = T0
        self._T_out = T0
        self._cum_vol = 0.0
        # Seed the transport history as if the line has always been full of T0: one anchor
        # far enough in the past that no reachable lookback in this run runs off the end of
        # it, plus the t=0 sample itself. Both read T0, so any lookup before real post-reset
        # volume has displaced a full line's worth returns T0 exactly -- see
        # ``_advance_transport`` and the conservation test in the test suite.
        self._buffer = [(-1.0e18, T0), (0.0, T0)]
        self._t = 0.0
        return self._observe()

    def step(self, u: NDArray[np.float64]) -> Observation:
        v = as_array(u, 1, "deadtime_process control")
        duty = float(v[0])
        dt = self.spec.sample_time / self._substeps
        T1 = self._T1
        T1_ss = self._params.T_supply + self._params.K_h * duty
        T_out = self._T_out
        for _ in range(self._substeps):
            k1 = self._dT1(T1, T1_ss)
            k2 = self._dT1(T1 + 0.5 * dt * k1, T1_ss)
            k3 = self._dT1(T1 + 0.5 * dt * k2, T1_ss)
            k4 = self._dT1(T1 + dt * k3, T1_ss)
            T1 = T1 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            T_out = self._advance_transport(T1, dt)
        self._T1 = T1
        self._T_out = T_out
        self._t += self.spec.sample_time
        return self._observe()

    def audit(self) -> Audit:
        x = np.array([self._T1, self._T_out], dtype=float)
        violated = tuple(
            c.name
            for c in self.spec.constraints
            if c.violated(float(x[self.spec.state_names.index(c.signal)]))
        )
        theta = self._params.V_line / self._params.q if self._params.q > 0 else float("inf")
        return Audit(t=self._t, x=x, violations=violated, notes={"theta_s": theta, "q": self._params.q})

    def set_params(self, overrides: Mapping[str, float]) -> None:
        """Apply a mid-run parameter change. This is how a rate change lands: a scheduler
        raising or lowering throughput steps ``q``, which steps the transport delay with it.
        """
        unknown = set(overrides) - {f.name for f in DeadtimeProcessParams.__dataclass_fields__.values()}
        if unknown:
            raise ValueError(f"deadtime_process has no parameter(s) {sorted(unknown)}")
        self._params = replace(self._params, **{k: float(v) for k, v in overrides.items()})

    def close(self) -> None:
        return None

    # -- internals ---------------------------------------------------------------

    @property
    def params(self) -> DeadtimeProcessParams:
        return self._params

    def nominal_params(self) -> DeadtimeProcessParams:
        """The *un-perturbed* parameters -- what a mismatch-tier brief publishes, deliberately
        not what :meth:`reset` drew."""
        return self._nominal

    def _draw_params(self, rng: np.random.Generator) -> DeadtimeProcessParams:
        if not self._mismatch:
            return self._nominal
        drawn = {}
        for name, rel_std in self._mismatch.items():
            base = getattr(self._nominal, name)
            # Log-normal: multiplicative, strictly positive -- the right shape for a volume,
            # a flow, a time constant or a gain, none of which can go negative.
            drawn[name] = float(base * np.exp(rng.normal(0.0, float(rel_std))))
        return replace(self._nominal, **drawn)

    def _dT1(self, T1: float, T1_ss: float) -> float:
        return (T1_ss - T1) / self._params.tau_h

    def _advance_transport(self, T1_new: float, dt_sub: float) -> float:
        """Advance the volume-indexed transport history by one substep and read back the
        line's outlet.

        The buffer is a list of ``(cumulative_volume, temperature)`` samples, one appended
        per substep, in strictly increasing volume order. The parcel now at the outlet is the
        one that entered when cumulative volume was ``V_now - V_line`` ago; interpolating
        between the two samples that bracket that volume is what keeps the delay continuous
        in ``q`` instead of snapping between whole substeps. Buffer entries older than the
        bracket are pruned every call, so its length stays O(1) rather than growing with the
        run -- the search for the bracket, and the prune, are both bounded by how many
        substeps have elapsed since the target last crossed a sample, which is one, except
        right after ``q`` changes.
        """
        q = self._params.q
        V_line = self._params.V_line
        self._cum_vol += q * dt_sub
        self._buffer.append((self._cum_vol, T1_new))
        target = self._cum_vol - V_line

        while len(self._buffer) >= 3 and self._buffer[1][0] <= target:
            self._buffer.pop(0)

        lo_cv, lo_T = self._buffer[0]
        hi_cv, hi_T = lo_cv, lo_T
        for cv, T in self._buffer[1:]:
            if cv <= target:
                lo_cv, lo_T = cv, T
            else:
                hi_cv, hi_T = cv, T
                break
        else:
            hi_cv, hi_T = lo_cv, lo_T

        if hi_cv <= lo_cv:
            return lo_T
        frac = (target - lo_cv) / (hi_cv - lo_cv)
        return lo_T + frac * (hi_T - lo_T)

    def _observe(self) -> Observation:
        return Observation(
            t=self._t,
            y=np.array([self._T_out], dtype=float),
            quality=np.ones(1, dtype=bool),
        )


# -- registry hooks -------------------------------------------------------------

PLANT_ID = "deadtime_process"


def build(cfg):
    """Construct a DeadtimeProcess from a task file's ``plant:`` block."""
    kwargs = dict(cfg)
    params = None
    if overrides := kwargs.pop("params", None):
        params = DeadtimeProcessParams(**overrides)
    return DeadtimeProcess(params, **kwargs)

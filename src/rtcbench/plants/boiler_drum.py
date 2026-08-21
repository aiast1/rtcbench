"""Boiler steam-drum level: shrink, swell, and a right-half-plane zero.

A natural-circulation drum boiler. Feedwater enters the drum, mixes with the water the
downcomers carry to the riser tubes, the risers boil it, and the two-phase mixture returns
to the drum where the steam separates and leaves for the turbine. The one signal a level
loop gets is the *indicated* drum level, which is not the water inventory: it is the height
of a two-phase surface whose steam content moves faster than the mass ever does.

That is the pathology this plant exists to exercise, and it is **inverse response**:

* **Shrink.** More feedwater is colder water. It subcools the downcomer flow, so the first
  stretch of riser tube is spent re-heating the water back to saturation instead of boiling
  it. The riser void collapses, the risers pull water out of the drum to replace it, and the
  indicated level *falls* — while the mass balance says it must eventually rise. A level
  controller that trusts the transmitter opens the valve further, deepens the shrink, and
  oscillates.
* **Swell.** A step increase in steam demand drops drum pressure faster than the firing rate
  can follow. Lower pressure means lighter steam, the riser void expands and the drum bubbles
  flash, and the level *rises* — at the exact moment the drum is losing mass. A controller
  that trusts the transmitter cuts feedwater into a boiler that is emptying.

Both are physics here, not a lead term bolted onto a first-order lag. The level is assembled
from a mass inventory and a void volume that are separate states with separate dynamics; the
right-half-plane zero is what falls out of adding them, and it moves with the operating point,
the feedwater temperature and the circulation rate. Nothing in this file computes a zero.

Two further traps are deliberate:

* **The trip that matters is not measured.** The low trip is on the *water inventory* — the
  level the drum would show if every bubble in the system collapsed. That is what starves
  the tubes, and no transmitter on this unit reads it. The indicated level reads above it
  during a swell, which is precisely when the inventory is draining. The high trip
  (carryover, water into the steam line) *is* on the indicated level.
* **Three-element control is available and is the intended answer.** Steam flow and feedwater
  flow are both instrumented. A single-element PI on level is bandwidth-limited by the RHP
  zero and cannot do better than the reference anchor; a controller that feeds the measured
  steam flow forward into the feedwater demand does not have to wait for the level to tell it
  the truth, and can beat it. That is the industrial answer to shrink and swell, and this
  task pays for it.

Model
-----
Structure follows K.J. Åström and R.D. Bell, "Drum-boiler dynamics", *Automatica* 36(3),
363-378 (2000): a global mass balance and a global energy balance solved together for drum
pressure and total water volume, a riser steam-mass balance carrying the exit quality, a
drum steam-volume state with a residence time, and the level assembled as

    level ~ water in the drum + steam bubbles under the surface,
    water in the drum = total water - downcomers - risers*(1 - void fraction)

which is the paper's equations (11)-(13) in substance. The average riser void fraction as a
function of exit quality is the paper's no-slip, linear-quality-profile expression

    alpha_v = rho_w/(rho_w - rho_s) * [1 - ln(1 + k*alpha_r)/(k*alpha_r)],  k = (rho_w-rho_s)/rho_s

**Where this deviates from the paper, and why.** Åström and Bell assume the drum water sits
at saturation and account for the subcooled feedwater as condensation of drum bubbles alone.
That reproduces shrink qualitatively, but at a realistic drum bubble hold-up the effect is a
couple of millimetres over a few seconds -- invisible next to the mass integration, and far
too weak to be the pathology this task is about. This model instead carries the drum water
enthalpy as a state (a mixing volume near the downcomer offtake, which is where the feedwater
distributor discharges) and charges the resulting subcooling to the risers, where the void
volume is twenty times larger. That is the non-boiling-length effect, standard evaporator
physics, and it makes the steady state *more* consistent rather than less: at equilibrium the
subcooling duty q_dc*(h_w - h_d) equals q_f*(h_w - h_fw) identically, so riser steam
production equals steam demand exactly with no tuning constant.

The firing side is modelled as a boiler master that is **not** part of the benchmark: fuel
demand is feedforward from both flows plus a proportional trim on pressure, through a
first-order combustion/mill lag. It is on automatic, it is deliberately slow, and its lag is
what makes a load change produce a real pressure sag and therefore a real swell. It carries
no integral action, so an upset it cannot see -- a feedwater heater trip -- leaves a standing
pressure droop, exactly as a P-only master does.

Water and steam saturation properties are linear fits in pressure over roughly 70-100 bar,
taken through the IAPWS saturation values at 80 and 90 bar (rho_w 722.5/705.2 kg/m3,
rho_s 42.5/48.8 kg/m3, h_w 1317.1/1363.7 kJ/kg, h_s 2758.0/2742.1 kJ/kg, t_s 295.1/303.4 C).
These fits are mine; the paper uses global fits over a wider range, which this task does not
need because the master holds pressure within a few bar of 85.

Geometry is a mid-size natural-circulation utility drum boiler of the paper's class (~50 kg/s
steam at 85 bar, ~86 MW): drum, downcomer and riser volumes, drum area at the level, and
metal mass are of the paper's order and are stated in :class:`DrumParams` rather than being
claimed as measurements of any particular unit. The circulation coefficient is set from a
circulation ratio of 8 at the duty point.

Units: pressure bar, volume m3, mass flow kg/s, enthalpy kJ/kg, heat kW, level mm.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import log1p, sqrt
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from ..plant import Audit, Channel, Constraint, Observation, PlantSpec, as_array

# -- saturated water/steam, linearized about 85 bar -----------------------------

P_REF = 85.0
RHO_W0, D_RHO_W = 713.8, -1.73     # kg/m3, kg/m3/bar
RHO_S0, D_RHO_S = 45.6, 0.63
H_W0, D_H_W = 1340.4, 4.66         # kJ/kg, kJ/kg/bar
H_S0, D_H_S = 2750.1, -1.59
T_S0, D_T_S = 299.3, 0.83          # degC, degC/bar


def rho_w(p: float) -> float:
    """Saturated water density, kg/m3."""
    return RHO_W0 + D_RHO_W * (p - P_REF)


def rho_s(p: float) -> float:
    """Saturated steam density, kg/m3."""
    return RHO_S0 + D_RHO_S * (p - P_REF)


def h_w(p: float) -> float:
    """Saturated water enthalpy, kJ/kg."""
    return H_W0 + D_H_W * (p - P_REF)


def h_s(p: float) -> float:
    """Saturated steam enthalpy, kJ/kg."""
    return H_S0 + D_H_S * (p - P_REF)


def void_fraction(alpha_r: float, rw: float, rs: float) -> float:
    """Average riser void fraction for exit steam quality ``alpha_r``.

    No slip, quality rising linearly along the heated length — Åström & Bell's expression.
    The series limit is used near zero quality, where ``ln(1+x)/x`` loses its digits.
    """
    k = (rw - rs) / rs
    x = k * max(alpha_r, 0.0)
    core = (0.5 * x - x * x / 3.0) if x < 1e-6 else (1.0 - log1p(x) / x)
    return min((rw / (rw - rs)) * core, 0.98)


def dvoid_dalpha(alpha_r: float, rw: float, rs: float) -> float:
    """d(void fraction)/d(exit quality). Series limit near zero for the same reason."""
    k = (rw - rs) / rs
    x = k * max(alpha_r, 1e-12)
    core = (0.5 - 2.0 * x / 3.0) if x < 1e-4 else (log1p(x) / (x * x) - 1.0 / (x * (1.0 + x)))
    return max((rw / (rw - rs)) * k * core, 1e-6)


@dataclass(frozen=True)
class DrumParams:
    """Geometry, circulation, and the two flow devices bracketing the drum."""

    # -- geometry
    V_d: float = 40.0
    """Drum volume, m3."""
    V_dc: float = 12.0
    """Downcomer volume, m3 — always water-filled."""
    V_r: float = 50.0
    """Riser volume, m3. The big void store, and therefore the shrink/swell lever."""
    A_d: float = 20.0
    """Drum free surface area at the level, m2."""
    V_wd0: float = 20.0
    """Drum water volume at the normal water level, m3 — the level datum."""
    V_mix: float = 3.0
    """Effective mixing volume between the feedwater distributor and the downcomer
    offtake, m3. Small because the distributor discharges next to the offtake: this sets
    how fast cold feedwater reaches the risers, i.e. how fast the shrink arrives."""

    m_t: float = 300000.0
    """Metal mass of drum, risers and downcomers, kg."""
    C_p: float = 0.55
    """Metal specific heat, kJ/(kg K). With m_t this dominates the pressure inertia."""

    # -- two-phase transport
    C_dc: float = 0.8527
    """Natural-circulation coefficient: q_dc = C_dc*sqrt((rho_w-rho_s)*alpha_v*rho_w).
    Set for a circulation ratio of 8 at the duty point."""
    T_d: float = 4.0
    """Residence time of steam bubbles under the drum level, s."""

    # -- feedwater
    k_fw: float = 18.78
    """Feedwater valve coefficient: q_f = k_fw*(u/100)*sqrt(p_pump - p). Sized for 50 kg/s
    at 45% open and 85 bar."""
    p_pump: float = 120.0
    """Feedwater pump discharge pressure, bar. The valve therefore passes more water when
    drum pressure sags, which is real, is why the loop gain is not constant, and is a
    positive feedback: filling the drum sags pressure, which fills it faster. The pump head
    sets how strong that is, and 35 bar across the regulating valve is what keeps it weaker
    than the master's pressure trim."""
    h_fw: float = 852.0
    """Feedwater enthalpy, kJ/kg (~200 degC). Drop it and the shrink gets worse — that is
    what a feedwater heater trip does."""

    # -- steam demand
    k_turbine: float = 0.588
    """Choked-nozzle constant: q_s = k_turbine*turbine_valve*p (kg/s per bar)."""
    turbine_valve: float = 1.0
    """Turbine governor valve, 1.0 = full load. The load disturbance moves this."""

    # -- boiler master (on automatic, not part of the benchmark)
    tau_fire: float = 50.0
    """Fuel/mill/furnace lag, s. This is what makes a load change swell the drum."""
    K_fire: float = 8000.0
    """Pressure trim gain, kW/bar. Large enough to dominate the pressure runaway that a
    drum losing inventory at constant firing would otherwise be."""
    T_fire_i: float = 400.0
    """Reset time on the pressure trim, s. Slow — a master that chased pressure hard would
    fight the turbine — but present, so a sustained flow imbalance does not leave a standing
    pressure offset quietly bleeding the drum on top of whatever the level loop is doing."""
    Q_max: float = 200000.0
    """Maximum firing rate, kW."""


NOMINAL = DrumParams()

STATE_NAMES = (
    "p", "V_wt", "alpha_r", "h_d", "V_sd", "Q", "I_fire",
    "l_ind", "l_inv", "q_f", "q_s", "alpha_v",
)
"""Seven integrated states, then five derived signals carried for the audit and the trend."""


class BoilerDrum:
    """Drum-boiler level dynamics, integrated with fixed-step RK4.

    Fixed-step on purpose: an adaptive solver makes a trajectory depend on floating-point
    accidents in its error controller, and every reproducibility claim in RTCbench rests on
    a plant being a pure function of ``(seed, control sequence)``.
    """

    def __init__(
        self,
        params: DrumParams | None = None,
        *,
        sample_time: float = 5.0,
        substeps: int = 10,
        nominal_u: float = 45.0,
        level_trip_low: float = -250.0,
        level_trip_high: float = 250.0,
        p_lo: float = 70.0,
        p_hi: float = 100.0,
        mismatch: Mapping[str, float] | None = None,
    ) -> None:
        self._nominal = params or NOMINAL
        self._params = self._nominal
        self._substeps = int(substeps)
        self._nominal_u = float(nominal_u)
        self._mismatch = dict(mismatch or {})

        measurements = (
            Channel(tag="LT-101", name="drum level, indicated", unit="mm", lo=-500.0, hi=500.0),
            Channel(tag="PT-102", name="drum pressure", unit="bar", lo=60.0, hi=110.0),
            Channel(tag="FT-103", name="main steam flow", unit="kg/s", lo=0.0, hi=100.0),
            Channel(tag="FT-104", name="feedwater flow", unit="kg/s", lo=0.0, hi=100.0),
        )
        actuators = (
            Channel(tag="FCV-201", name="feedwater valve", unit="%", lo=0.0, hi=100.0),
        )

        self.spec = PlantSpec(
            plant_id="boiler_drum",
            measurements=measurements,
            actuators=actuators,
            sample_time=float(sample_time),
            state_names=STATE_NAMES,
            constraints=(
                # Low water is on the INVENTORY, not on the transmitter. This is the trip
                # that wrecks tubes and it is the one nobody can see.
                Constraint(name="low_water", signal="l_inv", lo=float(level_trip_low)),
                # Carryover is on the indicated level: it is the two-phase surface that
                # reaches the separators, not the collapsed inventory.
                Constraint(name="carryover", signal="l_ind", hi=float(level_trip_high)),
                Constraint(name="pressure_envelope", signal="p", lo=float(p_lo), hi=float(p_hi)),
            ),
            initial_u=(float(nominal_u),),
            description=(
                "Natural-circulation drum boiler at ~85 bar. The feedwater valve is the only "
                "actuator; firing is on a boiler master outside your loop. The level "
                "transmitter reads a two-phase surface, so it shrinks when cold feedwater "
                "collapses the riser void and swells when a load increase drops pressure. "
                "The low-water trip is on the water inventory, which is not instrumented; "
                "the carryover trip is on the indicated level, which is. Steam flow and "
                "feedwater flow are both measured."
            ),
        )

        self._x = np.zeros(7, dtype=float)
        self._t = 0.0
        self._valve = float(nominal_u)
        """Last applied valve position — the feedwater flow transmitter reads what the
        valve actually passed, so it has to be remembered between steps."""
        # Datums and commissioned master constants, all fixed at reset.
        self._V_wt0 = 0.0
        self._ref_ind = 0.0
        self._p_set = P_REF
        self._h_fw_ff = self._nominal.h_fw

    # -- protocol ---------------------------------------------------------------

    def reset(self, seed: int) -> Observation:
        rng = np.random.default_rng((seed, 0xB0117E))
        self._params = self._draw_params(rng)
        p = self._params
        x0, V_wt0, ref_ind = equilibrium(p, self._nominal_u)
        self._x = x0
        self._V_wt0 = V_wt0
        self._ref_ind = ref_ind
        self._p_set = float(x0[0])
        self._h_fw_ff = p.h_fw     # what the master was commissioned with
        self._valve = self._nominal_u
        self._t = 0.0
        return self._observe()

    def step(self, u: NDArray[np.float64]) -> Observation:
        v = as_array(u, 1, "boiler_drum control")
        valve = float(v[0])
        self._valve = valve
        dt = self.spec.sample_time / self._substeps
        x = self._x
        for _ in range(self._substeps):
            k1 = self._dx(x, valve)
            k2 = self._dx(x + 0.5 * dt * k1, valve)
            k3 = self._dx(x + 0.5 * dt * k2, valve)
            k4 = self._dx(x + dt * k3, valve)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            x = self._clamp(x)
        self._x = x
        self._t += self.spec.sample_time
        return self._observe()

    def audit(self) -> Audit:
        x = self._full_state()
        violated = tuple(
            c.name for c in self.spec.constraints if c.violated(float(x[STATE_NAMES.index(c.signal)]))
        )
        return Audit(t=self._t, x=x, violations=violated)

    def set_params(self, overrides: Mapping[str, float]) -> None:
        """Apply a parameter change mid-run — this is how disturbances land.

        ``turbine_valve`` is a load change; ``h_fw`` is a feedwater heater trip; ``C_dc``
        is a fouled or gas-bound downcomer. The boiler master keeps its commissioned
        feedforward constants, so an upset it was not told about shows up as a pressure
        droop as well as a level excursion.
        """
        unknown = set(overrides) - set(DrumParams.__dataclass_fields__)
        if unknown:
            raise ValueError(f"boiler_drum has no parameter(s) {sorted(unknown)}")
        self._params = replace(self._params, **{k: float(v) for k, v in overrides.items()})

    def close(self) -> None:
        return None

    # -- internals --------------------------------------------------------------

    @property
    def params(self) -> DrumParams:
        return self._params

    def nominal_params(self) -> DrumParams:
        """The *un-perturbed* parameters — what a mismatch-tier brief publishes.
        Deliberately not what :meth:`reset` drew."""
        return self._nominal

    def _draw_params(self, rng: np.random.Generator) -> DrumParams:
        if not self._mismatch:
            return self._nominal
        drawn = {}
        for name, rel_std in self._mismatch.items():
            base = getattr(self._nominal, name)
            # Log-normal: multiplicative, strictly positive, symmetric in ratio terms —
            # the right shape for areas, volumes, valve gains and time constants.
            drawn[name] = float(base * np.exp(rng.normal(0.0, float(rel_std))))
        return replace(self._nominal, **drawn)

    def _clamp(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Keep the integrator inside the region where the model means anything.

        This is physics, not constraint evasion: the level trips are on ``l_inv`` and
        ``l_ind``, which are free to run as far past their limits as the boiler takes them.
        By the time water volume reaches this floor the low-water trip has long fired.
        """
        p = self._params
        x[0] = min(max(x[0], 20.0), 160.0)                      # pressure
        x[1] = min(max(x[1], p.V_dc + 0.5), p.V_dc + p.V_r + p.V_d - 1.0)
        x[2] = min(max(x[2], 1e-6), 0.9)                        # riser exit quality
        x[3] = min(max(x[3], 100.0), h_w(x[0]))                 # drum water enthalpy
        x[4] = max(x[4], 0.0)                                   # bubble volume
        x[5] = min(max(x[5], 0.0), p.Q_max)                     # firing
        x[6] = min(max(x[6], -p.Q_max), p.Q_max)                # master reset, anti-windup
        return x

    def _dx(self, x: NDArray[np.float64], valve: float) -> NDArray[np.float64]:
        p_ = self._params
        p, V_wt, a_r, h_d, V_sd, Q, I_fire = (float(v) for v in x)

        rw, rs = rho_w(p), rho_s(p)
        hw, hs = h_w(p), h_s(p)
        hc = hs - hw
        V_t = p_.V_d + p_.V_dc + p_.V_r
        V_st = max(V_t - V_wt, 1e-3)

        # -- flows across the boundary
        q_f = p_.k_fw * (max(valve, 0.0) / 100.0) * sqrt(max(p_.p_pump - p, 0.0))
        q_s = p_.k_turbine * p_.turbine_valve * p

        # -- circulation and boiling
        a_v = void_fraction(a_r, rw, rs)
        q_dc = p_.C_dc * sqrt(max((rw - rs) * a_v * rw, 0.0))
        sub = max(hw - h_d, 0.0)                     # downcomer subcooling, kJ/kg
        q_evap = max((Q - q_dc * sub) / hc, 0.0)     # riser steam production, kg/s

        # -- global mass and energy balance, solved together for dp and dV_wt
        e11 = rw - rs
        e12 = V_st * D_RHO_S + V_wt * D_RHO_W
        e21 = rw * hw - rs * hs
        e22 = (
            V_wt * (hw * D_RHO_W + rw * D_H_W)
            + V_st * (hs * D_RHO_S + rs * D_H_S)
            - 100.0 * V_t                            # d(p*V)/dp, bar -> kJ/m3
            + p_.m_t * p_.C_p * D_T_S
        )
        b1 = q_f - q_s
        b2 = Q + q_f * p_.h_fw - q_s * hs
        det = e11 * e22 - e12 * e21
        dV_wt = (b1 * e22 - e12 * b2) / det
        dp = (e11 * b2 - e21 * b1) / det

        # -- riser steam mass balance: d(rho_s*alpha_v*V_r)/dt = q_evap - alpha_r*q_dc
        d_a_r = (q_evap - a_r * q_dc - a_v * p_.V_r * D_RHO_S * dp) / (
            rs * p_.V_r * dvoid_dalpha(a_r, rw, rs)
        )

        # -- drum mixing volume: feedwater against saturated water off the risers
        d_h_d = (q_f * (p_.h_fw - h_d) + q_dc * (1.0 - a_r) * (hw - h_d)) / (rw * p_.V_mix)

        # -- bubbles under the level: in from the risers, out through the surface
        d_V_sd = ((a_r * q_dc - rs * V_sd / p_.T_d) - V_sd * D_RHO_S * dp) / rs

        # -- boiler master: flow feedforward + PI pressure trim, through the firing lag
        trim = p_.K_fire * (self._p_set - p)
        Q_dem = q_s * hs - q_f * self._h_fw_ff + trim + I_fire
        saturated = Q_dem > p_.Q_max or Q_dem < 0.0
        Q_dem = min(max(Q_dem, 0.0), p_.Q_max)
        dQ = (Q_dem - Q) / p_.tau_fire
        dI = 0.0 if saturated else trim / p_.T_fire_i

        return np.array([dp, dV_wt, d_a_r, d_h_d, d_V_sd, dQ, dI], dtype=float)

    def _levels(self) -> tuple[float, float, float]:
        """(indicated level mm, inventory level mm, riser void fraction)."""
        p_ = self._params
        p, V_wt, a_r = float(self._x[0]), float(self._x[1]), float(self._x[2])
        V_sd = float(self._x[4])
        a_v = void_fraction(a_r, rho_w(p), rho_s(p))
        V_wd = V_wt - p_.V_dc - p_.V_r * (1.0 - a_v)
        l_ind = 1000.0 * (V_wd + V_sd - self._ref_ind) / p_.A_d
        l_inv = 1000.0 * (V_wt - self._V_wt0) / p_.A_d
        return l_ind, l_inv, a_v

    def _flows(self, valve: float | None = None) -> tuple[float, float]:
        p_ = self._params
        p = float(self._x[0])
        # The measured feedwater flow is what the last applied valve position produced.
        v = self._valve if valve is None else valve
        q_f = p_.k_fw * (max(v, 0.0) / 100.0) * sqrt(max(p_.p_pump - p, 0.0))
        q_s = p_.k_turbine * p_.turbine_valve * p
        return q_f, q_s

    def _full_state(self) -> NDArray[np.float64]:
        l_ind, l_inv, a_v = self._levels()
        q_f, q_s = self._flows()
        return np.array([*self._x, l_ind, l_inv, q_f, q_s, a_v], dtype=float)

    def _observe(self) -> Observation:
        l_ind, _l_inv, _a_v = self._levels()
        q_f, q_s = self._flows()
        return Observation(
            t=self._t,
            y=np.array([l_ind, float(self._x[0]), q_s, q_f], dtype=float),
            quality=np.ones(4, dtype=bool),
        )


def equilibrium(p_: DrumParams, valve: float) -> tuple[NDArray[np.float64], float, float]:
    """Exact steady state for a held feedwater valve position.

    Solved rather than simulated, so a scenario starts stationary to the bit. Three scalar
    steps, each a bisection on a monotone function:

    1. Pressure balances the two flow devices: ``k_fw*(u/100)*sqrt(p_pump - p) = k_t*v*p``.
       The feedwater valve stiffens as pressure rises, the turbine passes more as pressure
       rises, so the crossing is unique.
    2. Circulation carries the steam it makes: ``alpha_r*q_dc(alpha_r) = q_f``.
    3. Everything else is algebra. The subcooling that follows from the drum mixing balance
       satisfies ``q_dc*(h_w - h_d) = q_f*(h_w - h_fw)`` identically, which is why riser
       steam production comes out equal to steam demand with nothing left over.
    """
    u = max(valve, 0.0) / 100.0

    def flow_gap(p: float) -> float:
        return p_.k_fw * u * sqrt(max(p_.p_pump - p, 0.0)) - p_.k_turbine * p_.turbine_valve * p

    p = _bisect(flow_gap, 5.0, p_.p_pump - 1e-9)
    rw, rs = rho_w(p), rho_s(p)
    hw, hs = h_w(p), h_s(p)
    hc = hs - hw

    q = p_.k_turbine * p_.turbine_valve * p          # = q_f = q_s at equilibrium

    def circulation_gap(a: float) -> float:
        return q - a * p_.C_dc * sqrt(max((rw - rs) * void_fraction(a, rw, rs) * rw, 0.0))

    a_r = _bisect(circulation_gap, 1e-8, 0.9)
    a_v = void_fraction(a_r, rw, rs)
    q_dc = p_.C_dc * sqrt(max((rw - rs) * a_v * rw, 0.0))

    sub = q * (hw - p_.h_fw) / (q + q_dc * (1.0 - a_r))
    h_d = hw - sub
    V_sd = p_.T_d * a_r * q_dc / rs
    V_wt = p_.V_wd0 + p_.V_dc + p_.V_r * (1.0 - a_v)
    Q = q * (hs - p_.h_fw)

    # The master's reset starts empty: its feedforward alone balances the duty point.
    x0 = np.array([p, V_wt, a_r, h_d, V_sd, Q, 0.0], dtype=float)
    return x0, V_wt, p_.V_wd0 + V_sd


def _bisect(f, lo: float, hi: float, iterations: int = 200) -> float:
    """Fixed-iteration bisection. Fixed on purpose: a tolerance-based loop makes the number
    of iterations, and therefore the last bits of the answer, depend on the arithmetic."""
    f_lo = f(lo)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0.0) == (f_lo > 0.0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# -- registry hooks -------------------------------------------------------------

PLANT_ID = "boiler_drum"


def build(cfg):
    """Construct a BoilerDrum from a task file's ``plant:`` block."""
    kwargs = dict(cfg)
    params = NOMINAL
    if overrides := kwargs.pop("params", None):
        params = replace(params, **{k: float(v) for k, v in overrides.items()})
    return BoilerDrum(params, **kwargs)

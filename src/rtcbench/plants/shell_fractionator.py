"""The Shell heavy oil fractionator -- Prett & Morari's Shell Standard Control Problem.

A three-product side-draw column (top draw, side draw, bottoms) whose three product-quality
loops are driven through a fully-populated 3x3 transfer-function matrix: every input affects
every output, each with its *own* independent gain, time constant and dead time. Two measured
disturbances (upstream reflux duties) enter the same three outputs through their own 3x2
matrix. This is the archetypal "constrained, heavily interacting, deadtime-ridden
multivariable problem" that motivated model-predictive control in the process industries, and
it remains the standard MPC textbook benchmark for exactly that reason.

    y1  top end-point composition     u1  top draw
    y2  side end-point composition    u2  side draw
    y3  bottoms reflux temperature    u3  bottoms reflux duty
    d1, d2  measured disturbances (upper / intermediate reflux duty -- see the naming note
            below): heat duties supplied by circulating loops this plant does not own.

All six signals are *deviation variables* around the plant's commissioned operating point, so
``u = 0`` is the trim duty for every actuator and ``y = 0`` is on-spec for every product --
which is also why the model's initial condition is always the origin regardless of the
mismatch draw (see the task file's mismatch-sizing note; this plant does not have four_tank's
"parameter draw moves the starting point toward a limit" problem at all).

**What breaks a naive controller here is not the pairing -- it is the deadtime dominance
combined with a manipulated variable whose own gains are enormous relative to its allowed
range.** ``K33 = 7.20`` on ``u3 in [-0.5, 0.5]``: half of u3's authority alone is worth more
than seven full units of y3, against a hard floor at y3 = -0.5. A controller that reacts to
u3's own loop error without respecting how little headroom that gain leaves, or that ignores
the 14-32 minute dead times on every other channel feeding y3, drives the bottoms reflux
temperature through its floor before the correction it already committed to has even arrived.
Steady-state interaction is comparatively tame: the published gain matrix's RGA (computed
below, and cross-checked against it -- see "Numeric provenance") has exactly one admissible
all-positive pairing, u1-y1 / u2-y2 / u3-y3, which is what this module and its task ship as the
default. The difficulty is dynamic, not structural.

Source
------
D.M. Prett and M. Morari (eds.), *The Shell Process Control Workshop*, Butterworths, 1987 --
the original problem statement. The specific first-order-plus-deadtime gain/delay matrix
implemented below (``G(s)`` and ``Gd(s)``, in minutes) is the one reproduced from D.M. Prett
and C.E. Garcia, *Fundamental Process Control*, Butterworths, 1988, in:

* Jusagemal, Setiawan & Setiyono, "Analisis dan Simulasi Shell Heavy Oil Fractionator (SHOF)
  Menggunakan Metode Kontrol PID", TRANSMISI 13(4), 2011, Table 1 -- clean, unambiguous
  transcription of the full 7-output x 5-input SHOF identification matrix.
* Araromi & Sulayman, "Gain Scheduling Control Design for Shell Heavy Oil Fractionator
  Column", Int. J. Energy & Environmental Research 3(1), 2015, Table 1 (same matrix) and
  Table 2 (the published gain-uncertainty structure, used below for the mismatch tier) and
  the constraint list quoted in their Methodology section (MV bounds, MV rate bound, the
  bottoms-reflux-temperature floor, minimum sample time).

Numeric provenance (why these two secondary sources are trusted over a from-scratch
transcription): the steady-state gain matrix K = G(0) implied by the FOPDT table below has
det(K) = 20.85 and an RGA of

    [[ 2.08, -0.73, -0.35],
     [ 3.42,  0.94, -3.36],
     [-4.50,  0.79,  4.71]]

both computed independently here from K alone -- and both match the determinant and RGA
figures Jusagemal et al. report for the identified SHOF column to the precision they publish.
That is a much stronger check than matching digits in a table transcription: an RGA is a
nonlinear function of all nine gains at once, so agreement to 3 significant figures across all
nine entries is not something a mis-transcribed gain could survive by accident.

Measured-disturbance naming caveat: Jusagemal et al. label d1 = intermediate reflux duty,
d2 = upper reflux duty; Araromi & Sulayman label the same two *columns of the same numeric
matrix* the other way around. The gain/delay numbers in each column agree between the two
papers; only which English name attaches to which column differs. This module follows
Araromi & Sulayman's labeling (d1 = upper reflux duty, d2 = intermediate reflux duty) purely
for definiteness -- swapping the label swaps nothing about the physics or the task.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from ..plant import Audit, Channel, Constraint, Observation, PlantSpec, as_array

# -- the published model ---------------------------------------------------------


@dataclass(frozen=True)
class ShellFractionatorParams:
    """Every element of G(s) (process) and Gd(s) (disturbance), FOPDT in minutes:
    ``g_ij(s) = K[i][j] * exp(-L[i][j] s) / (TAU[i][j] s + 1)``.

    Rows are outputs (0=y1 top end-point composition, 1=y2 side end-point composition,
    2=y3 bottoms reflux temperature). G's columns are manipulated inputs (0=u1 top draw,
    1=u2 side draw, 2=u3 bottoms reflux duty); Gd's columns are measured disturbances
    (0=d1, 1=d2 -- see the module docstring's naming caveat). Values are Prett & Garcia's
    published SHOF identification, transcribed from Jusagemal et al. (2011) Table 1.
    """

    K: tuple[tuple[float, float, float], ...] = (
        (4.05, 1.77, 5.88),
        (5.39, 5.72, 6.90),
        (4.38, 4.42, 7.20),
    )
    TAU: tuple[tuple[float, float, float], ...] = (
        (50.0, 60.0, 50.0),
        (50.0, 60.0, 40.0),
        (33.0, 44.0, 19.0),
    )
    L: tuple[tuple[float, float, float], ...] = (
        (27.0, 28.0, 27.0),
        (18.0, 14.0, 15.0),
        (20.0, 22.0, 0.0),
    )

    KD: tuple[tuple[float, float], ...] = ((1.20, 1.44), (1.52, 1.83), (1.14, 1.26))
    TAUD: tuple[tuple[float, float], ...] = ((45.0, 40.0), (25.0, 20.0), (27.0, 32.0))
    LD: tuple[tuple[float, float], ...] = ((27.0, 27.0), (15.0, 15.0), (27.0, 32.0))


P_NOMINAL = ShellFractionatorParams()


def steady_state_gain() -> NDArray[np.float64]:
    """K = G(0): the FOPDT dead time and lag never affect DC gain, so this is just ``K``,
    exposed as an array for the RGA / determinant cross-check in the module docstring and
    in the tests."""
    return np.array(P_NOMINAL.K, dtype=float)


def rga(k: NDArray[np.float64]) -> NDArray[np.float64]:
    """Relative Gain Array, K .* (K^-1)^T -- used only to cross-check the published matrix
    (see the module docstring) and to justify the shipped diagonal pairing."""
    return k * np.linalg.inv(k).T


# -- named-element overrides, for both the mismatch draw and set_params ---------

_FIELD_OF_PREFIX = {"kd": "KD", "taud": "TAUD", "ld": "LD", "k": "K", "tau": "TAU", "l": "L"}
_ELEMENT_NAME = re.compile(r"^(kd|taud|ld|k|tau|l)([1-3])([1-3])$", re.IGNORECASE)
_MATRIX_FIELDS = ("K", "TAU", "L", "KD", "TAUD", "LD")


def _parse_element_name(name: str) -> tuple[str, int, int]:
    m = _ELEMENT_NAME.match(name)
    if not m:
        raise ValueError(
            f"shell_fractionator has no parameter {name!r}; expected one of k/tau/l (G, e.g. "
            "'k12') or kd/taud/ld (Gd, e.g. 'kd31') followed by a 1-based row and column"
        )
    prefix, i, j = m.group(1).lower(), int(m.group(2)) - 1, int(m.group(3)) - 1
    field = _FIELD_OF_PREFIX[prefix]
    ncols = 3 if field in ("K", "TAU", "L") else 2
    if j >= ncols:
        raise ValueError(f"{name!r}: {field} only has {ncols} column(s)")
    return field, i, j


def _apply_element_overrides(
    params: ShellFractionatorParams, overrides: Mapping[str, float]
) -> ShellFractionatorParams:
    rows = {f: [list(row) for row in getattr(params, f)] for f in _MATRIX_FIELDS}
    for name, value in overrides.items():
        field, i, j = _parse_element_name(name)
        rows[field][i][j] = float(value)
    return replace(params, **{f: tuple(tuple(r) for r in v) for f, v in rows.items()})


# -- the plant --------------------------------------------------------------------


class ShellFractionator:
    """The Shell heavy oil fractionator, realized as 15 independent first-order lags (9 for
    G, 6 for Gd) each fed through its own explicit transport-delay buffer, integrated with
    fixed-step RK4. No adaptive solver anywhere: same seed, same control sequence, same
    trajectory, bit for bit.

    A delay is a buffer lookup, not a filter approximation: each of ``u1, u2, u3, d1, d2``
    gets its own ring buffer of *held* (zero-order-hold) values, one push per control period;
    element (i, j)'s dead time is realized by reading that channel's buffer ``L[i][j] /
    sample_time`` samples back before it ever reaches element (i, j)'s own lag ODE. Because
    every published delay here is a whole number of minutes and the default sample time is
    one minute, that lookback is always an exact integer -- no interpolation, no rounding
    error accumulating across a run.
    """

    def __init__(
        self,
        params: ShellFractionatorParams | None = None,
        *,
        sample_time: float = 1.0,
        substeps: int = 10,
        mismatch: Mapping[str, float] | None = None,
    ) -> None:
        self._nominal = params or ShellFractionatorParams()
        self._params = self._nominal
        self._substeps = int(substeps)
        self._mismatch = dict(mismatch or {})

        self.spec = PlantSpec(
            plant_id="shell_fractionator",
            measurements=(
                Channel(tag="AI-101", name="top end-point composition", unit="dev", lo=-0.6, hi=0.6),
                Channel(tag="AI-102", name="side end-point composition", unit="dev", lo=-0.6, hi=0.6),
                Channel(tag="TI-103", name="bottoms reflux temperature", unit="dev", lo=-0.6, hi=0.6),
            ),
            actuators=(
                # Published bound: -0.5 < u_i < 0.5 for i = 1, 2, 3 (Prett & Morari 1987, as
                # quoted by Araromi & Sulayman 2015). u3's headroom against K33=7.20 is what
                # makes this loop the hard one -- see the module docstring.
                Channel(tag="FCV-201", name="top draw", unit="dev", lo=-0.5, hi=0.5),
                Channel(tag="FCV-202", name="side draw", unit="dev", lo=-0.5, hi=0.5),
                Channel(tag="FCV-203", name="bottoms reflux duty", unit="dev", lo=-0.5, hi=0.5),
            ),
            sample_time=float(sample_time),
            state_names=("y1", "y2", "y3"),
            constraints=(
                # Published bound: y7 (bottoms reflux temperature, this plant's y3) > -0.5,
                # one-sided -- there is no product-quality spec on the bottoms draw, only this
                # operating constraint (Araromi & Sulayman 2015, constraint (c)).
                Constraint(name="bottoms_reflux_temperature_floor", signal="y3", lo=-0.5, hi=np.inf),
            ),
            initial_u=(0.0, 0.0, 0.0),
            description=(
                "A three-product side-draw fractionator. FCV-201/202 set the top and side "
                "product draws; FCV-203 sets the bottoms reflux (reboil) duty. AI-101/102 are "
                "the top and side end-point composition analyzers; TI-103 is the bottoms "
                "reflux temperature, which carries no product spec of its own but has a hard "
                "floor. Two upstream circulating-loop heat duties (not manipulated by this "
                "loop) enter all three outputs as measured disturbances. Every input-output "
                "pair has its own gain, time constant and dead time (14-32 minutes)."
            ),
        )

        self._K = self._TAU = self._KD = self._TAUD = None
        self._n = np.zeros((3, 3), dtype=int)
        self._nd = np.zeros((3, 2), dtype=int)
        self._buf_u: list[deque] = []
        self._buf_d: list[deque] = []
        self._d_level = np.zeros(2, dtype=float)
        self._x = np.zeros(15, dtype=float)
        self._y = np.zeros(3, dtype=float)
        self._t = 0.0
        self._sync_arrays()

    # -- protocol ---------------------------------------------------------------

    def reset(self, seed: int) -> Observation:
        rng = np.random.default_rng((seed, 0x5FE11))
        self._params = self._draw_params(rng)
        self._sync_arrays()

        ts = self.spec.sample_time
        self._n = np.array(
            [[max(0, int(round(self._params.L[i][j] / ts))) for j in range(3)] for i in range(3)]
        )
        self._nd = np.array(
            [[max(0, int(round(self._params.LD[i][k] / ts))) for k in range(2)] for i in range(3)]
        )
        depth_u = [int(self._n[:, j].max()) + 1 for j in range(3)]
        depth_d = [int(self._nd[:, k].max()) + 1 for k in range(2)]
        self._buf_u = [deque([0.0] * depth_u[j], maxlen=depth_u[j]) for j in range(3)]
        self._buf_d = [deque([0.0] * depth_d[k], maxlen=depth_d[k]) for k in range(2)]

        self._d_level = np.zeros(2, dtype=float)
        self._x = np.zeros(15, dtype=float)
        self._y = np.zeros(3, dtype=float)
        self._t = 0.0
        return self._observe()

    def step(self, u: NDArray[np.float64]) -> Observation:
        u_req = as_array(u, 3, "shell_fractionator control")
        for j in range(3):
            self._buf_u[j].append(float(u_req[j]))
        for k in range(2):
            self._buf_d[k].append(float(self._d_level[k]))

        u_delay = np.array(
            [[self._buf_u[j][-(int(self._n[i, j]) + 1)] for j in range(3)] for i in range(3)]
        )
        d_delay = np.array(
            [[self._buf_d[k][-(int(self._nd[i, k]) + 1)] for k in range(2)] for i in range(3)]
        )

        dt = self.spec.sample_time / self._substeps
        x = self._x
        for _ in range(self._substeps):
            k1 = self._dx(x, u_delay, d_delay)
            k2 = self._dx(x + 0.5 * dt * k1, u_delay, d_delay)
            k3 = self._dx(x + 0.5 * dt * k2, u_delay, d_delay)
            k4 = self._dx(x + dt * k3, u_delay, d_delay)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self._x = x
        self._t += self.spec.sample_time
        self._y = self._compute_y(x)
        return self._observe()

    def audit(self) -> Audit:
        violated = tuple(
            c.name
            for c in self.spec.constraints
            if c.violated(float(self._y[self.spec.state_names.index(c.signal)]))
        )
        return Audit(t=self._t, x=self._y.copy(), violations=violated)

    def set_params(self, overrides: Mapping[str, float]) -> None:
        """Apply a mid-run change. Two kinds are accepted:

        * ``d1`` / ``d2`` -- step the *held* level of a measured disturbance. This is how a
          disturbance event lands: a circulating-loop heat duty this plant does not own steps
          to a new value, and (like every other signal here) it then reaches the outputs
          through the plant's own delay buffers and lags, not instantaneously.
        * a named G/Gd element (``k12``, ``tau33``, ``kd21``, ...) -- a gain or time-constant
          drift. Delay elements (``l*`` / ``ld*``) are refused here: changing a dead time
          mid-run would mean resizing an in-flight buffer, which is not supported. Delay
          *mismatch* still exists -- it is drawn once at :meth:`reset`, before any buffer is
          sized, exactly like the gain draw.
        """
        d_over: dict[str, float] = {}
        other: dict[str, float] = {}
        for name, value in overrides.items():
            key = name.lower()
            if key in ("d1", "d2"):
                d_over[key] = float(value)
            else:
                other[name] = value

        for name in other:
            field, _, _ = _parse_element_name(name)
            if field in ("L", "LD"):
                raise ValueError(
                    f"{name!r}: transport delays are fixed for the life of an episode "
                    "(changing one mid-run would require resizing its delay buffer); use the "
                    "task's plant.mismatch to perturb delays at reset() instead"
                )
        if other:
            self._params = _apply_element_overrides(self._params, {k: float(v) for k, v in other.items()})
            self._sync_arrays()

        if "d1" in d_over:
            self._d_level[0] = d_over["d1"]
        if "d2" in d_over:
            self._d_level[1] = d_over["d2"]

    def close(self) -> None:
        return None

    # -- introspection, mainly for tests -----------------------------------------

    @property
    def params(self) -> ShellFractionatorParams:
        return self._params

    def nominal_params(self) -> ShellFractionatorParams:
        """The *un-perturbed* parameters -- what a mismatch-tier brief publishes, deliberately
        not what :meth:`reset` drew."""
        return self._nominal

    def delay_samples(self, i: int, j: int, *, disturbance: bool = False) -> int:
        """Transport delay of element (i, j), in whole control periods, as actually used by
        the ODE integration -- exposed so tests assert against the real mechanism rather than
        recomputing it independently."""
        return int((self._nd if disturbance else self._n)[i, j])

    # -- internals ---------------------------------------------------------------

    def _sync_arrays(self) -> None:
        self._K = np.array(self._params.K, dtype=float)
        self._TAU = np.array(self._params.TAU, dtype=float)
        self._KD = np.array(self._params.KD, dtype=float)
        self._TAUD = np.array(self._params.TAUD, dtype=float)

    def _draw_params(self, rng: np.random.Generator) -> ShellFractionatorParams:
        if not self._mismatch:
            return self._nominal
        drawn: dict[str, float] = {}
        for name, rel_std in self._mismatch.items():
            field, i, j = _parse_element_name(name)
            base = getattr(self._nominal, field)[i][j]
            # Log-normal, multiplicative: symmetric in ratio terms, keeps a delay >= 0 exactly
            # (a published zero delay, g33, stays exactly zero: 0 * anything = 0).
            drawn[name] = float(base * np.exp(rng.normal(0.0, float(rel_std))))
        return _apply_element_overrides(self._nominal, drawn)

    def _dx(
        self, x: NDArray[np.float64], u_delay: NDArray[np.float64], d_delay: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        xg = x[:9].reshape(3, 3)
        xd = x[9:].reshape(3, 2)
        dxg = (-xg + self._K * u_delay) / self._TAU
        dxd = (-xd + self._KD * d_delay) / self._TAUD
        return np.concatenate([dxg.reshape(-1), dxd.reshape(-1)])

    def _compute_y(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        xg = x[:9].reshape(3, 3)
        xd = x[9:].reshape(3, 2)
        return xg.sum(axis=1) + xd.sum(axis=1)

    def _observe(self) -> Observation:
        return Observation(t=self._t, y=self._y.copy(), quality=np.ones(3, dtype=bool))


# -- registry hooks -------------------------------------------------------------

PLANT_ID = "shell_fractionator"


def build(cfg):
    """Construct a ShellFractionator from a task file's ``plant:`` block."""
    kwargs = dict(cfg)
    params = None
    if overrides := kwargs.pop("params", None):
        params = _apply_element_overrides(ShellFractionatorParams(), overrides)
    return ShellFractionator(params, **kwargs)

"""T1 oracle — does each plant have the property it is in the suite for?

Every plant in the pack exists to defeat a *specific* naive controller, and its docstring
names which: a right-half-plane zero, a sign-reversing gain, an unstable operating point,
severe ill-conditioning. Those are published, checkable characterisations, and they are what
a reader is really trusting when they read a score.

This is the check nothing else performs. `tests/` verifies each plant against its own author's
expectations, and `integration.py` verifies the arithmetic converges — but a plant can pass
both while quietly *not having the pathology it claims*, at which point its column is
measuring something nobody named.

The measurements here are made through the public `step` interface only: step responses in,
gain matrices and response directions out. No private attributes, no re-implementation of
the physics, no shared code with the plant. That is what makes it an independent check rather
than a restatement.

Why not Cantera for the reacting plants? It was installed and considered. Building a
Cantera CSTR for a lumped liquid-phase reactor means hand-configuring its thermodynamics to
match the same rho/Cp/dH constants the plant already uses, so a transcription error in those
constants would be faithfully reproduced by the "independent" oracle. It would check the
reactor bookkeeping and buy less independence than it appears to. A genuine second
implementation needs a second author; see README.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rtcbench.plants import build_plant


@dataclass
class Spectral:
    plant_id: str
    claim: str
    measured: str
    passed: bool
    detail: dict = field(default_factory=dict)

    def line(self) -> str:
        return (f"  {self.plant_id:<22} {self.claim:<34} {self.measured:<26} "
                f"[{'confirmed' if self.passed else 'NOT CONFIRMED'}]")


def _settle(plant, u, steps):
    for _ in range(steps):
        obs = plant.step(u)
    return obs.y.copy()


def dc_gain(plant_id: str, cfg: dict, *, seed: int = 11, bump: float = 0.04,
            settle: int = 900) -> np.ndarray:
    """Steady-state gain matrix, measured by bumping each actuator in turn.

    Uses only `step`. Each column is (y_settled_up - y_settled_down) / (2 * delta), a central
    difference so a mild nonlinearity does not bias the estimate.
    """
    probe = build_plant({"kind": plant_id, **cfg})
    probe.reset(seed)
    u0 = probe.spec.initial_actuation()
    lo, hi = probe.spec.actuator_lo(), probe.spec.actuator_hi()
    n_y, n_u = probe.spec.n_y, probe.spec.n_u
    probe.close()

    G = np.zeros((n_y, n_u))
    for j in range(n_u):
        delta = bump * (hi[j] - lo[j])
        ys = []
        for sign in (+1, -1):
            p = build_plant({"kind": plant_id, **cfg})
            p.reset(seed)
            u = u0.copy()
            u[j] = float(np.clip(u0[j] + sign * delta, lo[j], hi[j]))
            ys.append(_settle(p, u, settle))
            p.close()
        G[:, j] = (ys[0] - ys[1]) / (2 * delta)
    return G


def rga(G: np.ndarray) -> np.ndarray:
    """Relative gain array. Large off-diagonal magnitudes mean the loops fight each other."""
    return G * np.linalg.pinv(G).T


def step_direction(plant_id: str, cfg: dict, actuator: int, channel: int, *,
                   seed: int = 11, bump: float = 0.15, early: int = 3,
                   settle: int = 900) -> tuple[float, float]:
    """Early and final movement of one output after a step on one input.

    Opposite signs are inverse response — a right-half-plane zero — which is exactly the
    trap a naive controller falls into, because it reacts to the wrong-way excursion.
    """
    p = build_plant({"kind": plant_id, **cfg})
    obs0 = p.reset(seed)
    y0 = obs0.y[channel]
    lo, hi = p.spec.actuator_lo(), p.spec.actuator_hi()
    u = p.spec.initial_actuation()
    u[actuator] = float(np.clip(u[actuator] + bump * (hi[actuator] - lo[actuator]),
                                lo[actuator], hi[actuator]))
    early_y = None
    for k in range(settle):
        y = p.step(u).y[channel]
        if k == early - 1:
            early_y = y
    p.close()
    return float(early_y - y0), float(y - y0)


# -- FOPDT identification, for building a defensible model-based anchor -------------


def fopdt_matrix(plant_id: str, cfg: dict, *, seed: int = 11, bump: float = 0.06,
                 horizon: int = 400) -> dict:
    """Fit gain / time-constant / deadtime per (output, input) pair from step tests.

    Everything here comes through `step`, which is what makes it usable by an ANCHOR: a
    reference controller may only use information a competitor could obtain, and bump tests
    are explicitly what the brief permits. A model lifted from the plant source would not be
    a reference, it would be an oracle.

    Deadtime is taken as the first sample whose movement exceeds a small fraction of the
    final change -- the practical definition, and the one that matters for control, since a
    response you cannot distinguish from noise is a response you cannot act on.
    """
    probe = build_plant({"kind": plant_id, **cfg})
    probe.reset(seed)
    u0 = probe.spec.initial_actuation()
    lo, hi = probe.spec.actuator_lo(), probe.spec.actuator_hi()
    n_y, n_u = probe.spec.n_y, probe.spec.n_u
    dt = probe.spec.sample_time
    probe.close()

    K = np.zeros((n_y, n_u)); TAU = np.zeros((n_y, n_u)); TH = np.zeros((n_y, n_u))
    for j in range(n_u):
        delta = bump * (hi[j] - lo[j])
        p = build_plant({"kind": plant_id, **cfg})
        y0 = p.reset(seed).y.copy()
        u = u0.copy()
        u[j] = float(np.clip(u0[j] + delta, lo[j], hi[j]))
        traj = np.array([p.step(u).y for _ in range(horizon)])
        p.close()

        for i in range(n_y):
            y = traj[:, i] - y0[i]
            final = float(y[-1])
            K[i, j] = final / delta
            if abs(final) < 1e-9:
                TAU[i, j], TH[i, j] = dt, 0.0
                continue
            moved = np.flatnonzero(np.abs(y) > 0.03 * abs(final))
            k_delay = int(moved[0]) if moved.size else 0
            TH[i, j] = k_delay * dt
            # 63.2% of the way to final, measured from the end of the delay.
            reached = np.flatnonzero(np.abs(y) >= 0.632 * abs(final))
            k63 = int(reached[0]) if reached.size else horizon - 1
            TAU[i, j] = max(dt, (k63 - k_delay) * dt)
    return {"gain": K.tolist(), "tau": TAU.tolist(), "theta": TH.tolist()}

"""T2 oracle — is the fixed-step integration converged?

Every plant marches with fixed-step RK4 and a per-task substep count, because reproducibility
requires it: an adaptive solver makes a trajectory depend on floating-point accidents inside
its error controller. The cost of that choice is that nothing checks whether the substep
count is *enough*.

That failure mode is nasty because it does not look like a failure. A stiff plant integrated
too coarsely produces a trajectory that is smooth, plausible, deterministic, reproducible and
wrong, and every test in `tests/` will happily confirm it is stationary at its own wrong
steady state.

**The check is a refinement study**, not a second solver. Run the plant at its shipped substep
count and again at 4x, over an identical control sequence, and measure how far the trajectory
moved. If refining the grid changes the answer, the shipped grid was too coarse.

A first version of this file re-integrated each plant's private `_dx` with scipy's Radau. It
was abandoned: the plants have heterogeneous derivative signatures (`_dx(x, u)`,
`_dx(x, u, d_delay)`, some with no separable RHS at all), so the oracle only worked on the
plants that happened to match, and it reached into private attributes to do it. Refinement
needs nothing but the public `step`, applies to every plant, and tests the actual question.

**What this cannot prove.** It says nothing about whether the equations transcribe the paper
correctly — a wrong model converges just as nicely as a right one. That is the other oracle's
job; see `cantera_cstr.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rtcbench.plants import build_plant

REFINEMENT = 4
"""How much finer the reference march is. RK4 is 4th order, so a converged plant should
improve by ~256x here; anything that moves materially was not converged."""


@dataclass
class Convergence:
    plant_id: str
    max_rel_error: float
    worst_state: str
    substeps: int
    refined: int
    tolerance: float
    passed: bool
    note: str = ""

    def line(self) -> str:
        if self.note:
            return f"  {self.plant_id:<22} {self.note}"
        verdict = "converged" if self.passed else "UNDER-RESOLVED"
        return (f"  {self.plant_id:<22} moved {self.max_rel_error:.2e} on "
                f"{self.worst_state:<10} at {self.substeps}->{self.refined} substeps  "
                f"[{verdict}]")


def _march(plant_id: str, cfg: dict, substeps: int, controls, seed: int) -> np.ndarray:
    plant = build_plant({"kind": plant_id, **{**cfg, "substeps": substeps}})
    plant.reset(seed)
    states = [plant.audit().x.copy()]
    for u in controls:
        plant.step(u)
        states.append(plant.audit().x.copy())
    plant.close()
    return np.array(states)


def check(plant_id: str, cfg: dict, n_steps: int = 40, tolerance: float = 5e-3,
          seed: int = 11) -> Convergence:
    """Refine the substep count and measure how far the trajectory moves."""
    cfg = dict(cfg)
    substeps = int(cfg.get("substeps", 0))
    if not substeps:
        return Convergence(plant_id, float("nan"), "", 0, 0, tolerance, True,
                           note="no substeps setting — nothing to refine")

    probe = build_plant({"kind": plant_id, **cfg})
    probe.reset(seed)
    lo, hi = probe.spec.actuator_lo(), probe.spec.actuator_hi()
    u0 = probe.spec.initial_actuation()
    names = probe.spec.state_names or tuple(f"x{i}" for i in range(len(probe.audit().x)))
    probe.close()

    # Exercise the plant rather than parking it at steady state: a coarse grid can look
    # perfect at equilibrium and fall apart during a transient.
    rng = np.random.default_rng((seed, 0xCAFE))
    controls = [np.clip(u0 + (hi - lo) * 0.12 * rng.normal(size=len(u0)), lo, hi)
                for _ in range(n_steps)]

    coarse = _march(plant_id, cfg, substeps, controls, seed)
    fine = _march(plant_id, cfg, substeps * REFINEMENT, controls, seed)

    scale = np.maximum(np.abs(fine).max(axis=0), 1e-9)
    rel = np.abs(coarse - fine) / scale
    worst = int(np.unravel_index(np.argmax(rel), rel.shape)[1])
    max_rel = float(rel.max())

    return Convergence(
        plant_id=plant_id,
        max_rel_error=max_rel,
        worst_state=names[worst] if worst < len(names) else f"x{worst}",
        substeps=substeps,
        refined=substeps * REFINEMENT,
        tolerance=tolerance,
        passed=max_rel <= tolerance,
    )

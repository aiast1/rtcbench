"""Tests for the open-loop-unstable exothermic CSTR.

This plant earns its place by being unstable, so the load-bearing test is the one that
proves the operating point genuinely does not hold itself. If the reactor quietly returns to
its setpoint with the actuator frozen, the task is a tuning exercise wearing a hazard label,
and the enormous hold-anchor gap it reports would be measuring nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from rtcbench.baselines import Hold, build_reference
from rtcbench.harness import run_ensemble
from rtcbench.plants import build_plant
from rtcbench.task import Task

TASK = Path(__file__).resolve().parents[1] / "tasks" / "unstable_cstr_v1.yaml"
T_INDEX = 1
"""Index of reactor temperature in the true state vector ('Ca', 'T')."""


@pytest.fixture(scope="module")
def task() -> Task:
    return Task.load(TASK)


def _bare_plant():
    """Ideal plant, no instrument layer — a divergence test must not be confounded by
    sensor noise or valve slew."""
    cfg = dict(yaml.safe_load(TASK.read_text(encoding="utf-8"))["plant"])
    return build_plant(cfg)


def _hold_from(plant, x0, steps: int) -> np.ndarray:
    """March with the actuator frozen at its duty point, from a chosen true state."""
    plant._x = np.asarray(x0, dtype=float).copy()
    u = np.asarray(plant.spec.initial_actuation())
    return np.array([plant.step(u).y for _ in range(steps)])


# -- the pathology ---------------------------------------------------------------


def test_the_operating_point_does_not_hold_itself():
    """Start exactly on the operating point, freeze the coolant, and it leaves.

    Measured behaviour, not assumed: T departs 350 K -> 388 K and stays there. The first
    draft of this test demanded a MONOTONE runaway and failed, because the departure
    overshoots to ~448 K before settling. Both are instability; only one is what this plant
    does, and the test now asserts the one that is true.
    """
    plant = _bare_plant()
    plant.reset(0)
    x_op = plant._x_op.copy()
    temps = _hold_from(plant, x_op, 200)[:, 0]

    departure = abs(temps[-1] - x_op[T_INDEX])
    assert departure > 25.0, (
        f"reactor stayed within {departure:.2f} K of its operating point with the actuator "
        "frozen - that point is stable, so this plant is not doing the job it is in the "
        "suite for"
    )
    # It must not come back: a stable point would return after its excursion.
    assert abs(temps[-1] - temps[-20]) < 1.0, "did not settle"
    assert abs(temps[-1] - x_op[T_INDEX]) > 25.0, "returned to the operating point"


def test_the_runaway_is_an_ignition_towards_a_hotter_branch():
    """Which way it goes, and how close the transient comes to the trip.

    The departure peaks near 448 K against a 470 K relief setting. That margin is why the
    safety gate on this task has teeth: a controller that reacts slowly does not merely
    track badly, it comes within ~20 K of lifting the vessel.
    """
    plant = _bare_plant()
    plant.reset(0)
    x_op = plant._x_op.copy()
    temps = _hold_from(plant, x_op, 200)[:, 0]

    assert temps[-1] > x_op[T_INDEX] + 25.0, "settled cooler, not hotter - expected ignition"
    trip = max(c.hi for c in plant.spec.constraints if c.signal == "T" and np.isfinite(c.hi))
    assert temps.max() < trip, "the uncontrolled transient already trips - task unwinnable"
    assert temps.max() > trip - 60.0, (
        f"peak {temps.max():.1f} K leaves more than 60 K of margin to the {trip:.0f} K trip; "
        "the safety gate would never bind and the envelope is decorative"
    )


def test_hold_is_expensive_because_the_reactor_leaves_setpoint(task: Task):
    """Doing nothing does NOT breach the envelope here -- it settles on the hot branch,
    inside the limits. It is ruinous for tracking, not for safety, and the anchor gap comes
    from that. An earlier draft of this test asserted a safety breach and was simply wrong.
    """
    records = run_ensemble(task, lambda b: Hold(b), seeds=task.seeds[:3])
    assert all(r.cost.iae > 0.05 for r in records), (
        "frozen actuators tracked setpoint acceptably, which an unstable plant should not "
        "permit"
    )


# -- determinism and physics -----------------------------------------------------


def test_plant_is_a_pure_function_of_seed_and_controls():
    controls = [np.array([300.0 + 4.0 * np.sin(k / 5)]) for k in range(50)]

    def trajectory():
        p = _bare_plant()
        p.reset(777)
        return np.array([p.step(u).y for u in controls])

    np.testing.assert_array_equal(trajectory(), trajectory())


def test_concentration_stays_physical_under_a_runaway():
    """Divergence must not produce a negative concentration or a non-finite state — the
    scoring arithmetic downstream cannot survive a NaN."""
    plant = _bare_plant()
    plant.reset(0)
    u_hot = np.array([plant.spec.actuator_hi()[0]])
    for _ in range(200):
        plant.step(u_hot)
    x = plant.audit().x
    assert np.all(np.isfinite(x)), f"state went non-finite during runaway: {x}"
    assert x[0] >= -1e-9, f"negative concentration {x[0]}"


def test_temperature_envelope_is_declared_on_true_state():
    signals = {c.signal for c in _bare_plant().spec.constraints}
    assert "T" in signals


# -- the task --------------------------------------------------------------------


def test_shipped_task_is_well_posed(task: Task):
    seeds = task.seeds[:4]
    hold = run_ensemble(task, lambda b: Hold(b), seeds=seeds)
    ref = run_ensemble(task, lambda b: build_reference(b, task.reference), seeds=seeds)
    for h, r in zip(hold, ref):
        assert not r.cost.violations, f"seed {r.seed}: the reference violates"
        assert not r.cost.duty_exceeded, f"seed {r.seed}: reference breaches its duty limit"
        assert r.cost.total < h.cost.total, f"seed {r.seed}: reference no better than hold"


def test_the_anchor_gap_is_wide_because_hold_diverges(task: Task):
    """Documents why this task's scores compress near the top.

    Because doing nothing runs the reactor away, the hold anchor is enormous and almost any
    stabilizing controller scores well above 0. The interesting question on this plant is
    entirely about the gap between a submission and 1.0, not between it and 0.0.
    """
    seeds = task.seeds[:3]
    hold = run_ensemble(task, lambda b: Hold(b), seeds=seeds)
    ref = run_ensemble(task, lambda b: build_reference(b, task.reference), seeds=seeds)
    ratio = np.mean([h.cost.total for h in hold]) / np.mean([r.cost.total for r in ref])
    assert ratio > 5.0, f"hold is only {ratio:.1f}x the reference; expected a wide gap here"

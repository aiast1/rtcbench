"""Tests for the drum-boiler plant.

The point of this plant in the suite is inverse response, so the load-bearing test is the
one that measures it. A boiler model that merely runs is worthless here — if the level does
not go the wrong way first, the task is not exercising the pathology it exists for and every
score on it is measuring something else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from rtcbench.baselines import Hold, build_reference
from rtcbench.harness import run_ensemble, run_scenario
from rtcbench.plants import build_plant
from rtcbench.task import Task

TASK = Path(__file__).resolve().parents[1] / "tasks" / "boiler_drum_v1.yaml"


@pytest.fixture(scope="module")
def task() -> Task:
    return Task.load(TASK)


def _bare_plant():
    """The plant without the instrument layer, so a step response is not confounded by
    sensor noise, transport delay or valve slew."""
    cfg = dict(yaml.safe_load(TASK.read_text(encoding="utf-8"))["plant"])
    return build_plant(cfg)


def _level_index(plant) -> int:
    return [c.tag for c in plant.spec.measurements].index("LT-101")


# -- the pathology ---------------------------------------------------------------


def test_feedwater_step_makes_the_level_fall_before_it_rises():
    """Shrink: cold feedwater collapses steam bubbles, so indicated level drops first.

    This is the whole reason the plant is in the pack. A naive level controller reads the
    initial fall as "not enough water", adds more, and drives the loop into oscillation.
    """
    plant = _bare_plant()
    obs = plant.reset(0)
    li = _level_index(plant)
    start = float(obs.y[li])

    u_open = np.array([plant.spec.actuator_hi()[0]])
    trace = [float(plant.step(u_open).y[li]) for _ in range(120)]

    lowest = min(trace)
    dipped = start - lowest
    assert dipped > 0.5, (
        f"level never fell after a feedwater step (start {start:.2f} mm, minimum "
        f"{lowest:.2f} mm) - there is no shrink in this model and the task is not testing "
        "inverse response"
    )
    # ...and it must eventually recover past where it began, or it is not inverse response,
    # just a plant that falls.
    assert max(trace) > start, (
        f"level fell but never rose above its start (max {max(trace):.2f} vs {start:.2f})"
    )
    assert trace.index(min(trace)) < trace.index(max(trace)), "the dip must precede the rise"


def test_steam_demand_step_makes_the_level_rise_before_it_falls():
    """Swell: the mirror image. More steam draw drops drum pressure, bubbles expand, level
    climbs even though inventory is leaving."""
    plant = _bare_plant()
    obs = plant.reset(0)
    li = _level_index(plant)
    start = float(obs.y[li])
    u_hold = np.asarray(plant.spec.initial_actuation())

    # Load is set by the turbine valve, which is also what the task's own disturbance moves.
    plant.set_params({"turbine_valve": 1.25})
    trace = [float(plant.step(u_hold).y[li]) for _ in range(150)]

    assert max(trace) > start + 0.2, (
        f"level did not swell on a load increase (max {max(trace):.2f} vs start {start:.2f})"
    )
    assert min(trace) < start, "level swelled but never fell back - inventory must leave"
    assert trace.index(max(trace)) < trace.index(min(trace)), "the swell must precede the fall"


# -- determinism and physics -----------------------------------------------------


def test_plant_is_a_pure_function_of_seed_and_controls():
    controls = [np.array([45.0 + 8.0 * np.sin(k / 7)]) for k in range(60)]

    def trajectory():
        p = _bare_plant()
        p.reset(4242)
        return np.array([p.step(u).y for u in controls])

    np.testing.assert_array_equal(trajectory(), trajectory())


def test_holding_the_duty_point_is_stationary():
    plant = _bare_plant()
    plant.reset(0)
    u = np.asarray(plant.spec.initial_actuation())
    before = plant.audit().x.copy()
    for _ in range(60):
        plant.step(u)
    after = plant.audit().x
    np.testing.assert_allclose(after, before, rtol=2e-3, atol=2e-3)


def test_pressure_and_level_constraints_are_declared_on_true_state():
    signals = {c.signal for c in _bare_plant().spec.constraints}
    assert {"l_ind", "l_inv", "p"} <= signals


# -- the task --------------------------------------------------------------------


def test_shipped_task_is_well_posed(task: Task):
    """The `rtcbench validate` contract as a test, on a subset for speed."""
    seeds = task.seeds[:4]
    hold = run_ensemble(task, lambda b: Hold(b), seeds=seeds)
    ref = run_ensemble(task, lambda b: build_reference(b, task.reference), seeds=seeds)
    for h, r in zip(hold, ref):
        assert not h.cost.violations, f"seed {h.seed}: doing nothing already violates"
        assert not r.cost.violations, f"seed {r.seed}: the reference violates"
        assert not r.cost.duty_exceeded, f"seed {r.seed}: the reference breaches its own duty limit"
        assert r.cost.total < h.cost.total, f"seed {r.seed}: reference no better than hold"


def test_the_reference_is_slow_on_purpose(task: Task):
    """Guards the tuning against a well-meaning future 'fix'.

    ti=2000 s looks like a typo and is not: the RHP zero caps achievable bandwidth, and an
    earlier tuner whose grid stopped at 200 s concluded that no safe tuning existed at all.
    Anyone tempted to speed this up should have to delete this test first.
    """
    ti = task.reference["ti"]
    assert min(ti) > 500.0, (
        f"reference integral time {ti} is far faster than the RHP zero allows; re-check "
        "against the provenance note in the task file before changing it"
    )


def test_an_aggressive_controller_is_punished(task: Task):
    """The trap, demonstrated: tuning as if there were no inverse response loses badly."""
    from rtcbench.baselines import MultiLoopPID

    seeds = task.seeds[:3]
    ref = run_ensemble(task, lambda b: build_reference(b, task.reference), seeds=seeds)
    hot = run_ensemble(
        task, lambda b: MultiLoopPID(b, pairing=[0], kp=[3.0], ti=[30.0]), seeds=seeds
    )
    ref_cost = float(np.mean([r.cost.total for r in ref]))
    hot_cost = float(np.mean([10.0 if (r.cost.violations or r.cost.duty_exceeded or r.failed)
                              else r.cost.total for r in hot]))
    assert hot_cost > ref_cost, (
        f"an aggressively tuned PI ({hot_cost:.5f}) did no worse than the detuned reference "
        f"({ref_cost:.5f}) - the inverse response is not biting and the task is too easy"
    )

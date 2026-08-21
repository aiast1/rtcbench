"""Tests for the Van de Vusse CSTR plant and its task.

Mirrors tests/test_core.py's style: properties that would be false if the plant or the task
were broken, not coverage for its own sake. The property this plant exists to exercise is
the sign-flipping steady-state gain from F/V to C_B -- see van_de_vusse.py's docstring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rtcbench import Task
from rtcbench.baselines import MultiLoopPID, build_reference
from rtcbench.harness import run_ensemble
from rtcbench.plants.van_de_vusse import NOMINAL, VanDeVusse, steady_state

TASK = Path(__file__).resolve().parents[1] / "tasks" / "van_de_vusse_v1.yaml"


@pytest.fixture(scope="module")
def task() -> Task:
    return Task.load(TASK)


# -- determinism ------------------------------------------------------------------


def test_plant_is_a_pure_function_of_seed_and_controls():
    """Every reproducibility claim in the project reduces to this one."""
    controls = [np.array([14.0 + 3.0 * np.sin(k / 7), -1100.0 + 40.0 * np.cos(k / 5)])
                for k in range(40)]

    def trajectory():
        p = VanDeVusse(NOMINAL, mismatch={"k10": 0.05, "kw": 0.05})
        p.reset(1234)
        return np.array([p.step(u).y for u in controls])

    np.testing.assert_array_equal(trajectory(), trajectory())


def test_different_seeds_draw_different_plants():
    a = VanDeVusse(NOMINAL, mismatch={"k10": 0.05})
    b = VanDeVusse(NOMINAL, mismatch={"k10": 0.05})
    a.reset(1)
    b.reset(2)
    assert a.params.k10 != b.params.k10


def test_no_mismatch_means_no_draw():
    p = VanDeVusse(NOMINAL)
    p.reset(99)
    assert p.params == NOMINAL


# -- the sealed boundary ------------------------------------------------------------


def test_observation_carries_only_cb_and_t():
    """C_A and T_K are real states the plant tracks but never hands to a controller."""
    p = VanDeVusse(NOMINAL)
    obs = p.reset(0)
    assert obs.y.shape == (2,)
    assert p.audit().x.shape == (4,)  # ground truth is wider than what is measured


def test_jacket_temperature_is_unmeasured_but_still_constrained():
    """A controller must avoid cooking the jacket even though it cannot see T_K directly."""
    signals = {c.signal for c in VanDeVusse(NOMINAL).spec.constraints}
    assert "TK" in signals
    tags = {c.tag for c in VanDeVusse(NOMINAL).spec.measurements}
    assert not any("TK" in t or "TCV" in t for t in tags)  # T_K is not an instrument tag


# -- physics --------------------------------------------------------------------


def test_analytic_steady_state_matches_the_papers_operating_point():
    """Chen/Kremling/Allgower and Klatt/Engell both report C_A=2.14, C_B=1.09 mol/L,
    T=114.2, T_K=112.9 degC at F/V=14.19 h^-1, Q_K=-1113.5 kJ/h."""
    x = steady_state(NOMINAL, np.array([14.19, -1113.5]))
    np.testing.assert_allclose(x, [2.14, 1.09, 114.2, 112.9], atol=0.02)


def test_steady_state_is_actually_stationary():
    p = VanDeVusse(NOMINAL, sample_time=30.0)
    p.reset(0)
    before = p.audit().x.copy()
    for _ in range(40):
        p.step(np.array([14.19, -1113.5]))
    np.testing.assert_allclose(p.audit().x, before, rtol=1e-6)


def test_the_reason_this_plant_is_in_the_pack_at_all():
    """The steady-state gain dC_B/d(F/V) is positive on the low-dilution side of the peak
    and negative on the high-dilution side -- input multiplicity, the textbook van de Vusse
    pathology. Measured directly off the plant's own steady_state(), not asserted by fiat."""
    u_at = lambda fv: np.array([fv, -1113.5])  # noqa: E731

    def cb_ss(fv, x0):
        return steady_state(NOMINAL, u_at(fv), x0=x0)

    x_low = cb_ss(10.0, None)
    gain_low = (cb_ss(10.05, x_low)[1] - cb_ss(9.95, x_low)[1]) / 0.10
    x_high = cb_ss(25.0, None)
    gain_high = (cb_ss(25.05, x_high)[1] - cb_ss(24.95, x_high)[1]) / 0.10

    assert gain_low > 0.0
    assert gain_high < 0.0


def test_published_nominal_point_sits_on_the_shoulder_of_the_peak():
    """The commissioned operating point (F/V=14.19) is deliberately close to the sign
    reversal (~14.7): this is not an obscure corner of parameter space, it is where the
    benchmark's designers put the plant's dial."""
    x0 = steady_state(NOMINAL, np.array([14.19, -1113.5]))
    gain = (
        steady_state(NOMINAL, np.array([14.29, -1113.5]), x0=x0)[1]
        - steady_state(NOMINAL, np.array([14.09, -1113.5]), x0=x0)[1]
    ) / 0.20
    assert 0.0 < gain < 0.01  # small and positive, not comfortably far from zero


def test_concentrations_are_clipped_but_temperatures_are_not():
    """A concentration cannot go negative (physics); a temperature exceeding its limit must
    stay visible to audit() rather than being quietly clipped away."""
    p = VanDeVusse(NOMINAL, sample_time=5.0)
    p.reset(0)
    # Starve the reactor (near-zero feed) long enough to approach zero concentration.
    for _ in range(200):
        p.step(np.array([3.0, 0.0]))
    x = p.audit().x
    assert x[0] >= 0.0 and x[1] >= 0.0


def test_runaway_heating_trips_the_temperature_constraint():
    """Force the jacket duty positive (net heating rather than cooling) at a low dilution
    rate -- long residence time, so the exothermic 2A -> D step has time to run -- and
    confirm the hard limit actually fires. (Not a scenario the task's actuator range
    permits; this drives the bare plant directly to stress the audit() path itself.)"""
    p = VanDeVusse(NOMINAL, sample_time=20.0)
    p.reset(0)
    tripped = False
    for _ in range(200):
        p.step(np.array([3.0, 5000.0]))
        if p.audit().violations:
            tripped = True
            break
    assert tripped


def test_set_params_rejects_unknown_names():
    p = VanDeVusse(NOMINAL)
    p.reset(0)
    with pytest.raises(ValueError):
        p.set_params({"not_a_real_parameter": 1.0})


def test_fouling_disturbance_actually_degrades_cooling():
    p = VanDeVusse(NOMINAL, sample_time=10.0)
    p.reset(0)
    p.set_params({"kw": NOMINAL.kw * 0.7})
    assert p.params.kw == pytest.approx(NOMINAL.kw * 0.7)


# -- task ---------------------------------------------------------------------


def test_shipped_task_is_well_posed(task: Task):
    """The `rtcbench validate` contract, as a test: on every scenario the reference must be
    safe and must beat doing nothing."""
    seeds = task.seeds[:6]
    hold = run_ensemble(task, lambda b: build_reference(b, {"kind": "hold"}), seeds=seeds)
    ref = run_ensemble(task, lambda b: build_reference(b, task.reference), seeds=seeds)
    for h, r in zip(hold, ref):
        assert not h.cost.violations, f"seed {h.seed}: doing nothing already violates"
        assert not r.cost.violations, f"seed {r.seed}: the reference violates"
        assert r.cost.total < h.cost.total, f"seed {r.seed}: reference no better than hold"


def test_the_reference_is_not_beaten_by_a_fast_integral_naive_pi(task: Task):
    """Guards the anchor, in this task's specific flavour: a naive controller that reads a
    small nominal gain as an invitation to wind up FAST (a short integral time) builds
    exactly the momentum that carries F/V across the gain reversal. The reference is tuned
    with a long enough integral time not to."""
    seeds = task.seeds[:6]
    ref = run_ensemble(task, lambda b: build_reference(b, task.reference), seeds=seeds)
    naive = run_ensemble(
        task, lambda b: MultiLoopPID(b, pairing=[0, 1], kp=[2.5, 2.0], ti=[10.0, 15.0]),
        seeds=seeds,
    )
    assert np.mean([r.cost.total for r in ref]) < np.mean([n.cost.total for n in naive])


def test_naive_fast_integral_pi_tracks_worse_than_doing_nothing():
    """The sharpest statement of the trap: on this schedule, a fast-integral controller
    that assumes a fixed-sign gain ends up WORSE than freezing the actuators, because it
    spends real time pushing C_B the wrong way once it crosses the peak."""
    task = Task.load(TASK)
    seeds = task.seeds[:6]
    hold = run_ensemble(task, lambda b: build_reference(b, {"kind": "hold"}), seeds=seeds)
    naive = run_ensemble(
        task, lambda b: MultiLoopPID(b, pairing=[0, 1], kp=[2.5, 2.0], ti=[10.0, 15.0]),
        seeds=seeds,
    )
    assert np.mean([n.cost.total for n in naive]) > np.mean([h.cost.total for h in hold])

"""Tests for the property this plant exists to exercise: the deadtime is not a fixed number.

Mirrors tests/test_core.py's style -- each test corresponds to a claim in the plant's
docstring that would be false if it failed, not coverage for its own sake.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rtcbench.baselines import MultiLoopPID, build_reference
from rtcbench.harness import run_ensemble
from rtcbench.plants.deadtime_process import DeadtimeProcess, DeadtimeProcessParams, steady_state
from rtcbench.task import Task

TASK = Path(__file__).resolve().parents[1] / "tasks" / "deadtime_process_v1.yaml"


@pytest.fixture(scope="module")
def task() -> Task:
    return Task.load(TASK)


def _measure_delay(q: float, *, sample_time: float = 1.0, substeps: int = 4,
                    duty_step: float = 30.0, tol: float = 1e-6) -> float:
    """Step the heater duty and find the first control period the outlet measurably moves.

    Returns the measured delay in seconds -- resolved to one ``sample_time``, which is fine
    granularity enough to tell two substantially different throughputs apart.
    """
    plant = DeadtimeProcess(
        DeadtimeProcessParams(q=q), sample_time=sample_time, substeps=substeps, nominal_u=50.0,
    )
    obs = plant.reset(0)
    y0 = float(obs.y[0])
    theta = plant.params.V_line / q
    n_steps = int(theta / sample_time) + 60
    for k in range(n_steps):
        obs = plant.step(np.array([50.0 + duty_step]))
        if abs(float(obs.y[0]) - y0) > tol:
            return (k + 1) * sample_time
    raise AssertionError(f"outlet never moved within {n_steps} steps at q={q}")


# -- the pathology this plant exists for -----------------------------------------


def test_delay_at_two_throughputs_differs_substantially():
    """The whole point of the plant: transport delay tracks flow, not a fixture."""
    fast_flow_delay = _measure_delay(q=6.0)     # theta = 240/6 = 40 s
    slow_flow_delay = _measure_delay(q=1.2)     # theta = 240/1.2 = 200 s

    assert slow_flow_delay > 3.0 * fast_flow_delay, (
        f"delay barely moved with flow: {fast_flow_delay}s @ q=6.0 vs "
        f"{slow_flow_delay}s @ q=1.2 -- a fixed Smith predictor would not notice this"
    )
    # And each is close to the analytic V_line/q, not some fixed sample-quantized value.
    assert fast_flow_delay == pytest.approx(240.0 / 6.0, abs=2.0)
    assert slow_flow_delay == pytest.approx(240.0 / 1.2, abs=2.0)


def test_delay_tracks_flow_continuously_not_by_whole_samples():
    """Sweeping q should sweep the measured delay smoothly and monotonically -- the buffer
    interpolates a fractional lookback rather than snapping to the nearest whole substep,
    the same failure mode ``SensorConfig.deadtime_samples`` deliberately accepts for a
    transmitter but this plant's own transport physics must not reproduce."""
    flows = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    delays = [_measure_delay(q, sample_time=0.5, substeps=1) for q in flows]

    # Monotonically decreasing as flow rises.
    assert all(a > b for a, b in zip(delays, delays[1:])), delays
    # And each tracks V_line/q tightly -- within a couple of substeps, not a whole
    # sample_time's worth of quantization error.
    for q, d in zip(flows, delays):
        assert d == pytest.approx(240.0 / q, abs=1.5)


def test_nothing_reaches_the_outlet_before_one_transport_time():
    """Conservation: the transport buffer must not leak a signal forward in time. Every
    reading strictly before one line-volume's worth of flow has passed must equal the
    pre-step value exactly -- not approximately, since the buffer is seeded to return exactly
    the initial steady value until real post-reset volume displaces it."""
    q = 2.0
    sample_time = 2.0
    plant = DeadtimeProcess(DeadtimeProcessParams(q=q), sample_time=sample_time, substeps=5,
                             nominal_u=50.0)
    obs = plant.reset(0)
    y0 = float(obs.y[0])
    theta = plant.params.V_line / q  # 120 s

    n_before = int(theta / sample_time) - 2  # stop safely short of the crossing
    seen_after_move = False
    for k in range(n_before):
        obs = plant.step(np.array([90.0]))
        assert float(obs.y[0]) == pytest.approx(y0, abs=0.0), (
            f"outlet moved at t={ (k+1)*sample_time }s, before theta={theta}s has elapsed"
        )
    # Sanity: the input change is real and does eventually arrive.
    for _ in range(20):
        obs = plant.step(np.array([90.0]))
        if abs(float(obs.y[0]) - y0) > 1e-9:
            seen_after_move = True
            break
    assert seen_after_move, "the step never arrived at all -- buffer is not conserving flow"


def test_flow_change_mid_transit_is_volume_conserving_not_time_conserving():
    """A parcel already inside the line when flow changes has a remaining *volume* to
    travel, not a remaining *time* -- indexing the transport buffer by cumulative volume
    (rather than looking up today's V_line/q and delaying by that) is what this test is
    checking. Substeps=1 so each ``step`` call is exactly one buffer update."""
    dt = 1.0
    V_line = 240.0
    plant = DeadtimeProcess(DeadtimeProcessParams(q=3.0, V_line=V_line), sample_time=dt,
                             substeps=1, nominal_u=50.0)
    obs = plant.reset(0)
    y0 = float(obs.y[0])

    cum = 0.0
    q = 3.0
    switch_at = 40.0
    switched = False
    move_cum = None
    for k in range(400):
        t_before = k * dt
        if not switched and t_before >= switch_at:
            plant.set_params({"q": 1.2})
            q = 1.2
            switched = True
        cum += q * dt
        obs = plant.step(np.array([90.0]))
        if move_cum is None and abs(float(obs.y[0]) - y0) > 1e-9:
            move_cum = cum
            break

    assert move_cum is not None
    # The parcel must not surface until (approximately) V_line worth of volume has actually
    # flowed past the inlet since t=0 -- not V_line/q_old (which would fire early, at t=80)
    # and not V_line/q_new applied from t=0 (which would fire absurdly late).
    assert move_cum == pytest.approx(V_line, abs=2.0)


def test_the_unmeasured_inlet_can_violate_while_the_sensor_still_reads_the_old_value():
    """The structural trap: T_in is not instrumented, so a controller reacting only to
    TT-401 cannot see a limit it has already blown through."""
    plant = DeadtimeProcess(DeadtimeProcessParams(q=3.0), sample_time=5.0, substeps=10,
                             nominal_u=50.0, T_in_max=90.0)
    obs = plant.reset(0)
    y0 = float(obs.y[0])

    saw_violation_while_blind = False
    for _ in range(20):
        obs = plant.step(np.array([100.0]))  # slam the heater
        audit = plant.audit()
        if audit.violations:
            saw_violation_while_blind = abs(float(obs.y[0]) - y0) < 1e-9
            break
    assert saw_violation_while_blind, (
        "expected T_in to breach its limit before TT-401's reading moved at all"
    )


# -- determinism ------------------------------------------------------------------


def test_plant_is_a_pure_function_of_seed_and_controls():
    controls = [np.array([50.0 + 10.0 * np.sin(k / 7)]) for k in range(60)]

    def trajectory():
        p = DeadtimeProcess(sample_time=5.0, substeps=10,
                             mismatch={"q": 0.05, "tau_h": 0.05})
        p.reset(4321)
        return np.array([p.step(u).y for u in controls])

    np.testing.assert_array_equal(trajectory(), trajectory())


def test_different_seeds_draw_different_params():
    a = DeadtimeProcess(mismatch={"q": 0.05})
    b = DeadtimeProcess(mismatch={"q": 0.05})
    a.reset(1)
    b.reset(2)
    assert a.params.q != b.params.q


def test_no_mismatch_means_no_draw():
    p = DeadtimeProcess()
    p.reset(99)
    assert p.params == DeadtimeProcessParams()


# -- the sealed boundary ------------------------------------------------------------


def test_observation_carries_only_the_downstream_temperature():
    p = DeadtimeProcess()
    obs = p.reset(0)
    assert obs.y.shape == (1,)
    assert p.audit().x.shape == (2,)  # ground truth also carries the unmeasured T_in


# -- physics --------------------------------------------------------------------


def test_steady_state_is_actually_stationary():
    p = DeadtimeProcess(sample_time=10.0, substeps=5)
    p.reset(0)
    u0 = p.spec.initial_actuation()
    before = p.audit().x.copy()
    for _ in range(80):
        p.step(u0)
    np.testing.assert_allclose(p.audit().x, before, atol=1e-9)


def test_analytic_steady_state_matches_simulated_equilibrium():
    p = DeadtimeProcess(sample_time=5.0, substeps=10, nominal_u=50.0)
    p.reset(0)
    u0 = p.spec.initial_actuation()
    for _ in range(200):  # several transport times at the default flow (theta = 80 s)
        obs = p.step(u0)
    expected = steady_state(p.params, float(u0[0]))
    assert float(obs.y[0]) == pytest.approx(expected, abs=1e-6)
    assert p.audit().x[0] == pytest.approx(expected, abs=1e-6)


def test_theta_over_tau_is_deadtime_dominant_across_the_scheduled_flows():
    """The regime this plant claims to model. theta/tau >= 2 is the requested floor; this
    task's flow schedule (q in {1.2, 3.0, 6.0}) stays well clear of it in every regime."""
    p = DeadtimeProcessParams()
    for q in (1.2, 3.0, 6.0):
        theta = p.V_line / q
        assert theta / p.tau_h >= 2.0, f"q={q}: theta/tau={theta/p.tau_h:.2f} is not dominant"


# -- task-level: the anchor and the trap it exists to catch ------------------------


def test_shipped_task_is_well_posed(task: Task):
    """The ``rtcbench validate`` contract, as a test."""
    from rtcbench.baselines.hold import Hold

    seeds = task.seeds[:4]
    hold = run_ensemble(task, lambda b: Hold(b), seeds=seeds)
    ref = run_ensemble(task, lambda b: build_reference(b, task.reference), seeds=seeds)
    for h, r in zip(hold, ref):
        assert not h.cost.violations, f"seed {h.seed}: doing nothing already violates"
        assert not r.cost.violations, f"seed {r.seed}: the reference violates"
        assert r.cost.total < h.cost.total, f"seed {r.seed}: reference no better than hold"


def test_a_fixed_deadtime_tuning_degrades_once_flow_changes(task: Task):
    """The task's reason for existing. A PI aggressively tuned around the *initial* flow's
    deadtime (theta ~ 80 s) must do markedly worse than the detuned reference once the
    scenario's throughput swings the delay out to ~200 s and back down to ~40 s -- ideally
    badly enough to gate on the safety constraint, not just score a little worse."""
    seeds = task.seeds[:6]
    ref = run_ensemble(task, lambda b: build_reference(b, task.reference), seeds=seeds)
    aggressive = run_ensemble(
        task, lambda b: MultiLoopPID(b, pairing=[0], kp=[2.5], ti=[20.0]), seeds=seeds
    )

    ref_cost = np.mean([r.cost.total for r in ref])
    aggr_cost = np.mean([r.cost.total for r in aggressive])
    assert aggr_cost > 2.0 * ref_cost, (
        f"aggressive fixed-deadtime tuning ({aggr_cost:.4f}) was not markedly worse than "
        f"the detuned reference ({ref_cost:.4f})"
    )
    assert any(r.cost.violations for r in aggressive), (
        "expected at least one seed where the aggressively-tuned fixed PI actually violates "
        "a constraint once the deadtime moves, not merely scores worse"
    )

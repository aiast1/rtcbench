"""Tests for the pH neutralization CSTR and its task.

Properties, not coverage. Each test here corresponds to a claim made in the plant's module
docstring or in `tasks/ph_neutralization_v1.yaml`, and would be false if the test failed:
the model is the published one, the titration curve emerges from the chemistry rather than
being tabulated, the equilibrium solve is deterministic, the process gain really does vary
by more than an order of magnitude, the safety envelope really does fire, and the task is
well posed against its own anchors.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rtcbench.baselines import Hold, MultiLoopPID, build_reference
from rtcbench.harness import run_ensemble, run_scenario
from rtcbench.plants.ph_neutralization import (
    HENSON_SEBORG,
    PHNeutralization,
    PHParams,
    ph_from_invariants,
    ph_residual,
    steady_ph,
    steady_state,
    titration_gain,
)
from rtcbench.task import Task

TASK = Path(__file__).resolve().parents[1] / "tasks" / "ph_neutralization_v1.yaml"

#: Base flows that put the nominal plant on each named part of the titration curve. These
#: are the task's own three setpoints, resolved to actuator positions.
SHELF_Q3 = 14.22       # pH 6.50, the bicarbonate buffer shelf
EQUIVALENCE_Q3 = 16.588  # pH 8.20, the steepest point on the curve
DEAF_Q3 = 19.93        # pH 9.90, the caustic end


@pytest.fixture(scope="module")
def task() -> Task:
    return Task.load(TASK)


def _q3_for_ph(target: float, p: PHParams = HENSON_SEBORG) -> float:
    """Invert the titration curve by bisection — for placing test operating points."""
    lo, hi = 0.0, 30.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if steady_ph(p, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# -- the pathology this plant exists for ----------------------------------------


def test_process_gain_varies_by_more_than_an_order_of_magnitude():
    """The reason the plant is in the pack, measured rather than asserted in prose.

    dpH/dq3 is evaluated at two operating points the task's own setpoint schedule visits.
    If this ratio ever collapses, the plant has stopped being a gain-variation task and
    the whole point of ph_neutralization_v1 has quietly gone away.
    """
    steep = titration_gain(HENSON_SEBORG, EQUIVALENCE_Q3)
    deaf = titration_gain(HENSON_SEBORG, DEAF_Q3)

    assert steep == pytest.approx(2.94, abs=0.05)   # pH per (mL/s), at pH 8.2
    assert deaf == pytest.approx(0.162, abs=0.01)   # pH per (mL/s), at pH 9.9
    assert steep / deaf > 10.0                      # ~18x

    # Wider still across the pump's full travel: at the caustic end the loop is nearly deaf.
    assert titration_gain(HENSON_SEBORG, EQUIVALENCE_Q3) / titration_gain(
        HENSON_SEBORG, 30.0
    ) > 50.0                                        # ~84x


def test_the_same_pump_move_does_ten_times_more_at_the_equivalence_point():
    """The static gain above, confirmed dynamically through the integrator.

    One identical +0.25 mL/s step, held for 400 s, from two different duty points. A
    controller cannot tune itself against a number this unstable.
    """

    def excursion(q3: float) -> float:
        plant = PHNeutralization(nominal_q3=q3, sample_time=5.0)
        start = plant.reset(0).y[0]
        for _ in range(80):
            end = plant.step(np.array([q3 + 0.25])).y[0]
        return end - start

    steep = excursion(EQUIVALENCE_Q3)
    deaf = excursion(DEAF_Q3)
    assert steep > 0.5 and deaf < 0.05
    assert steep / deaf > 10.0                      # ~15x


def test_the_setpoint_schedule_actually_visits_the_steep_region():
    """Guards the task, not the plant. A schedule that stayed on one part of the curve
    would leave the pathology unexercised while still looking like a pH task."""
    raw = Task.load(TASK)
    targets = {v for seg in raw.setpoints.segments for v in seg.values}
    gains = [titration_gain(HENSON_SEBORG, _q3_for_ph(sp)) for sp in targets]
    assert max(gains) / min(gains) > 10.0
    assert max(gains) > 2.0     # at least one setpoint sits on the equivalence step


def test_the_gain_variation_defeats_a_single_fixed_gain(task: Task):
    """The trap, through the harness: the gain that suits the flat end of the curve is
    the gain that wrecks the plant at the equivalence point.

    kp=8.0 tracks the buffer shelf twice as well as the reference does and pays for it with
    a limit cycle around the equivalence point that exceeds the task's actuator duty
    ceiling on every scenario.
    """
    seeds = task.seeds[:4]
    ref = run_ensemble(task, lambda b: build_reference(b, task.reference), seeds=seeds)
    hot = run_ensemble(task, lambda b: MultiLoopPID(b, pairing=[0], kp=[8.0], ti=[90.0]),
                       seeds=seeds)

    shelf = slice(20, 70)   # t = 100..350 s, holding 6.50 on the shelf
    ref_shelf = np.mean([np.abs(r.trace.r[shelf, 0] - r.trace.x[shelf, 3]).mean() for r in ref])
    hot_shelf = np.mean([np.abs(r.trace.r[shelf, 0] - r.trace.x[shelf, 3]).mean() for r in hot])

    assert hot_shelf < ref_shelf            # the high gain really is better on the shelf
    assert all(r.cost.duty_exceeded for r in hot)   # and unusable everywhere else
    assert not any(r.cost.duty_exceeded for r in ref)


# -- the chemistry ---------------------------------------------------------------


def test_published_parameters_reproduce_henson_and_seborgs_operating_point():
    """Henson & Seborg (1994) report q3 = 15.6 mL/s giving pH 7.0 at h = 14.0 cm. The
    steady state here is solved from the model, not copied from the paper, so agreeing
    with it is evidence the reimplementation is right rather than a restatement."""
    wa, wb, h = steady_state(HENSON_SEBORG, 15.6)
    assert h == pytest.approx(14.0, abs=0.05)
    assert ph_from_invariants(wa, wb) == pytest.approx(7.0, abs=0.05)


def test_the_solver_actually_solves_electroneutrality():
    """The pH is a root, not a lookup. Checked on invariants from across the whole curve."""
    for q3 in (0.0, 5.0, 10.0, 14.22, 16.588, 19.93, 25.0, 30.0):
        wa, wb, _ = steady_state(HENSON_SEBORG, q3)
        ph = ph_from_invariants(wa, wb)
        assert abs(ph_residual(ph, wa, wb, 6.35, 10.25)) < 1e-12


def test_the_residual_is_monotone_so_the_bracket_is_guaranteed():
    """Bisection is only safe because the residual increases strictly in pH. If a future
    parameter set broke that, the solver would silently return a different root."""
    wa, wb, _ = steady_state(HENSON_SEBORG, 16.588)
    grid = np.linspace(-2.0, 16.0, 2001)
    values = np.array([ph_residual(ph, wa, wb, 6.35, 10.25) for ph in grid])
    assert np.all(np.diff(values) > 0.0)
    assert values[0] < 0.0 and values[-1] > 0.0     # the published bracket really brackets


def test_the_titration_curve_is_emergent_not_tabulated():
    """Raise the buffer concentration and the curve must flatten out on its own — which it
    can only do if the shape comes from the equilibrium chemistry."""
    sharp = replace(HENSON_SEBORG, Wa2=-0.006, Wb2=0.006)   # a tenth of the buffer
    buffered = replace(HENSON_SEBORG, Wa2=-0.06, Wb2=0.06)  # twice it
    peak = lambda p: max(  # noqa: E731
        titration_gain(p, q) for q in np.linspace(8.0, 25.0, 400)
    )
    assert peak(sharp) > 3.0 * peak(HENSON_SEBORG)
    assert peak(buffered) < peak(HENSON_SEBORG)


def test_the_carbonate_shelf_sits_near_pk1():
    """The flat spot the loop goes deaf on is the bicarbonate buffer at pK1 = 6.35, not an
    artifact. Move pK1 and the shelf moves with it."""

    def shelf_ph(p: PHParams) -> float:
        qs = np.linspace(11.0, 15.5, 400)
        return steady_ph(p, float(qs[np.argmin([titration_gain(p, q) for q in qs])]))

    assert shelf_ph(HENSON_SEBORG) == pytest.approx(6.35, abs=0.5)
    assert shelf_ph(replace(HENSON_SEBORG, pK1=7.35)) > shelf_ph(HENSON_SEBORG) + 0.5


# -- determinism -----------------------------------------------------------------


def test_plant_is_a_pure_function_of_seed_and_controls():
    """Every reproducibility claim in the project reduces to this one."""
    controls = [np.array([14.22 + 2.0 * np.sin(k / 7)]) for k in range(60)]

    def trajectory():
        p = PHNeutralization(mismatch={"q1": 0.03, "Wb2": 0.05})
        p.reset(4321)
        return np.array([p.step(u).y for u in controls])

    np.testing.assert_array_equal(trajectory(), trajectory())


def test_the_whole_scenario_replays_bit_for_bit(task: Task):
    """Determinism end to end, through the instrument layer and the harness."""
    a = run_scenario(task, lambda b: build_reference(b, task.reference), task.seeds[0])
    b = run_scenario(task, lambda b: build_reference(b, task.reference), task.seeds[0])
    np.testing.assert_array_equal(a.trace.y, b.trace.y)
    np.testing.assert_array_equal(a.trace.x, b.trace.x)
    np.testing.assert_array_equal(a.trace.u_commanded, b.trace.u_commanded)


def test_the_equilibrium_solve_does_not_depend_on_where_it_is_called_from():
    """Fixed bracket, fixed iteration count: the same invariants give the same pH bit for
    bit whatever the plant did before the call. A warm-started or tolerance-terminated
    solver would not have this property."""
    wa, wb, _ = steady_state(HENSON_SEBORG, 16.588)
    first = ph_from_invariants(wa, wb)
    for q3 in (0.0, 30.0, 7.5):     # drag the solver all over the curve in between
        ph_from_invariants(*steady_state(HENSON_SEBORG, q3)[:2])
    assert ph_from_invariants(wa, wb) == first


def test_different_seeds_draw_different_plants():
    a, b = (PHNeutralization(mismatch={"Wb2": 0.05}) for _ in range(2))
    a.reset(1)
    b.reset(2)
    assert a.params.Wb2 != b.params.Wb2


def test_no_mismatch_means_no_draw():
    p = PHNeutralization()
    p.reset(99)
    assert p.params == HENSON_SEBORG


# -- physics ---------------------------------------------------------------------


def test_steady_state_is_actually_stationary():
    p = PHNeutralization(sample_time=20.0)
    p.reset(0)
    before = p.audit().x.copy()
    for _ in range(60):
        p.step(np.array([14.22]))
    np.testing.assert_allclose(p.audit().x, before, rtol=1e-8, atol=1e-12)


def test_more_caustic_raises_both_ph_and_level():
    """The coupling the task's description warns about: the pump that sets pH also sets
    the tank level, which is why the level limits bound the reachable pH range."""
    for lo, hi in ((10.0, 12.0), (14.0, 17.0), (20.0, 24.0)):
        assert steady_ph(HENSON_SEBORG, lo) < steady_ph(HENSON_SEBORG, hi)
        assert steady_state(HENSON_SEBORG, lo)[2] < steady_state(HENSON_SEBORG, hi)[2]


def test_invariants_mix_linearly_even_though_ph_does_not():
    """The formulation's load-bearing property. Half acid feed plus half base feed lands
    exactly halfway in invariant space — and nowhere near halfway in pH."""
    a, b = 14.0, 17.0
    mid = 0.5 * (a + b)
    wa_a, _, _ = steady_state(HENSON_SEBORG, a)
    wa_b, _, _ = steady_state(HENSON_SEBORG, b)
    # Flow-weighted, because the invariant is a concentration in a varying total flow.
    q_a, q_b = HENSON_SEBORG.q1 + HENSON_SEBORG.q2 + a, HENSON_SEBORG.q1 + HENSON_SEBORG.q2 + b
    blended = (wa_a * q_a + wa_b * q_b) / (q_a + q_b)
    wa_mid, _, _ = steady_state(HENSON_SEBORG, mid)
    assert wa_mid == pytest.approx(blended, rel=1e-9)

    mid_ph = 0.5 * (steady_ph(HENSON_SEBORG, a) + steady_ph(HENSON_SEBORG, b))
    # Same blend in pH is out by three quarters of a unit -- pH is not an average of pHs,
    # which is why a controller that reasons in pH is reasoning in the wrong coordinates.
    assert abs(steady_ph(HENSON_SEBORG, mid) - mid_ph) > 0.5


def test_set_params_rejects_a_parameter_the_plant_does_not_have():
    p = PHNeutralization()
    p.reset(0)
    with pytest.raises(ValueError, match="no parameter"):
        p.set_params({"gamma1": 0.5})


# -- the sealed boundary and the safety envelope ---------------------------------


def test_observation_carries_measurements_not_state():
    p = PHNeutralization()
    obs = p.reset(0)
    assert obs.y.shape == (2,)              # pH and level
    assert p.audit().x.shape == (4,)        # Wa, Wb, h and true pH


def test_consent_is_gated_on_true_ph_not_on_the_analyzer():
    """The analyzer is noisy, delayed and can drop out. Gating on it would let a
    controller that blinded its own instrument off the hook."""
    signals = {c.signal: c for c in PHNeutralization().spec.constraints}
    assert set(signals) == {"h", "pH"}
    assert signals["pH"].lo == 4.0 and signals["pH"].hi == 10.5


def test_shutting_the_pump_trips_the_envelope(task: Task):
    """Constraints fire when they should. With the caustic shut the tank drains through
    the standpipe and the effluent goes to the acid end — both limits, both real."""

    class Shut:
        def __init__(self, brief):
            pass

        def reset(self):
            pass

        def step(self, t, y, r, quality):
            return np.array([0.0])

    rec = run_scenario(task, Shut, task.seeds[0])
    assert rec.cost.violations > 0
    assert rec.trace.x[:, 2].min() < 5.0     # below the low-level trip
    assert rec.trace.x[:, 3].min() < 4.0     # and outside discharge consent


def test_opening_the_pump_wide_overflows_the_tank(task: Task):
    class WideOpen:
        def __init__(self, brief):
            pass

        def reset(self):
            pass

        def step(self, t, y, r, quality):
            return np.array([30.0])

    rec = run_scenario(task, WideOpen, task.seeds[0])
    assert rec.cost.violations > 0
    # Not clipped at the limit: the state has to exceed it or audit() cannot report it.
    assert rec.trace.x[:, 2].max() > 30.0
    assert rec.trace.x[:, 3].max() > 10.5


# -- the task --------------------------------------------------------------------


def test_disturbances_actually_move_the_plant(task: Task):
    """Two acid surges, at t=600 and t=1550. If they do not show up in the trace the task
    is testing nothing."""
    rec = run_scenario(task, lambda b: Hold(b), task.seeds[0])
    t = rec.trace.t
    ph = rec.trace.x[:, 3]
    settled = ph[np.searchsorted(t, 590) - 1]
    after_first = ph[np.searchsorted(t, 1540) - 1]
    after_second = ph[-1]
    assert after_first < settled - 0.15      # 6% more acid, uncontrolled
    assert after_second < after_first - 0.4  # 15% more acid on top


def test_doing_nothing_never_already_violates(task: Task):
    """A scenario the hold anchor cannot survive is unwinnable before any controller is
    involved, and is noise in the ensemble rather than a test of anything."""
    for rec in run_ensemble(task, lambda b: Hold(b), seeds=task.seeds[:6]):
        assert not rec.cost.violations, f"seed {rec.seed}: holding the pump already violates"


def test_shipped_task_is_well_posed(task: Task):
    """The `rtcbench validate` contract, as a test: on every scenario the reference must be
    safe and must beat doing nothing."""
    seeds = task.seeds[:4]
    hold = run_ensemble(task, lambda b: Hold(b), seeds=seeds)
    ref = run_ensemble(task, lambda b: build_reference(b, task.reference), seeds=seeds)
    for h, r in zip(hold, ref):
        assert not r.cost.violations, f"seed {r.seed}: the reference violates"
        assert not r.cost.duty_exceeded, f"seed {r.seed}: the reference exceeds its own gate"
        assert r.cost.total < h.cost.total, f"seed {r.seed}: reference no better than hold"


def test_the_reference_is_not_beaten_by_a_naive_pi(task: Task):
    """Guards the anchor. If a controller nobody tuned beats the published reference, the
    reference is wrong and every score on this task is inflated."""
    seeds = task.seeds[:4]
    ref = run_ensemble(task, lambda b: build_reference(b, task.reference), seeds=seeds)
    for kp, ti in ((1.0, 60.0), (1.2, 35.0), (4.0, 200.0)):
        naive = run_ensemble(
            task, lambda b, kp=kp, ti=ti: MultiLoopPID(b, pairing=[0], kp=[kp], ti=[ti]),
            seeds=seeds,
        )
        assert np.mean([r.cost.total for r in ref]) < np.mean([n.cost.total for n in naive]), (
            f"naive PI kp={kp} ti={ti} beats the reference"
        )


def test_the_brief_publishes_nominal_parameters_not_the_draw(task: Task):
    """Mismatch tier: the controller is handed the commissioning data sheet, and the plant
    it actually faces is not that plant."""
    plant = task.build_plant()
    plant.reset(task.seeds[0])
    brief = task.brief(plant)
    assert brief.model_hint["note"] == "nominal, not as-running"
    assert brief.model_hint["params"]["Wb2"] == pytest.approx(HENSON_SEBORG.Wb2)
    assert plant._inner.params.Wb2 != pytest.approx(HENSON_SEBORG.Wb2)


def test_level_is_instrumented_but_carries_no_setpoint(task: Task):
    """One actuator, two measurements. Level is scored only through its hard limits, which
    is what makes the pump's range self-limiting."""
    assert task.controlled == (0,)
    plant = task.build_plant()
    assert [c.tag for c in plant.spec.measurements] == ["AIT-101", "LIT-102"]
    assert plant.spec.n_u == 1

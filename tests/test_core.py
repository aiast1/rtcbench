"""Tests for the properties the benchmark's claims actually rest on.

Not coverage for its own sake. Each test here corresponds to a sentence in DESIGN.md that
would be false if it failed — determinism, the sealed boundary, the safety gate, the
anchoring, and the record being able to support its own score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from rtcbench import (
    Cost,
    InstrumentConfig,
    InstrumentedPlant,
    Observation,
    RunRecord,
    SensorConfig,
    Task,
    anchored_score,
    cvar,
    run_scenario,
)
from rtcbench.baselines import Hold, MultiLoopPID, build_reference
from rtcbench.harness import run_ensemble, score_against_anchors
from rtcbench.instruments import ActuatorConfig
from rtcbench.plants.four_tank import P_MINUS, P_PLUS, FourTank, steady_state
from rtcbench.score import score_scenario

TASK = Path(__file__).resolve().parents[1] / "tasks" / "four_tank_v1.yaml"


@pytest.fixture(scope="module")
def task() -> Task:
    return Task.load(TASK)


# -- determinism ----------------------------------------------------------------


def test_plant_is_a_pure_function_of_seed_and_controls():
    """Every reproducibility claim in the project reduces to this one."""
    controls = [np.array([3.0 + 0.1 * np.sin(k / 5), 3.0]) for k in range(50)]

    def trajectory():
        p = FourTank(P_MINUS, mismatch={"a1": 0.05, "k1": 0.05})
        p.reset(1234)
        return np.array([p.step(u).y for u in controls])

    np.testing.assert_array_equal(trajectory(), trajectory())


def test_different_seeds_draw_different_plants():
    a, b = FourTank(P_MINUS, mismatch={"a1": 0.05}), FourTank(P_MINUS, mismatch={"a1": 0.05})
    a.reset(1)
    b.reset(2)
    assert a.params.a1 != b.params.a1


def test_no_mismatch_means_no_draw():
    p = FourTank(P_MINUS)
    p.reset(99)
    assert p.params == P_MINUS


def test_instrument_noise_is_independent_of_the_parameter_draw():
    """Changing what the plant does must not reshuffle the measurement noise, or scenarios
    stop being comparable across the mismatch ensemble."""

    def noise_stream(mismatch):
        plant = InstrumentedPlant(
            FourTank(P_MINUS, mismatch=mismatch),
            InstrumentConfig(sensors=SensorConfig(noise_std=0.1)),
        )
        plant.reset(7)
        ideal = FourTank(P_MINUS, mismatch=mismatch)
        ideal.reset(7)
        return np.array([plant.step(np.array([3.0, 3.0])).y - ideal.step(np.array([3.0, 3.0])).y
                         for _ in range(20)])

    np.testing.assert_allclose(noise_stream({"a1": 0.05}), noise_stream({"a1": 0.20}))


# -- the sealed boundary --------------------------------------------------------


def test_observation_carries_no_state():
    """The controller sees measurements, never state. If this ever grows a field, the
    benchmark has quietly changed what it measures."""
    assert set(Observation.__dataclass_fields__) == {"t", "y", "quality"}


def test_only_lower_tanks_are_measured():
    p = FourTank(P_MINUS)
    obs = p.reset(0)
    assert obs.y.shape == (2,)
    assert p.audit().x.shape == (4,)  # ground truth is wider than what is measured


def test_upper_tanks_are_still_constrained():
    """A controller must avoid overflowing tanks it cannot see."""
    signals = {c.signal for c in FourTank(P_MINUS).spec.constraints}
    assert {"h3", "h4"} <= signals


# -- physics --------------------------------------------------------------------


def test_steady_state_is_actually_stationary():
    p = FourTank(P_MINUS, sample_time=10.0)
    p.reset(0)
    before = p.audit().x.copy()
    for _ in range(50):
        p.step(np.array([3.0, 3.0]))
    np.testing.assert_allclose(p.audit().x, before, rtol=1e-6)


def test_the_zero_moves_between_the_two_presets():
    """The reason this plant is in the pack at all."""
    assert P_MINUS.minimum_phase
    assert not P_PLUS.minimum_phase


def test_analytic_steady_state_matches_johanssons_operating_point():
    h = steady_state(P_MINUS, np.array([3.0, 3.0]))
    np.testing.assert_allclose(h[:2], [12.26, 12.78], atol=0.05)


# -- instrument layer -----------------------------------------------------------

def test_deadtime_delays_the_reading():
    plant = InstrumentedPlant(
        FourTank(P_MINUS), InstrumentConfig(sensors=SensorConfig(deadtime_samples=2))
    )
    first = plant.reset(0)
    seen = [plant.step(np.array([6.0, 3.0])).y[0] for _ in range(4)]
    # Two scans of transport delay: the first two readings still show the initial level.
    assert seen[0] == pytest.approx(first.y[0])
    assert seen[1] == pytest.approx(first.y[0])
    assert seen[2] > first.y[0]


def test_rate_limit_caps_actuator_movement():
    plant = InstrumentedPlant(
        FourTank(P_MINUS, sample_time=1.0),
        InstrumentConfig(actuators=ActuatorConfig(rate_limit=0.5)),
    )
    plant.reset(0)
    plant.step(np.array([10.0, 10.0]))
    # Starts at the duty point (3.0 V) and may move 0.5 V/s * 1 s.
    np.testing.assert_allclose(plant.u_actual, [3.5, 3.5])


def test_stiction_blocks_small_moves():
    plant = InstrumentedPlant(
        FourTank(P_MINUS), InstrumentConfig(actuators=ActuatorConfig(deadband=0.5))
    )
    plant.reset(0)
    plant.step(np.array([3.1, 3.0]))
    np.testing.assert_allclose(plant.u_actual, [3.0, 3.0])  # request too small to break stiction
    plant.step(np.array([4.0, 3.0]))
    np.testing.assert_allclose(plant.u_actual, [4.0, 3.0])


def test_dropout_flags_quality_and_holds_the_last_good_value():
    plant = InstrumentedPlant(
        FourTank(P_MINUS), InstrumentConfig(sensors=SensorConfig(dropout_prob=1.0))
    )
    first = plant.reset(3)
    obs = plant.step(np.array([9.0, 9.0]))
    assert not obs.quality.any()
    np.testing.assert_allclose(obs.y, first.y)


def test_actuation_starts_at_the_duty_point_not_at_zero():
    plant = InstrumentedPlant(FourTank(P_MINUS))
    plant.reset(0)
    np.testing.assert_allclose(plant.u_actual, [3.0, 3.0])


# -- scoring --------------------------------------------------------------------


def test_anchors_map_to_zero_and_one():
    assert anchored_score(1.0, hold=1.0, reference=0.2) == pytest.approx(0.0)
    assert anchored_score(0.2, hold=1.0, reference=0.2) == pytest.approx(1.0)
    assert anchored_score(0.1, hold=1.0, reference=0.2) > 1.0  # beating the reference


def test_degenerate_anchors_raise_rather_than_return_nonsense():
    with pytest.raises(ValueError, match="degenerate"):
        anchored_score(0.5, hold=0.3, reference=0.3)


def test_a_violation_gates_the_scenario_however_good_the_cost():
    perfect_but_unsafe = Cost(iae=0.0, tv=0.0, violations=1, overruns=0, total=0.0)
    s = score_scenario(seed=1, submission=perfect_but_unsafe, hold=1.0, reference=0.5)
    assert s.gated and s.score == 0.0


def test_cvar_reports_the_bad_tail_not_the_average():
    scores = [1.0] * 9 + [-5.0]  # 10 scenarios, so the worst 10% is exactly one of them
    assert cvar(scores, 0.10) == pytest.approx(-5.0)
    assert np.mean(scores) > 0  # which is exactly why the mean is not the headline


def test_cvar_averages_the_whole_tail_not_just_the_single_worst():
    scores = [1.0] * 18 + [0.0, -5.0]  # 20 scenarios -> the tail is two of them
    assert cvar(scores, 0.10) == pytest.approx(-2.5)


def test_cvar_always_counts_at_least_one_scenario():
    assert cvar([0.4], 0.10) == pytest.approx(0.4)


# -- harness --------------------------------------------------------------------


def test_hold_scores_zero_and_the_reference_scores_one(task: Task):
    """The definition of the scale, checked end to end rather than assumed."""
    seeds = task.seeds[:3]
    hold_factory = lambda b: Hold(b)  # noqa: E731
    ref_factory = lambda b: build_reference(b, task.reference)  # noqa: E731

    hold = run_ensemble(task, hold_factory, seeds=seeds)
    ref = run_ensemble(task, ref_factory, seeds=seeds)

    as_hold = score_against_anchors(task, hold, hold, ref)
    as_ref = score_against_anchors(task, ref, hold, ref)
    assert as_hold.mean == pytest.approx(0.0, abs=1e-9)
    assert as_ref.mean == pytest.approx(1.0, abs=1e-9)


def test_a_controller_that_raises_fails_its_run_rather_than_vanishing(task: Task):
    class Exploding:
        def __init__(self, brief):
            pass

        def reset(self):
            pass

        def step(self, t, y, r, quality):
            raise RuntimeError("boom")

    rec = run_scenario(task, Exploding, task.seeds[0])
    assert rec.failed and "boom" in rec.failure
    assert rec.cost.violations > 0  # gated, not silently dropped from the ensemble


def test_a_controller_that_throws_in_init_is_recorded_not_raised(task: Task):
    """A submission that misreads the interface must score zero, not abort the ensemble.

    Found when a model wrote `brief['control_period']` against a dataclass: the constructor
    raised, the exception escaped run_scenario, and one bad submission destroyed every other
    scenario's result in the same run.
    """

    class BadInterface:
        def __init__(self, brief):
            raise TypeError("'TaskBrief' object is not subscriptable")

    rec = run_scenario(task, BadInterface, task.seeds[0])
    assert rec.failed
    assert rec.cost.violations > 0          # gated like any other failure
    assert rec.meta["steps"] == 0
    assert "not subscriptable" in rec.failure


def test_one_broken_submission_does_not_abort_the_ensemble(task: Task):
    class BadInterface:
        def __init__(self, brief):
            raise KeyError("control_period")

    records = run_ensemble(task, BadInterface, seeds=task.seeds[:3])
    assert len(records) == 3               # every seed still produced a record
    assert all(r.failed for r in records)


def test_a_controller_returning_the_wrong_shape_fails_clearly(task: Task):
    class WrongShape:
        def __init__(self, brief):
            pass

        def reset(self):
            pass

        def step(self, t, y, r, quality):
            return np.array([1.0])  # plant has two actuators

    rec = run_scenario(task, WrongShape, task.seeds[0])
    assert rec.failed and "expected 2" in rec.failure


def test_scan_overruns_hold_the_last_output_instead_of_crashing(task: Task):
    import time

    class Slow:
        def __init__(self, brief):
            self.u = np.asarray(brief.initial_u, dtype=float)

        def reset(self):
            pass

        def step(self, t, y, r, quality):
            time.sleep(task.budget.step_seconds * 1.5)
            return self.u + 5.0  # would be a huge kick if it were ever applied

    rec = run_scenario(task, Slow, task.seeds[0])
    assert rec.cost.overruns > 0
    # The overrunning output was discarded, so the actuators never took the kick.
    assert np.allclose(rec.trace.u_commanded[0], [3.0, 3.0])


def test_disturbances_actually_move_the_plant(task: Task):
    """The leak at t=700 must show up in the trace, or the task is testing nothing."""
    rec = run_scenario(task, lambda b: Hold(b), task.seeds[0])
    k = int(700 / task.sample_time)
    before = rec.trace.x[k - 5, 0]
    after = rec.trace.x[-1, 0]
    assert after < before - 1.0  # tank 1 drains through the enlarged orifice


# -- records --------------------------------------------------------------------


def test_record_round_trips_through_json(tmp_path, task: Task):
    rec = run_scenario(task, lambda b: Hold(b), task.seeds[0], controller_id="hold")
    loaded = RunRecord.load(rec.save(tmp_path / "r.json"))
    assert loaded.cost.total == pytest.approx(rec.cost.total)
    np.testing.assert_allclose(loaded.trace.y, rec.trace.y)
    np.testing.assert_array_equal(np.isnan(loaded.trace.r), np.isnan(rec.trace.r))


def test_a_record_supports_its_own_score(task: Task):
    """A published score must be recomputable from the record, not taken on faith."""
    from rtcbench.metrics import scenario_cost

    rec = run_scenario(task, lambda b: build_reference(b, task.reference), task.seeds[0])
    plant = task.build_plant()
    y_spans = np.array([c.span for c in plant.spec.measurements])
    u_spans = np.array([c.span for c in plant.spec.actuators])
    tr = rec.trace
    again = scenario_cost(
        tr.y, tr.r, tr.u_commanded, rec.controlled, y_spans, u_spans, rec.sample_time,
        w_error=task.scoring.w_error, w_effort=task.scoring.w_effort,
        violations=int(tr.violations.sum()), overruns=int(tr.overrun.sum()),
    )
    assert again.total == pytest.approx(rec.cost.total, rel=1e-12)


# -- tasks ----------------------------------------------------------------------


def test_task_hash_changes_with_content():
    base = {
        "task_id": "t", "tier": "blind", "plant": {"kind": "four_tank"}, "horizon": 10.0,
        "controlled": [0], "setpoints": [{"t": 0, "values": [1.0]}],
        "scenarios": {"seeds": [1]},
    }
    a = Task.from_dict(base)
    b = Task.from_dict({**base, "horizon": 20.0})
    assert a.content_hash != b.content_hash


def test_blind_tier_hands_over_no_model(task: Task):
    blind = Task.from_dict({**_raw(task), "tier": "blind"})
    plant = blind.build_plant()
    assert blind.brief(plant).model_hint == {}


def test_mismatch_tier_publishes_the_nominal_not_the_draw(task: Task):
    plant = task.build_plant()
    plant.reset(task.seeds[0])
    brief = task.brief(plant)
    assert brief.model_hint["note"] == "nominal, not as-running"
    assert brief.model_hint["params"]["a1"] == pytest.approx(P_MINUS.a1)
    assert plant._inner.params.a1 != pytest.approx(P_MINUS.a1)  # the plant really did move


def test_setpoint_ramp_interpolates():
    t = Task.from_dict({
        "task_id": "t", "tier": "blind", "plant": {"kind": "four_tank"}, "horizon": 100.0,
        "controlled": [0],
        "setpoints": [{"t": 0, "values": [10.0]}, {"t": 50, "values": [20.0], "ramp": 10}],
        "scenarios": {"seeds": [1]},
    })
    assert t.setpoints.at(49.0) == pytest.approx([10.0])
    assert t.setpoints.at(55.0) == pytest.approx([15.0])
    assert t.setpoints.at(70.0) == pytest.approx([20.0])


def test_a_sealed_task_refuses_to_invent_an_ensemble():
    with pytest.raises(ValueError, match="Refusing to invent"):
        Task.from_dict({
            "task_id": "t", "tier": "blind", "plant": {"kind": "four_tank"}, "horizon": 10.0,
            "controlled": [0], "setpoints": [{"t": 0, "values": [1.0]}],
            "scenarios": {"count": 5},
        })


def test_sealed_seeds_are_reproducible_from_the_salt():
    spec = {
        "task_id": "t", "tier": "blind", "plant": {"kind": "four_tank"}, "horizon": 10.0,
        "controlled": [0], "setpoints": [{"t": 0, "values": [1.0]}],
        "scenarios": {"count": 8, "salt": "held-by-maintainers"},
    }
    assert Task.from_dict(spec).seeds == Task.from_dict(spec).seeds
    other = {**spec, "scenarios": {"count": 8, "salt": "different"}}
    assert Task.from_dict(spec).seeds != Task.from_dict(other).seeds


def test_shipped_task_is_well_posed(task: Task):
    """The `rtcbench validate` contract, as a test: on every scenario the reference must be
    safe and must beat doing nothing."""
    seeds = task.seeds[:4]
    hold = run_ensemble(task, lambda b: Hold(b), seeds=seeds)
    ref = run_ensemble(task, lambda b: build_reference(b, task.reference), seeds=seeds)
    for h, r in zip(hold, ref):
        assert not h.cost.violations, f"seed {h.seed}: doing nothing already violates"
        assert not r.cost.violations, f"seed {r.seed}: the reference violates"
        assert r.cost.total < h.cost.total, f"seed {r.seed}: reference no better than hold"


def test_the_reference_is_not_beaten_by_a_naive_pi(task: Task):
    """Guards the anchor. The first draft of four_tank_v1 shipped gains that the example
    submission beat, which silently inflated every score on the task."""
    seeds = task.seeds[:4]
    ref = run_ensemble(task, lambda b: build_reference(b, task.reference), seeds=seeds)
    naive = run_ensemble(
        task, lambda b: MultiLoopPID(b, pairing=[0, 1], kp=1.2, ti=35.0), seeds=seeds
    )
    assert np.mean([r.cost.total for r in ref]) < np.mean([n.cost.total for n in naive])


def _raw(task: Task) -> dict:
    import yaml

    return yaml.safe_load(TASK.read_text(encoding="utf-8"))


# -- the blind pair, and where the floor lives ------------------------------------


def _blind() -> Task:
    return Task.load(TASK.parent / "four_tank_blind_v1.yaml")


def test_the_blind_tier_withholds_the_model(task: Task):
    blind = _blind()
    plant = blind.build_plant()
    plant.reset(blind.seeds[0])
    assert blind.brief(plant).model_hint == {}, "a blind brief must carry no model"
    # ...while its mismatch twin publishes the nominal, so the pair differ in exactly this.
    mplant = task.build_plant()
    mplant.reset(task.seeds[0])
    assert task.brief(mplant).model_hint != {}


def test_the_blind_pair_shares_its_anchors_exactly(task: Task):
    """The whole point of the pair is that only the INFORMATION differs.

    If the anchors moved, a score difference between four_tank_v1 and four_tank_blind_v1
    would confound 'the model was withheld' with 'the task changed', and the comparison
    would be worthless.
    """
    blind = _blind()
    seeds = task.seeds[:4]
    for build in (lambda b: Hold(b), lambda b: build_reference(b, task.reference)):
        a = run_ensemble(task, build, seeds=seeds)
        b = run_ensemble(blind, build, seeds=seeds)
        for x, y in zip(a, b):
            assert x.cost.total == pytest.approx(y.cost.total, rel=1e-12), (
                f"seed {x.seed}: anchors differ between the mismatch and blind twins"
            )


def test_the_blind_description_does_not_leak_the_model():
    """A blind brief that names the parameters in prose is not blind."""
    blind = _blind()
    plant = blind.build_plant()
    text = blind.brief(plant).description.lower()
    for leak in ("gamma", "0.071", "0.057", "3.33", "3.35", "cross-section", "a1", "k1"):
        assert leak not in text, f"the blind description leaks {leak!r}"


def test_a_scenario_keeps_its_spread_but_a_task_is_clipped_into_the_suite():
    """The floor belongs in the aggregation, not in the measurement.

    An earlier version clipped every scenario at -1.0. That bounded the suite mean, which was
    the real problem, but it also flattened four_tank_nmp's whole field -- which genuinely
    spread from -0.59 to -6.80 -- into a column of identical -1.000s that ranked nothing.
    """
    from rtcbench.score import SCENARIO_FLOOR, TASK_FLOOR, suite_score

    assert SCENARIO_FLOOR < TASK_FLOOR, "a scenario must be free to spread further than a task"

    catastrophic = Cost(iae=9.0, tv=0.0, violations=0, overruns=0, total=9.0)
    s = score_scenario(seed=1, submission=catastrophic, hold=0.10, reference=0.02)
    assert s.score < -50 or s.score == pytest.approx(SCENARIO_FLOOR), (
        "a scenario score should keep its magnitude down to the sanity clamp"
    )

    # One disastrous task cannot drag the suite below its own floor divided by task count.
    assert suite_score([1.0, -6.8, 1.0]) == pytest.approx((1.0 - 1.0 + 1.0) / 3)
    assert suite_score([-6.8]) == pytest.approx(TASK_FLOOR)
    # ...and a merely-bad task is NOT clipped, so ordering survives above the floor.
    assert suite_score([0.0, -0.5]) == pytest.approx(-0.25)


def test_core_never_imports_the_validators():
    """`validators/` is dev-only and depends on scipy and Cantera.

    If anything under src/ imported it, `pip install rtcbench` would stop being a
    two-package install and the promise that the suite runs on any laptop would quietly
    become false. Cheaper to assert than to discover from a user's traceback.
    """
    root = Path(__file__).resolve().parents[1] / "src" / "rtcbench"
    offenders = [
        f"{p.relative_to(root)}:{i}"
        for p in root.rglob("*.py")
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if line.startswith(("import validators", "from validators"))
        or "import scipy" in line or "import cantera" in line
    ]
    assert not offenders, f"core must not reach into the validator tree: {offenders}"


def test_no_task_file_contains_the_reference_gains():
    """The anchor lives in tasks/references/, not in the file a competitor is handed.

    An agent with filesystem access read `kp` and `ti` out of a task file and shipped them
    as its own tuning, scoring exactly +1.000. Splitting the anchor out does not stop a
    determined submission (an absolute path still reads anything) but it does stop the
    obvious route, and it stops anyone being handed the answer by accident.
    """
    tasks = sorted((TASK.parent).glob("*_v*.yaml"))
    assert tasks, "no task files found"
    for t in tasks:
        raw = yaml.safe_load(t.read_text(encoding="utf-8")) or {}
        assert "reference" not in raw, (
            f"{t.name} carries its reference inline; move it to tasks/references/{t.name}"
        )


def test_every_task_has_a_reference_sidecar():
    for t in sorted((TASK.parent).glob("*_v*.yaml")):
        side = TASK.parent / "references" / t.name
        assert side.is_file(), f"{t.name} has no reference anchor in tasks/references/"
        assert (yaml.safe_load(side.read_text(encoding="utf-8")) or {}).get("reference"), (
            f"{side.name} has no reference block"
        )


def test_two_anchors_for_one_task_is_refused():
    """Ambiguity here would silently pick one anchor and re-denominate a task's scores."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "references").mkdir()
        body = (TASK.read_text(encoding="utf-8")
                + "\nreference:\n  kind: pid\n  pairing: [0, 1]\n  kp: [9.0, 9.0]\n"
                  "  ti: [9.0, 9.0]\n")
        (root / "x_v1.yaml").write_text(body, encoding="utf-8")
        (root / "references" / "x_v1.yaml").write_text(
            "reference:\n  kind: pid\n  pairing: [0, 1]\n  kp: [1.0, 1.0]\n  ti: [1.0, 1.0]\n",
            encoding="utf-8")
        with pytest.raises(ValueError, match="Two anchors"):
            Task.load(root / "x_v1.yaml")

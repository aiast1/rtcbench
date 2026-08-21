"""Tests for Column A — the pack's ill-conditioned plant.

Properties, not coverage. Each test here is a sentence about the plant that would be false
if it failed: that it reproduces Skogestad's published operating point, that its two
directions really do differ by two orders of magnitude in gain, that the direction a naive
controller reaches for is the one that destroys it, and that the task built on it is
well-posed.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rtcbench.baselines import Hold, MultiLoopPID, build_reference
from rtcbench.harness import run_ensemble, run_scenario
from rtcbench.plants import build_plant
from rtcbench.plants.column_a import (
    FEED_STAGE,
    N_STAGES,
    NOMINAL_LV,
    P_COLUMN_A,
    ColumnA,
    ColumnAParams,
    steady_state,
)
from rtcbench.task import Task

TASK = Path(__file__).resolve().parents[1] / "tasks" / "column_a_v1.yaml"

U0 = np.array(NOMINAL_LV)
"""Skogestad's duty point: L = 2.70629, V = 3.20629 kmol/min."""


@pytest.fixture(scope="module")
def task() -> Task:
    return Task.load(TASK)


@pytest.fixture(scope="module")
def gain_matrix() -> np.ndarray:
    """The 2x2 steady-state gain from (L, V) to (yD, xB), by central differences.

    Estimated numerically from the plant's own equilibrium map rather than read off a
    published table, so it tests this implementation and not the paper.
    """
    base = steady_state(P_COLUMN_A, U0)
    eps = 1e-5
    g = np.zeros((2, 2))
    for j, step in enumerate(((eps, 0.0), (0.0, eps))):
        hi = steady_state(P_COLUMN_A, U0 + step, guess=base)
        lo = steady_state(P_COLUMN_A, U0 - step, guess=base)
        g[0, j] = (hi[-1] - lo[-1]) / (2 * eps)     # d yD / du_j
        g[1, j] = (hi[0] - lo[0]) / (2 * eps)       # d xB / du_j
    return g


# -- the published operating point -----------------------------------------------


def test_reproduces_skogestads_operating_point():
    """41 stages, feed on 21, alpha 1.5, L = 2.70629, V = 3.20629 -> 99% / 1%, D = B = 0.5.

    This is the number every other claim about Column A is quoted against.
    """
    x = steady_state(P_COLUMN_A, U0)
    assert x.shape == (41,)
    assert x[-1] == pytest.approx(0.99, abs=1e-4)     # yD, distillate
    assert x[0] == pytest.approx(0.01, abs=1e-4)      # xB, bottoms
    plant = ColumnA()
    plant.reset(0)
    audit = plant.audit()
    assert audit.x[-2] == pytest.approx(0.5, abs=1e-6)   # D
    assert audit.x[-1] == pytest.approx(0.5, abs=1e-6)   # B


def test_the_column_is_the_published_size():
    plant = ColumnA()
    assert (N_STAGES, FEED_STAGE) == (41, 21)
    # 39 trays plus reboiler and condenser, then the two net product rates.
    assert len(plant.spec.state_names) == 41 + 2
    assert plant.reset(0).y.shape == (2,)


def test_the_dominant_time_constant_is_about_194_minutes():
    """Skogestad's column A is quoted at tau_1 = 194 min. Measured here off the linearised
    balances, which is also what sets the RK4 substep count at the other end of the
    spectrum."""
    from rtcbench.plants.column_a import _flows, _holdup_inverse, _jacobian

    x = steady_state(P_COLUMN_A, U0)
    liquid, vapour, distillate, _ = _flows(P_COLUMN_A, U0, N_STAGES, FEED_STAGE)
    j = _jacobian(x, liquid, vapour, distillate, P_COLUMN_A, _holdup_inverse(P_COLUMN_A, 41))
    eig = np.linalg.eigvals(j).real
    assert -1.0 / eig.max() == pytest.approx(194.0, rel=0.02)
    # ...and the fastest mode is what h = 0.05 min has to stay stable against.
    assert np.abs(eig).max() * 0.05 < 2.78


# -- the pathology: ill-conditioning ---------------------------------------------


def test_the_steady_state_gain_matrix_is_severely_ill_conditioned(gain_matrix):
    """The reason this plant is in the pack at all.

    Skogestad's published values for column A are cond(G) = 141.7 and RGA lambda_11 = 35.1;
    recomputed from these equations they come out at 145 and 35.9.
    """
    sigma = np.linalg.svd(gain_matrix, compute_uv=False)
    condition = sigma[0] / sigma[1]
    assert condition > 50.0, "a plant this well conditioned would not test directionality"
    assert condition == pytest.approx(141.7, rel=0.05)

    rga = gain_matrix * np.linalg.inv(gain_matrix).T
    assert rga[0, 0] > 20.0
    assert rga[0, 0] == pytest.approx(35.1, rel=0.05)
    # Negative off-diagonal RGA: the other pairing is not merely worse, it is forbidden.
    assert rga[0, 1] < 0.0


def test_the_two_directions_differ_by_two_orders_of_magnitude(gain_matrix):
    """Same size of composition move, hundredfold difference in the flow move it costs."""
    _, sigma, right = np.linalg.svd(gain_matrix)
    assert sigma[0] / sigma[1] > 50.0
    # A 0.005 composition move costs 0.005/sigma of input travel in each direction.
    cheap, dear = 0.005 / sigma[0], 0.005 / sigma[1]
    assert cheap < 0.01 and dear > 0.3
    # The expensive direction is the one where both flows move together.
    weak = right[1] / np.abs(right[1]).max()
    assert weak[0] * weak[1] > 0.9


def test_a_small_input_error_is_a_large_composition_error():
    """0.4% on one flow, and the bottoms impurity more than doubles.

    This is the pathology stated as a step test rather than as a singular value: nothing
    about the plant is unstable or slow here, it is simply enormously sensitive to which
    way the two flows move relative to each other.
    """
    imbalanced = _hold_for(ColumnA(), U0 + np.array([0.01, 0.0]), minutes=400)
    balanced = _hold_for(ColumnA(), U0 + np.array([0.4, 0.4]), minutes=400)
    assert imbalanced[0] > 2.0 * 0.010     # xB: 0.010 -> 0.024 for a 0.01 kmol/min nudge
    assert abs(balanced[0] - 0.010) < 0.005  # 40x the flow move, a quarter of the effect


def test_an_inverse_based_controller_is_destroyed_by_a_calibration_error():
    """Skogestad's headline result for this column, reproduced closed-loop.

    A controller that inverts G reaches the sluggish direction by commanding two large
    moves whose *difference* is the small quantity it actually wants (G^-1 has entries of
    ~40 here). Give the two flow controllers a 5% span error and that difference is not
    what gets delivered; the same factor of 40 turns the leftover into a large split change,
    the controller reads it as needing yet more separation, and it walks both flows into
    their limits. The decentralized reference, which never inverts anything, does not care.
    """
    miscalibrated = replace(P_COLUMN_A, gain_L=1.05, gain_V=0.95)
    setpoint = np.array([0.994, 0.994])          # both products purer
    g_model = _measured_gain_in_purity_coordinates()

    def inverse_based(y, u):
        return u + 0.05 * (np.linalg.inv(g_model) @ (setpoint - y))

    def decentralized(y, u, state={"i": np.zeros(2)}):  # noqa: B006 - a closure's memory
        state["i"] += (3.5 / 5.0) * (setpoint - y)
        return U0 + 3.5 * (setpoint - y) + state["i"]

    inverted = _closed_loop(miscalibrated, inverse_based, minutes=300)
    diagonal = _closed_loop(miscalibrated, decentralized, minutes=300)

    assert "bottoms_offspec" in inverted.violations
    assert inverted.x[0] > 0.15                    # bottoms driven far off-spec
    assert not diagonal.violations
    assert abs(diagonal.x[0] - 0.006) < 0.005      # the decentralized loop just works

    # ...and on a perfectly calibrated column the inverse-based controller is the better
    # one, which is exactly why it is the tempting answer.
    perfect = _closed_loop(P_COLUMN_A, inverse_based, minutes=300)
    assert abs(perfect.x[0] - 0.006) < 0.001


def test_the_flow_calibration_error_does_not_move_the_starting_point():
    """Span error, not zero error: it is applied to the deviation from the duty point.

    Load-bearing for the ensemble's fairness. Applied to the flow itself, a 5% error would
    put some draws a long way off spec before the controller had done anything, and the
    authoring guide is explicit that a scenario decided by its initial condition is noise.
    """
    calibrated = ColumnA().reset(0)
    skewed = ColumnA(params=replace(P_COLUMN_A, gain_L=1.10, gain_V=0.90)).reset(0)
    np.testing.assert_allclose(skewed.y, calibrated.y, atol=1e-9)


# -- constraints ------------------------------------------------------------------


def test_squeezing_the_two_flows_together_trips_the_distillate_draw():
    """D = V - L is not measured and not defended by anything except the controller."""
    plant = ColumnA()
    plant.reset(0)
    plant.step(np.array([3.2, 3.2]))               # equal flows: no net distillate
    audit = plant.audit()
    assert audit.x[-2] == pytest.approx(0.0, abs=1e-9)
    assert "distillate_flow" in audit.violations


def test_off_spec_product_is_a_violation_of_the_true_state_not_of_the_reading():
    """Reflux up, boilup barely up: the distillate draw is starved, most of the light key
    leaves at the bottom, and the bottoms goes off-spec while the draws stay legal."""
    final = _closed_loop(P_COLUMN_A, lambda y, u: np.array([3.1, 3.25]), minutes=400)
    assert final.x[0] > 0.15                        # true bottoms composition
    assert final.x[-2] > 0.05 and final.x[-1] > 0.05  # ...and not because a draw was lost
    assert final.violations == ("bottoms_offspec",)


def test_compositions_are_clipped_to_physics_but_the_limits_are_not():
    """Mole fractions stay in [0, 1] because that is what a mole fraction is; the 0.85/0.15
    product limits sit well inside and are never clipped away."""
    plant = ColumnA()
    plant.reset(0)
    for _ in range(200):
        plant.step(np.array([4.5, 2.0]))            # maximum reflux, minimum boilup
    x = plant.audit().x[:41]
    assert x.min() >= 0.0 and x.max() <= 1.0
    assert plant.audit().violations                 # and it is loudly off-spec


# -- determinism and integration --------------------------------------------------


def test_plant_is_a_pure_function_of_seed_and_controls():
    controls = [U0 + np.array([0.05 * np.sin(k / 6), 0.05 * np.cos(k / 9)]) for k in range(40)]

    def trajectory():
        p = ColumnA(mismatch={"alpha": 0.02, "F": 0.02, "gain_L": 0.05})
        p.reset(4242)
        return np.array([p.step(u).y for u in controls])

    np.testing.assert_array_equal(trajectory(), trajectory())


def test_different_seeds_draw_different_columns():
    a, b = (ColumnA(mismatch={"alpha": 0.02}) for _ in range(2))
    a.reset(1)
    b.reset(2)
    assert a.params.alpha != b.params.alpha


def test_no_mismatch_means_no_draw():
    plant = ColumnA()
    plant.reset(99)
    assert plant.params == P_COLUMN_A


def test_the_steady_state_is_actually_stationary():
    """Newton, not "integrate until it looks settled". On a plant whose slowest mode is
    194 minutes the difference is visible for the whole scenario."""
    plant = ColumnA()
    before = plant.reset(0).y.copy()
    for _ in range(120):
        plant.step(U0)
    np.testing.assert_allclose(plant.audit().x[:41], steady_state(P_COLUMN_A, U0), atol=1e-9)
    np.testing.assert_allclose(plant.reset(0).y, before)


def test_the_substep_count_is_enough_for_the_integration_to_have_converged():
    """Guards the one number in the plant that can be lowered to make it faster."""
    coarse = _hold_for(ColumnA(substeps=20), np.array([3.1, 3.6]), minutes=60)
    fine = _hold_for(ColumnA(substeps=80), np.array([3.1, 3.6]), minutes=60)
    assert abs(coarse[0] - fine[0]) < 1e-8


def test_a_full_scenario_costs_a_fraction_of_a_second():
    """41 states x 400 periods x 20 RK4 substeps, and the harness runs 60+ scenarios per
    scoring pass — validate alone is 40 of them, and a scoring pass with both anchors is 60.

    A whole scenario integrates in about 0.3 s. The bound is 25x that because this is the
    one test here that measures the machine as well as the code: it exists to catch the
    stage balance being rewritten as a Python loop over 41 trays inside four RK4 stages,
    which costs about 20x and would turn a scoring pass from a minute into half an hour.
    """
    plant = ColumnA()
    plant.reset(0)
    started = time.perf_counter()
    for _ in range(400):
        plant.step(U0)
    assert time.perf_counter() - started < 8.0


def test_set_params_rejects_a_parameter_the_column_does_not_have():
    plant = ColumnA()
    plant.reset(0)
    with pytest.raises(ValueError, match="no parameter"):
        plant.set_params({"reflux_ratio": 2.0})


# -- the sealed boundary ----------------------------------------------------------


def test_only_the_two_products_are_measured():
    plant = ColumnA()
    obs = plant.reset(0)
    assert obs.y.shape == (2,)
    assert plant.audit().x.shape == (43,)          # 41 compositions + D + B
    # The 39 tray compositions carry the column's whole future response and none of them
    # reach the controller.
    assert len(plant.spec.measurements) == 2


def test_the_bottoms_analyser_reads_purity_not_impurity():
    """Both loops direct-acting. A reverse-acting bottoms loop would fail on sign, which is
    not the thing this plant is here to test."""
    plant = ColumnA()
    obs = plant.reset(0)
    assert obs.y[1] == pytest.approx(1.0 - plant.audit().x[0])
    assert plant.spec.measurements[1].name.endswith("(heavy key)")


# -- the task ---------------------------------------------------------------------


def test_the_task_builds_the_plant_it_claims():
    plant = build_plant({"kind": "column_a", "sample_time": 1.0})
    assert plant.spec.plant_id == "column_a"


def test_every_draw_starts_far_from_the_trips(task: Task):
    """The authoring guide's rule: size the mismatch against the initial condition.

    This column is savagely sensitive at fixed flows, so the draws start a long way off
    setpoint on purpose — but a draw that starts *at* a trip is a scenario no controller
    could win, and that is noise in the ensemble rather than a robustness test.
    """
    starts = []
    for seed in task.seeds:
        plant = task.build_plant()
        plant.reset(seed)
        starts.append(plant.audit().x[[40, 0]])
        plant.close()
    yd, xb = np.array(starts).T
    assert yd.min() > 0.90, f"a draw starts at yD = {yd.min():.3f}, too near the 0.85 trip"
    assert xb.max() < 0.075, f"a draw starts at xB = {xb.max():.3f}, too near the 0.15 trip"
    assert xb.max() > 0.02, "the draws are too tame to be a robustness test"


def test_the_setpoint_moves_are_the_expensive_ones(task: Task):
    """Both setpoint changes must sit in the sluggish direction, where a four-thousandth
    composition move costs a 16% change in internal traffic. A schedule that only asked for
    split changes would be satisfied by moves so small the plant would look easy."""
    g = _measured_gain_in_purity_coordinates()
    values = np.array([s.values for s in task.setpoints.segments])
    for move in (values[1] - values[0], values[2] - values[1]):
        travel = np.abs(np.linalg.solve(g, move)).min()
        assert travel > 0.1, f"{move} is reachable with only {travel:.4f} kmol/min of flow"
    # The first is pure separation: both products purer at constant split.
    assert (values[1] - values[0])[0] == pytest.approx((values[1] - values[0])[1])


def test_the_shipped_task_is_well_posed(task: Task):
    """The `rtcbench validate` contract as a test: on every scenario the reference must be
    safe and must beat doing nothing."""
    seeds = task.seeds[:3]
    hold = run_ensemble(task, lambda b: Hold(b), seeds=seeds)
    ref = run_ensemble(task, lambda b: build_reference(b, task.reference), seeds=seeds)
    for h, r in zip(hold, ref):
        assert not h.cost.violations, f"seed {h.seed}: doing nothing already violates"
        assert not r.cost.violations, f"seed {r.seed}: the reference violates"
        assert r.cost.total < h.cost.total, f"seed {r.seed}: reference no better than hold"
        assert not r.cost.duty_exceeded, f"seed {r.seed}: the reference trips its own duty gate"


def test_the_reference_is_not_beaten_by_a_naive_pi(task: Task):
    """Guards the anchor from both sides. An under-tuned reference inflates every score on
    the task forever, so the two obvious hand tunings both have to lose:

    * a cautious one loses on cost — it is still crawling when the scenario ends;
    * an aggressive one wins on cost and *trips a constraint*, which gates it to zero. That
      asymmetry is this plant's whole character, so it is asserted rather than averaged
      away: cost falls monotonically with gain right up to the cliff, and only the safety
      gate says where the cliff is.
    """
    seeds = task.seeds[:3]
    ref = run_ensemble(task, lambda b: build_reference(b, task.reference), seeds=seeds)
    cost = np.mean([r.cost.total for r in ref])
    assert not any(r.cost.violations for r in ref)

    def naive(kp, ti):
        return run_ensemble(
            task, lambda b, kp=kp, ti=ti: MultiLoopPID(b, pairing=[0, 1], kp=kp, ti=ti),
            seeds=seeds,
        )

    cautious = naive(0.8, 20.0)
    assert not any(r.cost.violations for r in cautious)
    assert cost < np.mean([r.cost.total for r in cautious]), "a cautious PI beats the anchor"

    aggressive = naive(8.0, 10.0)
    assert any(r.cost.violations for r in aggressive), "an aggressive PI is supposed to trip"


def test_the_duty_limit_catches_chatter_without_punishing_the_reference(task: Task):
    """The actuator duty gate, checked from both sides rather than assumed from a ratio."""
    limit = task.scoring.max_total_variation
    assert limit is not None

    ref = run_ensemble(task, lambda b: build_reference(b, task.reference), seeds=task.seeds[:3])
    assert not any(r.cost.duty_exceeded for r in ref)
    assert max(r.cost.tv for r in ref) < 0.75 * limit, "no headroom above the reference"

    class Dither:
        """Holds setpoint on average by shaking both flow controllers every period."""

        def __init__(self, brief):
            self.u = np.asarray(brief.initial_u, dtype=float)
            self.k = 0

        def reset(self):
            self.k = 0

        def step(self, t, y, r, quality):
            self.k += 1
            return self.u + (0.05 if self.k % 2 else -0.05)

    assert run_scenario(task, Dither, task.seeds[0]).cost.duty_exceeded


# -- helpers ----------------------------------------------------------------------


def _hold_for(plant: ColumnA, u: np.ndarray, *, minutes: int) -> tuple[float, float]:
    """Hold ``u`` and return the true ``(xB, yD)`` reached."""
    plant.reset(0)
    for _ in range(minutes):
        plant.step(u)
    x = plant.audit().x
    return float(x[0]), float(x[40])


def _closed_loop(params: ColumnAParams, control, *, minutes: int):
    """Run a bare closed loop against the ideal plant (no instruments) and return its audit."""
    plant = ColumnA(params=params)
    obs = plant.reset(0)
    u = U0.copy()
    for _ in range(minutes):
        u = np.clip(control(obs.y, u), [1.5, 2.0], [4.5, 5.0])
        obs = plant.step(u)
    return plant.audit()


def _measured_gain_in_purity_coordinates() -> np.ndarray:
    """The gain a controller would identify from (L, V) to (yD, bottoms purity)."""
    base = steady_state(P_COLUMN_A, U0)
    eps = 1e-5
    g = np.zeros((2, 2))
    for j, step in enumerate(((eps, 0.0), (0.0, eps))):
        hi = steady_state(P_COLUMN_A, U0 + step, guess=base)
        lo = steady_state(P_COLUMN_A, U0 - step, guess=base)
        g[0, j] = (hi[-1] - lo[-1]) / (2 * eps)
        g[1, j] = ((1.0 - hi[0]) - (1.0 - lo[0])) / (2 * eps)
    return g

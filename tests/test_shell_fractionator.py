"""Tests for the Shell heavy oil fractionator plant.

Mirrors ``tests/test_core.py``'s style: each test corresponds to a claim in the module
docstring or the task file that would be false if it failed, not coverage for its own sake.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rtcbench import InstrumentConfig, InstrumentedPlant, SensorConfig, Task
from rtcbench.baselines import Hold, build_reference
from rtcbench.harness import run_ensemble
from rtcbench.plants.shell_fractionator import (
    P_NOMINAL,
    ShellFractionator,
    rga,
    steady_state_gain,
)

TASK = Path(__file__).resolve().parents[1] / "tasks" / "shell_fractionator_v1.yaml"


@pytest.fixture(scope="module")
def task() -> Task:
    return Task.load(TASK)


# -- determinism ----------------------------------------------------------------


def test_plant_is_a_pure_function_of_seed_and_controls():
    controls = [np.array([0.1 * np.sin(k / 7), 0.05 * np.cos(k / 5), -0.03]) for k in range(60)]

    def trajectory():
        p = ShellFractionator(mismatch={"k11": 0.1, "l21": 0.08})
        p.reset(2024)
        return np.array([p.step(u).y for u in controls])

    np.testing.assert_array_equal(trajectory(), trajectory())


def test_different_seeds_draw_different_plants():
    a = ShellFractionator(mismatch={"k11": 0.2})
    b = ShellFractionator(mismatch={"k11": 0.2})
    a.reset(1)
    b.reset(2)
    assert a.params.K[0][0] != b.params.K[0][0]


def test_no_mismatch_means_no_draw():
    p = ShellFractionator()
    p.reset(99)
    assert p.params == P_NOMINAL


# -- the sealed boundary ----------------------------------------------------------


def test_observation_carries_all_three_outputs():
    p = ShellFractionator()
    obs = p.reset(0)
    assert obs.y.shape == (3,)
    assert p.audit().x.shape == (3,)


# -- physics: the published matrix -----------------------------------------------


def test_analytic_dc_gain_matches_the_published_gain_matrix():
    """FOPDT dead time and lag never affect DC gain, so a long constant hold must converge
    to exactly K @ u -- the direct, from-first-principles check that the state-space
    realization reproduces the published transfer-function matrix rather than some nearby
    approximation of it."""
    p = ShellFractionator(sample_time=1.0, substeps=10)
    p.reset(0)
    u = np.array([0.2, -0.1, 0.05])
    for _ in range(800):  # >> the largest time constant (60 min) plus the largest delay (32 min)
        obs = p.step(u)
    K = steady_state_gain()
    np.testing.assert_allclose(obs.y, K @ u, atol=1e-4)


def test_published_gain_matrix_reproduces_the_papers_own_rga_and_determinant():
    """Cross-checks the transcribed gain matrix against two independently-reported numbers
    (Jusagemal et al. 2011): det(K) = 20.85 and the paper's own RGA table. Agreement to the
    precision they publish, across all nine RGA entries at once, is strong evidence the nine
    gains were transcribed correctly (see the module docstring's "Numeric provenance")."""
    K = steady_state_gain()
    assert np.linalg.det(K) == pytest.approx(20.85, abs=0.01)
    expected = np.array(
        [[2.08, -0.73, -0.35], [3.42, 0.94, -3.36], [-4.50, 0.79, 4.71]]
    )
    np.testing.assert_allclose(rga(K), expected, atol=0.01)


def test_the_only_all_positive_pairing_is_diagonal():
    """The reason the shipped reference pairs u1-y1 / u2-y2 / u3-y3: row 1 of the RGA has
    exactly one positive entry, so y1 must pair with u1, which forces the rest."""
    lam = rga(steady_state_gain())
    assert lam[0].argmax() == 0 and lam[0, 0] > 0
    assert all(lam[0, j] < 0 for j in range(3) if j != 0)


def test_g33_has_zero_published_delay():
    p = ShellFractionator()
    p.reset(0)
    assert p.delay_samples(2, 2) == 0


# -- the pathology: every element's own dead time is honoured --------------------


@pytest.mark.parametrize("j", [0, 1, 2])
def test_each_elements_delay_is_honoured(j: int):
    """Output element i must not respond to input j before element (i, j)'s own published
    dead time has elapsed -- checked by superposition: stepping only u_j from rest isolates
    row i's response to exactly g_ij, since every other column stays at zero."""
    p = ShellFractionator(sample_time=1.0, substeps=10)
    p.reset(0)
    u = np.zeros(3)
    u[j] = 1.0
    n = [p.delay_samples(i, j) for i in range(3)]
    n_max = max(n)

    seen_zero_until = [True, True, True]
    first_nonzero = [None, None, None]
    for k in range(1, n_max + 3):
        obs = p.step(u)
        for i in range(3):
            if k <= n[i]:
                assert obs.y[i] == pytest.approx(0.0, abs=1e-12), (
                    f"y{i+1} responded to u{j+1} at step {k}, before its published delay "
                    f"of {n[i]} sample(s) had elapsed"
                )
            elif first_nonzero[i] is None and obs.y[i] != pytest.approx(0.0, abs=1e-12):
                first_nonzero[i] = k

    for i in range(3):
        assert first_nonzero[i] == n[i] + 1, (
            f"y{i+1} should first move on the sample right after its u{j+1} delay of "
            f"{n[i]} elapses (step {n[i] + 1}), but first moved at step {first_nonzero[i]}"
        )


def test_measured_disturbance_also_respects_its_own_delay():
    p = ShellFractionator(sample_time=1.0, substeps=10)
    p.reset(0)
    n = p.delay_samples(2, 0, disturbance=True)  # y3 vs d1
    p.set_params({"d1": 1.0})
    u = np.zeros(3)
    for k in range(1, n + 1):
        obs = p.step(u)
        assert obs.y[2] == pytest.approx(0.0, abs=1e-12)
    obs = p.step(u)
    assert obs.y[2] != pytest.approx(0.0, abs=1e-12)


# -- steady state -----------------------------------------------------------------


def test_a_held_steady_state_is_actually_stationary():
    p = ShellFractionator(sample_time=1.0, substeps=10)
    p.reset(0)
    u = np.array([0.15, -0.1, 0.02])
    for _ in range(1200):  # >> 20x the largest time constant (60 min)
        p.step(u)
    before = p.audit().x.copy()
    for _ in range(20):
        p.step(u)
    np.testing.assert_allclose(p.audit().x, before, rtol=1e-6)


def test_zero_input_and_zero_disturbance_is_the_trim_point():
    """Deviation-variable model: u=0, d=0 is on-spec (y=0) *regardless* of the gain/delay
    mismatch draw -- unlike four_tank, this plant's initial condition can never start closer
    to a constraint just because the draw was unlucky."""
    p = ShellFractionator(mismatch={"k11": 0.3, "k21": 0.3, "k31": 0.3, "l11": 0.2})
    for seed in (1, 2, 3, 4, 5):
        obs = p.reset(seed)
        np.testing.assert_array_equal(obs.y, np.zeros(3))
        assert not p.audit().violations


# -- constraints --------------------------------------------------------------------


def test_the_bottoms_reflux_temperature_floor_fires():
    """u3's own gain (K33 = 7.20) against its ±0.5 range is what makes this the constraint
    that bites -- see the module docstring."""
    p = ShellFractionator(sample_time=1.0, substeps=10)
    p.reset(0)
    u = np.array([0.0, 0.0, -0.5])
    violated = False
    for _ in range(60):
        p.step(u)
        if p.audit().violations:
            violated = True
            break
    assert violated
    assert "bottoms_reflux_temperature_floor" in p.audit().violations


def test_actuators_are_clipped_to_the_published_mv_range():
    plant = InstrumentedPlant(ShellFractionator(), InstrumentConfig())
    plant.reset(0)
    plant.step(np.array([5.0, -5.0, 5.0]))
    np.testing.assert_allclose(plant.u_actual, [0.5, -0.5, 0.5])


def test_set_params_refuses_a_mid_run_delay_change():
    p = ShellFractionator()
    p.reset(0)
    with pytest.raises(ValueError, match="fixed for the life of an episode"):
        p.set_params({"l12": 40.0})


def test_set_params_accepts_a_gain_drift_and_a_disturbance_step():
    p = ShellFractionator()
    p.reset(0)
    p.set_params({"k13": 6.0})
    assert p.params.K[0][2] == pytest.approx(6.0)
    p.set_params({"d2": 0.3})  # should not raise


# -- instrument layer (shared machinery, plant-specific sanity) -------------------


def test_sensor_deadtime_delays_the_reading():
    plant = InstrumentedPlant(
        ShellFractionator(), InstrumentConfig(sensors=SensorConfig(deadtime_samples=2))
    )
    plant.reset(0)
    seen = [plant.step(np.array([0.3, 0.0, 0.0])).y[0] for _ in range(4)]
    assert seen[0] == pytest.approx(0.0)
    assert seen[1] == pytest.approx(0.0)


# -- task-level checks --------------------------------------------------------------


def test_disturbance_event_actually_moves_the_plant(task: Task):
    rec = run_ensemble(task, lambda b: Hold(b), seeds=task.seeds[:1])[0]
    k = int(next(iter(task.disturbances)).t / task.sample_time)
    before = rec.trace.y[k - 2, 2]
    after = rec.trace.y[-1, 2]
    assert abs(after - before) > 0.01


def test_shipped_task_is_well_posed(task: Task):
    """The ``rtcbench validate`` contract, exercised directly: on every dev seed the
    reference must be safe and must beat doing nothing."""
    seeds = task.seeds[:4]
    hold = run_ensemble(task, lambda b: Hold(b), seeds=seeds)
    ref = run_ensemble(task, lambda b: build_reference(b, task.reference), seeds=seeds)
    for h, r in zip(hold, ref):
        assert not h.cost.violations, f"seed {h.seed}: doing nothing already violates"
        assert not r.cost.violations, f"seed {r.seed}: the reference violates"
        assert r.cost.total < h.cost.total, f"seed {r.seed}: reference no better than hold"


def test_mismatch_tier_publishes_the_nominal_not_the_draw(task: Task):
    plant = task.build_plant()
    plant.reset(task.seeds[0])
    brief = task.brief(plant)
    assert brief.model_hint["note"] == "nominal, not as-running"
    assert tuple(brief.model_hint["params"]["K"]) == P_NOMINAL.K
    # the as-running draw must actually have moved at least one gain away from nominal
    inner = plant._inner
    assert inner.params.K != inner.nominal_params().K

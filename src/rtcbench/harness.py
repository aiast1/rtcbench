"""The sealed loop.

One function does the actual benchmarking: :func:`run_scenario` marches a plant forward,
calls the controller once per control period, and writes down everything that happened.
Everything else in RTCbench is setup or arithmetic around it.

Three behaviours here are deliberate and worth reading before trusting a number:

* **The controller sees only ``(t, y, r, quality)``.** Not the state, not the parameters,
  not the plant object. Enforced by construction — the harness simply never passes them.
* **A blown compute budget is a scan overrun, not a crash.** The previous output is held
  and the period is flagged, exactly as a DCS behaves when a block runs long. Enough of
  them fails the run. Modelling this as an exception would make the benchmark reward fast
  controllers over good ones at the cliff edge instead of degrading the way real systems do.
* **A controller that raises fails its run and stays in the ensemble** with a gated score.
  Dropping crashed runs would score a controller only on the scenarios it survived.
"""

from __future__ import annotations

import time
import traceback
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .controller import Controller, TaskBrief
from .metrics import Cost, scenario_cost
from .record import RunRecord, Trace
from .sandbox import ScanOverrun
from .score import ScenarioScore, TaskScore, score_scenario, score_task
from .task import Task

ControllerFactory = Callable[[TaskBrief], Controller]


def run_scenario(
    task: Task,
    factory: ControllerFactory,
    seed: int,
    *,
    controller_id: str = "unknown",
) -> RunRecord:
    """Run one scenario end to end and return its record."""
    plant = task.build_plant()
    obs = plant.reset(seed)
    brief = task.brief(plant)

    n = task.n_steps
    n_y, n_u = plant.spec.n_y, plant.spec.n_u
    nx = len(plant.spec.state_names)

    # A controller that throws while being CONSTRUCTED is a failed submission in exactly
    # the way one that throws on its first step is, and must be recorded the same way.
    # Letting it propagate aborts the whole ensemble and takes every other scenario's
    # result with it -- so one submission misreading the interface would destroy the run
    # rather than score zero. Found when a model wrote brief['control_period'] against a
    # dataclass.
    try:
        controller = factory(brief)
        controller.reset()
    except Exception:
        plant.close()
        return _construction_failure(
            task, seed, controller_id, plant.spec, n_y, n_u, nx,
            "controller construction failed: " + traceback.format_exc(limit=6),
        )

    t_log = np.zeros(n)
    y_log = np.zeros((n, n_y))
    r_log = np.zeros((n, n_y))
    uc_log = np.zeros((n, n_u))
    ua_log = np.zeros((n, n_u))
    x_log = np.zeros((n, nx))
    q_log = np.ones((n, n_y), dtype=bool)
    v_log = np.zeros(n, dtype=bool)
    o_log = np.zeros(n, dtype=bool)

    u_prev = plant.spec.initial_actuation()
    pending = sorted(task.disturbances, key=lambda d: d.t)
    overruns = 0
    failed = False
    failure = ""
    steps_done = 0

    for k in range(n):
        t = k * task.sample_time

        while pending and pending[0].t <= t:
            plant.set_params(pending.pop(0).params)

        r = task.setpoint_vector(t, n_y)

        started = time.perf_counter()
        enforced_overrun = False
        try:
            u = np.asarray(
                controller.step(t, obs.y.copy(), r.copy(), obs.quality.copy()), dtype=float
            ).reshape(-1)
            if u.shape != (n_u,) or not np.all(np.isfinite(u)):
                raise ValueError(f"controller returned {u!r}, expected {n_u} finite values")
        except ScanOverrun:
            # A sandboxed controller has its budget enforced from outside and says so
            # explicitly, rather than the parent inferring it from a stopwatch. Same
            # consequence either way -- this is a late block, not a broken one.
            enforced_overrun = True
            u = u_prev
        except Exception:
            failed = True
            failure = traceback.format_exc(limit=6)
            break
        elapsed = time.perf_counter() - started

        if enforced_overrun or elapsed > task.budget.step_seconds:
            overruns += 1
            o_log[k] = True
            u = u_prev  # hold last output, as a DCS does on a scan overrun
            if overruns > task.budget.max_overruns:
                failed = True
                failure = (
                    f"exceeded scan budget on {overruns} periods "
                    f"(limit {task.budget.max_overruns}, {task.budget.step_seconds*1e3:.0f} ms each)"
                )
                break

        obs = plant.step(u)
        audit = plant.audit()

        t_log[k] = t
        y_log[k] = obs.y
        r_log[k] = r
        uc_log[k] = u
        ua_log[k] = getattr(plant, "u_actual", u)
        x_log[k] = audit.x
        q_log[k] = obs.quality
        v_log[k] = bool(audit.violations)
        u_prev = u
        steps_done = k + 1

    plant.close()
    # A sandboxed controller owns a subprocess; leaking one per scenario would exhaust the
    # machine over a 20-seed ensemble.
    closer = getattr(controller, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            pass

    sl = slice(0, steps_done)
    trace = Trace(
        t=t_log[sl], y=y_log[sl], r=r_log[sl], u_commanded=uc_log[sl],
        u_actual=ua_log[sl], x=x_log[sl], quality=q_log[sl],
        violations=v_log[sl], overrun=o_log[sl],
    )

    y_spans = np.array([c.span for c in plant.spec.measurements])
    u_spans = np.array([c.span for c in plant.spec.actuators])

    if steps_done == 0:
        cost = Cost(iae=np.inf, tv=np.inf, violations=1, overruns=overruns, total=np.inf)
    else:
        cost = scenario_cost(
            trace.y, trace.r, trace.u_commanded, task.controlled, y_spans, u_spans,
            task.sample_time,
            w_error=task.scoring.w_error, w_effort=task.scoring.w_effort,
            max_total_variation=task.scoring.max_total_variation,
            # A run that ended early failed; it is gated like a violation so it cannot
            # score well by virtue of having stopped before things got bad.
            violations=int(v_log[sl].sum()) + (1 if failed else 0),
            overruns=overruns,
        )

    return RunRecord(
        task_id=task.task_id, task_hash=task.content_hash, tier=task.tier, seed=seed,
        controller_id=controller_id, plant_id=plant.spec.plant_id,
        sample_time=task.sample_time, controlled=task.controlled,
        measurement_tags=tuple(c.tag for c in plant.spec.measurements),
        actuator_tags=tuple(c.tag for c in plant.spec.actuators),
        cost=cost, trace=trace, failed=failed, failure=failure,
        meta={"steps": steps_done, "planned_steps": n},
    )


def _construction_failure(
    task: Task, seed: int, controller_id: str, spec, n_y: int, n_u: int, nx: int, why: str
) -> RunRecord:
    """A zero-length, gated record for a submission that never got as far as running."""
    empty = Trace(
        t=np.zeros(0), y=np.zeros((0, n_y)), r=np.zeros((0, n_y)),
        u_commanded=np.zeros((0, n_u)), u_actual=np.zeros((0, n_u)),
        x=np.zeros((0, nx)), quality=np.zeros((0, n_y), dtype=bool),
        violations=np.zeros(0, dtype=bool), overrun=np.zeros(0, dtype=bool),
    )
    return RunRecord(
        task_id=task.task_id, task_hash=task.content_hash, tier=task.tier, seed=seed,
        controller_id=controller_id, plant_id=spec.plant_id, sample_time=task.sample_time,
        controlled=task.controlled,
        measurement_tags=tuple(c.tag for c in spec.measurements),
        actuator_tags=tuple(c.tag for c in spec.actuators),
        cost=Cost(iae=np.inf, tv=np.inf, violations=1, overruns=0, total=np.inf),
        trace=empty, failed=True, failure=why,
        meta={"steps": 0, "planned_steps": task.n_steps},
    )


def run_ensemble(
    task: Task,
    factory: ControllerFactory,
    *,
    controller_id: str = "unknown",
    seeds: Sequence[int] | None = None,
) -> list[RunRecord]:
    """Run every scenario in the task's ensemble."""
    return [
        run_scenario(task, factory, seed, controller_id=controller_id)
        for seed in (task.seeds if seeds is None else seeds)
    ]


def score_against_anchors(
    task: Task,
    records: Sequence[RunRecord],
    hold: Sequence[RunRecord],
    reference: Sequence[RunRecord],
) -> TaskScore:
    """Turn three ensembles of records into the task's score.

    Anchors are matched per seed, not pooled: a submission is compared against the hold and
    reference controllers *on the same plant draw it faced*. Pooling would let a lucky draw
    in the anchor set flatter or punish a submission for reasons that have nothing to do
    with its controller.
    """
    by_seed = {r.seed: r for r in records}
    hold_by_seed = {r.seed: r for r in hold}
    ref_by_seed = {r.seed: r for r in reference}

    scored: list[ScenarioScore] = []
    for seed in task.seeds:
        if seed not in by_seed:
            continue
        scored.append(
            score_scenario(
                seed=seed,
                submission=by_seed[seed].cost,
                hold=hold_by_seed[seed].cost.total,
                reference=ref_by_seed[seed].cost.total,
            )
        )
    return score_task(task.task_id, scored, alpha=task.scoring.cvar_alpha)

"""``rtcbench`` command line.

    rtcbench run   --task tasks/four_tank_v1.yaml --controller builtin:pid
    rtcbench score --task tasks/four_tank_v1.yaml --controller my/controller.py --trends out/
    rtcbench show  --task tasks/four_tank_v1.yaml
    rtcbench replay runs/four_tank_v1_seed7.json

``run`` is the fast single-scenario loop for iterating on a controller. ``score`` is the
real thing: the whole ensemble, both anchors, the gated CVaR headline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .baselines import build_reference
from .harness import run_ensemble, run_scenario, score_against_anchors
from .metrics import scenario_cost
from .record import RunRecord
from .submission import load_controller
from .task import Task
from .trend import trend_svg, write_trends


def _anchors(task: Task):
    hold_factory, _ = load_controller("builtin:hold")
    ref_factory = lambda brief: build_reference(brief, task.reference)  # noqa: E731
    return hold_factory, ref_factory


def cmd_show(args: argparse.Namespace) -> int:
    task = Task.load(args.task)
    plant = task.build_plant()
    brief = task.brief(plant)
    print(f"{task.task_id}  [{task.tier}]  hash={task.content_hash}")
    print(f"  plant       {plant.spec.plant_id}   Ts={task.sample_time}s  "
          f"horizon={task.horizon}s  ({task.n_steps} periods)")
    print(f"  scenarios   {len(task.seeds)}  seeds={list(task.seeds[:6])}"
          f"{'...' if len(task.seeds) > 6 else ''}")
    print("  measured   ", ", ".join(f"{c.tag} ({c.name}, {c.unit})" for c in brief.measurements))
    print("  actuators  ", ", ".join(f"{c.tag} ({c.name}, {c.unit})" for c in brief.actuators))
    print("  controlled ", [brief.measurements[i].tag for i in brief.controlled])
    print("  envelope   ", ", ".join(f"{c.signal} in [{c.lo},{c.hi}]" for c in brief.constraints))
    if task.disturbances:
        print("  upsets     ", ", ".join(f"t={d.t:g}s {dict(d.params)}" for d in task.disturbances))
    print(f"  model hint  {brief.model_hint.get('note', 'none (blind tier)')}")
    plant.close()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    task = Task.load(args.task)
    factory, cid = load_controller(args.controller)
    seed = args.seed if args.seed is not None else task.seeds[0]

    rec = run_scenario(task, factory, seed, controller_id=cid)
    c = rec.cost
    status = "FAILED" if rec.failed else ("GATED" if c.violations else "ok")
    print(f"{task.task_id} seed={seed} [{status}]")
    print(f"  IAE(norm)  {c.iae:.5f}")
    print(f"  TV(norm)   {c.tv:.5f}")
    print(f"  cost       {c.total:.5f}   violations={c.violations}  overruns={c.overruns}")
    if rec.failed:
        print("\n" + rec.failure, file=sys.stderr)

    if args.out:
        p = rec.save(Path(args.out) / f"{task.task_id}_seed{seed}.json")
        Path(str(p).replace(".json", ".svg")).write_text(trend_svg(rec), encoding="utf-8")
        print(f"  record     {p}")
    return 1 if rec.failed else 0


def cmd_score(args: argparse.Namespace) -> int:
    task = Task.load(args.task)
    factory, cid = load_controller(args.controller)
    hold_factory, ref_factory = _anchors(task)

    print(f"{task.task_id} [{task.tier}]  {len(task.seeds)} scenarios  hash={task.content_hash}")
    hold = run_ensemble(task, hold_factory, controller_id="builtin:hold")
    ref = run_ensemble(task, ref_factory, controller_id="reference")
    sub = run_ensemble(task, factory, controller_id=cid)

    result = score_against_anchors(task, sub, hold, ref)

    print(f"\n  anchors   hold cost {np.mean([r.cost.total for r in hold]):.5f}"
          f"   reference cost {np.mean([r.cost.total for r in ref]):.5f}")
    print(f"  submission          {np.mean([r.cost.total for r in sub]):.5f}\n")
    print("  seed        score   IAE      TV       flags")
    for s in result.scenarios:
        rec = next(r for r in sub if r.seed == s.seed)
        flags = []
        if s.gated:
            flags.append("GATED")
        if rec.failed:
            flags.append("FAILED")
        if rec.cost.overruns:
            flags.append(f"overrun x{rec.cost.overruns}")
        print(f"  {s.seed:<10} {s.score:+.3f}  {s.cost.iae:.5f}  {s.cost.tv:.5f}  "
              f"{' '.join(flags)}")
    print(f"\n  {result.summary()}")
    print("  (0.0 = actuators held, 1.0 = reference controller, higher is better)")

    if args.trends:
        for path in write_trends(sub, args.trends, limit=args.trend_count):
            print(f"  trend      {path}")
    if args.out:
        for rec in sub:
            rec.save(Path(args.out) / f"{task.task_id}_seed{rec.seed}.json")
        print(f"  records    {args.out}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Check that a task is well-posed before anyone is scored against it.

    A benchmark's credibility rests as much on its tasks being *fair* as on its harness
    being correct, and the failure modes are quiet ones. Four conditions, each of which has
    already caught a real defect in this repo's own task file:

    1. **Doing nothing must not violate.** If the hold anchor trips a constraint, the
       scenario is unwinnable before any controller is involved and only adds noise.
    2. **The reference must not violate.** Same argument, one level up: if the maintainers'
       own tuned controller cannot keep the plant safe, the ensemble is punishing draws,
       not controllers.
    3. **The reference must beat hold on every scenario.** Otherwise the 0-to-1 scale is
       inverted for that draw and its scores are meaningless.
    4. **The anchors must be well separated.** A reference only marginally better than
       doing nothing turns the score into a division by almost zero.
    """
    task = Task.load(args.task)
    hold_factory, ref_factory = _anchors(task)

    print(f"validating {task.task_id} [{task.tier}]  hash={task.content_hash}")
    hold = {r.seed: r for r in run_ensemble(task, hold_factory, controller_id="hold")}
    ref = {r.seed: r for r in run_ensemble(task, ref_factory, controller_id="reference")}

    problems: list[str] = []
    print("\n  seed        hold      reference  separation")
    for seed in task.seeds:
        h, r = hold[seed], ref[seed]
        sep = h.cost.total - r.cost.total
        flags = []
        if h.cost.violations:
            flags.append("HOLD-VIOLATES")
            problems.append(f"seed {seed}: holding the actuators already violates a constraint")
        if r.cost.violations:
            flags.append("REF-VIOLATES")
            problems.append(f"seed {seed}: the reference controller violates a constraint")
        if r.failed:
            flags.append("REF-FAILED")
            problems.append(f"seed {seed}: the reference controller failed to run")
        if sep <= 0:
            flags.append("INVERTED")
            problems.append(f"seed {seed}: reference is no better than doing nothing")
        elif sep < 0.02 * h.cost.total:
            flags.append("NARROW")
            problems.append(f"seed {seed}: anchors separated by only {sep:.2e}")
        print(f"  {seed:<10} {h.cost.total:.5f}   {r.cost.total:.5f}    "
              f"{sep:+.5f}  {' '.join(flags)}")

    if problems:
        print(f"\n  FAIL — {len(problems)} problem(s):")
        for p in problems[:12]:
            print(f"    - {p}")
        if len(problems) > 12:
            print(f"    ... and {len(problems) - 12} more")
        return 1

    print(f"\n  PASS — {len(task.seeds)} scenarios, all winnable, anchors well separated")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Recompute a record's metrics from its own trace.

    This is the check that makes a published score falsifiable: the numbers are rederived
    from the stored time series, not read back out of the file that claims them.
    """
    rec = RunRecord.load(args.record)
    task = Task.load(args.task) if args.task else None
    tr = rec.trace

    if task is None:
        print("pass --task to recompute against the task's scoring weights", file=sys.stderr)
        return 2

    plant = task.build_plant()
    y_spans = np.array([c.span for c in plant.spec.measurements])
    u_spans = np.array([c.span for c in plant.spec.actuators])
    plant.close()

    recomputed = scenario_cost(
        tr.y, tr.r, tr.u_commanded, rec.controlled, y_spans, u_spans, rec.sample_time,
        w_error=task.scoring.w_error, w_effort=task.scoring.w_effort,
        max_total_variation=task.scoring.max_total_variation,
        violations=int(tr.violations.sum()) + (1 if rec.failed else 0),
        overruns=int(tr.overrun.sum()),
    )
    ok = np.isclose(recomputed.total, rec.cost.total, rtol=1e-9, atol=1e-12)
    print(f"stored     cost {rec.cost.total:.9f}  IAE {rec.cost.iae:.9f}  TV {rec.cost.tv:.9f}")
    print(f"recomputed cost {recomputed.total:.9f}  IAE {recomputed.iae:.9f}  "
          f"TV {recomputed.tv:.9f}")
    print("MATCH" if ok else "MISMATCH — the record does not support its own score")
    if args.svg:
        Path(args.svg).write_text(trend_svg(rec), encoding="utf-8")
        print(f"trend      {args.svg}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    # A Windows console defaults to cp1252 and raises on anything outside it. A benchmark
    # CLI that dies on a character in its own output is not portable, so make the encoding
    # explicit rather than relying on the terminal's locale.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="rtcbench", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("show", help="print a task's brief")
    p.add_argument("--task", required=True)
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("run", help="run one scenario")
    p.add_argument("--task", required=True)
    p.add_argument("--controller", required=True, help="builtin:pid | builtin:hold | path.py")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None, help="directory for the run record + trend")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("score", help="run the full ensemble against both anchors")
    p.add_argument("--task", required=True)
    p.add_argument("--controller", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--trends", default=None, help="directory for SVG trend plots")
    p.add_argument("--trend-count", type=int, default=3)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("validate", help="check a task is well-posed before scoring anyone on it")
    p.add_argument("--task", required=True)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("replay", help="recompute a record's metrics from its own trace")
    p.add_argument("record")
    p.add_argument("--task", default=None)
    p.add_argument("--svg", default=None)
    p.set_defaults(func=cmd_replay)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

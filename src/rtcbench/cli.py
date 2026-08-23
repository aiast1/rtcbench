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
    factory, cid = load_controller(
        args.controller, sandbox=args.sandbox, step_seconds=task.budget.step_seconds)
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
    factory, cid = load_controller(
        args.controller, sandbox=args.sandbox, step_seconds=task.budget.step_seconds)
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


MIN_ANCHOR_RATIO = 0.35
"""How much better than doing nothing the reference must be, as a fraction of hold cost.

The check three separate tasks needed and nobody had written. four_tank_v1, van_de_vusse_v1
and shell_fractionator_v1 each shipped with a gap so narrow that the 0-to-1 scale went
hypersensitive and an ordinary mediocre submission scored -6 or worse. Shell was the clearest:
hold 0.064 against a reference of 0.0435, so control bought 32%, and eight of twenty models
bottomed out at the score clamp on a task that was merely under-specified rather than hard.

0.35 admits every healthy task in the pack and rejects the three that had to be repaired.
"""


def cmd_check(args: argparse.Namespace) -> int:
    """Pre-flight a controller against the interface before spending a scoring run on it.

    `validate` checks a TASK is well-posed; this checks a SUBMISSION is well-formed. It was
    added after measuring how much of the leaderboard was decided by interface errors rather
    than by control: crashes correlate with suite score at -0.51, and one model lost 40 of
    its 200 scenarios to `channel.min` (the field is `.lo`) and `brief.actuator_ranges`
    (which does not exist). Both are reasonable guesses from someone who has never seen the
    class, which means the benchmark was partly measuring whether a submission guessed this
    codebase's naming conventions.

    Fixing that has two halves. The brief now documents the exact dataclass, and this
    command lets anyone confirm their controller constructs, resets and steps before
    submitting. Available to everyone equally, so it changes what the benchmark measures --
    control rather than API telepathy -- without favouring anybody.
    """
    task = Task.load(args.task)
    plant = task.build_plant()
    brief = task.brief(plant)
    print(f"checking {args.controller} against {task.task_id}")

    try:
        factory, cid = load_controller(args.controller, sandbox=args.sandbox,
                                       step_seconds=task.budget.step_seconds)
    except Exception as exc:
        print(f"  FAIL  module will not import: {type(exc).__name__}: {exc}")
        return 1

    try:
        controller = factory(brief)
    except Exception as exc:
        print(f"  FAIL  __init__ raised: {type(exc).__name__}: {exc}")
        print("        The brief is a dataclass. Its exact fields are:")
        for line in _brief_fields():
            print(f"          {line}")
        plant.close()
        return 1
    print("  ok    constructs")

    try:
        controller.reset()
    except Exception as exc:
        print(f"  FAIL  reset() raised: {type(exc).__name__}: {exc}")
        plant.close()
        return 1
    print("  ok    reset()")

    obs = plant.reset(task.seeds[0])
    n_u = plant.spec.n_u
    for k in range(5):
        t = k * task.sample_time
        r = task.setpoint_vector(t, plant.spec.n_y)
        try:
            u = np.asarray(controller.step(t, obs.y.copy(), r.copy(), obs.quality.copy()),
                           dtype=float).reshape(-1)
        except Exception as exc:
            print(f"  FAIL  step() raised on period {k}: {type(exc).__name__}: {exc}")
            plant.close()
            return 1
        if u.shape != (n_u,):
            print(f"  FAIL  step() returned shape {u.shape}, expected ({n_u},)")
            plant.close()
            return 1
        if not np.all(np.isfinite(u)):
            print(f"  FAIL  step() returned a non-finite value on period {k}: {u}")
            plant.close()
            return 1
        obs = plant.step(u)
    print(f"  ok    step() x5, returns {n_u} finite values")

    lo, hi = plant.spec.actuator_lo(), plant.spec.actuator_hi()
    if np.any(u < lo - 1e-9) or np.any(u > hi + 1e-9):
        print(f"  warn  last output {u} is outside the actuator limits {lo}..{hi}. The "
              "harness does not clip for you; the plant will.")
    plant.close()
    print("\n  PASS - the interface is satisfied.")
    print("         This says nothing about whether it controls well; "
          "run `rtcbench score` for that.")
    return 0


def _brief_fields() -> list[str]:
    import dataclasses

    from .controller import TaskBrief
    from .plant import Channel

    out = [f"brief.{f.name}" for f in dataclasses.fields(TaskBrief)]
    out.append("brief.n_y, brief.n_u, brief.tags()")
    out.append("each Channel has: " + ", ".join(f.name for f in dataclasses.fields(Channel))
               + ", .span")
    return out


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
    gaps: list[float] = []
    holds: list[float] = []
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
        gaps.append(sep)
        holds.append(h.cost.total)
        print(f"  {seed:<10} {h.cost.total:.5f}   {r.cost.total:.5f}    "
              f"{sep:+.5f}  {' '.join(flags)}")

    # The ensemble-level check. A task can pass every per-seed test above and still rank
    # nothing, because the SCALE is set by the anchor gap rather than by any single seed.
    if gaps and holds and np.mean(holds):
        ratio = float(np.mean(gaps) / np.mean(holds))
        verdict = "ok" if ratio >= MIN_ANCHOR_RATIO else "TOO NARROW"
        print(f"\n  anchor gap is {ratio * 100:.0f}% of the hold cost  [{verdict}]")
        if ratio < MIN_ANCHOR_RATIO:
            print(f"    a submission costing 3x hold would score {(1 - 3.0) / ratio:+.1f} here")
            problems.append(
                f"anchor gap is only {ratio * 100:.0f}% of the hold cost (want >= "
                f"{MIN_ANCHOR_RATIO * 100:.0f}%). The 0-to-1 scale is hypersensitive: an "
                "ordinary mediocre submission scores in the negative tens and this task's "
                "column swamps every aggregate it enters. Make the task demand more control "
                "-- longer setpoint travel, a stronger upset -- and retune. Do NOT fix it by "
                "weakening the plant."
            )

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
    p.add_argument("--sandbox", action="store_true",
                   help="run the controller in a separate process that cannot "
                        "import the plant or open a socket (see rtcbench.sandbox "
                        "for what this does and does not guarantee)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("score", help="run the full ensemble against both anchors")
    p.add_argument("--task", required=True)
    p.add_argument("--controller", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--trends", default=None, help="directory for SVG trend plots")
    p.add_argument("--trend-count", type=int, default=3)
    p.add_argument("--sandbox", action="store_true",
                   help="run the controller in a separate process that cannot "
                        "import the plant or open a socket (see rtcbench.sandbox "
                        "for what this does and does not guarantee)")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("check", help="pre-flight a controller against the interface")
    p.add_argument("--task", required=True)
    p.add_argument("--controller", required=True)
    p.add_argument("--sandbox", action="store_true")
    p.set_defaults(func=cmd_check)

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

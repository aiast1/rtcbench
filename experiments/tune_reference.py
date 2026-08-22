"""Tune a task's reference anchor by coordinate descent.

The reference defines 1.0 on a task's leaderboard, so how well it is tuned sets the meaning
of every score that task will ever produce. Doing that by hand does not scale past one task
and is not reproducible by a reviewer, so it lives here as a command.

Two properties this enforces that hand-tuning did not:

* **It refuses a tuning that violates.** A reference that trips a constraint on any draw
  makes the task unwinnable; such candidates score infinite and cannot win.
* **It reports flatness.** A gain set that is 1% better than its neighbours is a fit to the
  tuning seeds; one that sits in a broad basin is a real optimum. The printed spread is the
  evidence, and it belongs in a comment above the task's `reference:` block.

    python experiments/tune_reference.py --task tasks/four_tank_v1.yaml [--apply]
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from rtcbench.baselines import MultiLoopPID
from rtcbench.harness import run_ensemble
from rtcbench.task import Task

ROOT = Path(__file__).resolve().parents[1]

KP_GRID = [0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0, 8.0, 15.0]
TI_GRID = [5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 70.0, 120.0, 200.0,
           350.0, 600.0, 1000.0, 2000.0]
"""Integral times must reach far past a few hundred seconds.

The first version stopped at 200 s and it silently broke two tasks: on a plant with a
right-half-plane zero or dominant deadtime the achievable bandwidth is tiny, the correct
integral time is thousands of seconds, and every in-grid candidate violated a constraint.
The tuner then reported that no safe tuning existed, which read as "this task is impossible"
when it actually meant "the grid does not contain the answer". A search grid that cannot
express the right answer is worse than no search, because it fails with false confidence.
"""

BETA_GRID = [0.4, 0.6, 0.8, 1.0]


def evaluate(task: Task, pairing, kp, ti, beta, seeds) -> float:
    def factory(brief):
        return MultiLoopPID(brief, pairing=list(pairing), kp=list(kp),
                            ti=list(ti), beta=list(beta))

    try:
        records = run_ensemble(task, factory, controller_id="tune", seeds=list(seeds))
    except Exception:
        return float("inf")
    # A reference that is unsafe on any draw is not a candidate at any cost.
    if any(r.failed or r.cost.violations or r.cost.duty_exceeded for r in records):
        return float("inf")
    return float(np.mean([r.cost.total for r in records]))


def tune(task: Task, pairing, seeds, rounds: int = 3, verbose: bool = True):
    n = len(task.controlled)
    kp, ti, beta = [1.0] * n, [30.0] * n, [1.0] * n

    # Seed the search from the best single scalar setting so coordinate descent does not
    # start somewhere unstable and immediately wall itself in behind inf.
    #
    # The coarse sweep stays on a 2-seed probe: it is a RANKING pass, not a measurement, and
    # paying full ensemble cost for ~180 evaluations made this command too slow to finish
    # inside an agent's turn budget -- which is how two tasks once shipped with placeholder
    # gains. The DESCENT below runs on every seed, because that is the part whose result gets
    # published, and a descent on a sample overfits: a DMC anchor tuned on 6 seeds scored
    # 0.0816 on those 6 and 0.1181 on all 20.
    probe = list(seeds[:2]) or list(seeds)
    best = float("inf")
    for k, t in itertools.product(KP_GRID, TI_GRID):
        c = evaluate(task, pairing, [k] * n, [t] * n, beta, probe)
        if c < best:
            best, kp, ti = c, [k] * n, [t] * n
    # Re-measure the winner on the real seed set before descent starts from it.
    best = evaluate(task, pairing, kp, ti, beta, seeds)
    if verbose:
        print(f"  coarse   kp={kp} ti={ti} -> {best:.6f}  (probe seeds {probe})")

    for rnd in range(rounds):
        improved = False
        for name, grid in (("kp", KP_GRID), ("ti", TI_GRID), ("beta", BETA_GRID)):
            values = {"kp": kp, "ti": ti, "beta": beta}[name]
            for i in range(n):
                base, local_best, local_arg = values[i], best, values[i]
                for v in grid:
                    values[i] = v
                    c = evaluate(task, pairing, kp, ti, beta, seeds)
                    if c < local_best - 1e-12:
                        local_best, local_arg = c, v
                values[i] = local_arg
                if local_best < best - 1e-12:
                    best, improved = local_best, True
                elif local_arg == base:
                    values[i] = base
        if verbose:
            print(f"  round {rnd}  kp={kp} ti={ti} beta={beta} -> {best:.6f}")
        if not improved:
            break
    return kp, ti, beta, best


def flatness(task: Task, pairing, kp, ti, beta, seeds, best: float) -> str:
    """How much worse are the neighbours? A narrow basin means an overfit tuning."""
    spreads = []
    for scale in (0.8, 1.25):
        c = evaluate(task, pairing, [k * scale for k in kp], ti, beta, seeds)
        if np.isfinite(c) and best > 0:
            spreads.append((c - best) / best)
    if not spreads:
        return "neighbours unstable - the optimum is a knife edge, which is a bad reference"
    return (f"+/-25% on kp costs {min(spreads)*100:.1f}% to {max(spreads)*100:.1f}% "
            f"-> {'broad basin, sound' if max(spreads) < 0.35 else 'narrow, treat with suspicion'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--tune-seeds", type=int, nargs="+", default=None,
                    help="default: ALL the task's scenario seeds")
    ap.add_argument("--pairing", type=int, nargs="+", default=None)
    ap.add_argument("--apply", action="store_true",
                    help="print a ready-to-paste reference block")
    args = ap.parse_args(argv)

    task = Task.load(args.task)
    seeds = args.tune_seeds or list(task.seeds)
    pairing = args.pairing or list(range(len(task.controlled)))

    print(f"tuning {task.task_id}: {len(task.controlled)} loop(s), pairing {pairing}, "
          f"seeds {seeds}")
    kp, ti, beta, best = tune(task, pairing, seeds)

    if not np.isfinite(best):
        print("\nNo safe tuning found on this grid. Either the pairing is wrong (try the "
              "off-diagonal), or the task's constraints/disturbance are too aggressive.")
        return 1

    print(f"\nbest mean cost {best:.6f}")
    print(f"flatness: {flatness(task, pairing, kp, ti, beta, seeds, best)}")

    full = run_ensemble(
        task, lambda b: MultiLoopPID(b, pairing=pairing, kp=kp, ti=ti, beta=beta),
        controller_id="tuned")
    tv = float(np.mean([r.cost.tv for r in full]))
    print(f"full ensemble: mean cost {np.mean([r.cost.total for r in full]):.6f}, "
          f"mean tv {tv:.5f}, gated {sum(1 for r in full if r.cost.violations)}")
    print(f"suggested max_total_variation: {tv * 4:.5f}")

    if args.apply:
        print("\n# paste over the task's reference: block\nreference:")
        print(f"  kind: pid\n  pairing: {pairing}")
        print(f"  kp: {[round(v, 4) for v in kp]}")
        print(f"  ti: {[round(v, 2) for v in ti]}")
        print(f"  beta: {[round(v, 2) for v in beta]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

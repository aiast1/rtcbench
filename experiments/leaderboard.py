"""Score every submission in experiments/agents/ against one shared set of anchors.

`rtcbench score` recomputes the hold and reference ensembles on every invocation, which is
right for a single submission and wasteful for a field of them — five submissions would mean
200 anchor runs to produce 40 useful ones. The anchors depend only on the task, so they are
computed once here and reused.

Also reports two things the single-submission view cannot:

* **rank on CVaR@10%**, the metric that actually orders a leaderboard, next to the mean, so
  a controller that wins on average while being fragile on one draw is visible as such;
* **an anchor-copy check.** The task file publishes the reference gains, so a submission can
  tie 1.000 by transcribing them. That is not cheating exactly — it is a floor, and the
  leaderboard is about beating the floor — but a score that lands on 1.000 to three decimals
  is worth looking at rather than celebrating.

    python experiments/leaderboard.py [--task tasks/four_tank_v1.yaml] [--trends out/]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from rtcbench.baselines import Hold, build_reference
from rtcbench.harness import run_ensemble, score_against_anchors
from rtcbench.submission import load_controller
from rtcbench.task import Task
from rtcbench.trend import overlay_svg, write_trends

ROOT = Path(__file__).resolve().parents[1]
CHATTER_REFERENCE = 0.0029
"""Roughly the reference controller's total variation. Submissions far above it are holding
setpoint by hammering the actuator -- behaviour a real plant rejects on wear alone, and which
the current w_effort=0.1 does not punish nearly hard enough."""

ANCHOR_TIE = 0.0005
"""How close to exactly 1.000 counts as suspiciously like the published reference."""


def discover(agents_dir: Path) -> list[tuple[str, Path]]:
    found = [
        (p.parent.name, p)
        for p in sorted(agents_dir.glob("*/controller.py"))
    ]
    example = ROOT / "examples" / "pi_controller.py"
    if example.is_file():
        # The naive PI is carried as a control, not a competitor: it is the "did anyone
        # actually beat a copy-paste loop" line on the board.
        found.append(("example-pi", example))
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=str(ROOT / "tasks" / "four_tank_v1.yaml"))
    ap.add_argument("--agents", default=str(ROOT / "experiments" / "agents"))
    ap.add_argument("--trends", default=None)
    ap.add_argument("--overlay", default=str(ROOT / "out" / "compare.svg"),
                    help="path for the side-by-side comparison plot")
    ap.add_argument("--overlay-seed", type=int, default=None,
                    help="force a scenario; default picks the most discriminating one")
    args = ap.parse_args(argv)

    task = Task.load(args.task)
    entries = discover(Path(args.agents))
    if not entries:
        print("no submissions found", file=sys.stderr)
        return 1

    print(f"{task.task_id} [{task.tier}]  {len(task.seeds)} scenarios  hash={task.content_hash}")
    print(f"scoring {len(entries)} submissions: {', '.join(n for n, _ in entries)}\n")

    hold = run_ensemble(task, lambda b: Hold(b), controller_id="hold")
    ref = run_ensemble(task, lambda b: build_reference(b, task.reference), controller_id="reference")
    print(f"anchors  hold {np.mean([r.cost.total for r in hold]):.5f}"
          f"   reference {np.mean([r.cost.total for r in ref]):.5f}\n")

    rows = []
    by_name: dict[str, dict[int, object]] = {}
    scores_by_name: dict[str, dict[int, float]] = {}
    for name, path in entries:
        started = time.perf_counter()
        try:
            factory, cid = load_controller(str(path))
            records = run_ensemble(task, factory, controller_id=cid)
        except Exception as exc:  # a module that will not even import still gets a row
            print(f"  {name:<28} IMPORT FAILED: {type(exc).__name__}: {exc}")
            rows.append((name, None, None, None, None, None, None, str(exc)))
            continue

        result = score_against_anchors(task, records, hold, ref)
        by_name[name] = {r.seed: r for r in records}
        scores_by_name[name] = {s.seed: s.score for s in result.scenarios}
        crashed = sum(1 for r in records if r.failed)
        rows.append((
            name,
            result.headline,
            result.mean,
            result.worst,
            result.gated_count,
            crashed,
            float(np.mean([r.cost.tv for r in records])),
            "",
        ))
        print(f"  {name:<28} scored in {time.perf_counter() - started:.1f}s")

        if args.trends:
            write_trends(records, Path(args.trends) / name, limit=2)

    ranked = sorted(
        rows, key=lambda r: (r[1] is None, -(r[1] if r[1] is not None else 0))
    )

    width = max(12, max(len(r[0]) for r in ranked) + 1)
    print(f"\n{'rank':<5} {'submission':<{width}} {'CVaR@10%':>9} {'mean':>8} {'worst':>8} "
          f"{'gated':>6} {'crash':>6} {'effort':>9}")
    print("-" * (5 + width + 52))
    for i, (name, head, mean, worst, gated, crashed, tv, err) in enumerate(ranked, 1):
        if head is None:
            print(f"{i:<5} {name:<{width}} {'--':>9} {'--':>8} {'--':>8} {'--':>6} {'--':>6} "
                  f"{'--':>9}   module did not import")
            continue
        note = ""
        if tv is not None and tv > 5 * CHATTER_REFERENCE:
            note = f"  <- {tv / CHATTER_REFERENCE:.0f}x the reference's valve movement"
        if abs(head - 1.0) < ANCHOR_TIE:
            note = "  <- ties the published reference"
        elif gated:
            note = f"  <- {gated} scenario(s) zeroed on safety"
        print(f"{i:<5} {name:<{width}} {head:+9.3f} {mean:+8.3f} {worst:+8.3f} {gated:>6} "
              f"{crashed:>6} {tv:9.5f}{note}")

    print("\n0.0 = actuators held, 1.0 = the maintainers' tuned reference, higher is better.")
    print("Ranked on CVaR@10% (the mean of each submission's worst two scenarios).")

    if args.overlay and len(by_name) > 1:
        # Plot the scenario the submissions disagree about most. Overlaying a draw they all
        # handle identically shows a single thick line and teaches nothing; the informative
        # scenario is the one that separates them.
        spread = {
            seed: max(s[seed] for s in scores_by_name.values())
            - min(s[seed] for s in scores_by_name.values())
            for seed in task.seeds
            if all(seed in s for s in scores_by_name.values())
        }
        seed = args.overlay_seed or max(spread, key=lambda k: spread[k])
        series = {"reference": {r.seed: r for r in ref}[seed]}
        series.update({name: recs[seed] for name, recs in by_name.items() if seed in recs})

        out = Path(args.overlay)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            overlay_svg(
                series,
                title=f"{task.task_id} seed={seed} - most discriminating scenario "
                f"(score spread {spread.get(seed, 0):.3f})",
            ),
            encoding="utf-8",
        )
        print(f"\noverlay    {out}  (seed {seed}, the draw that separates them most)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

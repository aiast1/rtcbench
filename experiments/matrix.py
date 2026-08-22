"""Score every model against every task and print the suite matrix.

A single-task leaderboard says who tuned one loop well. The suite matrix says something much
harder to fake: whether a model's control ability generalizes across plants that fail in
*different* ways. A submission can tie the reference on the minimum-phase four-tank by
guessing reasonable PI gains; it cannot do that on a plant whose correct pairing is
off-diagonal, whose deadtime moves with throughput, and whose operating point is open-loop
unstable, all at once.

Aggregation rule: a model's suite score is the mean of its per-task CVaR@10%, with each
task's contribution clipped at `TASK_FLOOR` so no single task can dominate. A task a model
produced no controller for scores 0.0 rather than being dropped — dropping it would reward
failing to answer, the same reason a crashed scenario stays in its ensemble.

The per-task columns are NOT clipped. They spread from about +1.1 down to -7 in practice,
and that spread is the diagnostic content: on four_tank_nmp the field ranges from -0.59 to
-6.80, a tenfold difference in how badly each model failed that an earlier per-scenario floor
flattened into a column of identical -1.000s.

    python experiments/matrix.py --tasks tasks/*.yaml --submissions experiments/submissions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rtcbench.baselines import Hold, build_reference
from rtcbench.harness import run_ensemble, score_against_anchors
from rtcbench.score import TASK_FLOOR, suite_score
from rtcbench.submission import load_controller
from rtcbench.task import Task

ROOT = Path(__file__).resolve().parents[1]
MISSING = 0.0
"""Score for a task a model produced no controller for. Not None, not dropped."""


def score_one_task(task: Task, submissions: dict[str, Path], verbose: bool = True):
    """Anchors once, then every submission against them."""
    hold = run_ensemble(task, lambda b: Hold(b), controller_id="hold")
    ref = run_ensemble(task, lambda b: build_reference(b, task.reference),
                       controller_id="reference")
    ref_cost = float(np.mean([r.cost.total for r in ref]))
    hold_cost = float(np.mean([r.cost.total for r in hold]))
    if verbose:
        print(f"  {task.task_id:<26} anchors: hold {hold_cost:.5f} / ref {ref_cost:.5f}")

    out: dict[str, dict] = {}
    for name, path in sorted(submissions.items()):
        try:
            factory, cid = load_controller(str(path))
            records = run_ensemble(task, factory, controller_id=cid)
        except Exception as exc:
            out[name] = {"score": MISSING, "note": f"import failed: {type(exc).__name__}"}
            continue
        result = score_against_anchors(task, records, hold, ref)
        out[name] = {
            "score": result.headline,
            "mean": result.mean,
            "gated": result.gated_count,
            "crashed": sum(1 for r in records if r.failed),
            "tv": float(np.mean([r.cost.tv for r in records])),
            "note": "",
        }
    return out


def discover(root: Path, task_id: str) -> dict[str, Path]:
    d = root / task_id
    if not d.is_dir():
        return {}
    return {p.parent.name: p for p in sorted(d.glob("*/controller.py"))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--submissions", default=str(ROOT / "experiments" / "submissions"))
    ap.add_argument("--out", default=str(ROOT / "out" / "matrix.json"))
    args = ap.parse_args(argv)

    tasks = [Task.load(t) for t in args.tasks]
    root = Path(args.submissions)

    print(f"scoring {len(tasks)} tasks\n")
    per_task: dict[str, dict] = {}
    models: set[str] = set()
    for task in tasks:
        subs = discover(root, task.task_id)
        models |= set(subs)
        per_task[task.task_id] = score_one_task(task, subs)

    names = sorted(models)
    if not names:
        print("\nno submissions found -- run experiments/commission.py first")
        return 1

    suite = {
        n: suite_score([per_task[t.task_id].get(n, {}).get("score", MISSING) for t in tasks])
        for n in names
    }
    ranked = sorted(names, key=lambda n: -suite[n])

    width = max(12, max(len(n) for n in names) + 1)
    heads = [t.task_id.replace("_v1", "") for t in tasks]
    colw = max(9, max(len(h) for h in heads) + 1)

    print(f"\n{'rank':<5} {'model':<{width}} {'SUITE':>8} " +
          " ".join(f"{h:>{colw}}" for h in heads))
    print("-" * (5 + width + 9 + (colw + 1) * len(heads)))
    for i, n in enumerate(ranked, 1):
        cells = []
        for t in tasks:
            e = per_task[t.task_id].get(n)
            if e is None:
                cells.append(f"{'--':>{colw}}")
            elif e.get("note"):
                cells.append(f"{'ERR':>{colw}}")
            elif e.get("gated"):
                cells.append(f"{e['score']:>{colw-1}.2f}*")
            else:
                cells.append(f"{e['score']:>{colw}.2f}")
        print(f"{i:<5} {n:<{width}} {suite[n]:>+8.3f} " + " ".join(cells))

    print(f"\nSUITE = mean of per-task CVaR@10%, each task clipped at {TASK_FLOOR:+.1f} so that "
          "no single task can dominate.")
    print("Per-task columns are UNCLIPPED - their spread is the diagnostic content.")
    print("A task with no controller scores 0.00, not dropped.")
    print("*  = at least one scenario gated (safety envelope or actuator duty).")
    print("-- = no submission for that task.   ERR = module would not import.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Pin the exact task versions. Run records already carry a content hash each; the
    # aggregate did not, so a published leaderboard could not prove which tasks produced it
    # -- and tasks do change (shell_fractionator_v1 was superseded by v2 within a day).
    Path(args.out).write_text(
        json.dumps({
            "suite": suite,
            "per_task": per_task,
            "tasks": {t.task_id: {"content_hash": t.content_hash, "tier": t.tier,
                                  "seeds": len(t.seeds)} for t in tasks},
        }, indent=2),
        encoding="utf-8")
    print(f"\nwritten {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

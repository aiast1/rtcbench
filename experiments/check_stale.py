"""Is a published results file still describing the tasks that exist?

This replaces task immutability as the mechanism that keeps the leaderboard honest.

Immutability was the original rule: a published task is frozen, and a change means cutting a
v2. It is the right rule once results are cited, and the wrong one for a project whose whole
suite re-scores for about a dollar. Freezing tasks to protect a comparison nobody is making
yet buys a hypothetical benefit at the cost of a real one — a better benchmark.

But the *invariant* underneath the rule still matters: **a results file must describe the
tasks that actually exist, or the leaderboard silently means something else.** Immutability
guaranteed that by forbidding change. Cheap reruns guarantee it by making change followed by
a rerun the normal path — and this script is what makes a missed rerun visible instead of
invisible.

    python experiments/check_stale.py                       # every results file
    python experiments/check_stale.py --fix-hint            # print the rerun command

Exit code 1 if anything is stale, so CI can hold the line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtcbench.task import Task

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaderboard", default=str(ROOT / "leaderboard"))
    ap.add_argument("--tasks", default=str(ROOT / "tasks"))
    ap.add_argument("--fix-hint", action="store_true")
    args = ap.parse_args(argv)

    current = {}
    for p in sorted(Path(args.tasks).glob("*_v*.yaml")):
        t = Task.load(p)
        current[t.task_id] = t.content_hash

    results = sorted(Path(args.leaderboard).glob("matrix_*.json"))
    if not results:
        print("no results files found")
        return 0

    stale_any = False
    for r in results:
        data = json.loads(r.read_text(encoding="utf-8"))
        pinned = data.get("tasks")
        print(f"\n{r.name}")
        if not pinned:
            print("  UNPINNED - this file records no task hashes, so nothing can be checked.")
            print("  Anything produced before matrix.py started pinning them is in this "
                  "position; treat its numbers as describing tasks of unknown vintage.")
            stale_any = True
            continue

        drift = []
        for task_id, meta in sorted(pinned.items()):
            now = current.get(task_id)
            if now is None:
                drift.append(f"{task_id}: task no longer exists")
            elif now != meta.get("content_hash"):
                drift.append(f"{task_id}: {meta['content_hash']} -> {now}")
        if drift:
            stale_any = True
            print(f"  STALE - {len(drift)} task(s) changed since this was measured:")
            for d in drift:
                print(f"    {d}")
        else:
            print(f"  current - all {len(pinned)} task hashes match")

    if stale_any:
        print("\nA stale results file is not a crime -- it is a rerun waiting to happen.")
        if args.fix_hint:
            tasks = " ".join(f"tasks/{t}.yaml" for t in sorted(current))
            print("\n  python experiments/commission.py --rounds 3 --concurrency 12 \\")
            print(f"      --tasks {tasks} --models <roster> --out experiments/submissions2")
            print(f"  python experiments/matrix.py --submissions experiments/submissions2 \\")
            print(f"      --tasks {tasks} --out leaderboard/matrix_<date>.json")
    return 1 if stale_any else 0


if __name__ == "__main__":
    raise SystemExit(main())

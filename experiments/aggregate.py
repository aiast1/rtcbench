"""Combine repeated runs into a result with an interval instead of a point.

Why this exists. Every leaderboard this project published before it was a single
commissioning run per model, and that turned out to be indefensible: re-running three models
three times gave suite-score spreads of 0.107, 0.265 and 0.407, against a top-six span of
0.297. One model's run-to-run noise exceeded the entire top of the table. A ranking whose
spread is larger than its gaps is not a ranking.

So a published result is now N runs, and the headline carries the spread.

**The aggregate is the MEAN, and the spread is shown beside it, never folded in.** Two
tempting alternatives were rejected:

* *Best of N* rewards buying more attempts, which is a budget axis, not a control axis.
* *Worst of N* is the same argument that makes CVaR the per-task headline, and it is
  defensible — but applied here it would compound: a submission already ranked on its worst
  scenarios would then be ranked on its worst run of those. Two layers of tail-weighting
  measure luck more than skill.

Ranks are reported as tied wherever the intervals overlap, because reporting an order the
data does not support is the failure this whole file exists to prevent.

    python experiments/aggregate.py --matrices out/v3/matrix_rep*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(paths: list[str]) -> list[dict]:
    return [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]


def aggregate(runs: list[dict]) -> dict:
    models = sorted({m for r in runs for m in r["suite"]})
    out = {}
    for m in models:
        vals = [r["suite"][m] for r in runs if m in r["suite"]]
        out[m] = {
            "runs": vals,
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "spread": float(np.max(vals) - np.min(vals)),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return out


def per_task(runs: list[dict]) -> dict:
    tasks = sorted({t for r in runs for t in r["per_task"]})
    out: dict = {}
    for t in tasks:
        out[t] = {}
        models = sorted({m for r in runs for m in r["per_task"].get(t, {})})
        for m in models:
            vals = [r["per_task"][t][m]["score"]
                    for r in runs if m in r["per_task"].get(t, {})]
            gated = [r["per_task"][t][m].get("gated", 0)
                     for r in runs if m in r["per_task"].get(t, {})]
            crashed = [r["per_task"][t][m].get("crashed", 0)
                       for r in runs if m in r["per_task"].get(t, {})]
            out[t][m] = {"score": float(np.mean(vals)), "runs": vals,
                         "spread": float(np.max(vals) - np.min(vals)),
                         "gated": int(np.mean(gated)) if gated else 0,
                         "crashed": int(np.mean(crashed)) if crashed else 0}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "out" / "aggregate.json"))
    args = ap.parse_args(argv)

    runs = load(args.matrices)
    agg = aggregate(runs)
    ranked = sorted(agg, key=lambda m: -agg[m]["mean"])

    width = max(12, max(len(m) for m in ranked) + 1)
    print(f"{len(runs)} repeats, {len(ranked)} models\n")
    print(f"{'rank':<5} {'model':<{width}} {'mean':>8} {'spread':>8} {'range':>18}  runs")
    print("-" * (5 + width + 44))

    prev_hi = None
    rank = 0
    for i, m in enumerate(ranked, 1):
        a = agg[m]
        # A rank is only claimed where this model's interval clears the one above it.
        if prev_hi is None or a["max"] < prev_hi:
            rank = i
            shown = f"{rank}"
        else:
            shown = f"={rank}"
        prev_hi = min(prev_hi, a["max"]) if prev_hi is not None else a["max"]
        rng = f"[{a['min']:+.3f}, {a['max']:+.3f}]"
        runs_s = " ".join(f"{v:+.3f}" for v in a["runs"])
        print(f"{shown:<5} {m[:width-1]:<{width}} {a['mean']:>+8.3f} {a['spread']:>8.3f} "
              f"{rng:>18}  {runs_s}")

    spreads = [agg[m]["spread"] for m in ranked]
    means = sorted((agg[m]["mean"] for m in ranked), reverse=True)
    top6 = means[0] - means[min(5, len(means) - 1)]
    print(f"\nmedian run-to-run spread {np.median(spreads):.3f}   "
          f"largest {max(spreads):.3f}   top-six span {top6:.3f}")
    if np.median(spreads) > top6 / 3:
        print("Spread is large relative to the gaps: treat the ordering as indicative, and\n"
              "quote the interval rather than the rank.")

    payload = {"suite": {m: agg[m]["mean"] for m in agg}, "aggregate": agg,
               "per_task": per_task(runs), "repeats": min(v["n"] for v in agg.values()) if agg else 0,
        # Per MODEL, not the number of input matrices. Merging a 20-model run with an
        # 8-model one gives six files but three runs each, and recording six there
        # would overstate the evidence behind every number in the file.
        "matrices": len(runs),
               "tasks": runs[0].get("tasks", {})}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

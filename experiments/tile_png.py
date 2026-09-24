"""Render the leaderboard tile as a PNG.

The published tile is an SVG, which is right for a README — it stays sharp and diffs as
text. It is the wrong format everywhere else: LinkedIn, Slack, a slide and most chat clients
will not render SVG at all, so the figure that carries every result in this project could not
be shown anywhere outside GitHub.

Rasterising the SVG is a dead end here, and both dead ends are the same one: cairo. cairosvg
wants a native libcairo DLL that Windows does not ship, and svglib parses this file perfectly
well before renderPM refuses it for want of the `rlPyCairo` backend -- which wants that same
library. Any converter in this family needs cairo installed first; do not re-litigate it one
package at a time.

So this draws the same figure directly, importing `cell_color`, `PRETTY` and `TASK_LABEL`
from `report` so the colours, names and column order cannot drift from the SVG it mirrors.

    python experiments/tile_png.py --matrix leaderboard/aggregate_2026-09-24.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from report import PRETTY, TASK_LABEL, cell_color  # noqa: E402

# Geometry in px, matching matrix_svg so the two renderings stay comparable.
LABEL_W, CELL_W, CELL_H, GAP = 150, 62, 24, 2
SUITE_W, PAD_X, TOP, BOTTOM = 128, 22, 156, 62  # TOP clears the rotated column headers


def render(matrix: dict, out: Path, dark: bool = False, scale: float = 2.0) -> Path:
    agg = matrix.get("aggregate") or {m: {"mean": v} for m, v in matrix["suite"].items()}
    per_task = matrix["per_task"]
    tasks = [t for t in TASK_LABEL if t in per_task]
    models = sorted(agg, key=lambda m: -agg[m]["mean"])

    fg = "#e8eaed" if dark else "#1a1a19"
    muted = "#9a9a94" if dark else "#6b6a66"
    bg = "#14171a" if dark else "#fcfcfb"

    w = LABEL_W + len(tasks) * (CELL_W + GAP) + SUITE_W + PAD_X
    h = TOP + len(models) * CELL_H + BOTTOM
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100 * scale, facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.axis("off"); ax.set_facecolor(bg)
    mono = ["DejaVu Sans Mono", "Consolas", "monospace"]

    n_rep = matrix.get("repeats", 1)
    ax.text(PAD_X, 30, f"RTCbench — closed-loop control, {len(models)} models "
            f"× {len(tasks)} plants × {n_rep} repeats",
            color=fg, fontsize=11.5, fontfamily=mono, fontweight="bold", va="center")
    ax.text(PAD_X, 50, "0 = actuators frozen  ·  1 = a well-tuned reference controller  "
            "·  higher is better  ·  each cell is CVaR@10% over 20 parameter draws",
            color=muted, fontsize=7.6, fontfamily=mono, va="center")

    for j, t in enumerate(tasks):                                   # column headers
        x = LABEL_W + j * (CELL_W + GAP) + CELL_W / 2
        ax.text(x, TOP - 8, TASK_LABEL.get(t, t), color=muted, fontsize=7.8,
                fontfamily=mono, rotation=35, rotation_mode="anchor",
                ha="left", va="bottom")

    for i, m in enumerate(models):
        y = TOP + i * CELL_H
        ax.text(LABEL_W - 10, y + CELL_H / 2, PRETTY.get(m, m), color=fg, fontsize=8.2,
                fontfamily=mono, ha="right", va="center")
        for j, t in enumerate(tasks):
            x = LABEL_W + j * (CELL_W + GAP)
            cell = per_task.get(t, {}).get(m)
            if cell is None:
                continue
            sc = cell["score"]
            ax.add_patch(Rectangle((x, y + 1), CELL_W, CELL_H - 3,
                                   facecolor=cell_color(sc, dark), edgecolor="none"))
            txt = f"{sc:.2f}" if abs(sc) < 10 else f"{sc:.0f}"
            ax.text(x + CELL_W / 2, y + CELL_H / 2, txt, color=fg, fontsize=7.6,
                    fontfamily=mono, ha="center", va="center")
            if cell.get("gated"):        # a scenario lost to safety or duty, not to control
                ax.add_patch(Circle((x + CELL_W - 6, y + 6), 2.0,
                                    facecolor=fg, edgecolor="none", alpha=.85))
        sx = LABEL_W + len(tasks) * (CELL_W + GAP) + 10
        a = agg[m]
        ax.text(sx, y + CELL_H / 2, f"{a['mean']:+.3f}", color=fg, fontsize=8.6,
                fontfamily=mono, fontweight="bold", va="center")
        if "spread" in a:
            ax.text(sx + 58, y + CELL_H / 2, f"±{a['spread']/2:.3f}", color=muted,
                    fontsize=7.2, fontfamily=mono, va="center")

    fy = TOP + len(models) * CELL_H + 22
    ax.text(PAD_X, fy, "● a scenario gated on safety or actuator duty", color=muted,
            fontsize=7.4, fontfamily=mono, va="center")
    ax.text(PAD_X, fy + 16, "github.com/aiast1/rtcbench", color=muted, fontsize=7.4,
            fontfamily=mono, va="center")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=bg, dpi=100 * scale)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(ROOT / "leaderboard" / "aggregate_2026-09-24.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "leaderboard"))
    ap.add_argument("--stem", default="matrix_2026-09-24")
    ap.add_argument("--scale", type=float, default=2.0)
    args = ap.parse_args(argv)

    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    for dark in (False, True):
        p = render(matrix, Path(args.out_dir) / f"{args.stem}-{'dark' if dark else 'light'}.png",
                   dark=dark, scale=args.scale)
        print(f"wrote {p}  ({p.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

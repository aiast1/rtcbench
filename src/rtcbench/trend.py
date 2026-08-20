"""Trend plots, in dependency-free SVG.

Every leaderboard row should link to the PV/SP/OP trend of the run behind it. It is the
view a control engineer reads first and the cheapest possible defence against a benchmark
being gamed: a controller that holds setpoint by hammering the valve, or that "wins" by
riding a constraint, is obvious in three seconds on a trend and invisible in a scalar.

No matplotlib. A plotting dependency is a real install cost for a project whose whole
promise is ``pip install`` and go, and axes-plus-polylines is not enough code to justify it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .record import RunRecord

_W = 920
_PANEL_H = 170
_PAD_L, _PAD_R, _PAD_T, _PAD_B = 62, 18, 26, 26

_STYLE = """
:root { --bg:#ffffff; --fg:#1d2129; --muted:#8b939c; --grid:#e6e9ec;
        --pv:#2563eb; --sp:#dc2626; --mv:#059669; --mv2:#f59e0b; --bad:#fecaca; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14171a; --fg:#e8eaed; --muted:#8b939c; --grid:#272c31;
          --pv:#60a5fa; --sp:#f87171; --mv:#34d399; --mv2:#fbbf24; --bad:#7f1d1d; }
}
text { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; fill: var(--fg); }
.ax { font-size: 10px; fill: var(--muted); }
.ti { font-size: 12px; }
.lg { font-size: 10px; }
"""


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class _Panel:
    def __init__(self, y0: int, t: NDArray[np.float64], lo: float, hi: float) -> None:
        self.y0 = y0
        self.t = t
        span = hi - lo
        pad = span * 0.08 if span > 0 else 1.0
        self.lo, self.hi = lo - pad, hi + pad
        self.x0, self.x1 = _PAD_L, _W - _PAD_R
        self.yt, self.yb = y0 + _PAD_T, y0 + _PANEL_H - _PAD_B
        self.t0 = float(t[0]) if t.size else 0.0
        self.t1 = float(t[-1]) if t.size else 1.0
        if self.t1 <= self.t0:
            self.t1 = self.t0 + 1.0

    def px(self, t: float) -> float:
        return self.x0 + (t - self.t0) / (self.t1 - self.t0) * (self.x1 - self.x0)

    def py(self, v: float) -> float:
        if self.hi <= self.lo:
            return (self.yt + self.yb) / 2
        return self.yb - (v - self.lo) / (self.hi - self.lo) * (self.yb - self.yt)

    def line(self, values: NDArray[np.float64], color: str, dash: str = "") -> str:
        pts = " ".join(
            f"{self.px(float(ti)):.1f},{self.py(float(v)):.1f}"
            for ti, v in zip(self.t, values)
            if np.isfinite(v)
        )
        if not pts:
            return ""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6"{d}/>'

    def frame(self, title: str, unit: str) -> str:
        out = [
            f'<rect x="{self.x0}" y="{self.yt}" width="{self.x1-self.x0}" '
            f'height="{self.yb-self.yt}" fill="none" stroke="var(--grid)"/>',
            f'<text class="ti" x="{self.x0}" y="{self.yt-8}">{_esc(title)}</text>',
        ]
        for frac in (0.0, 0.5, 1.0):
            v = self.lo + frac * (self.hi - self.lo)
            y = self.py(v)
            out.append(
                f'<line x1="{self.x0}" y1="{y:.1f}" x2="{self.x1}" y2="{y:.1f}" '
                f'stroke="var(--grid)" stroke-width="1"/>'
            )
            out.append(
                f'<text class="ax" x="{self.x0-6}" y="{y+3:.1f}" text-anchor="end">'
                f"{v:.4g}</text>"
            )
        out.append(
            f'<text class="ax" x="{self.x0-6}" y="{self.yt-4}" text-anchor="end">{_esc(unit)}</text>'
        )
        return "".join(out)

    def shade(self, mask: NDArray[np.bool_], color: str) -> str:
        """Shade the periods where ``mask`` is true — used for constraint violations."""
        out = []
        in_run = False
        start = 0.0
        for i, flag in enumerate(mask):
            if flag and not in_run:
                in_run, start = True, float(self.t[i])
            elif not flag and in_run:
                in_run = False
                out.append(self._band(start, float(self.t[i])))
        if in_run:
            out.append(self._band(start, float(self.t[-1])))
        return "".join(out).replace("__C__", color)

    def _band(self, a: float, b: float) -> str:
        x0, x1 = self.px(a), self.px(max(b, a + 1e-9))
        return (
            f'<rect x="{x0:.1f}" y="{self.yt}" width="{max(x1-x0,1.5):.1f}" '
            f'height="{self.yb-self.yt}" fill="__C__" opacity="0.45"/>'
        )


def trend_svg(record: RunRecord, *, title: str | None = None) -> str:
    """Render one run record as a standalone SVG trend."""
    tr = record.trace
    if tr.t.size == 0:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="60"></svg>'

    n_panels = len(record.controlled) + 1
    height = n_panels * _PANEL_H + 34
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="{height}" '
        f'viewBox="0 0 {_W} {height}">',
        f"<style>{_STYLE}</style>",
        f'<rect width="{_W}" height="{height}" fill="var(--bg)"/>',
    ]

    head = title or (
        f"{record.task_id} [{record.tier}] seed={record.seed}  {record.controller_id}"
    )
    parts.append(f'<text class="ti" x="{_PAD_L}" y="18">{_esc(head)}</text>')

    y0 = 26
    for slot, ch in enumerate(record.controlled):
        pv, sp = tr.y[:, ch], tr.r[:, ch]
        finite = np.concatenate([pv[np.isfinite(pv)], sp[np.isfinite(sp)]])
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        p = _Panel(y0, tr.t, lo, hi)
        tag = record.measurement_tags[ch] if ch < len(record.measurement_tags) else f"y{ch}"
        parts.append(p.frame(f"{tag}   PV / SP", ""))
        parts.append(p.shade(tr.violations, "var(--bad)"))
        parts.append(p.line(sp, "var(--sp)", dash="5 3"))
        parts.append(p.line(pv, "var(--pv)"))
        if slot == 0:
            parts.append(
                f'<text class="lg" x="{p.x1-150}" y="{p.yt-8}">'
                f'<tspan fill="var(--pv)">--- PV</tspan>  '
                f'<tspan fill="var(--sp)">--- SP</tspan></text>'
            )
        y0 += _PANEL_H

    u_all = np.concatenate([tr.u_commanded.ravel(), tr.u_actual.ravel()])
    p = _Panel(y0, tr.t, float(u_all.min()), float(u_all.max()))
    parts.append(p.frame("OP  (commanded solid / actual dashed)", ""))
    parts.append(p.shade(tr.overrun, "var(--bad)"))
    for j in range(tr.u_commanded.shape[1]):
        color = "var(--mv)" if j % 2 == 0 else "var(--mv2)"
        parts.append(p.line(tr.u_commanded[:, j], color))
        if not np.allclose(tr.u_commanded[:, j], tr.u_actual[:, j]):
            parts.append(p.line(tr.u_actual[:, j], color, dash="3 3"))
    parts.append(
        f'<text class="ax" x="{p.x1}" y="{p.yb+16}" text-anchor="end">'
        f"time (s) -> {tr.t[-1]:.0f}</text>"
    )

    parts.append("</svg>")
    return "".join(parts)


def write_trends(records: Sequence[RunRecord], out_dir, limit: int = 3) -> list[str]:
    """Write trends for the first ``limit`` records. Defaults to a handful because the
    point is a spot check a human will actually open, not an archive nobody reads."""
    from pathlib import Path

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    written = []
    for rec in list(records)[:limit]:
        path = d / f"{rec.task_id}_seed{rec.seed}.svg"
        path.write_text(trend_svg(rec), encoding="utf-8")
        written.append(str(path))
    return written

"""Render a suite matrix as a readable HTML report.

A 14x8 grid of signed floats is technically complete and practically unreadable, which
defeats the point of publishing results at all. This turns one `matrix.json` into two
charts that answer the two questions a reader actually has: *who won*, and *what is each
model bad at*.

Chart choices, and why:

* **The score has polarity, not just magnitude.** Zero means "no better than freezing the
  actuators" and 1.0 means "matched a well-tuned reference". So both charts use a DIVERGING
  scale — two hues either side of a neutral midpoint — rather than a sequential one. A
  sequential ramp would make -0.9 and +0.1 look like neighbours, when one is a failure and
  the other is not.
* **A grid of one measure across two dimensions is a heatmap**, and the column patterns are
  the finding: a task where most of the field is red is a task that works.
* **Cells are not all labelled.** Only the ones that carry an argument — beat the reference,
  or hit the floor. The exact numbers live in the table below, which also serves as the
  accessibility view, so identity is never carried by colour alone.

Colour is computed rather than chosen: the red arm's lightness and chroma are matched to
the blue arm's step-for-step in OKLCH, so neither side reads as louder than the other. Dark
mode is a separate selection stepped for the dark surface, not an inversion.

    python experiments/report.py --matrix leaderboard/matrix_2026-08-21.json --out out/report.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Diverging ramp, negative -> positive. Light arms taken from the reference palette's blue
# sequential steps; the red arm is those same OKLCH lightness/chroma values at red's hue.
RED_LIGHT = ["#b13f3c", "#d75853", "#e4857e", "#f1aea8", "#fad6d2"]
BLUE_LIGHT = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf"]
MID_LIGHT = "#f0efec"

RED_DARK = ["#ff7c75", "#dd6a64", "#b45d57", "#8d4f4a", "#68413e"]
BLUE_DARK = ["#384e6b", "#416593", "#487cbd", "#5094e9", "#5cacff"]
MID_DARK = "#383835"

PRETTY = {
    "us-anthropic-claude-fable-5": "Fable 5",
    "us-anthropic-claude-sonnet-5": "Sonnet 5",
    "us-anthropic-claude-haiku-4-5-20251001-v1-0": "Haiku 4.5",
    "us-anthropic-claude-opus-5": "Opus 5",
    "moonshot-kimi-k2-thinking": "Kimi K2 Thinking",
    "google-gemma-3-27b-it": "Gemma 3 27B",
    "zai-glm-5": "GLM-5",
    "qwen-qwen3-coder-next": "Qwen3 Coder Next",
    "mistral-mistral-large-3-675b-instruct": "Mistral Large 3",
    "minimax-minimax-m2-5": "MiniMax M2.5",
    "meta-llama3-70b-instruct-v1-0": "Llama 3 70B",
    "deepseek-v3-2": "DeepSeek V3.2",
    "amazon-nova-pro-v1-0": "Nova Pro",
    "nvidia-nemotron-super-3-120b": "Nemotron Super 3",
    "openai-gpt-oss-120b-1-0": "gpt-oss 120B",
}

TASK_LABEL = {
    "four_tank_v1": "Four-tank",
    "four_tank_nmp_v1": "Four-tank NMP",
    "column_a_v1": "Column A",
    "ph_neutralization_v1": "pH",
    "deadtime_process_v1": "Deadtime",
    "van_de_vusse_v1": "Van de Vusse",
    "boiler_drum_v1": "Boiler drum",
    "unstable_cstr_v1": "Unstable CSTR",
}

TASK_TRAP = {
    "four_tank_v1": "minimum phase; the gentle one",
    "four_tank_nmp_v1": "RHP zero - the obvious pairing is wrong",
    "column_a_v1": "ill-conditioned 41-stage column",
    "ph_neutralization_v1": "gain varies by orders of magnitude",
    "deadtime_process_v1": "deadtime moves with throughput",
    "van_de_vusse_v1": "steady-state gain reverses sign",
    "boiler_drum_v1": "inverse response (shrink and swell)",
    "unstable_cstr_v1": "open-loop unstable operating point",
}

FLOOR = -1.0


def bucket(score: float, arms: int = 5) -> int:
    """Map a score to a signed ramp index. 0 is the neutral midpoint."""
    if abs(score) < 0.02:
        return 0
    span = 1.15  # a touch past the best observed score, so the top arm is reachable
    step = min(arms, max(1, int(abs(score) / span * arms) + 1))
    return step if score > 0 else -step


def cell_color(score: float, dark: bool) -> str:
    b = bucket(score)
    if b == 0:
        return MID_DARK if dark else MID_LIGHT
    if b > 0:
        arm = BLUE_DARK if dark else BLUE_LIGHT
        return arm[min(b - 1, len(arm) - 1)]
    arm = RED_DARK if dark else RED_LIGHT
    return arm[max(0, len(arm) + b)]


def render(matrix: dict, task_order: list[str], generated: str) -> str:
    suite = matrix["suite"]
    per_task = matrix["per_task"]
    models = sorted(suite, key=lambda m: -suite[m])
    tasks = [t for t in task_order if t in per_task]

    def name(m: str) -> str:
        return PRETTY.get(m, m)

    def entry(task: str, model: str) -> dict:
        return per_task.get(task, {}).get(model, {})

    beat = sum(1 for t in tasks for m in models if entry(t, m).get("score", 0) > 1.0)
    floored = sum(1 for t in tasks for m in models
                  if entry(t, m).get("score", 0) <= FLOOR + 1e-9)
    total_cells = len(tasks) * len(models)
    champion = models[0]

    # ---- ranked diverging bars -------------------------------------------------
    bar_rows = []
    lo, hi = min(suite.values()), max(suite.values())
    reach = max(abs(lo), abs(hi), 0.2) * 1.12
    for i, m in enumerate(models, 1):
        v = suite[m]
        frac = abs(v) / reach * 50.0
        left = 50.0 - frac if v < 0 else 50.0
        cls = "neg" if v < 0 else "pos"
        bar_rows.append(
            f'<div class="row" tabindex="0" data-tip="{html.escape(name(m))} &mdash; '
            f'suite {v:+.3f}">'
            f'<div class="rank">{i}</div>'
            f'<div class="who">{html.escape(name(m))}</div>'
            f'<div class="track"><div class="zero"></div>'
            f'<div class="bar {cls}" style="left:{left:.2f}%;width:{frac:.2f}%"></div></div>'
            f'<div class="val {cls}">{v:+.3f}</div></div>'
        )

    # ---- heatmap ---------------------------------------------------------------
    head = "".join(
        f'<th scope="col" title="{html.escape(TASK_TRAP.get(t, ""))}">'
        f'<span>{html.escape(TASK_LABEL.get(t, t))}</span></th>'
        for t in tasks
    )
    grid = []
    for m in models:
        cells = []
        for t in tasks:
            e = entry(t, m)
            s = e.get("score")
            if s is None:
                cells.append('<td class="cell none" data-tip="no submission">&middot;</td>')
                continue
            gated = e.get("gated", 0)
            # Label only the cells that carry an argument: beat the reference, or bottomed out.
            label = ""
            if s > 1.0:
                label = f"{s:.2f}"
            elif s <= FLOOR + 1e-9:
                label = "floor"
            tip = (f"{html.escape(name(m))} &middot; {html.escape(TASK_LABEL.get(t, t))}<br>"
                   f"score {s:+.3f}" + (f" &middot; {gated} scenario(s) gated" if gated else ""))
            mark = ' <span class="gate" aria-hidden="true">&#9670;</span>' if gated else ""
            cells.append(
                f'<td class="cell" tabindex="0" data-tip="{tip}" '
                f'style="--c-l:{cell_color(s, False)};--c-d:{cell_color(s, True)}">'
                f'<span class="lbl">{label}</span>{mark}</td>'
            )
        grid.append(
            f'<tr><th scope="row">{html.escape(name(m))}</th>{"".join(cells)}</tr>'
        )

    # ---- legend swatches -------------------------------------------------------
    legend = []
    for b, lab in ((-5, "−1.0"), (-3, ""), (-1, ""), (0, "0"),
                   (1, ""), (3, ""), (5, "+1.1")):
        v = 0.0 if b == 0 else (b / 5) * 1.1
        legend.append(
            f'<div class="sw" style="--c-l:{cell_color(v, False)};'
            f'--c-d:{cell_color(v, True)}"></div><div class="swl">{lab}</div>'
        )

    # ---- table view ------------------------------------------------------------
    thead = "".join(f"<th>{html.escape(TASK_LABEL.get(t, t))}</th>" for t in tasks)
    tbody = []
    for m in models:
        tds = "".join(
            f"<td>{entry(t, m).get('score', float('nan')):+.3f}</td>" for t in tasks
        )
        tbody.append(f"<tr><th>{html.escape(name(m))}</th>"
                     f"<td class='st'>{suite[m]:+.3f}</td>{tds}</tr>")

    return TEMPLATE.format(
        generated=html.escape(generated),
        champion=html.escape(name(champion)),
        champion_score=f"{suite[champion]:+.3f}",
        n_models=len(models),
        n_tasks=len(tasks),
        beat=beat,
        total_cells=total_cells,
        floored=floored,
        bars="\n".join(bar_rows),
        head=head,
        grid="\n".join(grid),
        legend="".join(legend),
        thead=thead,
        tbody="\n".join(tbody),
    )


TEMPLATE = """<title>RTCbench Suite Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?\
family=IBM+Plex+Mono:wght@400;500;600&\
family=IBM+Plex+Sans:wght@400;500;600;700&\
family=IBM+Plex+Sans+Condensed:wght@600;700&display=swap">
<style>
  /* Typography is IBM Plex throughout -- a family drawn for engineering documentation,
     which is what this is. Every numeral on the page is monospaced and tabular, because
     an instrument readout that does not align in a column is misread. */
  .viz-root {{
    --font-display:"IBM Plex Sans Condensed", "Helvetica Neue", Arial, sans-serif;
    --font-body:"IBM Plex Sans", -apple-system, "Segoe UI", Roboto, sans-serif;
    --font-mono:"IBM Plex Mono", ui-monospace, "SF Mono", Consolas, monospace;
    color-scheme: light;
    --surface-1:#fcfcfb; --surface-2:#f4f3f0; --line:#e3e2de;
    --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#8a8984;
    --pos:#256abf; --neg:#b13f3c;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz-root {{
      color-scheme: dark;
      --surface-1:#1a1a19; --surface-2:#232322; --line:#33322f;
      --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8a8984;
      --pos:#5cacff; --neg:#ff7c75;
    }}
  }}
  :root[data-theme="dark"] .viz-root {{
    color-scheme: dark;
    --surface-1:#1a1a19; --surface-2:#232322; --line:#33322f;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8a8984;
    --pos:#5cacff; --neg:#ff7c75;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--surface-1); }}
  .viz-root {{
    background:var(--surface-1); color:var(--text-primary);
    font-family:var(--font-body);
    padding: 44px 28px 72px; max-width: 1060px; margin: 0 auto;
    -webkit-font-smoothing: antialiased;
  }}
  .masthead {{ border-top:2px solid var(--text-primary); padding-top:14px; }}
  .eyebrow {{ font-family:var(--font-mono); font-size:11px; letter-spacing:0.14em;
              text-transform:uppercase; color:var(--text-muted); margin-bottom:10px; }}
  h1 {{ font-family:var(--font-display); font-size: clamp(28px, 5vw, 42px);
        margin:0 0 10px; letter-spacing:-0.01em; font-weight:700;
        text-wrap:balance; line-height:1.05; }}
  .sub {{ color:var(--text-secondary); font-size:15px; margin:0 0 30px;
          line-height:1.6; max-width:66ch; }}
  h2 {{ font-family:var(--font-mono); font-size:11.5px; text-transform:uppercase;
        letter-spacing:0.14em; color:var(--text-muted); margin:46px 0 6px;
        font-weight:500; padding-bottom:8px; border-bottom:1px solid var(--line); }}
  .note {{ color:var(--text-secondary); font-size:13px; margin:12px 0 20px;
           line-height:1.6; max-width:74ch; }}

  /* A summary strip rather than cards: three readings divided by hairlines, the way a
     control-room header bar carries its key figures. */
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
            border-top:1px solid var(--line); border-bottom:1px solid var(--line); }}
  .stat {{ padding:16px 20px 15px; border-left:1px solid var(--line); }}
  .stat:first-child {{ border-left:none; padding-left:0; }}
  .stat .n {{ font-family:var(--font-mono); font-size:23px; font-weight:600;
              letter-spacing:-0.02em; font-variant-numeric:tabular-nums;
              line-height:1.15; }}
  .stat .k {{ font-size:12px; color:var(--text-secondary); margin-top:6px;
              line-height:1.45; }}

  .row {{ display:grid; grid-template-columns:26px 148px 1fr 62px; align-items:center;
          gap:10px; padding:2px 0; }}
  .row:focus {{ outline:2px solid var(--text-secondary); outline-offset:2px; border-radius:4px; }}
  .rank {{ color:var(--text-muted); font-size:11px; text-align:right;
           font-family:var(--font-mono); font-variant-numeric:tabular-nums; }}
  .who {{ font-size:13px; color:var(--text-primary); white-space:nowrap;
          overflow:hidden; text-overflow:ellipsis; font-weight:500; }}
  .track {{ position:relative; height:15px; }}
  .zero {{ position:absolute; left:50%; top:-1px; bottom:-1px; width:1px;
           background:var(--line); }}
  .bar {{ position:absolute; top:2px; height:11px; }}
  .bar.pos {{ background:var(--pos); border-radius:0 4px 4px 0; }}
  .bar.neg {{ background:var(--neg); border-radius:4px 0 0 4px; }}
  .val {{ font-size:12.5px; text-align:right; font-family:var(--font-mono);
          font-variant-numeric:tabular-nums; color:var(--text-secondary);
          font-weight:500; }}

  .hm {{ border-collapse:separate; border-spacing:2px; width:100%; margin-top:6px; }}
  .hm th {{ font-weight:500; font-size:11px; color:var(--text-secondary);
             font-family:var(--font-mono); letter-spacing:0.02em; }}
  /* One display declaration, not two -- an earlier draft set block then flex on the same
     rule, which is the kind of silent cascade collision that only shows up in the render. */
  .hm thead th span {{ display:flex; align-items:flex-end; height:70px; padding-left:5px;
                       transform:rotate(-34deg); transform-origin:left bottom;
                       white-space:nowrap; }}
  .hm th[scope="row"] {{ text-align:right; padding-right:10px; white-space:nowrap;
                         font-size:12.5px; color:var(--text-primary); width:150px;
                         font-family:var(--font-body); font-weight:500; }}
  .cell {{ height:27px; border-radius:2px; background:var(--c-l);
           text-align:center; font-size:10px; font-family:var(--font-mono);
           font-variant-numeric:tabular-nums;
           color:var(--text-primary); position:relative; }}
  .cell.none {{ background:var(--surface-2); color:var(--text-muted); }}
  .cell:focus {{ outline:2px solid var(--text-primary); outline-offset:1px; }}
  /* The gate marker has to be readable at a glance or it is not doing its job -- at 6px
     it was an invisible speck. */
  .cell .gate {{ position:absolute; top:0; right:2px; font-size:9px; line-height:1;
                 color:var(--text-primary); opacity:.55; }}
  .lbl {{ font-weight:600; }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .cell {{ background:var(--c-d); }}
    :root:where(:not([data-theme="light"])) .sw {{ background:var(--c-d); }}
  }}
  :root[data-theme="dark"] .cell {{ background:var(--c-d); }}
  :root[data-theme="dark"] .sw {{ background:var(--c-d); }}

  .legend {{ display:grid; grid-template-columns:repeat(7,26px); gap:2px;
             margin:16px 0 0 158px; }}
  .sw {{ height:11px; border-radius:2px; background:var(--c-l); }}
  .swl {{ font-size:10px; color:var(--text-muted); text-align:center;
          font-family:var(--font-mono); font-variant-numeric:tabular-nums; }}
  .legkey {{ font-size:12px; color:var(--text-secondary); margin:8px 0 0 158px; }}

  table.tv {{ border-collapse:collapse; width:100%; font-size:12px; margin-top:10px; }}
  table.tv th, table.tv td {{ border-bottom:1px solid var(--line); padding:5px 7px;
                              text-align:right; font-family:var(--font-mono);
                              font-variant-numeric:tabular-nums;
                              color:var(--text-secondary); }}
  table.tv th:first-child {{ text-align:left; color:var(--text-primary);
                             font-family:var(--font-body); font-weight:500; }}
  table.tv thead th {{ color:var(--text-muted); font-weight:500; }}
  table.tv td.st {{ color:var(--text-primary); font-weight:600; }}
  details {{ margin-top:20px; }}
  summary {{ cursor:pointer; font-size:13px; color:var(--text-secondary); }}

  #tip {{ position:fixed; font-family:var(--font-body); pointer-events:none; z-index:50; opacity:0;
          background:var(--text-primary, #0b0b0b); color:var(--surface-1, #fcfcfb);
          padding:6px 9px; border-radius:6px; font-size:12px; line-height:1.4;
          transition:opacity .09s; max-width:260px; }}
  footer {{ margin-top:48px; padding-top:16px; border-top:1px solid var(--line);
            color:var(--text-muted); font-size:12px; line-height:1.65;
            max-width:74ch; }}
  footer code {{ font-family:var(--font-mono); font-size:11.5px; }}
  @media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; }} }}
</style>

<div class="viz-root">
  <div class="masthead">
    <div class="eyebrow">RTCbench &middot; run 2026-08-21 &middot; 8 plants &middot; 20 scenarios each</div>
    <h1>Which models can actually commission a control loop?</h1>
  </div>
  <p class="sub">{n_models} models commissioned a control system for {n_tasks} dynamic
  process plants, each through the identical harness. Every score is anchored:
  <strong>0.0</strong> means no better than freezing the actuators,
  <strong>1.0</strong> means matching a well-tuned reference controller. Higher is better;
  above 1.0 beats the reference.</p>

  <div class="stats">
    <div class="stat"><div class="n">{champion}</div>
      <div class="k">leads the suite at {champion_score}</div></div>
    <div class="stat"><div class="n">{beat} / {total_cells}</div>
      <div class="k">submissions that beat their task&rsquo;s reference</div></div>
    <div class="stat"><div class="n">{floored}</div>
      <div class="k">results at the floor &mdash; worse than doing nothing</div></div>
  </div>

  <h2>Suite score</h2>
  <p class="note">Mean of each model&rsquo;s per-task CVaR@10% &mdash; the average of its
  worst two scenarios on every task. Ranking on the bad tail rather than the mean is
  deliberate: a controller that is excellent on most parameter draws and unsafe on a few is
  the one nobody wants commissioned.</p>
  {bars}

  <h2>Where each model struggles</h2>
  <p class="note">One cell per model per plant. Blue beats doing nothing, red is worse than
  doing nothing, grey is indistinguishable from it. A diamond marks a task where at least
  one scenario was gated outright &mdash; a safety-envelope breach or an actuator worn out by
  chattering. Read the columns: a plant where most of the field is red is a plant that
  works.</p>
  <table class="hm">
    <thead><tr><th></th>{head}</tr></thead>
    <tbody>
    {grid}
    </tbody>
  </table>
  <div class="legend">{legend}</div>
  <div class="legkey">&#9670; at least one scenario gated on safety or actuator duty</div>

  <details>
    <summary>Table view &mdash; exact scores</summary>
    <table class="tv">
      <thead><tr><th>Model</th><th>Suite</th>{thead}</tr></thead>
      <tbody>{tbody}</tbody>
    </table>
  </details>

  <footer>
    Generated {generated} from <code>leaderboard/matrix_2026-08-21.json</code>.
    Every model was given the same brief, three revision rounds, and feedback on held-out
    scenario seeds; submissions are sealed controller files scored by the ordinary harness.
    Two caveats recorded at the time: the Claude models ran in a later batch after a
    parameter bug was fixed, and Opus 5 is absent rather than zero &mdash; Bedrock returned
    an access denial for it on this account.
  </footer>

  <!-- Inside .viz-root on purpose: the tokens are scoped to it, and a tooltip parked
       outside that scope resolves var(--surface-1) to nothing and renders unstyled. -->
  <div id="tip" role="status"></div>
</div>
<script>
(function () {{
  var tip = document.getElementById('tip');
  function show(e) {{
    var el = e.target.closest('[data-tip]');
    if (!el) return;
    tip.innerHTML = el.getAttribute('data-tip');
    tip.style.opacity = '1';
    var r = el.getBoundingClientRect();
    var x = Math.min(window.innerWidth - 270, Math.max(8, r.left));
    var y = r.top > 90 ? r.top - 8 - tip.offsetHeight : r.bottom + 8;
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  }}
  function hide() {{ tip.style.opacity = '0'; }}
  document.addEventListener('mouseover', show);
  document.addEventListener('mouseout', hide);
  document.addEventListener('focusin', show);
  document.addEventListener('focusout', hide);
}})();
</script>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(ROOT / "leaderboard" / "matrix_2026-08-21.json"))
    ap.add_argument("--out", default=str(ROOT / "out" / "report.html"))
    ap.add_argument("--generated", default="2026-08-21")
    args = ap.parse_args(argv)

    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    order = list(TASK_LABEL)
    page = render(matrix, order, args.generated)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}  ({len(page)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

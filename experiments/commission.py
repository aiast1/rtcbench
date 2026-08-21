"""Drive Bedrock models through the controller-commissioning task, across the whole suite.

This is the piece RTCbench v0.1 was missing: the thing that turns "a harness that scores a
controller file" into "a benchmark that measures models". A model is handed the task brief,
writes a controller, sees how it did on a few held-out scenarios, and revises. What it emits
is an ordinary submission, scored by the ordinary harness with no special case.

Design notes worth keeping:

* **The brief is generated from `TaskBrief`, never hand-written.** Whatever a tier redacts is
  redacted here automatically, so this runner cannot leak a model the tier says to withhold.
* **Feedback comes from a held-out replica, not the scored ensemble.** Seeds default to
  outside the task's own list — the design's "training replica: same topology, different
  draw". Pointing it at the scored seeds is tuning on the test set.
* **Every episode keeps the best revision, not the last.** Models routinely make a controller
  worse on a later round while sounding confident about it.
* **Output token limits are discovered, not tabled.** Bedrock model caps vary by two orders
  of magnitude (Llama3-70b: 2048, Nova Pro: 10000, most reasoning models: 32k+). A hardcoded
  table goes stale; instead the first ValidationException is parsed for the real limit and
  the call is retried. An earlier version silently lost two models to this.
* **boto3 lives here, not in core.** `rtcbench` promises two dependencies and a laptop
  install; an AWS SDK in the scored path would break that.

    python experiments/commission.py --models nvidia.nemotron-super-3-120b --tasks tasks/*.yaml
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rtcbench.harness import run_ensemble
from rtcbench.submission import load_controller
from rtcbench.task import Task

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEEDBACK_SEEDS = (1001, 1002, 1003, 1004, 1005)
"""Deliberately outside every shipped task's scored ensemble."""

DEFAULT_MAX_TOKENS = 32768

INTERFACE = '''
class Controller:
    def __init__(self, brief): ...      # the TaskBrief described above
    def reset(self): ...                # called once before each scenario
    def step(self, t, y, r, quality):
        """t: seconds (float). y: numpy array of measured values, one per measurement.
        r: numpy array of setpoints, nan for channels that are not scored.
        quality: numpy bool array, False means that reading is stale/bad.
        Returns: numpy array of control values, one per actuator."""
'''

SYSTEM = (
    "You are a control engineer commissioning a loop on a real process. You write "
    "production PID/MPC/adaptive control code. Respond with ONE fenced ```python block "
    "containing a complete, self-contained module defining class Controller. No prose "
    "outside the code block. Use only numpy. Do not import anything from rtcbench. "
    "brief is a dataclass: use attribute access (brief.sample_time), never brief['...']."
)


def build_brief_text(task: Task) -> str:
    """Render the operating manual a competitor is entitled to, straight from TaskBrief."""
    plant = task.build_plant()
    b = task.brief(plant)
    plant.close()

    lines = [
        f"# Commissioning brief: {b.task_id}  (tier: {b.tier})",
        "",
        b.description.strip(),
        "",
        "## Measurements (what you can read), indices into `y`",
    ]
    for i, c in enumerate(b.measurements):
        scored = "  [SCORED - has a setpoint]" if i in b.controlled else "  [not scored]"
        lines.append(f"  [{i}] {c.tag}  {c.name}  in {c.unit}, range {c.lo} to {c.hi}{scored}")

    lines += ["", "## Actuators (what you can move), indices into your returned vector"]
    for j, c in enumerate(b.actuators):
        lines.append(f"  [{j}] {c.tag}  {c.name}  in {c.unit}, hard limits {c.lo} to {c.hi}")

    lines += [
        "",
        "## Operating data",
        f"  control period     {b.sample_time} s  (step() is called every {b.sample_time} s)",
        f"  scenario length    {b.horizon} s",
        f"  actuators start at {list(b.initial_u)} - the plant is in service and at steady",
        f"                     state there, so a bumpless start means your first output",
        f"                     should be close to this",
        "",
        "## Safety envelope (a violation zeroes the whole scenario)",
    ]
    for c in b.constraints:
        lines.append(f"  {c.signal} must stay within [{c.lo}, {c.hi}]")
    lines.append(
        "  NOTE: some constrained signals may NOT be measured. You must keep them safe "
        "without being able to see them."
    )

    lines += ["", "## Setpoint schedule (values are for the SCORED channels, in order)"]
    for seg in task.setpoints.segments:
        kind = f"ramp over {seg.ramp:g}s" if seg.ramp else "step"
        lines.append(f"  t={seg.t:>7g}s  -> {list(seg.values)}   ({kind})")

    lines += [
        "",
        "## Instrumentation (already applied to what you see)",
        "  sensor noise, transport delay, quantization and occasional dropped samples;",
        "  actuator slew limits and valve stiction. Check the `quality` flags.",
        "",
        "## What is NOT constant",
        "  Plant parameters are drawn per scenario and differ from any nominal you are given.",
        "  A disturbance occurs partway through each run. ONE fixed controller design must",
        "  handle every draw - you do not get to detect the scenario and switch strategy.",
    ]

    if b.model_hint:
        lines += [
            "",
            f"## Model hint ({b.model_hint.get('note', '')})",
            "  " + json.dumps(b.model_hint.get("params", {}), indent=2).replace("\n", "\n  "),
            "  These are NOMINAL values. The plant you are scored on is a draw around them.",
        ]
    return "\n".join(lines)


def first_prompt(task: Task) -> str:
    return f"""{build_brief_text(task)}

## Your interface
```python{INTERFACE}```

## Scoring
Lower cost is better. Cost is w_error * tracking_error + w_effort * actuator_travel, with
w_effort = {task.scoring.w_effort}. Any safety violation zeroes the scenario, and so does
exceeding the actuator duty limit{
    f" of {task.scoring.max_total_variation}" if task.scoring.max_total_variation else ""
} - chattering the actuator to hold setpoint is not a
trade, it fails outright. You are judged on your WORST scenarios, not your average, so
robustness beats aggression.

Write the controller. Think about: integral action to remove the steady-state offset you
start with, anti-windup because the actuators saturate, and how much derivative action is
safe on a noisy delayed measurement.
"""


def revision_prompt(report: str, round_no: int, rounds: int) -> str:
    return f"""Your controller was run on the plant. Results (round {round_no} of {rounds}):

{report}

Revise it to lower the cost. If a scenario shows a violation, a duty breach or a crash, fix
that first - any of those scores zero regardless of tracking. Reply with the complete
revised module in one ```python block.
"""


_THINK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_TOKEN_LIMIT = re.compile(r"model limit of (\d+)")


def extract_code(text: str) -> str | None:
    """Pull the module out of a reply, tolerating reasoning traces and stray prose."""
    cleaned = _THINK.sub("", text)
    cleaned = re.sub(r"<(think|thinking|reasoning)>.*", "", cleaned, flags=re.DOTALL | re.I)
    blocks = _FENCE.findall(cleaned)
    candidates = [b for b in blocks if "class Controller" in b]
    if candidates:
        return candidates[-1].strip()
    if "class Controller" in cleaned:
        start = cleaned.index("class Controller")
        imports = "\n".join(
            ln for ln in cleaned[:start].splitlines() if ln.startswith(("import ", "from "))
        )
        return (imports + "\n\n" + cleaned[start:]).strip()
    return None


def check_syntax(code: str) -> str:
    """Return a description of the syntax error, or "" if the code parses.

    A reasoning model that spends its whole budget thinking gets cut off mid-block and the
    fence regex then cheerfully returns prose. Writing that to a .py file and reporting
    "cost inf" hides a truncation behind what looks like a bad controller — two failures
    needing opposite feedback.
    """
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"line {exc.lineno}: {exc.msg}"
    return ""


@dataclass
class Episode:
    model_id: str
    task_id: str
    slug: str
    best_cost: float = float("inf")
    best_code: str = ""
    rounds_run: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    history: list[str] = field(default_factory=list)
    error: str = ""
    token_cap: int = DEFAULT_MAX_TOKENS


def evaluate(task: Task, path: Path, seeds) -> tuple[float, str]:
    """Run a candidate on the feedback replica. Returns (cost, human-readable report)."""
    try:
        factory, cid = load_controller(str(path))
    except Exception:
        return float("inf"), f"The module failed to import:\n{traceback.format_exc(limit=4)}"
    try:
        records = run_ensemble(task, factory, controller_id=cid, seeds=list(seeds))
    except Exception:
        return float("inf"), f"The run crashed:\n{traceback.format_exc(limit=4)}"

    lines = [f"{'seed':>7} {'cost':>10} {'track':>10} {'effort':>10}  notes"]
    costs = []
    for r in records:
        notes = []
        if r.failed:
            notes.append("CRASHED: " + r.failure.strip().splitlines()[-1][:90])
        if r.cost.violations:
            notes.append(f"SAFETY VIOLATION on {r.cost.violations} periods -> scores zero")
        if r.cost.duty_exceeded:
            notes.append("ACTUATOR DUTY LIMIT EXCEEDED -> scores zero")
        bad = r.cost.violations or r.failed or r.cost.duty_exceeded
        costs.append(10.0 if bad else r.cost.total)
        lines.append(
            f"{r.seed:>7} {r.cost.total:>10.5f} {r.cost.iae:>10.5f} {r.cost.tv:>10.5f}  "
            + "; ".join(notes)
        )
    mean = float(np.mean(costs))
    lines.append(f"\nmean cost {mean:.5f}")
    return mean, "\n".join(lines)


_local = threading.local()


def _client(region: str):
    """One boto3 client per worker thread — sharing one across threads is not supported."""
    if not hasattr(_local, "client"):
        import boto3
        from botocore.config import Config

        _local.client = boto3.client(
            "bedrock-runtime", region_name=region,
            config=Config(read_timeout=900, connect_timeout=30, retries={"max_attempts": 2}),
        )
    return _local.client


def _converse(client, model_id: str, messages, cap: int, attempts: int = 4):
    """Call Converse, discovering the model's real output cap if ours is too high."""
    delay = 4.0
    for attempt in range(attempts):
        try:
            return client.converse(
                modelId=model_id,
                messages=messages,
                system=[{"text": SYSTEM}],
                inferenceConfig={"maxTokens": cap, "temperature": 0.2},
            ), cap
        except Exception as exc:
            text = str(exc)
            if found := _TOKEN_LIMIT.search(text):
                # The model told us its ceiling; take it and retry rather than losing the run.
                cap = int(found.group(1))
                continue
            name = type(exc).__name__
            retriable = any(k in name for k in ("Throttl", "Timeout", "ServiceUnavailable"))
            if not retriable or attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def commission(model_id: str, task: Task, rounds: int, feedback_seeds, out_root: Path,
               max_tokens: int, region: str) -> Episode:
    slug = re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")
    ep = Episode(model_id=model_id, task_id=task.task_id, slug=slug, token_cap=max_tokens)
    work = out_root / task.task_id / slug
    work.mkdir(parents=True, exist_ok=True)
    path = work / "controller.py"

    client = _client(region)
    messages = [{"role": "user", "content": [{"text": first_prompt(task)}]}]
    started = time.perf_counter()

    for rnd in range(1, rounds + 1):
        try:
            resp, ep.token_cap = _converse(client, model_id, messages, ep.token_cap)
        except Exception as exc:
            ep.error = f"{type(exc).__name__}: {str(exc)[:180]}"
            break

        reply = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
        ep.input_tokens += resp["usage"]["inputTokens"]
        ep.output_tokens += resp["usage"]["outputTokens"]
        ep.rounds_run = rnd

        truncated = resp.get("stopReason") == "max_tokens"
        code = extract_code(reply)
        problem = ""
        if code is None:
            problem = "I could not find a ```python block defining class Controller."
        elif syntax := check_syntax(code):
            problem = f"That code does not parse: {syntax}."
        if truncated and problem:
            problem += (" Your reply hit the output token limit and was cut off mid-way."
                        " Spend far fewer tokens reasoning and emit the code.")

        if problem:
            ep.history.append(f"r{rnd}:{'trunc' if truncated else 'unusable'}")
            messages.append({"role": "assistant", "content": [{"text": reply[-1500:]}]})
            messages.append({"role": "user", "content": [
                {"text": problem + " Reply with ONLY the complete code block."}]})
            continue

        path.write_text(code, encoding="utf-8")
        cost, report = evaluate(task, path, feedback_seeds)
        ep.history.append(f"r{rnd}:{cost:.5f}" if np.isfinite(cost) else f"r{rnd}:fail")
        if cost <= ep.best_cost:
            ep.best_cost, ep.best_code = cost, code
        if rnd < rounds:
            messages.append({"role": "assistant",
                             "content": [{"text": f"```python\n{code}\n```"}]})
            messages.append({"role": "user",
                             "content": [{"text": revision_prompt(report, rnd, rounds)}]})

    ep.seconds = time.perf_counter() - started
    if ep.best_code:
        path.write_text(ep.best_code, encoding="utf-8")
    elif path.exists():
        path.unlink()
    return ep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "submissions"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--feedback-seeds", type=int, nargs="+",
                    default=list(DEFAULT_FEEDBACK_SEEDS))
    args = ap.parse_args(argv)

    tasks = [Task.load(t) for t in args.tasks]
    out_root = Path(args.out)
    jobs = [(m, t) for t in tasks for m in args.models]

    print(f"{len(args.models)} models x {len(tasks)} tasks = {len(jobs)} episodes, "
          f"{args.rounds} rounds each, {args.concurrency}-way concurrent")
    print(f"feedback seeds {args.feedback_seeds} (held out from every scored ensemble)\n")

    episodes: list[Episode] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(commission, m, t, args.rounds, args.feedback_seeds, out_root,
                        args.max_tokens, args.region): (m, t)
            for m, t in jobs
        }
        for i, fut in enumerate(as_completed(futures), 1):
            model_id, task = futures[fut]
            try:
                ep = fut.result()
            except Exception as exc:
                ep = Episode(model_id=model_id, task_id=task.task_id, slug="",
                             error=f"{type(exc).__name__}: {exc}")
            episodes.append(ep)
            status = ep.error[:44] or (f"best {ep.best_cost:.5f}" if ep.best_code
                                       else "no usable code")
            print(f"  [{i:>3}/{len(jobs)}] {ep.task_id:<24} {ep.model_id:<38} "
                  f"{status:<30} {ep.seconds:5.0f}s  [{' '.join(ep.history)}]")

    log = out_root / "commission_log.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps([e.__dict__ | {"best_code": ""} for e in episodes], indent=2),
                   encoding="utf-8")

    tin = sum(e.input_tokens for e in episodes)
    tout = sum(e.output_tokens for e in episodes)
    usable = sum(1 for e in episodes if e.best_code)
    print(f"\n{usable}/{len(episodes)} episodes produced a usable controller")
    print(f"{tin/1000:.0f}k input / {tout/1000:.0f}k output tokens in "
          f"{(time.perf_counter()-started)/60:.1f} min wall clock")
    print(f"log  {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

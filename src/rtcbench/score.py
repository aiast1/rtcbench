"""Anchored scoring.

A raw cost number is meaningless across plants and unfalsifiable within one: nobody can
tell whether an IAE of 0.031 on a distillation column is good. So every scenario is scored
*relative to two controllers the maintainers publish and anyone can re-run*:

    0.0  = the hold anchor      — freeze the actuators at their initial position
    1.0  = the reference anchor — the maintainers' well-tuned controller for this task

Scores above 1.0 are possible and are the entire point of the exercise.

The property that matters is not the arithmetic, it is what the arithmetic forecloses. Since
the unit of measurement *is* a well-tuned baseline, there is no way to make a submission look
good by comparing it against a weak one. A strawman baseline does not merely weaken a
comparison, it invalidates it, and anchoring removes the opportunity structurally instead of
leaving it to reviewer vigilance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .metrics import Cost, cvar

SAFETY_GATE_SCORE = 0.0
"""What a scenario scores if any hard constraint was violated. A gate, not a penalty."""

SCENARIO_FLOOR = -10.0
"""Sanity clamp on a single scenario, NOT a ranking device.

Its only job is to stop one pathological cost — a controller that diverges, on a task whose
anchor gap happens to be narrow — from producing something like -1e4 and swamping an
otherwise informative mean. Set far enough out that ordinary failures keep their spread.
"""

TASK_FLOOR = -1.0
"""How far below zero a single TASK may drag a model's SUITE score.

This constant exists because the first version put the floor in the wrong place. Per-scenario
flooring at -1.0 did fix the real problem — a task's denominator is its anchor gap, which
ranges from ~0.10 on four_tank to ~0.013 on van_de_vusse, so an identical failure scored 8x
worse on the narrow-anchor task and one task was dominating every suite mean — but it paid
for that by destroying the per-task ranking as well.

Measured on the first full run: unfloored, the field's four_tank_nmp scores spread from
-0.59 to -6.80, a tenfold difference in how badly each model failed. Floored, all of them
read -1.000 and the task ranked nothing.

The two jobs are separate, so they now get separate constants. A per-task score keeps its
full spread and stays diagnostic; the SUITE aggregate clips each task's contribution at
TASK_FLOOR so no single task can dominate. Aggregation is bounded, diagnosis is not.
"""

SCORE_FLOOR = SCENARIO_FLOOR
"""Deprecated alias, kept so an older analysis script does not silently change meaning."""

_DEGENERATE_EPS = 1e-12


@dataclass(frozen=True)
class ScenarioScore:
    seed: int
    score: float
    gated: bool
    """True if a safety violation zeroed this scenario, whatever its cost was."""

    cost: Cost


@dataclass(frozen=True)
class TaskScore:
    task_id: str
    scenarios: tuple[ScenarioScore, ...]
    headline: float
    """CVaR@10% across the ensemble — the ranked number."""

    mean: float
    worst: float
    gated_count: int

    def summary(self) -> str:
        return (
            f"{self.task_id}: CVaR@10%={self.headline:+.3f}  mean={self.mean:+.3f}  "
            f"worst={self.worst:+.3f}  gated={self.gated_count}/{len(self.scenarios)}"
        )


def anchored_score(submission: float, hold: float, reference: float) -> float:
    """Map a raw cost onto the hold/reference scale.

    A degenerate anchor pair — where the reference controller is no better than doing
    nothing — means the task itself is broken, not that every submission is perfect. It
    raises rather than returning a number nobody could interpret.
    """
    denom = hold - reference
    if abs(denom) < _DEGENERATE_EPS:
        raise ValueError(
            "degenerate anchors: the reference controller does no better than holding. "
            "The task is mis-specified (no disturbance, or setpoints already at steady state)."
        )
    return float((hold - submission) / denom)


def score_scenario(
    seed: int,
    submission: Cost,
    hold: float,
    reference: float,
) -> ScenarioScore:
    if submission.violations > 0 or submission.duty_exceeded:
        # Chattering a valve to hold setpoint is not a clever trade against tracking error,
        # it is a controller a plant will reject on actuator wear. A weight lets it be
        # bought off; a gate does not.
        return ScenarioScore(seed=seed, score=SAFETY_GATE_SCORE, gated=True, cost=submission)
    return ScenarioScore(
        seed=seed,
        score=max(SCENARIO_FLOOR, anchored_score(submission.total, hold, reference)),
        gated=False,
        cost=submission,
    )


def score_task(task_id: str, scenarios: Sequence[ScenarioScore], alpha: float = 0.10) -> TaskScore:
    values = [s.score for s in scenarios]
    return TaskScore(
        task_id=task_id,
        scenarios=tuple(scenarios),
        headline=cvar(values, alpha),
        mean=float(np.mean(values)) if values else float("nan"),
        worst=float(np.min(values)) if values else float("nan"),
        gated_count=sum(1 for s in scenarios if s.gated),
    )


def suite_score(task_scores: Sequence[float], floor: float = TASK_FLOOR) -> float:
    """Aggregate per-task scores into one number, bounding each task's contribution.

    Clipping happens HERE and not in the per-task score, so a task keeps its diagnostic
    spread while no single task can dominate the aggregate. A task a model produced no
    controller for is the caller's job to include as 0.0 — dropping it would reward failing
    to answer.
    """
    if not len(task_scores):
        return float("nan")
    return float(np.mean([max(floor, s) for s in task_scores]))

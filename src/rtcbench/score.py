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

SCORE_FLOOR = -1.0
"""Worst score a single scenario can contribute.

The anchored scale is unbounded below, and that turned out to be a real defect. A task's
denominator is its anchor gap, which varies hugely across the suite: on four_tank it is
~0.10, on van_de_vusse ~0.013. A submission that cost 0.25 on van_de_vusse therefore scored
-15.6, and two models scored around -32 there -- numbers that then dominated their suite
means and effectively made one task the whole benchmark.

Flooring fixes the aggregation without losing information that matters. "Worse than doing
nothing" is one failure category; ranking degrees of catastrophe inside it is not
meaningful, because how far below hold you land depends mostly on how narrow that task's
anchors happen to be. -1.0 reads as "as far below the hold anchor as the reference is above
it, or worse". The raw cost stays in the record for anyone who wants the unclamped number.
"""

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
        score=max(SCORE_FLOOR, anchored_score(submission.total, hold, reference)),
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

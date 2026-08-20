"""RTCbench — a benchmark for LLMs that commission real closed-loop control systems.

The whole library is four ideas:

1. A plant is anything satisfying :class:`~rtcbench.plant.Plant`, and it hands out
   measurements, never state.
2. A submission is a :class:`~rtcbench.controller.Controller` — a sealed artifact, not a
   transcript — so it can be re-scored on tasks written years later.
3. A task is data (:class:`~rtcbench.task.Task`), content-hashed and immutable once published.
4. A score is anchored between doing nothing and a well-tuned reference, gated on safety,
   and ranked on the bad tail rather than the mean.

See DESIGN.md for why each of those is the way it is.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .controller import Controller, TaskBrief
from .harness import run_ensemble, run_scenario, score_against_anchors
from .instruments import ActuatorConfig, InstrumentConfig, InstrumentedPlant, SensorConfig
from .metrics import Cost, cvar, scenario_cost
from .plant import Audit, Channel, Constraint, Observation, Plant, PlantSpec
from .record import RunRecord, Trace
from .score import ScenarioScore, TaskScore, anchored_score, score_task
from .task import Task
from .trend import trend_svg

__all__ = [
    "__version__",
    "ActuatorConfig",
    "Audit",
    "Channel",
    "Constraint",
    "Controller",
    "Cost",
    "InstrumentConfig",
    "InstrumentedPlant",
    "Observation",
    "Plant",
    "PlantSpec",
    "RunRecord",
    "ScenarioScore",
    "SensorConfig",
    "Task",
    "TaskBrief",
    "TaskScore",
    "Trace",
    "anchored_score",
    "cvar",
    "run_ensemble",
    "run_scenario",
    "scenario_cost",
    "score_against_anchors",
    "score_task",
    "trend_svg",
]

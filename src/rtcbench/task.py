"""Tasks are data, not code.

A task is a YAML document describing a plant, a scenario ensemble, and how the result is
scored. Keeping it declarative is what makes the leaderboard re-runnable: a submission from
a year ago can be scored against a task pack written tomorrow, because neither one is code
the other has to import.

Two rules hold the whole scheme together:

* **A published task is immutable.** You never edit ``four_tank_v1``; you cut ``v2``. The
  content hash in every run record is what makes a violation of this detectable rather than
  a slow, silent drift in what a number means.
* **The seed list is part of the task.** Dev tasks publish their seeds. Sealed tasks publish
  only a count and derive seeds from a salt the maintainers hold, so the ensemble is
  reproducible by whoever has the salt and unguessable by everyone else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from numpy.typing import NDArray

from .controller import TaskBrief
from .instruments import ActuatorConfig, InstrumentConfig, InstrumentedPlant, SensorConfig
from .plant import Plant
from .plants import build_plant

TIERS = ("nominal", "mismatch", "blind")


@dataclass(frozen=True)
class SetpointSegment:
    """One entry in the setpoint schedule. ``values`` are ordered like ``Task.controlled``."""

    t: float
    values: tuple[float, ...]
    ramp: float = 0.0
    """Seconds to travel from the previous values to these. ``0`` is a step.

    Ramps matter: a plant that a controller handles on a step it may fail on a slow ramp
    (integral windup has time to build), and vice versa. Tasks should contain both."""


@dataclass(frozen=True)
class SetpointProfile:
    segments: tuple[SetpointSegment, ...]

    def at(self, t: float) -> NDArray[np.float64]:
        prev = self.segments[0]
        current = self.segments[0]
        for seg in self.segments:
            if seg.t <= t:
                prev, current = current, seg
            else:
                break
        target = np.asarray(current.values, dtype=float)
        if current.ramp > 0.0 and t < current.t + current.ramp:
            start = np.asarray(prev.values, dtype=float)
            frac = max(0.0, (t - current.t) / current.ramp)
            return start + frac * (target - start)
        return target


@dataclass(frozen=True)
class DisturbanceEvent:
    """A parameter change applied at ``t``. This is how the plant is kicked mid-run."""

    t: float
    params: Mapping[str, float]


@dataclass(frozen=True)
class Budget:
    step_seconds: float = 0.05
    """Wall-clock a controller gets per control period before it counts as a scan overrun."""

    max_overruns: int = 20
    """Overruns tolerated before the run is failed outright."""


@dataclass(frozen=True)
class Scoring:
    w_error: float = 1.0
    w_effort: float = 0.5
    """Weight on actuator travel. Raised from an initial 0.1 after the first full field:
    at 0.1 the effort term was ~5% of cost, and a submission moving the valve 24x more than
    the reference still scored +0.507."""

    cvar_alpha: float = 0.10

    max_total_variation: float | None = None
    """Actuator duty ceiling, in the same normalized units as ``Cost.tv``. Exceeding it
    gates the scenario. Published per task; set it a few multiples above a well-tuned
    controller so it catches chatter without punishing necessary control action."""


@dataclass(frozen=True)
class Task:
    task_id: str
    tier: str
    plant_config: Mapping[str, Any]
    horizon: float
    controlled: tuple[int, ...]
    setpoints: SetpointProfile
    seeds: tuple[int, ...]
    instruments: InstrumentConfig = field(default_factory=InstrumentConfig)
    disturbances: tuple[DisturbanceEvent, ...] = ()
    scoring: Scoring = field(default_factory=Scoring)
    budget: Budget = field(default_factory=Budget)
    reference: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""
    content_hash: str = ""

    # -- construction ------------------------------------------------------------

    @staticmethod
    def load(path: str | Path) -> "Task":
        """Load a task, merging its reference anchor from ``tasks/references/<name>``.

        The anchor lives in a sibling file rather than in the task itself. It is still
        published — an anchor nobody can argue with is worse than a weak one — but it is not
        in the file a competitor is handed, because a submission with filesystem access will
        read that file. One already did: an agent lifted `kp` and `ti` out of a task and
        reported them as its own tuning.

        This is a speed bump, not a wall (an absolute path still reads anything), and it is
        the cheap half of the fix. The other half is that the scored tasks should not be on
        the scoring machine at all — see the sealed test set in ROADMAP.md M3.
        """
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        sidecar = path.parent / "references" / path.name
        if sidecar.is_file():
            side = yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {}
            if "reference" in side:
                if raw.get("reference"):
                    raise ValueError(
                        f"{path.name} defines a reference AND has one in references/. "
                        "Two anchors for one task is ambiguous; delete the inline one."
                    )
                raw["reference"] = side["reference"]
        return Task.from_dict(raw)

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> "Task":
        tier = str(raw.get("tier", "blind"))
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")

        segments = tuple(
            SetpointSegment(
                t=float(s["t"]),
                values=tuple(float(v) for v in s["values"]),
                ramp=float(s.get("ramp", 0.0)),
            )
            for s in raw["setpoints"]
        )
        if not segments or segments[0].t > 0.0:
            raise ValueError("setpoints must include a segment at t=0")

        inst = raw.get("instruments", {}) or {}
        instruments = InstrumentConfig(
            sensors=SensorConfig(**(inst.get("sensors", {}) or {})),
            actuators=ActuatorConfig(**(inst.get("actuators", {}) or {})),
        )

        task = Task(
            task_id=str(raw["task_id"]),
            tier=tier,
            plant_config=dict(raw["plant"]),
            horizon=float(raw["horizon"]),
            controlled=tuple(int(i) for i in raw["controlled"]),
            setpoints=SetpointProfile(segments),
            seeds=_resolve_seeds(raw.get("scenarios", {}) or {}),
            instruments=instruments,
            disturbances=tuple(
                DisturbanceEvent(t=float(d["t"]), params=dict(d["params"]))
                for d in (raw.get("disturbances", []) or [])
            ),
            scoring=Scoring(**(raw.get("scoring", {}) or {})),
            budget=Budget(**(raw.get("budget", {}) or {})),
            reference=dict(raw.get("reference", {}) or {}),
            description=str(raw.get("description", "")),
        )
        return dataclass_with_hash(task, raw)

    # -- use ---------------------------------------------------------------------

    @property
    def n_steps(self) -> int:
        return int(round(self.horizon / self.sample_time))

    @property
    def sample_time(self) -> float:
        return float(self.plant_config.get("sample_time", 1.0))

    def build_plant(self) -> Plant:
        """A fresh plant for one scenario, wrapped in this task's instrument layer."""
        cfg = dict(self.plant_config)
        if self.tier == "nominal":
            # The nominal tier is the honest tuning exercise: the controller is handed the
            # true model, so there must not be a hidden parameter draw to be robust to.
            cfg.pop("mismatch", None)
        return InstrumentedPlant(build_plant(cfg), self.instruments)

    def brief(self, plant: Plant) -> TaskBrief:
        """The operating manual handed to a controller, redacted for this task's tier."""
        spec = plant.spec
        hint: dict[str, Any] = {}
        if self.tier != "blind":
            inner = getattr(plant, "_inner", plant)
            source = getattr(inner, "nominal_params", None)
            if source is not None:
                hint = {"params": _as_plain(source()), "note": "nominal, not as-running"}
                if self.tier == "nominal":
                    hint = {"params": _as_plain(getattr(inner, "params")), "note": "exact"}

        return TaskBrief(
            task_id=self.task_id,
            tier=self.tier,
            measurements=spec.measurements,
            actuators=spec.actuators,
            controlled=self.controlled,
            sample_time=spec.sample_time,
            horizon=self.horizon,
            initial_u=tuple(float(v) for v in spec.initial_actuation()),
            constraints=spec.constraints,
            description=self.description or spec.description,
            model_hint=hint,
        )

    def setpoint_vector(self, t: float, n_y: int) -> NDArray[np.float64]:
        """Full-width setpoint vector; ``nan`` for channels that carry no setpoint."""
        r = np.full(n_y, np.nan, dtype=float)
        r[list(self.controlled)] = self.setpoints.at(t)
        return r


def _resolve_seeds(scenarios: Mapping[str, Any]) -> tuple[int, ...]:
    """Explicit seeds for a dev task, salt-derived seeds for a sealed one."""
    if "seeds" in scenarios:
        return tuple(int(s) for s in scenarios["seeds"])
    count = int(scenarios.get("count", 20))
    salt = scenarios.get("salt")
    if salt is None:
        raise ValueError(
            "scenarios needs either an explicit 'seeds' list (dev task) or a 'salt' "
            "(sealed task). Refusing to invent an ensemble."
        )
    return tuple(
        int.from_bytes(hashlib.sha256(f"{salt}:{i}".encode()).digest()[:4], "big") & 0x7FFFFFFF
        for i in range(count)
    )


def _as_plain(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        return {f: getattr(obj, f) for f in obj.__dataclass_fields__}
    return dict(obj)


def dataclass_with_hash(task: "Task", raw: Mapping[str, Any]) -> "Task":
    """Stamp the task with a content hash of its source document.

    Hashing the *source* rather than the parsed object is deliberate: it is the thing a
    maintainer edits and a reviewer diffs, so it is the thing whose identity a run record
    should pin.
    """
    canonical = json.dumps(raw, sort_keys=True, default=str).encode()
    digest = hashlib.sha256(canonical).hexdigest()[:16]
    return Task(**{**task.__dict__, "content_hash": digest})


def load_tasks(paths: Sequence[str | Path]) -> list[Task]:
    return [Task.load(p) for p in paths]

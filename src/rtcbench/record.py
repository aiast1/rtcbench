"""Run records — the unit of evidence.

Every scenario produces one of these, and it carries enough to (a) redraw the trend a
control engineer would ask to see, (b) recompute every metric from scratch without trusting
the harness that produced it, and (c) replay the run bit-for-bit.

(b) is the one that matters for a public leaderboard. A score you cannot recompute from the
record is a score the maintainers are asking you to take on faith.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .metrics import Cost

SCHEMA = "rtcbench.record/1"


@dataclass
class Trace:
    """The time series of one scenario. Rows are control periods."""

    t: NDArray[np.float64]
    y: NDArray[np.float64]
    """Measured, as the controller saw them — post-noise, post-deadtime, post-quantization."""

    r: NDArray[np.float64]
    u_commanded: NDArray[np.float64]
    u_actual: NDArray[np.float64]
    """Where the valves actually went. Plotted against ``u_commanded``, this is how
    stiction shows itself."""

    x: NDArray[np.float64]
    """True state. Recorded for plots and for auditing the safety gate; never given to a
    controller during the run."""

    quality: NDArray[np.bool_]
    violations: NDArray[np.bool_]
    overrun: NDArray[np.bool_]

    def to_json(self) -> dict[str, Any]:
        return {
            "t": self.t.tolist(),
            "y": self.y.tolist(),
            "r": np.where(np.isnan(self.r), None, self.r).tolist(),
            "u_commanded": self.u_commanded.tolist(),
            "u_actual": self.u_actual.tolist(),
            "x": self.x.tolist(),
            "quality": self.quality.tolist(),
            "violations": self.violations.tolist(),
            "overrun": self.overrun.tolist(),
        }

    @staticmethod
    def from_json(d: Mapping[str, Any]) -> "Trace":
        def arr(key: str, dtype: Any = float) -> NDArray[Any]:
            raw = [[np.nan if v is None else v for v in row] if isinstance(row, list) else row
                   for row in d[key]]
            return np.asarray(raw, dtype=dtype)

        return Trace(
            t=np.asarray(d["t"], dtype=float),
            y=arr("y"),
            r=arr("r"),
            u_commanded=arr("u_commanded"),
            u_actual=arr("u_actual"),
            x=arr("x"),
            quality=arr("quality", bool),
            violations=np.asarray(d["violations"], dtype=bool),
            overrun=np.asarray(d["overrun"], dtype=bool),
        )


@dataclass
class RunRecord:
    """One scenario, fully described."""

    task_id: str
    task_hash: str
    tier: str
    seed: int
    controller_id: str
    plant_id: str
    sample_time: float
    controlled: tuple[int, ...]
    measurement_tags: tuple[str, ...]
    actuator_tags: tuple[str, ...]
    cost: Cost
    trace: Trace
    failed: bool = False
    failure: str = ""
    """Populated when the controller raised, or blew past ``max_overruns``. A failed run is
    reported as a failure, never silently dropped from an ensemble — dropping the runs a
    controller crashed on would score it on the subset it happened to survive."""

    meta: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA

    def to_json(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k not in ("trace", "cost")}
        d["cost"] = asdict(self.cost)
        d["trace"] = self.trace.to_json()
        d["controlled"] = list(self.controlled)
        d["measurement_tags"] = list(self.measurement_tags)
        d["actuator_tags"] = list(self.actuator_tags)
        return d

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return p

    @staticmethod
    def load(path: str | Path) -> "RunRecord":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("schema") != SCHEMA:
            raise ValueError(f"expected {SCHEMA}, got {d.get('schema')!r}")
        return RunRecord(
            task_id=d["task_id"],
            task_hash=d["task_hash"],
            tier=d["tier"],
            seed=int(d["seed"]),
            controller_id=d["controller_id"],
            plant_id=d["plant_id"],
            sample_time=float(d["sample_time"]),
            controlled=tuple(d["controlled"]),
            measurement_tags=tuple(d["measurement_tags"]),
            actuator_tags=tuple(d["actuator_tags"]),
            cost=Cost(**d["cost"]),
            trace=Trace.from_json(d["trace"]),
            failed=bool(d.get("failed", False)),
            failure=d.get("failure", ""),
            meta=d.get("meta", {}),
        )

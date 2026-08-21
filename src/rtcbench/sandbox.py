"""Run a submitted controller in a separate process, at arm's length.

**Read this before trusting it.** A pure-Python sandbox is a speed bump, not a security
boundary. It stops a submission from *casually* reaching things it should not — importing the
plant, opening a socket, running forever — and it does not stop a determined attacker, who
has `ctypes` and a hundred other doors. Real isolation is a container or a VM. What this
buys is that an ordinary submission cannot cheat by accident or by obvious intent, and that a
runaway one cannot hang the harness. Treat it as the honest-mistake barrier it is, and run
untrusted code in a container regardless.

What it does enforce:

* **The plant is unreachable.** The child never imports `rtcbench` at all. The brief crosses
  the process boundary as plain data and is rebuilt on the far side as a look-alike, so there
  is no path from the controller to the physics — not even a transitive one. This matters:
  `rtcbench/__init__` pulls in `task`, which pulls in every plant module, so a child that
  imported the package for its dataclasses would have the answers sitting in `sys.modules`.
* **A step budget enforced from outside.** The parent waits `step_seconds` for a reply; past
  that it is a scan overrun, exactly as a DCS treats a block that runs long — the previous
  output is held. The child is not killed for one overrun, because a controller that is
  occasionally slow is a real thing; it is killed once it exceeds `kill_factor` times the
  budget, at which point it is not slow, it is stuck.
* **No network on the obvious path.** `socket` is neutered in the child.
* **A scratch working directory.** The child is chdir'd away from the repository root, so a
  *relative* path stops reaching the task files. The cheat this was built for was literally
  ``Path("tasks/four_tank_v1.yaml").read_text()``, and that now raises.

What it does NOT enforce — stated plainly, because a half-understood sandbox is more
dangerous than a known-absent one:

* **Filesystem confinement.** Python cannot do it portably. An *absolute* path still reads
  anything the harness user can read, task files included. Until the scored tasks are simply
  absent from the scoring machine (the sealed test set) or the run happens inside a
  container, a determined submission can still find the answer key. Verified, not assumed:
  see ``tests/test_sandbox.py::test_an_absolute_path_can_still_reach_the_task_file``.
* ``ctypes``, ``subprocess``, and every other escape hatch in the standard library.
* Memory or CPU limits beyond the per-step wall clock.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .controller import TaskBrief

DEFAULT_KILL_FACTOR = 20.0
"""Multiple of the step budget after which a silent child is presumed hung, not slow."""

BLOCKED_PREFIXES = ("rtcbench",)
"""Import prefixes refused inside the child. `rtcbench` in full: importing the package for
its dataclasses would drag in every plant module as a side effect."""


class ScanOverrun(RuntimeError):
    """The controller did not answer within its step budget.

    Not a failure on its own — the harness holds the previous output and logs it, the way a
    DCS does. Enough of them fails the run.
    """


class SandboxError(RuntimeError):
    """The child process died, or never started."""


# --------------------------------------------------------------------------------
# Child side. Nothing here may import rtcbench.
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Chan:
    """Stand-in for `rtcbench.plant.Channel`, rebuilt from plain data in the child."""

    tag: str
    name: str
    unit: str
    lo: float
    hi: float

    @property
    def span(self) -> float:
        return self.hi - self.lo


@dataclass(frozen=True)
class _Cons:
    name: str
    signal: str
    lo: float
    hi: float


class _Brief:
    """Attribute-compatible with :class:`~rtcbench.controller.TaskBrief`.

    Controllers only ever read attributes off the brief, so a look-alike is indistinguishable
    from the real thing — and it lets the child stay free of any rtcbench import.
    """

    def __init__(self, d: Mapping[str, Any]) -> None:
        self.task_id = d["task_id"]
        self.tier = d["tier"]
        self.measurements = tuple(_Chan(**c) for c in d["measurements"])
        self.actuators = tuple(_Chan(**c) for c in d["actuators"])
        self.controlled = tuple(d["controlled"])
        self.sample_time = d["sample_time"]
        self.horizon = d["horizon"]
        self.initial_u = tuple(d["initial_u"])
        self.constraints = tuple(_Cons(**c) for c in d["constraints"])
        self.description = d["description"]
        self.model_hint = d["model_hint"]

    @property
    def n_y(self) -> int:
        return len(self.measurements)

    @property
    def n_u(self) -> int:
        return len(self.actuators)

    def tags(self) -> dict[str, int]:
        return {c.tag: i for i, c in enumerate(self.measurements)}


def _harden() -> None:
    """Refuse the imports and the network access a submission has no business wanting."""
    import builtins
    import os
    import tempfile

    # Move off the repository root so a RELATIVE path stops resolving to it. The observed
    # cheat was literally `Path("tasks/four_tank_v1.yaml").read_text()`; after this it
    # raises. An ABSOLUTE path still works -- Python cannot confine a filesystem portably --
    # so this narrows the attack, it does not close it. See the module docstring.
    try:
        os.chdir(tempfile.mkdtemp(prefix="rtcb_sub_"))
    except Exception:
        pass

    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root in BLOCKED_PREFIXES:
            raise ImportError(
                f"'{name}' is not importable from a sandboxed controller. The plant, the "
                "task file and the harness are all off limits; everything you are entitled "
                "to is on the brief."
            )
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded

    try:
        import socket

        def _no_network(*a, **k):
            raise OSError("network access is disabled inside the sandbox")

        socket.socket = _no_network  # type: ignore[assignment]
        socket.create_connection = _no_network  # type: ignore[assignment]
    except Exception:
        pass


def _load(module_path: str, kwargs: Mapping[str, Any], brief: _Brief):
    import importlib.util

    path = Path(module_path)
    spec = importlib.util.spec_from_file_location(f"rtcb_sub_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cls = getattr(module, "Controller", None)
    if cls is None:
        raise AttributeError(
            f"{path} defines no 'Controller' class. A submission must expose one."
        )
    return cls(brief, **dict(kwargs))


def _worker(conn, module_path: str, brief_payload: Mapping[str, Any],
            kwargs: Mapping[str, Any]) -> None:
    """Child entry point. Loads the submission, then answers steps until told to stop."""
    try:
        brief = _Brief(brief_payload)
        # Harden BEFORE the submission's module body runs -- it executes arbitrary code at
        # import time, so blocking imports afterwards would be too late.
        _harden()
        controller = _load(module_path, kwargs, brief)
        conn.send(("ok", None))
    except BaseException:
        # BaseException, not Exception: a submission that calls sys.exit() or raises
        # KeyboardInterrupt would otherwise kill the child without a word, and the parent
        # would surface a bare EOFError instead of saying what happened.
        try:
            conn.send(("construct_failed", traceback.format_exc(limit=6)))
        except Exception:
            pass
        return

    n_u = brief.n_u
    while True:
        try:
            msg = conn.recv()
        except (EOFError, KeyboardInterrupt):
            return
        kind = msg[0]
        if kind == "close":
            return
        try:
            if kind == "reset":
                controller.reset()
                conn.send(("ok", None))
            elif kind == "step":
                _, t, y, r, q = msg
                u = controller.step(
                    float(t), np.asarray(y, dtype=float), np.asarray(r, dtype=float),
                    np.asarray(q, dtype=bool),
                )
                arr = np.asarray(u, dtype=float).reshape(-1)
                if arr.shape != (n_u,) or not np.all(np.isfinite(arr)):
                    raise ValueError(
                        f"controller returned {u!r}, expected {n_u} finite values"
                    )
                conn.send(("ok", arr.tolist()))
            else:
                conn.send(("raised", f"unknown message {kind!r}"))
        except BaseException:
            try:
                conn.send(("raised", traceback.format_exc(limit=6)))
            except Exception:
                return


# --------------------------------------------------------------------------------
# Parent side
# --------------------------------------------------------------------------------


def _brief_payload(brief: TaskBrief) -> dict[str, Any]:
    """Flatten a brief to plain data. Anything not representable here is, by construction,
    something the controller was never entitled to see."""
    def chan(c):
        return {"tag": c.tag, "name": c.name, "unit": c.unit,
                "lo": float(c.lo), "hi": float(c.hi)}

    return {
        "task_id": brief.task_id,
        "tier": brief.tier,
        "measurements": [chan(c) for c in brief.measurements],
        "actuators": [chan(c) for c in brief.actuators],
        "controlled": list(brief.controlled),
        "sample_time": float(brief.sample_time),
        "horizon": float(brief.horizon),
        "initial_u": [float(v) for v in brief.initial_u],
        "constraints": [{"name": c.name, "signal": c.signal,
                         "lo": float(c.lo), "hi": float(c.hi)} for c in brief.constraints],
        "description": brief.description,
        "model_hint": dict(brief.model_hint),
    }


class SandboxedController:
    """Proxy for a controller running in its own process. Satisfies the Controller protocol."""

    def __init__(
        self,
        brief: TaskBrief,
        module_path: str,
        *,
        step_seconds: float = 0.05,
        startup_seconds: float = 30.0,
        kill_factor: float = DEFAULT_KILL_FACTOR,
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self._n_u = brief.n_u
        self._step_budget = float(step_seconds)
        self._kill_after = float(step_seconds) * float(kill_factor)
        self._last_u = np.asarray(brief.initial_u, dtype=float)
        if self._last_u.size != self._n_u:
            self._last_u = np.zeros(self._n_u)

        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(
            target=_worker,
            args=(child, str(module_path), _brief_payload(brief), dict(kwargs or {})),
            daemon=True,
        )
        self._proc.start()
        child.close()

        if not self._conn.poll(startup_seconds):
            self.close()
            raise SandboxError(
                f"controller did not finish loading within {startup_seconds:g}s"
            )
        try:
            status, payload = self._conn.recv()
        except (EOFError, OSError) as exc:
            self.close()
            raise SandboxError(
                "controller process died while loading, without reporting why "
                f"(exit code {self._proc.exitcode})"
            ) from exc
        if status != "ok":
            self.close()
            raise SandboxError(f"controller construction failed:\n{payload}")

    # -- protocol ----------------------------------------------------------------

    def reset(self) -> None:
        self._send(("reset",), self._step_budget * 10)

    def step(
        self,
        t: float,
        y: NDArray[np.float64],
        r: NDArray[np.float64],
        quality: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        # nan does not survive a JSON-ish round trip cleanly in every transport, but the Pipe
        # pickles, so unscored setpoints keep their nan and the controller sees what the
        # in-process path shows it.
        payload = ("step", float(t), y.tolist(), r.tolist(), quality.tolist())
        u = self._send(payload, self._step_budget)
        self._last_u = np.asarray(u, dtype=float)
        return self._last_u.copy()

    def close(self) -> None:
        try:
            if self._proc.is_alive():
                try:
                    self._conn.send(("close",))
                except Exception:
                    pass
                self._proc.join(timeout=1.0)
        finally:
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=1.0)
            try:
                self._conn.close()
            except Exception:
                pass

    # -- transport ---------------------------------------------------------------

    def _send(self, payload: Sequence[Any], budget: float):
        if not self._proc.is_alive():
            raise SandboxError("controller process is gone")
        try:
            self._conn.send(tuple(payload))
        except Exception as exc:
            raise SandboxError(f"could not reach the controller: {exc}") from exc

        if not self._conn.poll(budget):
            # Late, not necessarily lost. Give it until the kill threshold and DRAIN the
            # answer when it comes: skipping it would leave the pipe one message out of
            # step, and every later reply would belong to the previous request.
            remaining = max(0.0, self._kill_after - budget)
            if self._conn.poll(remaining):
                try:
                    self._conn.recv()
                except Exception:
                    pass
                raise ScanOverrun("controller exceeded its step budget")
            self.close()
            raise SandboxError(
                f"controller stopped responding (silent for {self._kill_after:g}s, "
                f"{self._kill_after / max(self._step_budget, 1e-9):.0f}x its step budget)"
            )

        try:
            status, value = self._conn.recv()
        except (EOFError, OSError) as exc:
            # A child that died without unwinding (os._exit, a segfault in a C extension)
            # leaves the pipe readable but empty. Surfacing the raw EOFError would put a
            # multiprocessing traceback in the run record instead of saying what happened.
            code = self._proc.exitcode
            self.close()
            raise SandboxError(
                f"controller process died mid-run without reporting why (exit code {code})"
            ) from exc
        if status == "ok":
            return value
        raise SandboxError(f"controller raised:\n{value}")


def sandboxed_factory(
    module_path: str,
    *,
    step_seconds: float = 0.05,
    kill_factor: float = DEFAULT_KILL_FACTOR,
    kwargs: Mapping[str, Any] | None = None,
):
    """Return a `factory(brief) -> SandboxedController` for the harness."""

    def build(brief: TaskBrief) -> SandboxedController:
        return SandboxedController(
            brief, module_path, step_seconds=step_seconds,
            kill_factor=kill_factor, kwargs=kwargs,
        )

    return build

"""Loading a controller — from a builtin, or from a submitted file.

A submission is a Python module exposing a class named ``Controller``. That is the entire
interface. Loading it by path rather than by package install is deliberate: a submission
should be a directory you can zip, archive next to its results, and re-run in five years
without resolving a dependency graph that no longer exists.

By default the module is imported in-process, which runs whatever it contains with the
harness's own privileges — fine for your own controllers and for the reference anchors, and
not fine for a submission from a stranger. Pass ``sandbox=True`` to load it behind
:mod:`rtcbench.sandbox` instead: a separate process that cannot import the plant, cannot
open a socket, and has its step budget enforced from outside.

Read the caveat at the top of :mod:`rtcbench.sandbox` before relying on that. It is an
honest-mistake barrier, not a security boundary; genuinely untrusted code belongs in a
container either way.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

from .baselines import BUILTINS
from .controller import TaskBrief
from .sandbox import DEFAULT_KILL_FACTOR, sandboxed_factory


def load_controller(
    spec: str,
    *,
    sandbox: bool = False,
    step_seconds: float = 0.05,
    kill_factor: float = DEFAULT_KILL_FACTOR,
    **kwargs: Any,
) -> tuple[Callable[[TaskBrief], Any], str]:
    """Resolve a controller spec into a ``(factory, controller_id)`` pair.

    ``spec`` is either ``builtin:<name>`` or a path to a ``.py`` file exposing ``Controller``.
    ``sandbox=True`` applies only to file submissions: the builtins are this project's own
    anchors, and isolating them would cost a subprocess per scenario to protect against code
    that ships in the same repository as the harness.
    """
    if spec.startswith("builtin:"):
        name = spec.split(":", 1)[1]
        if name not in BUILTINS:
            raise ValueError(f"unknown builtin {name!r}; have {sorted(BUILTINS)}")
        cls = BUILTINS[name]
        return (lambda brief: cls(brief, **kwargs)), spec

    path = Path(spec).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no controller at {path}")

    if sandbox:
        return (
            sandboxed_factory(str(path), step_seconds=step_seconds,
                              kill_factor=kill_factor, kwargs=kwargs),
            f"{path} [sandboxed]",
        )

    module_name = f"rtcbench_submission_{path.stem}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)

    cls = getattr(module, "Controller", None)
    if cls is None:
        raise AttributeError(
            f"{path} defines no 'Controller' class. A submission must expose one — see "
            "rtcbench.controller.Controller for the protocol."
        )
    return (lambda brief: cls(brief, **kwargs)), str(path)

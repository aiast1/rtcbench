"""Loading a controller — from a builtin, or from a submitted file.

A submission is a Python module exposing a class named ``Controller``. That is the entire
interface. Loading it by path rather than by package install is deliberate: a submission
should be a directory you can zip, archive next to its results, and re-run in five years
without resolving a dependency graph that no longer exists.

**This module does not sandbox anything yet.** ``import`` runs whatever the file contains,
with the harness's own privileges. That is fine for running your own controllers and for
the reference anchors, and it is *not* fine for accepting submissions from strangers. The
subprocess isolation described in DESIGN.md §6 is v0.2 work; until it lands, the honest
statement is that this loader trusts the code it is given.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

from .baselines import BUILTINS
from .controller import TaskBrief


def load_controller(spec: str, **kwargs: Any) -> tuple[Callable[[TaskBrief], Any], str]:
    """Resolve a controller spec into a ``(factory, controller_id)`` pair.

    ``spec`` is either ``builtin:<name>`` or a path to a ``.py`` file exposing ``Controller``.
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

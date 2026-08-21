"""Tests for the sandboxed submission loader.

These assert the things the sandbox *claims*, one test per claim, because a sandbox whose
guarantees are not individually pinned is a sandbox nobody should trust. The claims are
deliberately modest — see the caveat at the top of `rtcbench.sandbox` — and the tests are
scoped to exactly them, not to a stronger boundary the module does not provide.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from rtcbench.harness import run_ensemble, run_scenario
from rtcbench.sandbox import SandboxError, sandboxed_factory
from rtcbench.submission import load_controller
from rtcbench.task import Task

TASK = Path(__file__).resolve().parents[1] / "tasks" / "four_tank_v1.yaml"


@pytest.fixture(scope="module")
def task() -> Task:
    return Task.load(TASK)


def write(tmp_path: Path, body: str, name: str = "controller.py") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


GOOD = """
    import numpy as np

    class Controller:
        def __init__(self, brief):
            self.u = np.asarray(brief.initial_u, dtype=float)
            self.n = brief.n_u
            self.tags = brief.tags()
        def reset(self):
            pass
        def step(self, t, y, r, quality):
            return self.u
    """


# -- it works at all --------------------------------------------------------------


def test_a_sandboxed_controller_runs_a_whole_scenario(tmp_path, task: Task):
    path = write(tmp_path, GOOD)
    factory, cid = load_controller(str(path), sandbox=True,
                                   step_seconds=task.budget.step_seconds)
    rec = run_scenario(task, factory, task.seeds[0], controller_id=cid)
    assert not rec.failed, rec.failure
    assert rec.meta["steps"] == rec.meta["planned_steps"]
    assert "[sandboxed]" in cid


def test_sandboxed_and_in_process_give_identical_results(tmp_path, task: Task):
    """The boundary must not perturb the measurement. If crossing a process changed a
    trajectory, every sandboxed score would be incomparable with every in-process one."""
    path = write(tmp_path, GOOD)
    plain, _ = load_controller(str(path))
    boxed, _ = load_controller(str(path), sandbox=True,
                               step_seconds=task.budget.step_seconds)
    a = run_scenario(task, plain, task.seeds[0])
    b = run_scenario(task, boxed, task.seeds[0])
    np.testing.assert_array_equal(a.trace.u_commanded, b.trace.u_commanded)
    np.testing.assert_array_equal(a.trace.y, b.trace.y)
    assert a.cost.total == pytest.approx(b.cost.total, rel=1e-12)


def test_the_brief_survives_the_boundary_intact(tmp_path, task: Task):
    """The child rebuilds the brief from plain data, so every field a controller might read
    has to arrive — including the derived ones."""
    path = write(tmp_path, """
        import numpy as np

        class Controller:
            def __init__(self, brief):
                assert brief.task_id == "four_tank_v1"
                assert brief.tier == "mismatch"
                assert brief.n_y == 2 and brief.n_u == 2
                assert brief.measurements[0].tag == "LT-101"
                assert brief.measurements[0].unit == "cm"
                assert brief.measurements[0].span == 20.0      # derived property
                assert brief.tags()["LT-102"] == 1
                assert brief.controlled == (0, 1)
                assert brief.sample_time == 2.0
                assert len(brief.constraints) == 4
                assert brief.constraints[0].signal == "h1"
                assert brief.model_hint["note"].startswith("nominal")
                assert "tank" in brief.description
                self.u = np.asarray(brief.initial_u, dtype=float)
            def reset(self): pass
            def step(self, t, y, r, quality): return self.u
        """)
    factory, _ = load_controller(str(path), sandbox=True)
    rec = run_scenario(task, factory, task.seeds[0])
    assert not rec.failed, rec.failure


def test_unscored_setpoints_still_arrive_as_nan(tmp_path, task: Task):
    path = write(tmp_path, """
        import numpy as np

        class Controller:
            def __init__(self, brief):
                self.u = np.asarray(brief.initial_u, dtype=float)
                self.controlled = set(brief.controlled)
                self.n_y = brief.n_y
            def reset(self): pass
            def step(self, t, y, r, quality):
                for i in range(self.n_y):
                    if i not in self.controlled:
                        assert np.isnan(r[i]), "unscored setpoint should be nan"
                return self.u
        """)
    factory, _ = load_controller(str(path), sandbox=True)
    assert not run_scenario(task, factory, task.seeds[0]).failed


# -- the claims -------------------------------------------------------------------


@pytest.mark.parametrize("statement", [
    "import rtcbench",
    "import rtcbench.plants",
    "from rtcbench.plants.four_tank import FourTank",
    "from rtcbench.task import Task",
    "__import__('rtcbench')",
])
def test_the_plant_is_unreachable(tmp_path, task: Task, statement: str):
    """Every route to the physics is refused, including the transitive one.

    `import rtcbench` matters most: the package __init__ pulls in `task`, which pulls in
    every plant module, so a child allowed to import the package for its dataclasses would
    have the answers sitting in sys.modules.
    """
    path = write(tmp_path, f"""
        import numpy as np

        class Controller:
            def __init__(self, brief):
                {statement}
                self.u = np.zeros(brief.n_u)
            def reset(self): pass
            def step(self, t, y, r, quality): return self.u
        """)
    factory, _ = load_controller(str(path), sandbox=True)
    with pytest.raises(SandboxError, match="not importable|construction failed"):
        factory(task.brief(task.build_plant()))


def test_the_block_is_installed_before_the_module_body_runs(tmp_path, task: Task):
    """A submission executes arbitrary code at import time, so hardening afterwards would
    be too late to matter."""
    path = write(tmp_path, """
        import rtcbench.plants          # module level, not inside __init__
        import numpy as np

        class Controller:
            def __init__(self, brief): self.u = np.zeros(brief.n_u)
            def reset(self): pass
            def step(self, t, y, r, quality): return self.u
        """)
    factory, _ = load_controller(str(path), sandbox=True)
    with pytest.raises(SandboxError, match="not importable|construction failed"):
        factory(task.brief(task.build_plant()))


def test_the_network_is_closed(tmp_path, task: Task):
    path = write(tmp_path, """
        import numpy as np, socket

        class Controller:
            def __init__(self, brief):
                socket.socket()
                self.u = np.zeros(brief.n_u)
            def reset(self): pass
            def step(self, t, y, r, quality): return self.u
        """)
    factory, _ = load_controller(str(path), sandbox=True)
    with pytest.raises(SandboxError, match="network access is disabled|construction failed"):
        factory(task.brief(task.build_plant()))


def test_a_slow_step_is_a_scan_overrun_not_a_crash(tmp_path, task: Task):
    """Budget enforced from outside. A late block holds its previous output, as a DCS does,
    and the actuators never take the value it eventually produced."""
    path = write(tmp_path, """
        import numpy as np, time

        class Controller:
            def __init__(self, brief):
                self.u = np.asarray(brief.initial_u, dtype=float)
            def reset(self): pass
            def step(self, t, y, r, quality):
                time.sleep(0.4)
                return self.u + 5.0      # a huge kick, if it were ever applied
        """)
    factory, _ = load_controller(str(path), sandbox=True,
                                 step_seconds=task.budget.step_seconds)
    rec = run_scenario(task, factory, task.seeds[0])
    assert rec.cost.overruns > 0
    np.testing.assert_allclose(rec.trace.u_commanded[0], [3.0, 3.0])


def test_a_wedged_controller_is_killed_rather_than_hanging_the_run(tmp_path, task: Task):
    path = write(tmp_path, """
        import numpy as np

        class Controller:
            def __init__(self, brief): self.u = np.asarray(brief.initial_u, dtype=float)
            def reset(self): pass
            def step(self, t, y, r, quality):
                while True:
                    pass
        """)
    factory, _ = load_controller(str(path), sandbox=True, step_seconds=0.05, kill_factor=4)
    rec = run_scenario(task, factory, task.seeds[0])
    assert rec.failed
    assert "stopped responding" in rec.failure


def test_a_controller_that_raises_mid_run_fails_its_scenario(tmp_path, task: Task):
    path = write(tmp_path, """
        import numpy as np

        class Controller:
            def __init__(self, brief):
                self.u = np.asarray(brief.initial_u, dtype=float)
                self.k = 0
            def reset(self): self.k = 0
            def step(self, t, y, r, quality):
                self.k += 1
                if self.k > 5:
                    raise RuntimeError("boom")
                return self.u
        """)
    factory, _ = load_controller(str(path), sandbox=True)
    rec = run_scenario(task, factory, task.seeds[0])
    assert rec.failed and "boom" in rec.failure
    assert rec.cost.violations > 0          # gated, not silently dropped


def test_a_wrong_shape_return_is_caught_in_the_child(tmp_path, task: Task):
    path = write(tmp_path, """
        import numpy as np

        class Controller:
            def __init__(self, brief): pass
            def reset(self): pass
            def step(self, t, y, r, quality): return np.array([1.0])
        """)
    factory, _ = load_controller(str(path), sandbox=True)
    rec = run_scenario(task, factory, task.seeds[0])
    assert rec.failed and "expected 2" in rec.failure


def test_a_module_with_no_controller_class_says_so(tmp_path, task: Task):
    path = write(tmp_path, "x = 1\n")
    factory, _ = load_controller(str(path), sandbox=True)
    with pytest.raises(SandboxError, match="no 'Controller' class|construction failed"):
        factory(task.brief(task.build_plant()))


# -- housekeeping -----------------------------------------------------------------


def test_an_ensemble_does_not_leak_processes(tmp_path, task: Task):
    """One subprocess per scenario, each closed. Leaking them would exhaust the machine
    over a 20-seed ensemble."""
    import multiprocessing as mp

    path = write(tmp_path, GOOD)
    factory, _ = load_controller(str(path), sandbox=True)
    before = len(mp.active_children())
    run_ensemble(task, factory, seeds=task.seeds[:3])
    assert len(mp.active_children()) <= before


def test_builtins_are_never_sandboxed(task: Task):
    """The anchors ship in this repository; isolating them would cost a subprocess per
    scenario to guard against our own code."""
    factory, cid = load_controller("builtin:hold", sandbox=True)
    assert "[sandboxed]" not in cid
    assert not run_scenario(task, factory, task.seeds[0]).failed


# -- the boundary's limits, pinned so they cannot be misremembered ------------------


def test_a_relative_path_no_longer_reaches_the_task_file(tmp_path, task: Task):
    """The observed cheat, verbatim, now fails.

    An agent with repository access read the reference gains out of the task file with a
    relative path and reported them as its own tuning. The child runs in a scratch directory,
    so that exact route is closed.
    """
    path = write(tmp_path, """
        import numpy as np, yaml, pathlib

        class Controller:
            def __init__(self, brief):
                cfg = yaml.safe_load(
                    pathlib.Path("tasks/four_tank_v1.yaml").read_text())
                raise SystemExit(f"READ IT: {cfg['reference']['kp']}")
            def reset(self): pass
            def step(self, t, y, r, quality): return np.zeros(2)
        """)
    factory, _ = load_controller(str(path), sandbox=True)
    with pytest.raises(SandboxError) as exc:
        factory(task.brief(task.build_plant()))
    assert "READ IT" not in str(exc.value), "the relative-path read succeeded"


def test_an_absolute_path_can_still_reach_the_task_file(tmp_path, task: Task):
    """A KNOWN, DELIBERATE HOLE — asserted so it cannot be quietly misremembered as closed.

    Python cannot confine a filesystem portably. This test documents the boundary's real
    edge; when the sealed test set or container isolation lands, it should start failing,
    and whoever fixes it should delete it with a note rather than discover the gap by
    accident.
    """
    absolute = str(TASK).replace("\\", "\\\\")
    path = write(tmp_path, f"""
        import numpy as np, yaml, pathlib

        class Controller:
            def __init__(self, brief):
                cfg = yaml.safe_load(pathlib.Path(r"{absolute}").read_text())
                raise SystemExit(f"READ IT: {{cfg['reference']['kp']}}")
            def reset(self): pass
            def step(self, t, y, r, quality): return np.zeros(2)
        """)
    factory, _ = load_controller(str(path), sandbox=True)
    with pytest.raises(SandboxError) as exc:
        factory(task.brief(task.build_plant()))
    assert "READ IT" in str(exc.value), (
        "an absolute path no longer reaches the task file -- if that is deliberate, delete "
        "this test and update the sandbox docstring, which currently promises the opposite"
    )


def test_a_submission_that_exits_reports_cleanly(tmp_path, task: Task):
    """SystemExit is a BaseException, so `except Exception` in the child would let it kill
    the process silently and the parent would surface a bare EOFError from the pipe."""
    path = write(tmp_path, """
        import sys
        class Controller:
            def __init__(self, brief): sys.exit("submission bailed out")
            def reset(self): pass
            def step(self, t, y, r, quality): pass
        """)
    factory, _ = load_controller(str(path), sandbox=True)
    with pytest.raises(SandboxError, match="submission bailed out|construction failed"):
        factory(task.brief(task.build_plant()))


def test_a_child_killed_mid_run_becomes_a_failed_scenario(tmp_path, task: Task):
    path = write(tmp_path, """
        import numpy as np, os

        class Controller:
            def __init__(self, brief):
                self.u = np.asarray(brief.initial_u, dtype=float)
                self.k = 0
            def reset(self): self.k = 0
            def step(self, t, y, r, quality):
                self.k += 1
                if self.k > 3:
                    os._exit(1)          # dies without unwinding, no message sent
                return self.u
        """)
    factory, _ = load_controller(str(path), sandbox=True)
    rec = run_scenario(task, factory, task.seeds[0])
    assert rec.failed and "died mid-run" in rec.failure

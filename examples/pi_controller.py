"""A worked submission — copy this file as the starting point for your own.

The contract is small enough to state in full: expose a class named ``Controller`` that
takes a :class:`~rtcbench.controller.TaskBrief`, implements ``reset()``, and returns one
control vector per call to ``step()``. Nothing else is imported, inspected or required.

What this example is doing, and why each part earns its place:

* reads channels **by tag** rather than by position, so it does not silently mis-pair if a
  task reorders its measurement list;
* starts from ``brief.initial_u`` so putting the loop in service is bumpless;
* stops integrating into a saturated limit (anti-windup) — without this, one setpoint step
  large enough to hit the pump ceiling poisons the rest of the run;
* holds output on a bad reading instead of integrating against a stale value.

It does *not* do the interesting part. It never identifies the plant, so its gains are
guesses that happen to be reasonable, and it will lose badly on the non-minimum-phase task
where the diagonal pairing is wrong. Beating the reference means doing better than this.
"""

from __future__ import annotations

import numpy as np


class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.ts = brief.sample_time
        self.controlled = list(brief.controlled)

        self.lo = np.array([c.lo for c in brief.actuators], dtype=float)
        self.hi = np.array([c.hi for c in brief.actuators], dtype=float)
        self.bias = np.asarray(brief.initial_u, dtype=float)
        if self.bias.size != brief.n_u:
            self.bias = self.lo.copy()

        # Diagonal pairing: first controlled measurement to first actuator, and so on.
        self.pairing = list(range(len(self.controlled)))
        self.kp = [1.2] * len(self.controlled)
        self.ti = [35.0] * len(self.controlled)

        self.integral = np.zeros(len(self.controlled))

    def reset(self):
        self.integral[:] = 0.0
        self.u = self.bias.copy()

    def step(self, t, y, r, quality):
        u = self.u.copy()

        for n, ch in enumerate(self.controlled):
            setpoint = float(r[ch])
            if not np.isfinite(setpoint) or not bool(quality[ch]):
                continue

            j = self.pairing[n]
            error = setpoint - float(y[ch])
            candidate = self.bias[j] + self.kp[n] * error + self.integral[n]

            increment = self.kp[n] * (self.ts / self.ti[n]) * error
            pushing_into_limit = (candidate >= self.hi[j] and increment > 0) or (
                candidate <= self.lo[j] and increment < 0
            )
            if not pushing_into_limit:
                self.integral[n] += increment
                candidate += increment

            u[j] = float(np.clip(candidate, self.lo[j], self.hi[j]))

        self.u = u
        return u

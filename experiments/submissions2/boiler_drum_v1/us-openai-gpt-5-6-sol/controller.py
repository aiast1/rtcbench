import numpy as np


class Controller:
    """
    Robust three-element drum-level controller.

    The controller uses steam-flow feedforward, an inferred water-inventory
    state from the feedwater/steam mass balance, slow reconciliation to drum
    level, pressure-compensated valve inversion, and conservative PI feedback.
    """

    def __init__(self, brief):
        self.dt = float(brief.sample_time)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            self.dt = 5.0
        self.reset()

    def reset(self):
        self.u = 45.0
        self.first_step = True

        self.level = 0.0
        self.pressure = 85.0
        self.steam = 50.0
        self.feedwater = 50.0

        self.prev_pressure = 85.0
        self.prev_steam = 50.0
        self.inventory = 0.0
        self.integral = 0.0

        self.prev_r = 0.0
        self.have_r = False
        self.k_valve = 18.78

    @staticmethod
    def _valid(value, good, low, high):
        return bool(good) and np.isfinite(value) and low <= value <= high

    @staticmethod
    def _filter(old, new, dt, tau):
        alpha = dt / (tau + dt)
        return old + alpha * (new - old)

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float).reshape(-1)
        r = np.asarray(r, dtype=float).reshape(-1)
        quality = np.asarray(quality, dtype=bool).reshape(-1)

        dt = self.dt

        q0 = quality[0] if quality.size > 0 else False
        q1 = quality[1] if quality.size > 1 else False
        q2 = quality[2] if quality.size > 2 else False
        q3 = quality[3] if quality.size > 3 else False

        raw_level = self.level
        if y.size > 0 and self._valid(y[0], q0, -550.0, 550.0):
            raw_level = float(y[0])
            self.level = self._filter(self.level, raw_level, dt, 5.0)

        if y.size > 1 and self._valid(y[1], q1, 50.0, 125.0):
            self.pressure = self._filter(
                self.pressure, float(y[1]), dt, 15.0
            )

        if y.size > 2 and self._valid(y[2], q2, -2.0, 110.0):
            self.steam = self._filter(
                self.steam, float(y[2]), dt, 10.0
            )

        if y.size > 3 and self._valid(y[3], q3, -2.0, 110.0):
            self.feedwater = self._filter(
                self.feedwater, float(y[3]), dt, 8.0
            )

        target = 0.0
        if r.size > 0 and np.isfinite(r[0]):
            target = float(np.clip(r[0], -200.0, 200.0))
        elif self.have_r:
            target = self.prev_r

        if self.first_step:
            self.inventory = self.level
            self.prev_pressure = self.pressure
            self.prev_steam = self.steam
            self.prev_r = target
            self.have_r = True

            head = np.sqrt(max(120.0 - self.pressure, 4.0))
            if self.u > 5.0 and self.feedwater > 2.0:
                estimate = self.feedwater / ((self.u / 100.0) * head)
                if np.isfinite(estimate):
                    self.k_valve = float(np.clip(estimate, 11.0, 29.0))

            self.first_step = False
            return np.array([45.0], dtype=float)

        # Equivalent collapsed-water level from the measured mass balance.
        # Roughly 14.6 kg changes collapsed inventory by one millimetre.
        mass_per_mm = 14.6
        self.inventory += (
            dt * (self.feedwater - self.steam) / mass_per_mm
        )
        self.inventory = float(np.clip(self.inventory, -450.0, 450.0))

        dpdt = (self.pressure - self.prev_pressure) / dt
        dsdt = (self.steam - self.prev_steam) / dt
        self.prev_pressure = self.pressure
        self.prev_steam = self.steam

        # Correct flow-integrator bias only slowly. Reconciliation is fastest
        # when pressure, load, and mass flow are approximately settled.
        quiet = (
            abs(dpdt) < 0.012
            and abs(dsdt) < 0.035
            and abs(self.feedwater - self.steam) < 2.5
        )
        reconcile_tau = 800.0 if quiet else 3200.0
        reconcile = dt / (reconcile_tau + dt)
        self.inventory += reconcile * (self.level - self.inventory)

        # Learn the installed valve coefficient from valid operating data.
        head = np.sqrt(max(120.0 - self.pressure, 4.0))
        if q3 and q1 and self.u > 12.0 and self.feedwater > 5.0:
            estimate = self.feedwater / ((self.u / 100.0) * head)
            if np.isfinite(estimate) and 10.0 <= estimate <= 32.0:
                alpha_k = dt / (250.0 + dt)
                self.k_valve += alpha_k * (estimate - self.k_valve)
                self.k_valve = float(np.clip(self.k_valve, 11.0, 29.0))

        # Inventory dominates during shrink/swell; indicated level supplies
        # enough correction to ensure the requested steady indicated level.
        control_level = 0.78 * self.inventory + 0.22 * self.level
        error = target - control_level

        # Setpoint-rate feedforward is enabled only for genuine ramps, not
        # schedule steps.
        setpoint_ff = 0.0
        if self.have_r:
            delta_r = target - self.prev_r
            rate_r = delta_r / dt
            if abs(delta_r) <= 8.0 and abs(rate_r) <= 1.5:
                setpoint_ff = mass_per_mm * rate_r
        self.prev_r = target
        self.have_r = True

        kp = 0.080
        ki = 0.00016

        proposed_integral = self.integral + ki * error * dt
        proposed_integral = float(np.clip(proposed_integral, -14.0, 14.0))

        pressure_bias = 0.0
        if self.pressure < 75.0:
            pressure_bias -= min(8.0, 1.8 * (75.0 - self.pressure))
        elif self.pressure > 97.0:
            pressure_bias += min(6.0, 1.5 * (self.pressure - 97.0))

        desired_flow_raw = (
            self.steam
            + setpoint_ff
            + kp * error
            + proposed_integral
            + pressure_bias
        )

        # Protect inferred inventory. This override is applied after the
        # pressure bias because low-water protection has priority.
        if self.inventory < -120.0:
            inventory_makeup = min(
                16.0, 0.13 * (-120.0 - self.inventory)
            )
            desired_flow_raw = max(
                desired_flow_raw, self.steam + inventory_makeup
            )

        # A high indicated level accompanied by normal/low inventory is swell,
        # not excess water. Close the valve only when both measures are high.
        if self.level > 185.0 and self.inventory > 90.0:
            excess = min(
                14.0,
                0.08 * (self.level - 185.0)
                + 0.05 * (self.inventory - 90.0),
            )
            desired_flow_raw -= excess

        # Near carryover with a large swell discrepancy, maintaining a small
        # positive feed imbalance promotes shrink while protecting inventory.
        if raw_level > 225.0 and (raw_level - self.inventory) > 75.0:
            desired_flow_raw = max(desired_flow_raw, self.steam + 2.0)

        desired_flow = float(np.clip(desired_flow_raw, 1.0, 98.0))

        # Conditional integration / anti-windup at flow saturation.
        saturated_high = desired_flow_raw > 98.0
        saturated_low = desired_flow_raw < 1.0
        if not (
            (saturated_high and error > 0.0)
            or (saturated_low and error < 0.0)
        ):
            self.integral = proposed_integral
        else:
            self.integral *= max(0.0, 1.0 - dt / 500.0)

        # Pressure-compensated inverse valve characteristic.
        denominator = max(self.k_valve * head, 1.0)
        valve_target = 100.0 * desired_flow / denominator

        # Small measured-flow trim compensates residual valve/model error.
        flow_error = desired_flow - self.feedwater
        valve_target += 0.14 * flow_error
        valve_target = float(np.clip(valve_target, 0.0, 100.0))

        # Smooth command generation, rate limiting, and a command deadband
        # suppress noise-driven travel and stiction chatter.
        alpha_u = dt / (18.0 + dt)
        candidate = self.u + alpha_u * (valve_target - self.u)
        max_move = 0.85 * dt / 5.0
        candidate = float(
            np.clip(candidate, self.u - max_move, self.u + max_move)
        )
        candidate = float(np.clip(candidate, 0.0, 100.0))

        if abs(candidate - self.u) >= 0.10:
            self.u = candidate

        return np.array([self.u], dtype=float)
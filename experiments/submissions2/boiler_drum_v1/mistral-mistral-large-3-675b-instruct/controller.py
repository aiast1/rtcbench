import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.u_start = np.array([45.0], dtype=np.float64)  # Hardcoded initial actuator position

        # PID parameters for level control (LT-101)
        self.Kp = 0.1
        self.Ki = 0.005
        self.Kd = 0.3
        self.tau_d = 10.0  # derivative filter time constant

        # Anti-windup and bumpless transfer
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.prev_output = self.u_start[0]
        self.prev_measurement = 0.0

        # Feedforward from steam flow (FT-103)
        self.ff_gain = 0.4  # kg/s steam -> % valve

        # Safety limits
        self.l_ind_max = 250.0
        self.l_inv_min = -250.0
        self.p_min = 70.0
        self.p_max = 100.0

        # State for inventory estimation
        self.inventory_estimate = 0.0
        self.steam_flow_prev = 0.0
        self.feedwater_flow_prev = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.prev_output = self.u_start[0]
        self.prev_measurement = 0.0
        self.steam_flow_prev = 0.0
        self.feedwater_flow_prev = 0.0
        self.inventory_estimate = 0.0

    def step(self, t, y, r, quality):
        # Extract measurements
        l_ind = y[0]  # LT-101
        p = y[1]      # PT-102
        steam_flow = y[2] if quality[2] else self.steam_flow_prev  # FT-103
        feedwater_flow = y[3] if quality[3] else self.feedwater_flow_prev  # FT-104

        # Update inventory estimate (simplified mass balance)
        delta_steam = steam_flow - self.steam_flow_prev
        delta_fw = feedwater_flow - self.feedwater_flow_prev
        self.inventory_estimate += (delta_fw - delta_steam) * self.sample_time
        self.steam_flow_prev = steam_flow
        self.feedwater_flow_prev = feedwater_flow

        # Safety checks
        if (self.inventory_estimate < self.l_inv_min or
            l_ind > self.l_ind_max or
            p < self.p_min or
            p > self.p_max):
            # Emergency action: close valve if inventory is low or level high
            u = np.array([max(0.0, self.prev_output - 10.0)], dtype=np.float64)
            self.prev_output = u[0]
            return u

        # PID control for level (LT-101)
        setpoint = r[0]
        error = setpoint - l_ind

        # Proportional term
        proportional = self.Kp * error

        # Integral term with anti-windup
        self.integral += self.Ki * error * self.sample_time
        integral = self.integral

        # Derivative term with filtering
        derivative = (error - self.prev_error) / self.sample_time if quality[0] else 0.0
        filtered_derivative = (self.tau_d * self.prev_derivative + self.sample_time * derivative) / (self.tau_d + self.sample_time)
        derivative_term = self.Kd * filtered_derivative
        self.prev_derivative = filtered_derivative
        self.prev_error = error if quality[0] else self.prev_error

        # Feedforward from steam flow
        feedforward = self.ff_gain * steam_flow if quality[2] else 0.0

        # Compute output
        u = proportional + integral + derivative_term + feedforward

        # Clamp output to [0, 100]
        u = np.clip(u, 0.0, 100.0)

        # Anti-windup: clamp integral term if output is saturated
        if u >= 100.0 or u <= 0.0:
            self.integral = self.prev_output - proportional - derivative_term - feedforward
            self.integral = np.clip(self.integral, -100.0 / self.Ki, 100.0 / self.Ki)

        # Bumpless transfer
        u = 0.95 * self.prev_output + 0.05 * u
        self.prev_output = u

        return np.array([u], dtype=np.float64)
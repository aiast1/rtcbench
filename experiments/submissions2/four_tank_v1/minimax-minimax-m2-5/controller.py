import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        # Calibrate feedforward gains from initial steady state
        # Initial setpoint at t=0: [12.26, 12.78]
        # Initial actuator values: [3.0, 3.0]
        self.K1 = 3.0 / np.sqrt(12.26)
        self.K2 = 3.0 / np.sqrt(12.78)
        # Controller parameters - more aggressive tuning
        self.Kp = 0.3  # Proportional gain
        self.Ki = 0.008  # Integral gain (higher to reduce steady-state error)
        self.max_integral = 2.0  # Limit integral term
        self.actuator_min = 0.0
        self.actuator_max = 10.0
        self.max_delta = 0.017  # Maximum change per step (actuator duty limit)
        self.filter_alpha = 0.5  # Low-pass filter coefficient (faster response)
        # State variables
        self.integral = np.array([0.0, 0.0])
        self.prev_u = np.array([3.0, 3.0])  # Initial actuator values
        self.filtered_y = None
        self.first_call = True

    def reset(self):
        self.integral = np.array([0.0, 0.0])
        self.prev_u = np.array([3.0, 3.0])
        self.filtered_y = None
        self.first_call = True

    def step(self, t, y, r, quality):
        # y: measured levels (2,)
        # r: setpoints (2,)
        # quality: bool array (2,)

        # Initialize filtered_y on first call
        if self.first_call:
            self.filtered_y = y.copy().astype(float)
            self.first_call = False

        # Update filtered measurements (low-pass filter)
        for i in range(2):
            if quality[i]:
                self.filtered_y[i] = self.filter_alpha * y[i] + (1 - self.filter_alpha) * self.filtered_y[i]

        # Compute feedforward term (sqrt relationship)
        # Handle potential negative setpoints (clip to zero)
        r1 = max(r[0], 0.0)
        r2 = max(r[1], 0.0)
        u1_ff = self.K1 * np.sqrt(r1)
        u2_ff = self.K2 * np.sqrt(r2)

        # Compute error
        e1 = r1 - self.filtered_y[0]
        e2 = r2 - self.filtered_y[1]

        # Update integral term (only if measurement is valid)
        if quality[0]:
            self.integral[0] += self.Ki * e1
            self.integral[0] = np.clip(self.integral[0], -self.max_integral, self.max_integral)
        if quality[1]:
            self.integral[1] += self.Ki * e2
            self.integral[1] = np.clip(self.integral[1], -self.max_integral, self.max_integral)

        # Compute desired control output (before rate limiting and actuator limits)
        u1_desired = u1_ff + self.Kp * e1 + self.integral[0]
        u2_desired = u2_ff + self.Kp * e2 + self.integral[1]

        # Apply anti-windup: clamp output to actuator limits and adjust integral
        u1 = np.clip(u1_desired, self.actuator_min, self.actuator_max)
        u2 = np.clip(u2_desired, self.actuator_min, self.actuator_max)

        # Adjust integral if actuator is saturated (anti-windup)
        if u1_desired > self.actuator_max:
            self.integral[0] = self.actuator_max - u1_ff - self.Kp * e1
        elif u1_desired < self.actuator_min:
            self.integral[0] = self.actuator_min - u1_ff - self.Kp * e1

        if u2_desired > self.actuator_max:
            self.integral[1] = self.actuator_max - u2_ff - self.Kp * e2
        elif u2_desired < self.actuator_min:
            self.integral[1] = self.actuator_min - u2_ff - self.Kp * e2

        # Apply rate limiting (actuator duty limit)
        delta1 = u1 - self.prev_u[0]
        delta2 = u2 - self.prev_u[1]
        if abs(delta1) > self.max_delta:
            u1 = self.prev_u[0] + np.sign(delta1) * self.max_delta
        if abs(delta2) > self.max_delta:
            u2 = self.prev_u[1] + np.sign(delta2) * self.max_delta

        # Update previous actuator values
        self.prev_u[0] = u1
        self.prev_u[1] = u2

        return np.array([u1, u2])
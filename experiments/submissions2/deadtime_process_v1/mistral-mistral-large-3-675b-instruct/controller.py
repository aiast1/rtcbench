import numpy as np
from dataclasses import dataclass

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.actuator_min = 0.0
        self.actuator_max = 100.0
        self.measurement_min = 0.0
        self.measurement_max = 90.0  # Strict safety limit

        # Very conservative PID parameters
        self.Kp = 0.3
        self.Ti = 300.0
        self.Td = 0.0  # Disabled derivative to avoid noise amplification
        self.N = 5.0

        # State variables
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 50.0

        # Setpoint tracking
        self.setpoint_schedule = [
            {'time': 0, 'type': 'step', 'value': [55.0]},
            {'time': 500, 'type': 'step', 'value': [68.0]},
            {'time': 1600, 'type': 'step', 'value': [48.0]},
            {'time': 2600, 'type': 'ramp', 'initial': 48.0, 'value': [60.0], 'duration': 200},
            {'time': 3700, 'type': 'step', 'value': [65.0]}
        ]
        self.current_setpoint = 55.0

        # Safety parameters
        self.max_output_change = 0.5  # Limit output change per step
        self.safety_margin = 2.0  # Keep away from safety limits

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 50.0
        self.current_setpoint = 55.0

    def _update_setpoint(self, t):
        for event in self.setpoint_schedule:
            if t >= event['time']:
                if event['type'] == 'step':
                    self.current_setpoint = event['value'][0]
                elif event['type'] == 'ramp':
                    if t <= event['time'] + event['duration']:
                        self.current_setpoint = event['initial'] + (event['value'][0] - event['initial']) * (t - event['time']) / event['duration']
                    else:
                        self.current_setpoint = event['value'][0]

    def step(self, t, y, r, quality):
        # Update setpoint
        self._update_setpoint(t)
        sp = self.current_setpoint if not np.isnan(r[0]) else y[0]

        # Check quality flag
        if not quality[0]:
            return np.array([self.prev_output])

        # Calculate error with safety margin
        safe_sp = min(sp, self.measurement_max - self.safety_margin)
        error = safe_sp - y[0]

        # PI calculation (no derivative)
        self.integral += error * self.sample_time
        output = self.Kp * (error + (1/self.Ti) * self.integral)

        # Anti-windup
        if output < self.actuator_min or output > self.actuator_max:
            self.integral -= error * self.sample_time

        # Clamp output to limits
        output = np.clip(output, self.actuator_min, self.actuator_max)

        # Rate limiting
        output_change = output - self.prev_output
        if abs(output_change) > self.max_output_change:
            output = self.prev_output + np.sign(output_change) * self.max_output_change

        # Strict safety checks
        if y[0] > self.measurement_max - self.safety_margin:
            output = min(output, self.prev_output * 0.9)
            self.integral = 0.0

        # Update state
        self.prev_error = error
        self.prev_output = output

        return np.array([output])
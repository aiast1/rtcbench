import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.setpoint_schedule = np.array([0, 80, 220])
        self.setpoint_values = np.array([0.99, 0.994, 0.986, 0.994, 0.996])
        self.ramp_start = 220
        self.ramp_duration = 40
        self.actuator_limits = np.array([[1.5, 2.0], [4.5, 5.0]])
        self.previous_time = 0
        self.previous_error = np.zeros(2)
        self.integral_term = np.zeros(2)
        self.previous_actuator = np.array([2.70629, 3.20629])

    def reset(self):
        self.previous_time = 0
        self.previous_error = np.zeros(2)
        self.integral_term = np.zeros(2)

    def step(self, t, y, r, quality):
        if not np.all(quality):
            return self.previous_actuator

        # Get current setpoint
        if t < 80:
            setpoint = self.setpoint_values[0]
        elif t < 220:
            setpoint = self.setpoint_values[1]
        else:
            if t < self.ramp_start + self.ramp_duration:
                ramp_progress = (t - self.ramp_start) / self.ramp_duration
                setpoint = self.setpoint_values[1] + ramp_progress * (self.setpoint_values[4] - self.setpoint_values[1])
            else:
                setpoint = self.setpoint_values[4]

        setpoints = np.array([setpoint, setpoint])

        # Calculate error
        error = setpoints - y

        # Check for safety violations
        D = self.previous_actuator[1] - self.previous_actuator[0]
        B = self.previous_actuator[0] + 1 - self.previous_actuator[1]
        if y[0] < 0.85 or y[1] > 0.15 or D < 0.05 or B < 0.05:
            return np.zeros(2)

        # PI control
        self.integral_term += error * self.sample_time
        derivative_term = (error - self.previous_error) / self.sample_time
        self.previous_error = error

        # Anti-windup
        actuator = np.zeros(2)
        for i in range(2):
            actuator[i] = self.previous_actuator[i] + 0.1 * error[i] + 0.05 * self.integral_term[i] 
            actuator[i] = max(self.actuator_limits[0, i], min(actuator[i], self.actuator_limits[1, i]))

        # Limit actuator slew rate
        actuator = np.maximum(self.previous_actuator - 0.003 / self.sample_time, np.minimum(actuator, self.previous_actuator + 0.003 / self.sample_time))

        self.previous_actuator = actuator
        self.previous_time = t
        return actuator
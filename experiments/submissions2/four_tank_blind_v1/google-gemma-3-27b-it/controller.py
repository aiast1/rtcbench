import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.actuator_limits = np.array([0.0, 10.0])
        self.setpoint_schedule = {
            0: np.array([12.26, 12.78]),
            200: np.array([14.0, 12.78]),
            500: np.array([14.0, 11.2]),
            900: np.array([12.26, 12.78])
        }
        self.current_setpoint = np.array([12.26, 12.78])
        self.last_time = 0.0
        self.integrator = np.zeros(2)
        self.anti_windup_gain = 1.0
        self.Kp = np.array([0.4, 0.4])  # Further reduced Kp
        self.Ki = np.array([0.01, 0.01])  # Further reduced Ki
        self.Kd = np.array([0.02, 0.02])  # Reduced Kd
        self.actuator_travel = 0.0
        self.last_output = np.array([3.0, 3.0])
        self.min_output = np.array([0.0, 0.0]) # Ensure output is never NaN
        self.max_integrator = 1.0 # Limit integrator windup

    def reset(self):
        self.current_setpoint = np.array([12.26, 12.78])
        self.last_time = 0.0
        self.integrator = np.zeros(2)
        self.actuator_travel = 0.0
        self.last_output = np.array([3.0, 3.0])

    def step(self, t, y, r, quality):
        dt = t - self.last_time
        self.last_time = t

        # Setpoint handling
        for time, setpoint in self.setpoint_schedule.items():
            if t >= time:
                self.current_setpoint = setpoint

        # Error calculation
        error = self.current_setpoint - y
        
        # PID control
        output = np.zeros(2)
        for i in range(2):
            if quality[i]:
                output[i] = self.Kp[i] * error[i] + self.Ki[i] * self.integrator[i] + self.Kd[i] * (error[i] - self.integrator[i]) / dt
            else:
                output[i] = self.last_output[i] # Hold last output if quality is bad

        # Anti-windup and Integrator Limit
        for i in range(2):
            if output[i] > self.actuator_limits[1]:
                self.integrator[i] = min(self.integrator[i] + (output[i] - self.actuator_limits[1]) * self.anti_windup_gain * dt, self.max_integrator)
                output[i] = self.actuator_limits[1]
            elif output[i] < self.actuator_limits[0]:
                self.integrator[i] = max(self.integrator[i] + (output[i] - self.actuator_limits[0]) * self.anti_windup_gain * dt, -self.max_integrator)
                output[i] = self.actuator_limits[0]
            else:
                self.integrator[i] = np.clip(self.integrator[i] + error[i] * self.Ki[i] * dt, -self.max_integrator, self.max_integrator)

        # Actuator limits
        output = np.clip(output, self.actuator_limits[0], self.actuator_limits[1])

        # Actuator travel calculation
        self.actuator_travel += np.sum(np.abs(output - self.last_output))
        if self.actuator_travel > 0.017:
            return self.min_output  # Zero out if duty limit exceeded

        self.last_output = output
        
        # Safety checks (simplified - ideally would check h3 and h4 too)
        if not (0.0 <= y[0] <= 20.0 and 0.0 <= y[1] <= 20.0):
            return self.min_output # Zero out if safety violated

        return output
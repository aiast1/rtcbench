import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.actuator_limits = (0.0, 10.0)
        self.initial_actuators = np.array([3.0, 3.0])
        
        # PID parameters - tuned for robustness
        self.Kp = np.array([0.8, 0.8])
        self.Ki = np.array([0.15, 0.15])
        self.Kd = np.array([0.05, 0.05])
        
        # Filter for derivative term
        self.tau_f = 1.0
        self.alpha = self.sample_time / (self.tau_f + self.sample_time)
        
        # Anti-windup parameters
        self.Kaw = np.array([0.5, 0.5])
        
    def reset(self):
        self.prev_error = np.zeros(2)
        self.integral = np.zeros(2)
        self.prev_measurement = np.zeros(2)
        self.prev_derivative = np.zeros(2)
        self.prev_output = self.initial_actuators.copy()
        
    def step(self, t, y, r, quality):
        # Handle bad measurements - use last good value
        if not quality[0]:
            y[0] = self.prev_measurement[0]
        if not quality[1]:
            y[1] = self.prev_measurement[1]
        
        # Store measurements for next step
        self.prev_measurement = y.copy()
        
        # Calculate errors (only for channels with setpoints)
        error = np.zeros(2)
        if not np.isnan(r[0]):
            error[0] = r[0] - y[0]
        if not np.isnan(r[1]):
            error[1] = r[1] - y[1]
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term with anti-windup
        self.integral += self.Ki * error * self.sample_time
        
        # Calculate tentative output for anti-windup
        tentative_output = P + self.integral
        
        # Anti-windup: clamp integral if output would saturate
        for i in range(2):
            if tentative_output[i] > self.actuator_limits[1]:
                self.integral[i] -= self.Kaw[i] * (tentative_output[i] - self.actuator_limits[1])
            elif tentative_output[i] < self.actuator_limits[0]:
                self.integral[i] -= self.Kaw[i] * (tentative_output[i] - self.actuator_limits[0])
        
        # Derivative term with filtering
        derivative = (error - self.prev_error) / self.sample_time
        filtered_derivative = self.alpha * derivative + (1 - self.alpha) * self.prev_derivative
        
        D = self.Kd * filtered_derivative
        
        # PID output
        output = P + self.integral + D
        
        # Clamp to actuator limits
        output = np.clip(output, self.actuator_limits[0], self.actuator_limits[1])
        
        # Store values for next step
        self.prev_error = error.copy()
        self.prev_derivative = filtered_derivative.copy()
        self.prev_output = output.copy()
        
        return output
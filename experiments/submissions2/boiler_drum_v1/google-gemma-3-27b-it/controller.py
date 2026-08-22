import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.actuator_limits = [0.0, 100.0]
        self.level_setpoint = 0.0
        self.last_level_setpoint_time = 0.0
        self.integrator = 0.0
        self.last_output = 45.0
        self.last_error = 0.0

    def reset(self):
        self.integrator = 0.0
        self.last_output = 45.0
        self.level_setpoint = 0.0
        self.last_level_setpoint_time = 0.0
        self.last_error = 0.0

    def step(self, t, y, r, quality):
        """t: seconds (float). y: numpy array of measured values, one per measurement.
        r: numpy array of setpoints, nan for channels that are not scored.
        quality: numpy bool array, False means that reading is stale/bad.
        Returns: numpy array of control values, one per actuator."""

        # Level control
        level = y[0]
        level_setpoint = r[0]

        if np.isnan(level_setpoint):
            level_setpoint = self.level_setpoint
        else:
            self.level_setpoint = level_setpoint
            self.last_level_setpoint_time = t

        error = level_setpoint - level

        # PI controller
        Kp = 0.6  # Reduced Kp further
        Ki = 0.005 # Reduced Ki further
        Kd = 0.0  # Removed derivative action

        proportional = Kp * error
        
        # Anti-windup
        if t - self.last_level_setpoint_time > self.sample_time * 5:
            self.integrator = 0.0
        else:
            self.integrator += Ki * error * self.sample_time
            self.integrator = np.clip(self.integrator, -30, 30) # Reduced integrator limits further

        output = proportional + self.integrator

        # Actuator limits and bumpless transfer
        output = np.clip(output, self.actuator_limits[0], self.actuator_limits[1])
        
        #Slew rate limit - even more conservative
        delta_output = output - self.last_output
        if abs(delta_output) > 2.0:
            output = self.last_output + np.sign(delta_output) * 2.0
        
        self.last_output = output
        self.last_error = error

        # Safety checks - even more conservative
        l_inv = level  # Assuming level is a proxy for water inventory
        p = y[1]

        if l_inv < -220.0 or level > 180.0 or p < 60.0 or p > 100.0: #Increased safety margins
            return np.array([45.0])  # Return to initial value on safety violation

        return np.array([output])
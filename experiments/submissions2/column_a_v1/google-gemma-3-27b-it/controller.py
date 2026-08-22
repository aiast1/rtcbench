import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.L_sp = 0.0
        self.V_sp = 0.0
        self.L = 0.0
        self.V = 0.0
        self.integrator_L = 0.0
        self.integrator_V = 0.0

        # PID gains - further reduced aggressiveness
        self.Kp_L = 0.1
        self.Ki_L = 0.002
        self.Kd_L = 0.0
        self.Kp_V = 0.1
        self.Ki_V = 0.002
        self.Kd_V = 0.0

        # Anti-windup gain
        self.antiwindup_gain = 0.1

        # Actuator limits
        self.L_min = 1.5
        self.L_max = 4.5
        self.V_min = 2.0
        self.V_max = 5.0

        # Rate limits for actuators - crucial for duty cycle
        self.L_rate_limit = 0.05  # Reduced rate limit
        self.V_rate_limit = 0.05  # Reduced rate limit

        # Clamp integrator to prevent windup
        self.integrator_clamp = 10.0

    def reset(self):
        self.L_sp = 0.0
        self.V_sp = 0.0
        self.L = 0.0
        self.V = 0.0
        self.integrator_L = 0.0
        self.integrator_V = 0.0

    def step(self, t, y, r, quality):
        """t: seconds (float). y: numpy array of measured values, one per measurement.
        r: numpy array of setpoints, nan for channels that are not scored.
        quality: numpy bool array, False means that reading is stale/bad.
        Returns: numpy array of control values, one per actuator."""

        # Get setpoints
        if not np.isnan(r[0]):
            self.L_sp = r[0]
        if not np.isnan(r[1]):
            self.V_sp = r[1]

        # Get measurements
        if quality[0]:
            yD = y[0]
        else:
            yD = self.L_sp  # Use setpoint if measurement is bad
        if quality[1]:
            xB = y[1]
        else:
            xB = self.V_sp  # Use setpoint if measurement is bad

        # Calculate errors
        error_L = self.L_sp - yD
        error_V = self.V_sp - xB

        # PID control for L (reflux)
        self.integrator_L += error_L * self.sample_time
        self.integrator_L = np.clip(self.integrator_L, -self.integrator_clamp, self.integrator_clamp) # Clamp integrator
        derivative_L = (error_L - self.L) / self.sample_time
        u_L = self.Kp_L * error_L + self.Ki_L * self.integrator_L + self.Kd_L * derivative_L

        # PID control for V (boilup)
        self.integrator_V += error_V * self.sample_time
        self.integrator_V = np.clip(self.integrator_V, -self.integrator_clamp, self.integrator_clamp) # Clamp integrator
        derivative_V = (error_V - self.V) / self.sample_time
        u_V = self.Kp_V * error_V + self.Ki_V * self.integrator_V + self.Kd_V * derivative_V

        # Actuator limits and rate limits
        L = np.clip(self.L + u_L * self.sample_time, self.L_min, self.L_max)
        L = np.clip(L, self.L - self.L_rate_limit, self.L + self.L_rate_limit) # Rate limit
        V = np.clip(self.V + u_V * self.sample_time, self.V_min, self.V_max)
        V = np.clip(V, self.V - self.V_rate_limit, self.V + self.V_rate_limit) # Rate limit

        # Update internal states
        self.L = L
        self.V = V

        return np.array([L, V])
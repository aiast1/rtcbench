import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.n_actuators = 2
        self.n_measurements = 2
        # Very conservative parameters to avoid safety violations and duty limit breaches
        self.Kp = np.array([[0.15, 0.02], [0.02, 0.15]])
        self.Ki = np.array([[0.005, 0.001], [0.001, 0.005]])
        self.Kd = np.array([[0.0, 0.0], [0.0, 0.0]])
        self.u_max = np.array([10.0, 10.0])
        self.u_min = np.array([0.0, 0.0])
        self.u_slew = np.array([0.08, 0.08])  # Very conservative slew limit
        self.integral = np.zeros(self.n_actuators)
        self.prev_u = np.array([3.0, 3.0])
        self.prev_y = np.zeros(self.n_measurements)
        self.last_update_time = -self.sample_time
        self.dropped_count = np.zeros(self.n_measurements)
        self.prev_error = np.zeros(self.n_measurements)
        
    def reset(self):
        self.integral = np.zeros(self.n_actuators)
        self.prev_u = np.array([3.0, 3.0])
        self.prev_y = np.zeros(self.n_measurements)
        self.last_update_time = -self.sample_time
        self.dropped_count = np.zeros(self.n_measurements)
        self.prev_error = np.zeros(self.n_measurements)
        
    def step(self, t, y, r, quality):
        # Handle dropped samples by using previous values
        y_valid = np.where(quality, y, self.prev_y)
        self.prev_y = y_valid.copy()
        
        # Compute time since last update
        dt = t - self.last_update_time
        if dt < 0.5 * self.sample_time:
            return self.prev_u
        dt = self.sample_time
        
        # Extract setpoints for measured channels
        r_valid = np.where(np.isnan(r), y_valid, r)
        
        # Compute errors
        error = r_valid[:self.n_measurements] - y_valid[:self.n_measurements]
        
        # Anti-windup: only integrate if actuator is not saturated or error opposes saturation
        u_sat = np.zeros(self.n_actuators, dtype=bool)
        for i in range(self.n_actuators):
            if self.prev_u[i] >= self.u_max[i] and error[i] > 0:
                u_sat[i] = True
            elif self.prev_u[i] <= self.u_min[i] and error[i] < 0:
                u_sat[i] = True
        
        # Update integral with anti-windup
        self.integral += error * dt * (1 - u_sat.astype(float))
        
        # Compute control
        u_prop = self.Kp @ error
        u_int = self.Ki @ self.integral
        
        u_raw = u_prop + u_int
        
        # Add feedforward to maintain current operating point
        u_ff = self.prev_u.copy()
        u_raw += u_ff
        
        # Apply actuator constraints with anti-windup
        u_new = np.clip(u_raw, self.u_min, self.u_max)
        
        # Apply slew rate limits
        du = u_new - self.prev_u
        du = np.clip(du, -self.u_slew, self.u_slew)
        u_new = self.prev_u + du
        
        # Final hard constraints
        u_new = np.clip(u_new, self.u_min, self.u_max)
        
        # Update state
        self.prev_u = u_new.copy()
        self.last_update_time = t
        
        return u_new
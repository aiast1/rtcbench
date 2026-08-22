import numpy as np

class Controller:
    def __init__(self, brief):
        self.dt = brief.sample_time
        self.n_u = 2  # two actuators
        self.n_y = 2  # two measured outputs
        
        # Setpoint tracking: only two outputs are scored
        self.sp = np.zeros(self.n_y)
        self.sp_prev = np.zeros(self.n_y)
        
        # Integral state for PI control
        self.xi = np.zeros(self.n_y)
        
        # Anti-windup: back-calculation gain
        self.k_aw = 0.05
        
        # Controller gains (reduced for robustness to prevent overshoot and constraint violation)
        # Based on nominal model: integrating process with low gain, slow dynamics
        # Reduced gains to minimize aggressive moves that could cause unmeasured tanks to overflow
        self.kp = np.array([0.04, 0.04])   # further reduced proportional gain
        self.ki = np.array([0.002, 0.002]) # reduced integral gain
        
        # Actuator constraints
        self.u_min = np.array([0.0, 0.0])
        self.u_max = np.array([10.0, 10.0])
        self.u_prev = np.array([3.0, 3.0])  # bumpless start: match initial operating point
        
        # Rate of change limit to avoid chattering and respect actuator slew and duty limit
        # Duty limit: 0.0037 per step -> max |delta_u| <= 0.0037 to avoid chattering penalty
        self.max_delta_u = 0.003  # conservative: slightly below duty limit to ensure compliance
        
        # Internal model of disturbance (for integral action)
        self.disturbance_estimate = np.zeros(self.n_y)
        
        # Filter for measurement noise (first-order)
        self.y_filtered = np.zeros(self.n_y)
        self.alpha = 0.1  # increased filtering to reduce noise sensitivity
        
        # Track if we have valid measurements
        self.last_valid_y = np.zeros(self.n_y)
        self.has_valid_measurement = np.array([False, False])
        
        # Deadzone to reduce unnecessary actuator movement (helps with duty limit)
        self.deadzone = 0.02  # cm: ignore small errors to reduce chattering
        
    def reset(self):
        # Reset integral state and filters
        self.xi = np.zeros(self.n_y)
        self.y_filtered = np.zeros(self.n_y)
        self.last_valid_y = np.zeros(self.n_y)
        self.has_valid_measurement = np.array([False, False])
        self.disturbance_estimate = np.zeros(self.n_y)
        self.u_prev = np.array([3.0, 3.0])  # reset to bumpless start
        
    def step(self, t, y, r, quality):
        # Update filtered measurement: handle noise and dropped samples
        for i in range(self.n_y):
            if quality[i]:  # good measurement
                self.y_filtered[i] = self.alpha * y[i] + (1 - self.alpha) * self.y_filtered[i]
                self.last_valid_y[i] = y[i]
                self.has_valid_measurement[i] = True
            else:  # bad measurement: hold last valid or use filtered value
                if self.has_valid_measurement[i]:
                    self.y_filtered[i] = self.y_filtered[i]  # no change, keep filtering
                else:
                    # No valid measurement yet: use placeholder (should not happen after init)
                    self.y_filtered[i] = 0.0
        
        # Use filtered measurement for control
        y_meas = self.y_filtered
        
        # Update setpoint: handle NaN (non-scored channels) by holding last valid
        for i in range(self.n_y):
            if not np.isnan(r[i]):
                self.sp[i] = r[i]
            # else: hold previous setpoint (should not happen for scored channels in this problem)
        
        # Compute error
        e = self.sp - y_meas
        
        # Apply deadzone: ignore small errors to reduce actuator movement and chattering
        e_deadzoned = np.where(np.abs(e) < self.deadzone, 0.0, e)
        
        # Integral update with anti-windup via back-calculation
        # Compute tentative integral increment (only if error outside deadzone)
        xi_tentative = self.xi + self.ki * e_deadzoned * self.dt
        
        # Compute tentative control output (before saturation)
        u_tentative = self.kp * e_deadzoned + self.xi
        
        # Saturate tentative control
        u_saturated = np.clip(u_tentative, self.u_min, self.u_max)
        
        # Anti-windup back-calculation: adjust integral state if saturated
        self.xi = xi_tentative + self.k_aw * (u_saturated - u_tentative) * self.dt
        
        # Final control output (use saturated value for actuator command)
        u = u_saturated
        
        # Apply rate limiting to avoid actuator chattering and duty limit violation
        delta_u = u - self.u_prev
        delta_u_clipped = np.clip(delta_u, -self.max_delta_u, self.max_delta_u)
        u = self.u_prev + delta_u_clipped
        
        # Ensure final u is still within bounds (should be, but clip again for safety)
        u = np.clip(u, self.u_min, self.u_max)
        
        # Store for next step
        self.u_prev = u
        
        return u
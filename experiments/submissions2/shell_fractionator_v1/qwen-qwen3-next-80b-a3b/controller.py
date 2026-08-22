import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.nu = 3  # number of actuators
        self.ny = 3  # number of measurements
        self.dt = brief.sample_time
        
        # Conservative tuning parameters
        self.Kp_gain = 0.10  # very conservative proportional gain
        self.Ki_gain = 0.005  # very slow integral action
        self.Kd_gain = 0.0   # no derivative - noise sensitive
        
        # Constraints
        self.actuator_limits = np.array([-0.5, 0.5])
        self.max_actuator_change = 0.016  # duty limit
        self.safety_margin = 0.10  # safety buffer for TI-103 (conservative)
        self.integral_windup_limit = 0.3  # tight anti-windup
        self.deadband = 0.005  # prevent chattering
        
        # State variables
        self.integral_terms = np.zeros(self.ny)
        self.prev_error = np.zeros(self.ny)
        self.prev_output = np.zeros(self.nu)
        self.prev_y = np.zeros(self.ny)
        self.last_u = np.zeros(self.nu)
        
        # Safety-focused state
        self.safety_priority = 0.0  # how much to prioritize safety over setpoint
        self.safety_timer = 0
        
    def reset(self):
        """Called once before each scenario"""
        self.integral_terms = np.zeros(self.ny)
        self.prev_error = np.zeros(self.ny)
        self.prev_output = np.zeros(self.nu)
        self.prev_y = np.zeros(self.ny)
        self.last_u = np.zeros(self.nu)
        self.safety_priority = 0.0
        self.safety_timer = 0
        
    def step(self, t, y, r, quality):
        # Ensure inputs are numpy arrays
        y = np.array(y, dtype=float)
        r = np.array(r, dtype=float)
        quality = np.array(quality, dtype=bool)
        
        # Handle bad measurements - use previous value if quality is False
        for i in range(self.ny):
            if not quality[i]:
                y[i] = self.prev_y[i]
            else:
                self.prev_y[i] = y[i]
        
        # Calculate error
        error = r - y
        
        # Integral action with aggressive anti-windup
        for i in range(self.ny):
            # Only integrate if actuators are safely away from limits
            if (self.prev_output[0] > self.actuator_limits[0] + 0.1 and 
                self.prev_output[0] < self.actuator_limits[1] - 0.1 and
                self.prev_output[1] > self.actuator_limits[0] + 0.1 and 
                self.prev_output[1] < self.actuator_limits[1] - 0.1 and
                self.prev_output[2] > self.actuator_limits[0] + 0.1 and 
                self.prev_output[2] < self.actuator_limits[1] - 0.1):
                self.integral_terms[i] += error[i] * self.dt
            else:
                # Aggressive anti-windup - reduce integral term when near saturation
                self.integral_terms[i] -= error[i] * self.dt * 0.5
            
            # Clamp integral term tightly
            self.integral_terms[i] = np.clip(self.integral_terms[i], 
                                            -self.integral_windup_limit, 
                                            self.integral_windup_limit)
        
        # Proportional and integral control
        u_p = np.zeros(self.nu)
        u_i = np.zeros(self.nu)
        
        # Proportional contribution (reduced gain)
        for i in range(self.ny):
            for j in range(self.nu):
                u_p[j] += self.Kp_gain * self.Kp[i, j] * error[i]
        
        # Integral contribution (very slow)
        for i in range(self.ny):
            for j in range(self.nu):
                u_i[j] += self.Ki_gain * self.Kp[i, j] * self.integral_terms[i]
        
        # Combine control actions
        u_raw = u_p + u_i
        
        # Slew rate limiting and saturation
        u_final = np.zeros(self.nu)
        for j in range(self.nu):
            # Apply slew limit
            delta_u = u_raw[j] - self.last_u[j]
            delta_u = np.clip(delta_u, -self.max_actuator_change, self.max_actuator_change)
            u_final[j] = self.last_u[j] + delta_u
            
            # Apply hard limits
            u_final[j] = np.clip(u_final[j], self.actuator_limits[0], self.actuator_limits[1])
        
        # SAFETY CRITICAL: TI-103 (bottoms reflux temperature) must be >= -0.5
        # FCV-203 (u[2]) is the primary control for TI-103
        if y[2] < -0.45:  # approaching safety limit
            self.safety_priority = 0.8
            self.safety_timer = 50  # maintain priority for 50 steps
        elif y[2] < -0.4:
            self.safety_priority = 0.5
            self.safety_timer = 30
        elif y[2] < -0.35:
            self.safety_priority = 0.2
            self.safety_timer = 15
        else:
            self.safety_timer = max(0, self.safety_timer - 1)
            self.safety_priority = max(0, self.safety_priority - 0.02)
        
        # Apply safety override: prioritize FCV-203 to increase temperature
        if self.safety_priority > 0:
            # Force FCV-203 to increase
            u_final[2] = max(u_final[2], 0.2)
            # Reduce other outputs to compensate and avoid total saturation
            if u_final[0] > 0.1:
                u_final[0] -= self.safety_priority * 0.1
            if u_final[1] > 0.1:
                u_final[1] -= self.safety_priority * 0.1
            # Re-apply limits after compensation
            u_final[0] = np.clip(u_final[0], self.actuator_limits[0], self.actuator_limits[1])
            u_final[1] = np.clip(u_final[1], self.actuator_limits[0], self.actuator_limits[1])
            u_final[2] = np.clip(u_final[2], self.actuator_limits[0], self.actuator_limits[1])
        
        # Prevent chattering with deadband
        for j in range(self.nu):
            if abs(u_final[j]) < self.deadband:
                u_final[j] = 0.0
        
        # Update state
        self.prev_error = error.copy()
        self.last_u = u_final.copy()
        self.prev_output = u_final.copy()
        
        return u_final
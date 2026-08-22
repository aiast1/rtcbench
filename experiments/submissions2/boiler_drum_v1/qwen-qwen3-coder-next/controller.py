import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.actuator_count = 1
        self.measurement_count = 4
        self.setpoint_count = 1
        
        # Safety limits (from brief)
        self.l_ind_max = 250.0      # LT-101 max (scored)
        self.l_inv_min = -250.0     # water inventory min (not measured)
        self.p_min = 70.0           # PT-102 min
        self.p_max = 100.0          # PT-102 max
        
        # Actuator limits
        self.u_min = 0.0
        self.u_max = 100.0
        self.u_initial = 45.0
        
        # Cost weights
        self.w_effort = 0.5
        
        # Controller parameters (highly conservative)
        self.kp = 0.15
        self.ki = 0.002
        self.kd = 0.0
        self.tau_d = 30.0
        
        # Anti-windup
        self.aw_gain = 0.1
        
        # Derivative filter state
        self.d_error_filtered = 0.0
        self.last_error = 0.0
        self.integral = 0.0
        self.last_u = self.u_initial
        
        # Setpoint tracking
        self.last_r = 0.0
        
        # Safety margins
        self.level_safety_margin = 50.0
        self.pressure_safety_margin = 5.0
        
    def reset(self):
        self.integral = 0.0
        self.d_error_filtered = 0.0
        self.last_error = 0.0
        self.last_u = self.u_initial
        self.last_r = 0.0
        
    def step(self, t, y, r, quality):
        # Extract measurements
        l_ind = y[0]  # drum level (indicated), mm
        p = y[1]      # drum pressure, bar
        steam_flow = y[2]  # not used directly
        fw_flow = y[3]     # not used directly
        
        # Extract setpoint (only first channel is scored)
        sp = r[0] if not np.isnan(r[0]) else self.last_r
        
        # Update setpoint tracking
        self.last_r = sp
        
        # Check quality - if bad, hold last control action
        if not quality[0]:
            return np.array([self.last_u])
        
        # Safety checks - clamp to safe range if possible
        safe_l_ind_max = self.l_ind_max - self.level_safety_margin
        
        # Compute error
        error = sp - l_ind
        
        # PID computation with derivative filter
        dt = self.sample_time
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        # Anti-windup: reduce integral if actuator saturated
        if self.last_u >= self.u_max and error > 0:
            self.integral -= self.aw_gain * (error * dt)
        elif self.last_u <= self.u_min and error < 0:
            self.integral -= self.aw_gain * (error * dt)
        i_term = self.integral * self.ki
        
        # Derivative term with low-pass filter (disabled for robustness)
        d_error = (error - self.last_error) / dt
        self.d_error_filtered = (1 - dt/self.tau_d) * self.d_error_filtered + (dt/self.tau_d) * d_error
        d_term = self.kd * self.d_error_filtered
        
        # Total PID
        u_pid = self.last_u + p_term + i_term + d_term
        
        # Apply actuator constraints
        u_clamped = np.clip(u_pid, self.u_min, self.u_max)
        
        # Slew rate limiting (very conservative to avoid duty limit)
        # Max slew ~0.5% per sample to stay under duty limit
        max_slew = 0.5
        u_slew_limited = np.clip(u_clamped, 
                                 self.last_u - max_slew, 
                                 self.last_u + max_slew)
        
        # Apply pressure-based safety: if pressure is outside safe range, hold valve
        if p < (self.p_min + self.pressure_safety_margin) or p > (self.p_max - self.pressure_safety_margin):
            u_slew_limited = self.last_u
        
        # Apply level-based safety: if level is too high, reduce feedwater aggressively
        if l_ind > safe_l_ind_max:
            u_slew_limited = min(u_slew_limited, self.last_u - 2.0)
        
        # Final control action
        u_final = np.clip(u_slew_limited, self.u_min, self.u_max)
        
        # Update state
        self.last_error = error
        self.last_u = u_final
        
        return np.array([u_final])
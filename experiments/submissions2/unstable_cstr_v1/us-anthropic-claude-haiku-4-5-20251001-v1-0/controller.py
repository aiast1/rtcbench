import numpy as np
from dataclasses import dataclass

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = brief.sample_time
        
        # PID gains - tuned for unstable CSTR with measurement noise and actuator constraints
        # Conservative approach: prioritize stability and safety over aggressive tracking
        self.Kp = 15.0      # Proportional gain
        self.Ki = 0.8       # Integral gain (modest to avoid windup)
        self.Kd = 2.0       # Derivative gain (light to avoid noise amplification)
        
        # State tracking
        self.integral_error = 0.0
        self.last_error = 0.0
        self.last_measurement = None
        self.last_output = 300.0  # Start at steady state
        
        # Anti-windup limits
        self.integral_max = 30.0
        self.integral_min = -30.0
        
        # Actuator constraints
        self.u_min = 270.0
        self.u_max = 340.0
        self.max_slew = 1.5 * self.dt  # 1.5 K/s * dt
        
        # Stiction handling
        self.stiction_band = 0.15
        self.last_commanded = 300.0
        
        # Safety margins
        self.T_max_safe = 465.0  # Well below relief at 470 K
        self.T_min_safe = 305.0  # Well above product precipitation at 300 K
        
    def reset(self):
        self.integral_error = 0.0
        self.last_error = 0.0
        self.last_measurement = None
        self.last_output = 300.0
        self.last_commanded = 300.0
        
    def step(self, t, y, r, quality):
        # Extract measurements
        T_reactor = y[0]  # TI-101: reactor temperature
        T_jacket = y[1]   # TI-102: jacket supply temperature (readback)
        
        # Use only good quality measurements
        if not quality[0]:
            # If reactor temp is bad, use last known or be conservative
            if self.last_measurement is not None:
                T_reactor = self.last_measurement
            else:
                T_reactor = 350.0
        else:
            self.last_measurement = T_reactor
        
        # Get setpoint for reactor temperature (index 0 is scored)
        T_setpoint = r[0]
        if np.isnan(T_setpoint):
            T_setpoint = 350.0
        
        # Calculate error
        error = T_setpoint - T_reactor
        
        # Proportional term
        P_term = self.Kp * error
        
        # Integral term with anti-windup
        self.integral_error += error * self.dt
        self.integral_error = np.clip(self.integral_error, self.integral_min, self.integral_max)
        I_term = self.Ki * self.integral_error
        
        # Derivative term (use error derivative, not measurement derivative, to reduce noise)
        if self.last_error is not None:
            error_rate = (error - self.last_error) / self.dt
            # Low-pass filter on derivative to reduce noise impact
            D_term = self.Kd * error_rate * 0.5  # Damping factor
        else:
            D_term = 0.0
        
        self.last_error = error
        
        # Compute raw control output
        u_raw = self.last_output + P_term + I_term + D_term
        
        # Apply actuator slew rate limit
        u_slewed = np.clip(u_raw, self.last_commanded - self.max_slew, 
                          self.last_commanded + self.max_slew)
        
        # Apply stiction: if command is very close to last command, hold it
        if abs(u_slewed - self.last_commanded) < self.stiction_band:
            u_slewed = self.last_commanded
        
        # Apply hard limits
        u_limited = np.clip(u_slewed, self.u_min, self.u_max)
        
        # Safety override: if reactor temperature is too high, maximize cooling
        if T_reactor > self.T_max_safe:
            u_limited = self.u_min
            # Reset integral to prevent windup during safety action
            self.integral_error = 0.0
        
        # Safety override: if reactor temperature is too low, minimize cooling
        if T_reactor < self.T_min_safe:
            u_limited = self.u_max
            # Reset integral to prevent windup during safety action
            self.integral_error = 0.0
        
        # Update state for next iteration
        self.last_output = u_limited
        self.last_commanded = u_limited
        
        return np.array([u_limited])
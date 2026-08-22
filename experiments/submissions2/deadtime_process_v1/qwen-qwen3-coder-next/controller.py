import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.actuator_limits = np.array([0.0, 100.0])
        self.safety_limits = {
            'T_in_max': 90.0,
            'T_out_max': 90.0,
            'T_out_min': 0.0
        }
        # PID parameters - tuned for robustness with delay and noise
        self.Kp = 0.8
        self.Ki = 0.005
        self.Kd = 0.1
        self.tau_d = 10.0  # derivative filter time constant
        self.integral_min = -20.0
        self.integral_max = 20.0
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_measurement = 0.0
        self.prev_derivative = 0.0
        self.last_u = 50.0  # initial actuator value (steady state)
        self.last_t = None
        
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_measurement = 0.0
        self.prev_derivative = 0.0
        self.last_u = 50.0
        self.last_t = None
        
    def step(self, t, y, r, quality):
        # Extract measurements and setpoints
        T_out = y[0]  # TT-401: line outlet temperature
        
        # Get current setpoint for scored channel
        sp = r[0] if not np.isnan(r[0]) else 55.0  # default to initial setpoint if NaN
        
        # Handle bad measurements
        if not quality[0]:
            # Use last known good value if measurement is bad
            T_out = self.prev_measurement if self.prev_measurement != 0.0 else 55.0
        
        # Store current measurement for next step
        self.prev_measurement = T_out
        
        # Calculate error
        error = sp - T_out
        
        # PID calculations with anti-windup and derivative filtering
        dt = self.sample_time
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term with anti-windup (clamping)
        if self.last_u is not None:
            # Only integrate if not saturated or error opposes saturation
            if not ((self.last_u >= self.actuator_limits[1] and error > 0) or 
                    (self.last_u <= self.actuator_limits[0] and error < 0)):
                self.integral += error * dt
                # Clamp integral to prevent windup
                self.integral = np.clip(self.integral, self.integral_min, self.integral_max)
        
        I = self.Ki * self.integral
        
        # Derivative term with filtered derivative (noise reduction)
        if self.last_t is not None:
            # Raw derivative
            raw_derivative = (T_out - self.prev_measurement) / dt if dt > 0 else 0.0
            # Filtered derivative using first-order filter
            alpha = dt / (self.tau_d + dt)
            self.prev_derivative = (1 - alpha) * self.prev_derivative + alpha * raw_derivative
        else:
            self.prev_derivative = 0.0
            
        D = self.Kd * self.prev_derivative
        
        # Calculate control output
        u = P + I + D
        
        # Add to base value (50% duty at steady state)
        u = 50.0 + u
        
        # Apply actuator limits
        u = np.clip(u, self.actuator_limits[0], self.actuator_limits[1])
        
        # Store for next iteration
        self.last_u = u
        self.last_t = t
        self.prev_error = error
        
        # Return control output as numpy array
        return np.array([u])
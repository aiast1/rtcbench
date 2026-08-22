import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.reset()
        
    def reset(self):
        # Initialize controller state
        self.integral_term = 0.0
        self.prev_error = 0.0
        self.prev_output = 50.0  # Start at nominal operating point
        self.prev_measurement = 55.0  # Assume initial steady state
        self.last_time = 0.0
        self.sample_time = self.brief.sample_time
        
        # PID parameters (tuned for robustness over varying deadtime)
        self.Kp = 0.6
        self.Ki = 0.005
        self.Kd = 0.05
        
        # Anti-windup and constraints
        self.max_output = 100.0
        self.min_output = 0.0
        self.max_slew_rate = 1.5  # % per sample (18% per minute, more conservative)
        self.integral_limit = 100.0 / self.Ki  # Anti-windup limit
        
        # Safety margins (to protect unmeasured T_in)
        self.safety_margin = 8.0  # More conservative: keep T_out at least 8°C below 90°C limit
        self.aggressive_reduction_threshold = 85.0  # Start reducing aggressively above this
        
        # Derivative filter coefficient (for noise suppression)
        self.df = 0.1  # Filter coefficient for derivative term
        
        # State for derivative filtering
        self.filtered_derivative = 0.0
        self.prev_T_out = 55.0
        
    def step(self, t, y, r, quality):
        # Extract measurements and setpoints
        T_out = y[0]  # TT-401: line outlet temperature
        setpoint = r[0]  # Setpoint for T_out
        
        # Handle bad measurements (use last good value if current is bad)
        if not quality[0]:
            T_out = self.prev_measurement
        else:
            self.prev_measurement = T_out
            
        # Calculate error
        error = setpoint - T_out
        
        # Time step
        dt = t - self.last_time
        if dt <= 0:
            dt = self.sample_time
            
        # Proportional term
        P = self.Kp * error
        
        # Integral term with anti-windup
        if self.prev_output >= self.max_output and error > 0:
            # Prevent windup when output is saturated high
            I = self.integral_term
        elif self.prev_output <= self.min_output and error < 0:
            # Prevent windup when output is saturated low
            I = self.integral_term
        else:
            I = self.integral_term + self.Ki * error * dt
            
        # Clamp integral term to prevent windup
        I = np.clip(I, -self.integral_limit, self.integral_limit)
        
        # Derivative term with noise filtering
        # Use filtered derivative: first-order low-pass filter
        dT_out = (T_out - self.prev_T_out) / dt
        self.filtered_derivative = (1 - self.df) * self.filtered_derivative + self.df * (-dT_out)
        D = self.Kd * self.filtered_derivative
        
        # Calculate raw output
        raw_output = P + I + D
        
        # Apply anti-windup: if output saturated, reduce integral contribution
        if raw_output >= self.max_output:
            I = (self.max_output - P - D)
            I = max(I, -self.integral_limit)
        elif raw_output <= self.min_output:
            I = (self.min_output - P - D)
            I = min(I, self.integral_limit)
            
        # Recalculate output with clamped integral
        output = P + I + D
        
        # Apply slew rate limiting (more conservative)
        output = np.clip(output, 
                        self.prev_output - self.max_slew_rate, 
                        self.prev_output + self.max_slew_rate)
        
        # Apply hard actuator limits
        output = np.clip(output, self.min_output, self.max_output)
        
        # Safety constraint: protect unmeasured T_in by being extremely conservative
        # If T_out is high, reduce output aggressively to prevent T_in from exceeding 90°C
        if T_out >= self.aggressive_reduction_threshold:
            # Reduce output by additional margin when near limit
            output = min(output, self.prev_output - 2.0)
        elif T_out >= (90.0 - self.safety_margin):
            # Moderate reduction when approaching safety margin
            output = min(output, self.prev_output - 1.0)
            
        # Additional safety: if setpoint is high and T_out is low, don't overshoot
        if setpoint > 70.0 and T_out < (setpoint - 10.0):
            # Prevent aggressive ramp-up when far from setpoint at high targets
            output = min(output, self.prev_output + 0.8)
            
        # Ensure output doesn't jump due to noise
        if abs(error) < 0.5 and dt == self.sample_time:
            # Very small error: dampen output changes
            output = self.prev_output + np.clip(output - self.prev_output, -0.5, 0.5)
            
        # Update state
        self.integral_term = I
        self.prev_output = output
        self.prev_T_out = T_out
        self.last_time = t
        
        return np.array([output])
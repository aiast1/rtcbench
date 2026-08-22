import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.actuator_limits = (0.0, 100.0)
        
        # Conservative controller tuning for robustness
        # Much more conservative to prevent violations
        self.Kp = 0.2  # Very low gain for safety
        self.Ti = 500.0  # Very slow integral
        self.Td = 0.0  # No derivative
        
        # Anti-windup
        self.Kt = 1.0 / self.Ti
        
        # Initialize state
        self.integral = 0.0
        self.last_output = 50.0
        self.last_error = 0.0
        self.last_measurement = None
        
        # Stronger filtering for noise
        self.alpha = 0.1  # Very strong filter
        
        # Output rate limiting to prevent duty limit violations
        self.max_output_rate = 0.001  # Maximum change per sample (0.5% per sample)
        self.output_buffer = np.zeros(10)
        self.buffer_index = 0
        
        # Safety monitoring
        self.safety_margin = 5.0  # Stay 5°C below limit
        self.max_setpoint = 90.0 - self.safety_margin  # Maximum allowed setpoint
        
        # For ramp handling
        self.last_setpoint = None
        
    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.last_output = 50.0
        self.last_error = 0.0
        self.last_measurement = None
        self.filtered_y = None
        self.output_buffer.fill(0.0)
        self.buffer_index = 0
        self.last_setpoint = None
        
    def step(self, t, y, r, quality):
        """Main control step - ultra conservative for safety"""
        # Handle bad quality
        if not quality[0] and self.last_measurement is not None:
            current_y = self.last_measurement
        else:
            current_y = y[0]
            self.last_measurement = current_y
        
        # Apply strong filtering
        if self.filtered_y is None:
            self.filtered_y = current_y
        else:
            self.filtered_y = self.alpha * current_y + (1 - self.alpha) * self.filtered_y
        
        # Get setpoint with safety limit
        setpoint = r[0]
        if np.isnan(setpoint):
            setpoint = self.filtered_y
        else:
            # Clamp setpoint to safety limit
            setpoint = min(setpoint, self.max_setpoint)
        
        # Calculate error
        error = setpoint - self.filtered_y
        
        # Clamp error to prevent aggressive control
        max_error = 2.0  # Maximum error to respond to
        error = np.clip(error, -max_error, max_error)
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term with anti-windup and clamping
        integral_update = error * self.Kt
        self.integral += integral_update * self.sample_time
        
        # Very tight integral clamping
        max_integral = 10.0  # Small integral contribution
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        
        I = self.Kp * self.integral / self.Ti
        
        # Calculate raw output
        raw_output = self.last_output + P + I
        
        # Apply hard limits
        clamped_output = np.clip(raw_output, self.actuator_limits[0], self.actuator_limits[1])
        
        # Rate limiting to prevent duty limit violations
        output_change = clamped_output - self.last_output
        max_change = self.max_output_rate * self.sample_time * 100.0  # Convert to percent
        
        if abs(output_change) > max_change:
            clamped_output = self.last_output + np.sign(output_change) * max_change
        
        # Store output for rate monitoring
        self.output_buffer[self.buffer_index] = clamped_output
        self.buffer_index = (self.buffer_index + 1) % len(self.output_buffer)
        
        # Check for duty limit violation (excessive movement)
        output_variance = np.var(self.output_buffer)
        if output_variance > 0.0001:  # If too much movement
            # Further reduce output change
            clamped_output = self.last_output + 0.5 * (clamped_output - self.last_output)
        
        # Additional smoothing
        final_output = 0.9 * clamped_output + 0.1 * self.last_output
        
        # Final safety clamp with margins
        final_output = np.clip(final_output, 
                              self.actuator_limits[0] + 5.0,  # Stay away from limits
                              self.actuator_limits[1] - 5.0)
        
        # Store for next step
        self.last_output = final_output
        self.last_error = error
        self.last_setpoint = setpoint
        
        return np.array([final_output])
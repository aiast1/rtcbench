import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.actuator_limits = (0.0, 10.0)
        self.level_limits = (0.0, 20.0)
        
        # Conservative PID parameters - safety violations indicate too aggressive control
        self.Kp = np.array([0.6, 0.6])
        self.Ki = np.array([0.1, 0.1])
        self.Kd = np.array([0.02, 0.02])
        
        # Derivative filter time constant
        self.tau_d = 3.0
        
        # Safety margins for unmeasured tanks
        self.safety_margin = 2.0  # Keep levels 2cm from limits
        
        # Initialize state
        self.reset()
        
    def reset(self):
        # Integral terms
        self.integral = np.array([0.0, 0.0])
        
        # Previous values
        self.prev_error = np.array([0.0, 0.0])
        self.prev_measurement = np.array([0.0, 0.0])
        self.prev_output = np.array([3.0, 3.0])
        
        # Filtered derivative states
        self.filtered_derivative = np.array([0.0, 0.0])
        
        # Anti-windup tracking
        self.saturated = np.array([False, False])
        
        # Safety monitoring
        self.safety_override_active = False
        self.safety_override_output = np.array([3.0, 3.0])
        
    def step(self, t, y, r, quality):
        # Handle bad measurements - use last good value
        if not quality[0]:
            y = y.copy()
            y[0] = self.prev_measurement[0]
        if not quality[1]:
            y = y.copy()
            y[1] = self.prev_measurement[1]
        
        # Calculate errors
        error = np.zeros(2)
        if not np.isnan(r[0]):
            error[0] = r[0] - y[0]
        if not np.isnan(r[1]):
            error[1] = r[1] - y[1]
        
        # SAFETY CHECK: Prevent unmeasured tank overflows
        # Based on four-tank dynamics: high h1/h2 can cause h3/h4 to overflow
        # Conservative approach: keep measured levels well within limits
        safety_override = False
        
        # Check if measured levels are approaching limits
        if y[0] > self.level_limits[1] - self.safety_margin:
            safety_override = True
            # Reduce pump 1 to lower tank 1 and indirectly tank 4
            error[0] = min(error[0], 0)  # Only allow negative errors (reduce level)
            
        if y[1] > self.level_limits[1] - self.safety_margin:
            safety_override = True
            # Reduce pump 2 to lower tank 2 and indirectly tank 3
            error[1] = min(error[1], 0)  # Only allow negative errors (reduce level)
            
        if y[0] < self.safety_margin:
            safety_override = True
            # Increase pump 1 to raise tank 1
            error[0] = max(error[0], 0)  # Only allow positive errors (increase level)
            
        if y[1] < self.safety_margin:
            safety_override = True
            # Increase pump 2 to raise tank 2
            error[1] = max(error[1], 0)  # Only allow positive errors (increase level)
        
        # Proportional term with safety-aware gain reduction
        if safety_override:
            safety_Kp = self.Kp * 0.5  # Reduce gain when near limits
            P = safety_Kp * error
        else:
            P = self.Kp * error
        
        # Integral term with conditional integration and clamping
        for i in range(2):
            if not self.saturated[i]:
                # Limit integral growth based on safety
                integral_increment = self.Ki[i] * error[i] * self.sample_time
                
                # Clamp integral term to prevent windup
                max_integral = 2.0  # Maximum integral contribution
                if abs(self.integral[i] + integral_increment) > max_integral:
                    integral_increment = np.sign(integral_increment) * max_integral - self.integral[i]
                
                self.integral[i] += integral_increment
        
        # Derivative term - filtered measurement rate
        measurement_rate = (y - self.prev_measurement) / self.sample_time
        
        # Filter the derivative
        alpha = self.sample_time / (self.tau_d + self.sample_time)
        self.filtered_derivative = alpha * measurement_rate + (1 - alpha) * self.filtered_derivative
        
        D = -self.Kd * self.filtered_derivative
        
        # Combine terms
        output = P + self.integral + D + np.array([3.0, 3.0])
        
        # Apply output limits
        output_clipped = np.clip(output, self.actuator_limits[0], self.actuator_limits[1])
        
        # Update saturation flags
        self.saturated = (output != output_clipped)
        
        # Apply very conservative rate limiting
        max_change = 0.008 * self.sample_time  # Very conservative: 0.008 vs 0.017 limit
        for i in range(2):
            change = output_clipped[i] - self.prev_output[i]
            if abs(change) > max_change:
                output_clipped[i] = self.prev_output[i] + np.sign(change) * max_change
        
        # Additional safety: if levels are very close to limits, hold at safe output
        if y[0] > self.level_limits[1] - 1.0 or y[1] > self.level_limits[1] - 1.0:
            # Reduce both pumps when any level is very high
            output_clipped = np.maximum(output_clipped - 0.5, self.actuator_limits[0])
        
        if y[0] < 1.0 or y[1] < 1.0:
            # Increase both pumps when any level is very low
            output_clipped = np.minimum(output_clipped + 0.5, self.actuator_limits[1])
        
        # Update state
        self.prev_error = error.copy()
        self.prev_measurement = y.copy()
        self.prev_output = output_clipped.copy()
        
        return output_clipped
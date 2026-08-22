import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        
        # Tuned controller parameters - more aggressive for better tracking
        self.Kp = np.array([3.5, 3.5])  # Increased proportional gains
        self.Ki = np.array([0.02, 0.02])  # Increased integral gains for faster offset removal
        self.Kd = np.array([0.05, 0.05])  # Reduced derivative gains (less noise sensitivity)
        
        # Filter time constant for derivative action
        self.tau_f = 15.0  # Increased for better noise filtering
        
        # Actuator limits
        self.u_min = np.array([1.5, 2.0])
        self.u_max = np.array([4.5, 5.0])
        
        # Initial steady-state values
        self.u0 = np.array([2.70629, 3.20629])
        
        # Safety margins for unmeasured variables
        self.D_min = 0.05
        self.B_min = 0.05
        self.yD_min = 0.85
        self.xB_max = 0.15
        
        # Store for anti-windup and filtering
        self.integral = None
        self.prev_error = None
        self.prev_measurement = None
        self.filtered_derivative = None
        self.prev_u = None
        self.prev_quality = None
        
        # For rate limiting
        self.max_rate = 0.003  # Maximum allowed rate of change per step
        
    def reset(self):
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.prev_measurement = np.array([0.99, 0.99])  # Initial setpoint
        self.filtered_derivative = np.zeros(2)
        self.prev_u = self.u0.copy()
        self.prev_quality = np.array([True, True])
        
    def step(self, t, y, r, quality):
        # Handle bad measurements by using filtered values
        current_y = y.copy()
        for i in range(2):
            if not quality[i]:
                # Use previous good measurement if current is bad
                if self.prev_quality[i]:
                    current_y[i] = self.prev_measurement[i]
                else:
                    # If consecutive bad readings, use setpoint as fallback
                    current_y[i] = r[i] if not np.isnan(r[i]) else 0.99
            else:
                # Apply simple low-pass filter to reduce noise
                alpha = 0.3  # Filter coefficient
                current_y[i] = alpha * y[i] + (1 - alpha) * self.prev_measurement[i]
        
        # Calculate errors
        error = r - current_y
        
        # Initialize if first step
        if self.prev_error is None:
            self.prev_error = error.copy()
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term with conditional integration and anti-windup
        # Only integrate when error is small to prevent overshoot
        integration_enabled = np.abs(error) < 0.02  # Only integrate near setpoint
        
        for i in range(2):
            if integration_enabled[i]:
                self.integral[i] += self.Ki[i] * error[i] * self.sample_time
            else:
                # Reset integral when far from setpoint to prevent windup
                self.integral[i] *= 0.9
        
        # Clamp integral to prevent windup
        for i in range(2):
            max_integral = (self.u_max[i] - self.u0[i]) / max(self.Ki[i], 1e-6)
            min_integral = (self.u_min[i] - self.u0[i]) / max(self.Ki[i], 1e-6)
            self.integral[i] = np.clip(self.integral[i], min_integral, max_integral)
        
        I = self.integral
        
        # Filtered derivative term with improved noise handling
        derivative = (error - self.prev_error) / self.sample_time
        
        # Apply stronger filtering to derivative
        alpha_d = self.sample_time / (self.tau_f + self.sample_time)
        self.filtered_derivative = (1 - alpha_d) * self.filtered_derivative + alpha_d * derivative
        
        # Only use derivative when error is changing slowly
        derivative_gain_factor = np.exp(-10.0 * np.abs(derivative))
        D = self.Kd * self.filtered_derivative * derivative_gain_factor
        
        # Compute raw control action
        u_raw = self.u0 + P + I + D
        
        # Apply hard limits
        u_clamped = np.clip(u_raw, self.u_min, self.u_max)
        
        # Safety constraints for unmeasured variables
        L = u_clamped[0]
        V = u_clamped[1]
        
        # Calculate unmeasured variables
        D_flow = V - L  # Distillate flow
        B_flow = L + 1.0 - V  # Bottoms flow (F = 1.0 kmol/min)
        
        # Estimate purities with more conservative margins
        # Use linear approximation but with safety buffer
        yD_est = 0.99 + 0.08 * (L - 2.70629) - 0.08 * (V - 3.20629)
        xB_est = 0.01 - 0.08 * (L - 2.70629) + 0.08 * (V - 3.20629)
        
        # Check safety constraints with conservative margins
        safety_violation = False
        
        # Add safety margins to constraints
        if D_flow < self.D_min * 1.2:  # 20% safety margin
            safety_violation = True
            # Adjust to increase distillate flow
            delta = (self.D_min * 1.2 - D_flow) / 2.0
            u_clamped[1] += delta  # Increase V
            u_clamped[0] -= delta * 0.5  # Decrease L slightly
        
        if B_flow < self.B_min * 1.2:  # 20% safety margin
            safety_violation = True
            # Adjust to increase bottoms flow
            delta = (self.B_min * 1.2 - B_flow) / 2.0
            u_clamped[0] += delta  # Increase L
            u_clamped[1] -= delta * 0.5  # Decrease V slightly
        
        if yD_est < self.yD_min * 1.02:  # 2% safety margin
            safety_violation = True
            # Adjust to increase distillate purity
            delta = (self.yD_min * 1.02 - yD_est) / 0.08
            u_clamped[0] += delta * 0.3
            u_clamped[1] -= delta * 0.3
        
        if xB_est > self.xB_max * 0.98:  # 2% safety margin
            safety_violation = True
            # Adjust to decrease bottoms impurity
            delta = (xB_est - self.xB_max * 0.98) / 0.08
            u_clamped[1] += delta * 0.3
            u_clamped[0] -= delta * 0.3
        
        # Re-clamp after safety adjustments
        u_clamped = np.clip(u_clamped, self.u_min, self.u_max)
        
        # Rate limiting with careful handling
        delta_u = u_clamped - self.prev_u
        delta_u_norm = np.abs(delta_u)
        
        # Apply rate limiting
        if np.any(delta_u_norm > self.max_rate):
            scale = self.max_rate / np.max(delta_u_norm)
            delta_u = delta_u * scale
        
        u_final = self.prev_u + delta_u
        
        # Final clamp to ensure within bounds
        u_final = np.clip(u_final, self.u_min, self.u_max)
        
        # Update stored values
        self.prev_error = error.copy()
        self.prev_measurement = current_y.copy()
        self.prev_u = u_final.copy()
        self.prev_quality = quality.copy()
        
        return u_final
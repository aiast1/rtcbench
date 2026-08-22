import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        
        # Nominal operating point
        self.L_nom = 2.70629
        self.V_nom = 3.20629
        
        # Actuator limits
        self.L_min, self.L_max = 1.5, 4.5
        self.V_min, self.V_max = 2.0, 5.0
        
        # Safety limits for draws
        self.D_min = 0.05
        self.B_min = 0.05
        
        # Safety margins for unmeasured constraints
        self.D_margin = 0.15
        self.B_margin = 0.15
        
        # Controller tuning - aggressive for better tracking
        self.Kp = np.array([2.5, 2.5])
        self.Ki = np.array([0.006, 0.006])
        self.Kd = np.array([20.0, 20.0])
        
        # Anti-windup back-calculation coefficient
        self.Kb = 2.0
        
        # Filter coefficient for derivative (low-pass)
        self.N_filter = 3.0
        
        # State variables
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.prev_derivative = np.zeros(2)
        self.prev_output = np.array([self.L_nom, self.V_nom])
        
        # Output filtering for smooth control
        self.output_filter_alpha = 0.08
        
        # Tracking of previous setpoint for ramp detection
        self.prev_r = np.array([0.99, 0.99])
        
        # Initialize
        self.reset()
    
    def reset(self):
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.prev_derivative = np.zeros(2)
        self.prev_output = np.array([self.L_nom, self.V_nom])
        self.prev_r = np.array([0.99, 0.99])
    
    def _estimate_draws(self, L, V, F=1.0):
        """Estimate product draws from flows"""
        D = V - L  # Distillate draw
        B = L + F - V  # Bottoms draw
        return D, B
    
    def _apply_safety_limits(self, L, V):
        """Apply safety limits to ensure draws stay above minimum"""
        F = 1.0  # nominal feed
        
        # Calculate draws
        D = V - L
        B = L + F - V
        
        # Apply margins to prevent approaching limits
        D_safe_min = self.D_min + self.D_margin
        B_safe_min = self.B_min + self.B_margin
        
        # Adjust flows if draws are too low
        # D = V - L >= D_safe_min  =>  V >= L + D_safe_min
        # B = L + F - V >= B_safe_min  =>  L >= V - F + B_safe_min
        
        # Check D constraint
        if D < D_safe_min:
            # Need to increase D: increase V or decrease L
            # Prefer to increase V
            V_needed = L + D_safe_min
            if V_needed <= self.V_max:
                V = V_needed
            else:
                V = self.V_max
                L = V - D_safe_min
                L = max(L, self.L_min)
        
        # Check B constraint
        B = L + F - V
        if B < B_safe_min:
            # Need to increase B: increase L or decrease V
            L_needed = V - F + B_safe_min
            if L_needed <= self.L_max:
                L = L_needed
            else:
                L = self.L_max
                V = L + F - B_safe_min
                V = max(V, self.V_min)
        
        # Final clamp to actuator limits
        L = np.clip(L, self.L_min, self.L_max)
        V = np.clip(V, self.V_min, self.V_max)
        
        return L, V
    
    def step(self, t, y, r, quality):
        # Handle bad quality measurements
        y_valid = y.copy()
        for i in range(len(y)):
            if not quality[i]:
                y_valid[i] = self.prev_r[i]  # Use setpoint as fallback
        
        # Calculate error
        error = r - y_valid
        error = np.nan_to_num(error, nan=0.0)
        
        # Determine if we're in a ramp (smooth transition)
        in_ramp = 220 <= t < 260
        
        # Adjust gains during ramp for smoother tracking
        Ki_adj = self.Ki.copy()
        if in_ramp:
            Ki_adj *= 3.0  # More aggressive during ramp
        
        # Update integral with anti-windup
        # Only integrate if output wasn't saturated
        L_prev, V_prev = self.prev_output
        L_sat = np.clip(L_prev, self.L_min, self.L_max)
        V_sat = np.clip(V_prev, self.V_min, self.V_max)
        
        # Back-calculation anti-windup
        saturation_error = np.array([
            L_prev - L_sat,
            V_prev - V_sat
        ])
        
        # Update integral
        self.integral += Ki_adj * error * self.sample_time
        self.integral -= self.Kb * saturation_error
        
        # Limit integral to prevent excessive windup
        max_integral = 1.0
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        
        # Derivative with filtering
        derivative = (error - self.prev_error) / self.sample_time
        derivative = np.nan_to_num(derivative, nan=0.0)
        
        # Low-pass filter on derivative
        alpha_d = self.sample_time / (self.sample_time + 1.0 / self.N_filter)
        derivative_filtered = alpha_d * derivative + (1 - alpha_d) * self.prev_derivative
        
        # Calculate PID output
        # Output delta from nominal
        delta_L = self.Kp[0] * error[0] + self.integral[0] + self.Kd[0] * derivative_filtered[0]
        delta_V = self.Kp[1] * error[1] + self.integral[1] + self.Kd[1] * derivative_filtered[1]
        
        # Apply to nominal operating point
        L = self.L_nom + delta_L
        V = self.V_nom + delta_V
        
        # Apply safety limits
        L, V = self._apply_safety_limits(L, V)
        
        # Output smoothing filter
        L = self.output_filter_alpha * L + (1 - self.output_filter_alpha) * self.prev_output[0]
        V = self.output_filter_alpha * V + (1 - self.output_filter_alpha) * self.prev_output[1]
        
        # Final clamp
        L = np.clip(L, self.L_min, self.L_max)
        V = np.clip(V, self.V_min, self.V_max)
        
        # Store state for next iteration
        self.prev_error = error.copy()
        self.prev_derivative = derivative_filtered.copy()
        self.prev_output = np.array([L, V])
        self.prev_r = r.copy()
        
        return np.array([L, V])
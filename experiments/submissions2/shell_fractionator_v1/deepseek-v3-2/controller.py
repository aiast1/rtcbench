import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        
        # FIXED: Access model_hint correctly from brief
        # The brief has model_hint as an attribute, not a key
        model_hint = brief.model_hint
        
        # Extract model parameters with safe defaults
        self.K = np.array(model_hint.get("K", [[4.05, 1.77, 5.88], [5.39, 5.72, 6.9], [4.38, 4.42, 7.2]]), dtype=np.float64)
        self.tau = np.array(model_hint.get("TAU", [[50.0, 60.0, 50.0], [50.0, 60.0, 40.0], [33.0, 44.0, 19.0]]), dtype=np.float64)
        self.L = np.array(model_hint.get("L", [[27.0, 28.0, 27.0], [18.0, 14.0, 15.0], [20.0, 22.0, 0.0]]), dtype=np.float64)
        self.Kd = np.array(model_hint.get("KD", [[1.2, 1.44], [1.52, 1.83], [1.14, 1.26]]), dtype=np.float64)
        self.taud = np.array(model_hint.get("TAUD", [[45.0, 40.0], [25.0, 20.0], [27.0, 32.0]]), dtype=np.float64)
        self.Ld = np.array(model_hint.get("LD", [[27.0, 27.0], [15.0, 15.0], [27.0, 32.0]]), dtype=np.float64)
        
        # Controller dimensions
        self.n_y = 3  # measurements
        self.n_u = 3  # actuators
        self.n_d = 2  # disturbances
        
        # Initialize controller parameters
        self._initialize_controller()
        
        # Initialize state variables
        self.reset()
    
    def _initialize_controller(self):
        """Initialize PID controller parameters with anti-windup."""
        # Conservative PID tuning based on nominal model
        # Using lambda tuning method (SIMC) for robustness
        lambda_factor = 2.0  # Robustness factor
        
        # Initialize PID parameters for each output
        self.kp = np.zeros((self.n_u, self.n_y))
        self.ki = np.zeros((self.n_u, self.n_y))
        self.kd = np.zeros((self.n_u, self.n_y))
        
        # Decentralized PID tuning - each output controlled by one input
        # y0 (top composition) controlled by u0 (top draw)
        # y1 (side composition) controlled by u1 (side draw)
        # y2 (temperature) controlled by u2 (bottoms reflux)
        
        # Pairings based on relative gain analysis (simplified)
        # Using diagonal pairing
        for i in range(self.n_y):
            # Use the diagonal process gain
            Kp = self.K[i, i]
            tau = self.tau[i, i]
            L = self.L[i, i]
            
            if tau > 0 and L > 0:
                # SIMC PI tuning rules
                tau_c = max(lambda_factor * L, tau/10)
                self.kp[i, i] = tau / (Kp * tau_c)
                self.ki[i, i] = self.kp[i, i] / min(tau, 4*tau_c)
            else:
                # Conservative default
                self.kp[i, i] = 0.5
                self.ki[i, i] = 0.1
        
        # Add small cross-coupling for better MIMO performance
        for i in range(self.n_y):
            for j in range(self.n_u):
                if i != j and abs(self.K[i, j]) > 0:
                    self.kp[i, j] = 0.1 * self.kp[i, i] * np.sign(self.K[i, j])
        
        # Anti-windup tracking time constant
        self.taw = 10.0
        
        # Derivative filter time constant
        self.tau_f = 2.0  # Increased for better noise filtering
        
        # Initialize disturbance compensation
        self.dist_comp_gain = np.zeros((self.n_u, self.n_d))
        for i in range(self.n_u):
            for j in range(self.n_d):
                if self.taud[i, j] > 0:
                    self.dist_comp_gain[i, j] = -self.Kd[i, j] / self.K[i, i] if abs(self.K[i, i]) > 1e-6 else 0
        
        # Setpoint filter time constant
        self.tau_sp = 5.0
        
        # Control limits
        self.u_min = np.array([-0.5, -0.5, -0.5])
        self.u_max = np.array([0.5, 0.5, 0.5])
        
        # Rate limits (per sample) - from brief: actuator duty limit of 0.016
        self.rate_limit = 0.016
        
        # Safety constraint for y[2] (temperature) - from brief: y3 must stay within [-0.5, inf]
        self.y2_min = -0.5
        
        # Initialize filtered values
        self.prev_u = np.zeros(self.n_u)
        self.prev_y = np.zeros(self.n_y)
        self.prev_r = np.zeros(self.n_y)
        self.prev_d = np.zeros(self.n_d)
        
        # Store previous actuator values for rate limiting
        self.last_u = np.zeros(self.n_u)
    
    def reset(self):
        """Reset controller state."""
        # Integral terms
        self.integral = np.zeros(self.n_y)
        
        # Previous errors for derivative action
        self.prev_error = np.zeros(self.n_y)
        
        # Previous measurements for filtering
        self.prev_ym = np.zeros(self.n_y)
        
        # Filtered setpoints
        self.filtered_r = np.zeros(self.n_y)
        
        # Filtered measurements
        self.filtered_y = np.zeros(self.n_y)
        
        # Derivative filter states
        self.deriv_state = np.zeros(self.n_y)
        
        # Anti-windup tracking states
        self.aw_state = np.zeros(self.n_u)
        
        # Previous control output
        self.prev_output = np.array([0.0, 0.0, 0.0])
        
        # Disturbance estimate
        self.dist_est = np.zeros(self.n_d)
        
        # Time since last good measurement
        self.bad_sample_count = np.zeros(self.n_y, dtype=int)
        
        # Initialize filtered values
        self.prev_u = np.zeros(self.n_u)
        self.prev_y = np.zeros(self.n_y)
        self.prev_r = np.zeros(self.n_y)
        self.prev_d = np.zeros(self.n_d)
        
        # Store previous actuator values for rate limiting
        self.last_u = np.array([0.0, 0.0, 0.0])
        
        # Safety monitoring
        self.safety_violation = False
        
        # Initialize with bumpless start - from brief: actuators start at [0.0, 0.0, 0.0]
        self.last_u = np.array([0.0, 0.0, 0.0])
    
    def step(self, t, y, r, quality):
        """Main control step."""
        # Handle bad measurements
        y_processed = self._handle_bad_measurements(y, quality)
        
        # Filter measurements
        alpha = self.sample_time / (self.tau_f + self.sample_time)
        self.filtered_y = alpha * y_processed + (1 - alpha) * self.filtered_y
        
        # Filter setpoints
        alpha_sp = self.sample_time / (self.tau_sp + self.sample_time)
        # Handle NaN setpoints - use previous filtered value
        r_valid = np.where(np.isnan(r), self.filtered_r, r)
        self.filtered_r = alpha_sp * r_valid + (1 - alpha_sp) * self.filtered_r
        
        # Calculate errors
        error = self.filtered_r - self.filtered_y
        
        # Update integral terms with anti-windup
        self._update_integral(error)
        
        # Calculate derivative term
        deriv = self._calculate_derivative(error)
        
        # Compute preliminary control action
        u_prelim = self._compute_control_action(error, deriv)
        
        # Apply safety constraint for temperature
        u_prelim = self._apply_safety_constraint(u_prelim, y_processed[2])
        
        # Apply rate limiting
        u_limited = self._apply_rate_limits(u_prelim)
        
        # Apply saturation with anti-windup tracking
        u_sat = np.clip(u_limited, self.u_min, self.u_max)
        
        # Update anti-windup states
        self.aw_state += (u_sat - u_limited) / self.taw
        
        # Store for next iteration
        self.prev_error = error.copy()
        self.prev_ym = y_processed.copy()
        self.prev_output = u_sat.copy()
        self.last_u = u_sat.copy()
        
        return u_sat
    
    def _handle_bad_measurements(self, y, quality):
        """Handle bad or stale measurements."""
        y_processed = y.copy()
        
        for i in range(self.n_y):
            if not quality[i]:
                # Bad measurement - use previous filtered value
                self.bad_sample_count[i] += 1
                if self.bad_sample_count[i] == 1:
                    # First bad sample, use last good measurement
                    y_processed[i] = self.filtered_y[i]
                else:
                    # Multiple bad samples, hold last value
                    y_processed[i] = self.filtered_y[i]
            else:
                # Good measurement
                self.bad_sample_count[i] = 0
        
        return y_processed
    
    def _update_integral(self, error):
        """Update integral terms with anti-windup."""
        for i in range(self.n_y):
            # Only integrate if channel has a setpoint (not NaN)
            if not np.isnan(error[i]):
                # Add anti-windup compensation
                aw_comp = 0.0
                for j in range(self.n_u):
                    aw_comp += self.kp[i, j] * self.aw_state[j]
                
                self.integral[i] += self.ki[i, i] * error[i] * self.sample_time + aw_comp * self.sample_time
                
                # Limit integral term to prevent windup
                int_max = 2.0 / max(abs(self.ki[i, i]), 1e-6)
                self.integral[i] = np.clip(self.integral[i], -int_max, int_max)
    
    def _calculate_derivative(self, error):
        """Calculate filtered derivative term."""
        deriv = np.zeros(self.n_y)
        
        for i in range(self.n_y):
            # Filter derivative term
            error_diff = error[i] - self.prev_error[i]
            self.deriv_state[i] = (error_diff / self.sample_time - self.deriv_state[i]) * self.sample_time / self.tau_f
            
            deriv[i] = self.kd[i, i] * self.deriv_state[i]
        
        return deriv
    
    def _compute_control_action(self, error, deriv):
        """Compute MIMO control action."""
        u = np.zeros(self.n_u)
        
        # Main diagonal control
        for i in range(self.n_u):
            # PID contribution
            pid_contrib = 0.0
            for j in range(self.n_y):
                if not np.isnan(error[j]):
                    pid_contrib += self.kp[i, j] * error[j]
            
            u[i] = pid_contrib + self.integral[i] + deriv[i]
        
        return u
    
    def _apply_safety_constraint(self, u, y_temp):
        """Apply safety constraint for temperature."""
        # If temperature is approaching lower limit, adjust bottoms reflux
        safety_margin = 0.05  # Reduced margin for faster response
        
        # Check if temperature is below minimum
        if y_temp < self.y2_min:
            # Emergency: temperature below minimum, increase reflux significantly
            u[2] = 0.3  # Moderate increase
        elif y_temp < self.y2_min + safety_margin:
            # Warning: temperature approaching minimum
            u[2] = max(u[2], 0.15)  # Ensure minimum reflux
        
        return u
    
    def _apply_rate_limits(self, u):
        """Apply rate limits to control signals."""
        u_limited = u.copy()
        
        for i in range(self.n_u):
            delta = u[i] - self.last_u[i]
            delta_limited = np.clip(delta, -self.rate_limit, self.rate_limit)
            u_limited[i] = self.last_u[i] + delta_limited
        
        return u_limited
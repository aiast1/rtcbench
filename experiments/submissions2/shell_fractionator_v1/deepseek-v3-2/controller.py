import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        
        # Number of inputs, outputs
        self.n_inputs = 3
        self.n_outputs = 3
        
        # Conservative PID tuning - fixed gains for robustness
        # More conservative to prevent safety violations
        self.kp = np.array([0.3, 0.3, 0.4])  # Proportional gains for each loop
        self.ki = np.array([0.01, 0.01, 0.015])  # Integral gains
        self.kd = np.array([0.05, 0.05, 0.08])  # Derivative gains
        
        # State variables
        self.prev_error = None
        self.integral = None
        self.prev_output = None
        self.prev_measurement = None
        self.filtered_measurement = None
        self.last_good_y = None
        self.prev_derivative = None
        self.safety_integral = None
        
        # Filter parameters - more filtering for safety
        self.filter_tau = 15.0  # Conservative measurement filter
        self.filter_alpha = 0.0
        
        # Derivative filter - more filtering
        self.derivative_tau = 8.0
        self.derivative_alpha = 0.0
        
        # Anti-windup parameters
        self.integral_max = 1.5
        self.integral_min = -1.5
        
        # Output constraints
        self.output_min = np.array([-0.5, -0.5, -0.5])
        self.output_max = np.array([0.5, 0.5, 0.5])
        
        # Safety constraint for TI-103 (y[2]) - must stay above -0.5
        self.temp_min = -0.5
        self.temp_safety_margin = 0.1  # Start acting at -0.4
        
        # Rate limiting - more conservative
        self.max_rate = 0.015
        
        # Safety controller parameters
        self.safety_kp = 0.8
        self.safety_ki = 0.05
        
        # Store setpoint schedule for reference
        self.setpoint_schedule = getattr(brief, 'setpoint_schedule', [])
        
    def reset(self):
        """Reset controller state for new scenario"""
        self.prev_error = np.zeros(self.n_outputs)
        self.integral = np.zeros(self.n_outputs)
        self.prev_output = np.zeros(self.n_inputs)
        self.prev_measurement = np.zeros(self.n_outputs)
        self.filtered_measurement = np.zeros(self.n_outputs)
        self.last_good_y = np.zeros(self.n_outputs)
        self.prev_derivative = np.zeros(self.n_outputs)
        self.safety_integral = 0.0
        
        # Calculate filter constants
        self.filter_alpha = np.exp(-self.sample_time / self.filter_tau)
        self.derivative_alpha = np.exp(-self.sample_time / self.derivative_tau)
        
        # Initialize outputs at steady state (bumpless transfer)
        self.prev_output[:] = 0.0
        
    def step(self, t, y, r, quality):
        """Main control step"""
        # Handle bad measurements - use last good value
        current_y = np.copy(y)
        for i in range(self.n_outputs):
            if not quality[i]:
                if self.last_good_y is not None:
                    current_y[i] = self.last_good_y[i]
                else:
                    current_y[i] = 0.0
            else:
                self.last_good_y[i] = y[i]
        
        # Apply measurement filtering
        if self.prev_measurement is not None:
            for i in range(self.n_outputs):
                self.filtered_measurement[i] = (self.filter_alpha * self.filtered_measurement[i] + 
                                               (1 - self.filter_alpha) * current_y[i])
        else:
            self.filtered_measurement = np.copy(current_y)
        
        # Calculate errors (only for scored channels)
        error = np.zeros(self.n_outputs)
        for i in range(self.n_outputs):
            if not np.isnan(r[i]):
                error[i] = r[i] - self.filtered_measurement[i]
            else:
                error[i] = 0.0
        
        # Initialize on first step
        if self.prev_error is None:
            self.prev_error = np.copy(error)
            self.integral = np.zeros(self.n_outputs)
            self.prev_derivative = np.zeros(self.n_outputs)
        
        # Update integral with conditional integration
        for i in range(self.n_outputs):
            # Only integrate if not near saturation and error is reasonable
            if abs(error[i]) < 0.3 and abs(self.integral[i]) < self.integral_max * 0.8:
                self.integral[i] += self.ki[i] * error[i] * self.sample_time
            
            # Clamp integral
            self.integral[i] = np.clip(self.integral[i], self.integral_min, self.integral_max)
        
        # Calculate derivative with filtering
        derivative = np.zeros(self.n_outputs)
        if self.prev_error is not None:
            raw_derivative = (error - self.prev_error) / self.sample_time
            
            for i in range(self.n_outputs):
                derivative[i] = (self.derivative_alpha * self.prev_derivative[i] + 
                               (1 - self.derivative_alpha) * raw_derivative[i])
        
        # Calculate PID output for each loop (diagonal control)
        u_pid = np.zeros(self.n_inputs)
        
        for i in range(self.n_inputs):
            if i < self.n_outputs:
                u_pid[i] = (self.kp[i] * error[i] + 
                          self.integral[i] + 
                          self.kd[i] * derivative[i])
        
        # Apply output constraints
        u = np.clip(u_pid, self.output_min, self.output_max)
        
        # Rate limiting
        if self.prev_output is not None:
            for i in range(self.n_inputs):
                change = u[i] - self.prev_output[i]
                if abs(change) > self.max_rate:
                    u[i] = self.prev_output[i] + np.sign(change) * self.max_rate
        
        # CRITICAL: Safety controller for TI-103 (bottoms reflux temperature)
        # This takes priority over normal control when temperature is low
        temp_measurement = self.filtered_measurement[2]
        
        # Calculate safety error (how far below minimum we are)
        safety_error = max(self.temp_min - temp_measurement, 0)
        
        # If we're getting close to the limit, start acting
        if temp_measurement < (self.temp_min + self.temp_safety_margin):
            # Update safety integral
            self.safety_integral += safety_error * self.sample_time
            self.safety_integral = min(self.safety_integral, 1.0)  # Limit integral
            
            # Calculate safety control action
            safety_action = (self.safety_kp * safety_error + 
                           self.safety_ki * self.safety_integral)
            
            # Apply safety action to bottoms reflux (actuator 2)
            # This OVERRIDES the normal PID control for this actuator
            u[2] = max(u[2], safety_action)
            
            # If we're in serious violation, use stronger action
            if temp_measurement < self.temp_min:
                # Emergency override - use significant positive action
                u[2] = max(u[2], 0.25)
        
        # Also prevent temperature from getting too high
        if temp_measurement > 0.4:  # Getting too hot
            u[2] = min(u[2], -0.05)  # Reduce reflux
        
        # Ensure we don't violate actuator constraints
        u = np.clip(u, self.output_min, self.output_max)
        
        # Additional safety: if temperature is critically low, use all actuators
        if temp_measurement < (self.temp_min - 0.1):  # Well below limit
            # Use top and side draws to help (reduce them to increase temperature)
            u[0] = max(u[0], -0.2)  # Reduce top draw
            u[1] = max(u[1], -0.2)  # Reduce side draw
        
        # Final constraint check
        u = np.clip(u, self.output_min, self.output_max)
        
        # Update state variables
        self.prev_error = np.copy(error)
        self.prev_output = np.copy(u)
        self.prev_measurement = np.copy(current_y)
        self.prev_derivative = np.copy(derivative)
        
        return u
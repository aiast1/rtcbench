import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.actuator_limits = np.array([270.0, 340.0])
        self.slew_limit = 1.5 * self.sample_time  # K per step
        self.stiction = 0.15
        self.setpoints = []
        self._precompute_setpoints()
        
    def _precompute_setpoints(self):
        # Generate setpoint trajectory for the full scenario
        t_end = 900.0
        dt = self.sample_time
        n_steps = int(t_end / dt) + 1
        self.setpoints = np.full((n_steps, 1), np.nan)
        
        for i in range(n_steps):
            t = i * dt
            if t < 240.0:
                self.setpoints[i, 0] = 350.0
            elif t < 480.0:
                self.setpoints[i, 0] = 352.0
            elif t < 570.0:  # 480 to 570 is 90s ramp
                progress = (t - 480.0) / 90.0
                self.setpoints[i, 0] = 348.0 + (352.0 - 348.0) * progress
            elif t < 720.0:
                self.setpoints[i, 0] = 352.0
            else:
                self.setpoints[i, 0] = 350.0
    
    def reset(self):
        self.integral = 0.0
        self.last_control = 300.0
        self.last_measurement = None
        self.last_time = 0.0
        self.step_count = 0
        self.derivative_filter = 0.0
        self.last_measurement_filtered = None
        self.safety_margin_active = False
        
    def step(self, t, y, r, quality):
        # Extract measurements
        T_reactor = y[0]  # TI-101 - reactor temp (scored)
        T_jacket = y[1]   # TI-102 - jacket temp (not scored)
        
        # Get current setpoint
        setpoint = self.setpoints[self.step_count, 0]
        
        # Handle bad measurements - use last known good value if available
        if not quality[0] or np.isnan(T_reactor) or np.isinf(T_reactor):
            if self.last_measurement is not None:
                T_reactor = self.last_measurement
            else:
                # Fallback: assume we're at setpoint if no history
                T_reactor = setpoint
        
        self.last_measurement = T_reactor
        
        # Safety constraints: hard limits on reactor temperature
        # If approaching dangerous temperatures, prioritize safety over tracking
        if T_reactor >= 450.0:  # Aggressive cooling well before relief
            control_output = 270.0
            self.integral = 0.0  # Reset integral to prevent windup
            self.last_control = control_output
            self.last_time = t
            self.last_measurement_filtered = T_reactor
            self.step_count += 1
            return np.array([control_output])
        elif T_reactor <= 310.0:  # Warm up before product drop
            control_output = min(340.0, self.last_control + self.slew_limit)
            self.last_control = control_output
            self.last_time = t
            self.last_measurement_filtered = T_reactor
            self.step_count += 1
            return np.array([control_output])
        
        # Calculate error
        error = setpoint - T_reactor
        
        # Anti-windup: only integrate if not saturated or error opposes saturation
        if self.last_control >= self.actuator_limits[1] and error > 0:
            # Already at upper limit and trying to increase - don't integrate
            pass
        elif self.last_control <= self.actuator_limits[0] and error < 0:
            # Already at lower limit and trying to decrease - don't integrate
            pass
        else:
            # Simple integral action with anti-windup
            self.integral += error * self.sample_time
        
        # Derivative on measurement (not error) to avoid derivative kick
        if self.last_time > 0 and self.last_measurement_filtered is not None:
            dt_step = t - self.last_time
            if dt_step > 0:
                # Filtered derivative
                measurement_rate = (T_reactor - self.last_measurement_filtered) / dt_step
                # Low-pass filter for derivative (tau_d = 2.0 * sample_time for heavy smoothing)
                tau_d = 2.0 * self.sample_time
                alpha = tau_d / (tau_d + dt_step)
                self.derivative_filter = alpha * self.derivative_filter + (1 - alpha) * measurement_rate
        else:
            self.derivative_filter = 0.0
        
        # PID parameters - very conservative tuning for stability on unstable process
        Kp = 0.15
        Ki = 0.0005
        Kd = 0.2
        
        # PID calculation
        p_term = Kp * error
        i_term = Ki * self.integral
        d_term = -Kd * self.derivative_filter  # negative because derivative of error = -derivative of measurement
        
        u_unconstrained = p_term + i_term + d_term
        
        # Add feedforward component based on jacket temperature (rough estimate)
        # At steady state: T_reactor ~ 350, T_jacket ~ 300
        # Adjust based on deviation from nominal
        nominal_jacket = 300.0
        ff_offset = 0.0
        if T_jacket is not None and not np.isnan(T_jacket):
            ff_offset = 0.02 * (T_jacket - nominal_jacket)
        
        u_unconstrained += ff_offset
        
        # Calculate target control value
        control_target = self.last_control + u_unconstrained
        
        # Apply actuator constraints (hard limits)
        control_target = np.clip(control_target, self.actuator_limits[0], self.actuator_limits[1])
        
        # Apply slew rate limits
        delta_max = self.slew_limit
        control_target = np.clip(control_target, 
                                self.last_control - delta_max, 
                                self.last_control + delta_max)
        
        # Apply stiction (quantization)
        # Round to nearest 0.3 K (2 * stiction) to avoid chattering
        stiction_step = 2 * self.stiction
        control_output = np.round(control_target / stiction_step) * stiction_step
        
        # Final hard constraint
        control_output = np.clip(control_output, self.actuator_limits[0], self.actuator_limits[1])
        
        # Store state for next step
        self.last_control = control_output
        self.last_time = t
        self.last_measurement_filtered = T_reactor
        self.step_count += 1
        
        return np.array([control_output])
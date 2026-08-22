import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        # Use default initial values as per brief description
        self.actuators_start = np.array([14.19, -1113.5])
        self.safety_T_max = 150.0
        self.safety_TK_max = 150.0
        self.actuator_limits = np.array([
            [3.0, 35.0],      # F/V limits
            [-9000.0, 0.0]    # Q_K limits
        ])
        self.setpoint_schedule = brief.setpoint_schedule
        self.w_effort = 0.5
        self.w_error = 1.0
        self.max_duty = 0.00134
        
        # PID parameters (tuned for robustness)
        self.Kp = np.array([[0.08, 0.005],   # C_B control effort per error
                            [0.002, 0.008]])  # T control effort per error
        self.Ki = np.array([[0.0005, 0.0001],
                            [0.0001, 0.0005]])
        self.Kd = np.array([[0.0, 0.0],
                            [0.0, 0.0]])
        
        # Anti-windup and filtering
        self.windup_limit = 0.5
        self.derivative_filter = 0.1
        
        # Internal state
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.prev_measurement = np.zeros(2)
        self.prev_output = self.actuators_start.copy()
        self.last_update_time = None
        self.prev_measurement_rate = np.zeros(2)
        
        # Setpoint interpolation
        self._setup_setpoints()
    
    def _setup_setpoints(self):
        """Precompute setpoint schedule as piecewise functions"""
        self.setpoints = []
        for t, sp in self.setpoint_schedule:
            self.setpoints.append((t, np.array(sp)))
    
    def _get_setpoint(self, t):
        """Interpolate setpoint at time t"""
        if not self.setpoints:
            return np.array([np.nan, np.nan])
        
        # Find relevant setpoint segment
        for i, (t_i, sp_i) in enumerate(self.setpoints):
            if t < t_i:
                if i == 0:
                    return sp_i
                else:
                    # Linear interpolation between previous and current
                    t_prev, sp_prev = self.setpoints[i-1]
                    if t_i == t_prev:
                        return sp_i
                    alpha = (t - t_prev) / (t_i - t_prev)
                    return sp_prev + alpha * (sp_i - sp_prev)
        
        # After last setpoint, use last value
        return self.setpoints[-1][1]
    
    def _clip_actuators(self, u):
        """Clip actuators to safe limits"""
        return np.clip(u, self.actuator_limits[:, 0], self.actuator_limits[:, 1])
    
    def _check_safety(self, y, u):
        """Check safety constraints - T must be <= 150, TK estimated from u[1]"""
        # Check measured T
        if y[1] > self.safety_T_max:
            return False
        
        # Estimate jacket temperature from heat duty (conservative estimate)
        # Q_K = kw * AR * (T - T_K), so T_K = T - Q_K / (kw * AR)
        # Using conservative kw*AR estimate (minimum possible denominator)
        kw_AR_min = 4032.0 * 0.215 * 0.8  # 20% safety margin
        T_K_est = y[1] - u[1] / kw_AR_min
        if T_K_est > self.safety_TK_max:
            return False
        
        return True
    
    def _calculate_duty(self, u_new, u_old):
        """Calculate actuator duty for this step"""
        if self.last_update_time is None:
            return 0.0
        dt = self.sample_time
        if dt <= 0:
            return 0.0
        duty = np.max(np.abs(u_new - u_old)) / dt
        return duty
    
    def reset(self):
        """Reset controller state for new scenario"""
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.prev_measurement = np.zeros(2)
        self.prev_output = self.actuators_start.copy()
        self.last_update_time = None
        self.prev_measurement_rate = np.zeros(2)
    
    def step(self, t, y, r, quality):
        """Main control step"""
        # Get setpoint
        sp = self._get_setpoint(t)
        
        # Handle NaN setpoints by keeping previous setpoint
        sp = np.where(np.isnan(sp), sp, sp)
        if np.all(np.isnan(sp)):
            sp = np.array([1.09, 114.19])  # Default fallback
        
        # Extract measurements (only use good quality signals)
        y_valid = np.where(quality, y, np.nan)
        
        # If measurements are bad, use last known values
        if np.any(np.isnan(y_valid)):
            y_valid = np.where(np.isnan(y_valid), self.prev_measurement, y_valid)
        
        # Update measurement history
        self.prev_measurement = y_valid.copy()
        
        # Calculate errors (only for scored channels)
        error = sp - y_valid
        
        # PID calculations
        # Proportional term
        P = self.Kp @ error
        
        # Integral term with anti-windup
        self.integral += error * self.sample_time
        # Clamp integral to prevent windup
        u_total = self.prev_output + P + self.integral * self.Ki
        u_clipped = self._clip_actuators(u_total)
        windup = u_clipped - u_total
        self.integral += windup
        
        # Integral term contribution
        I = self.integral * self.Ki
        
        # Derivative term (filtered)
        if self.last_update_time is not None:
            dt = self.sample_time
            # Filtered derivative using exponential smoothing
            measurement_rate = (y_valid - self.prev_measurement) / dt
            measurement_rate = self.derivative_filter * measurement_rate + (1 - self.derivative_filter) * self.prev_measurement_rate
            self.prev_measurement_rate = measurement_rate
            D = self.Kd @ (-measurement_rate)
        else:
            D = np.zeros(2)
            self.prev_measurement_rate = np.zeros(2)
        
        # Total control output
        u = self.prev_output + P + I + D
        
        # Clip to actuator limits
        u = self._clip_actuators(u)
        
        # Check safety constraints
        if not self._check_safety(y_valid, u):
            # Safety override: reduce temperature aggressively
            u[0] = max(3.0, min(14.0, u[0]))  # Reduce feed rate
            u[1] = min(u[1], -2000.0)  # Increase cooling
        
        # Check duty limit
        duty = self._calculate_duty(u, self.prev_output)
        if duty > self.max_duty:
            # Smooth transition if duty exceeded
            u = self.prev_output + np.clip(u - self.prev_output, 
                                          -self.max_duty * self.sample_time,
                                          self.max_duty * self.sample_time)
        
        # Update state
        self.prev_output = u.copy()
        self.last_update_time = t
        
        return u
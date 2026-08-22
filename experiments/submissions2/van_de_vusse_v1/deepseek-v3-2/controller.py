import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        
        # Store actuator initial values for bumpless start
        self.u_init = np.array([14.19, -1113.5])
        
        # Advanced PID parameters with gain scheduling
        # Base gains for C_B control
        self.kp_cb_base = 10.0
        self.ki_cb_base = 0.1
        self.kd_cb_base = 0.0
        
        # Base gains for temperature control
        self.kp_temp_base = -70.0
        self.ki_temp_base = -1.0
        self.kd_temp_base = -10.0
        
        # Gain scheduling based on operating point
        self.cb_setpoint_history = []
        self.temp_setpoint_history = []
        
        # Anti-windup with dynamic limits
        self.integral_max = np.array([6.0, 300.0])
        self.integral_min = np.array([-6.0, -300.0])
        
        # Derivative filtering
        self.derivative_filter = 0.3
        
        # Safety parameters
        self.max_temp = 150.0
        self.temp_margin = 15.0
        
        # Setpoint tracking
        self.last_setpoint = np.array([1.09, 114.19])
        self.setpoint_change_threshold = 0.05
        
        # Output rate limiting
        self.max_rate = 0.00134
        self.output_limits = {
            'F/V': {'min': 3.0, 'max': 35.0},
            'Q_K': {'min': -9000.0, 'max': 0.0}
        }
        
        # Adaptive deadband based on measurement quality
        self.deadband_base = np.array([0.005, 0.05])
        
        # State variables
        self.prev_error = None
        self.prev_measurement = None
        self.integral = None
        self.prev_output = None
        self.error_integral = None
        self.control_effort = None
        
        # Model-based compensation (nominal model)
        self.nominal_params = {
            'CA0': 5.1,
            'T0': 104.9
        }
        
    def reset(self):
        # Reset all state variables
        self.prev_error = np.zeros(2)
        self.prev_measurement = np.array([1.09, 114.19])
        self.integral = np.zeros(2)
        self.prev_output = self.u_init.copy()
        self.last_setpoint = np.array([1.09, 114.19])
        self.error_integral = np.zeros(2)
        self.control_effort = np.zeros(2)
        self.cb_setpoint_history = [1.09]
        self.temp_setpoint_history = [114.19]
        
    def calculate_gain_schedule(self, setpoint_cb, setpoint_temp):
        """Adjust gains based on operating point"""
        # For C_B: higher gains at lower concentrations (more sensitive region)
        cb_factor = 1.0 + (1.2 - setpoint_cb) * 0.5  # Higher gain at lower C_B
        
        # For temperature: higher gains near safety limit
        temp_safety = max(0, (setpoint_temp - 110) / 10)  # 0 at 110°C, 1 at 120°C
        temp_factor = 1.0 + temp_safety * 0.3  # Higher gain near limit
        
        kp_cb = self.kp_cb_base * cb_factor
        ki_cb = self.ki_cb_base * cb_factor
        kp_temp = self.kp_temp_base * temp_factor
        ki_temp = self.ki_temp_base * temp_factor
        kd_temp = self.kd_temp_base * temp_factor
        
        return kp_cb, ki_cb, kp_temp, ki_temp, kd_temp
    
    def model_based_feedforward(self, setpoint_cb, setpoint_temp):
        """Simple model-based feedforward compensation"""
        # Based on nominal model relationships
        # Higher C_B setpoint requires lower F/V (residence time)
        ff_cb = -2.5 * (setpoint_cb - 1.0)  # Empirical relationship
        
        # Temperature control: cooling needed increases with temperature
        ff_temp = -50.0 * (setpoint_temp - 110.0)
        
        return np.array([ff_cb, ff_temp])
    
    def step(self, t, y, r, quality):
        # Initialize with previous output
        u = self.prev_output.copy()
        
        # Enhanced measurement filtering
        y_filtered = np.zeros_like(y)
        for i in range(len(y)):
            if not quality[i]:
                # Use prediction if measurement is bad
                y_filtered[i] = self.prev_measurement[i] + 0.1 * self.prev_error[i]
            else:
                # Adaptive filtering based on noise level
                if self.prev_measurement is not None:
                    noise_estimate = abs(y[i] - self.prev_measurement[i])
                    filter_factor = min(0.8, 0.3 + noise_estimate * 5.0)
                    y_filtered[i] = filter_factor * y[i] + (1 - filter_factor) * self.prev_measurement[i]
                else:
                    y_filtered[i] = y[i]
        
        # Update setpoint history for gain scheduling
        if not np.isnan(r[0]):
            self.cb_setpoint_history.append(r[0])
            if len(self.cb_setpoint_history) > 10:
                self.cb_setpoint_history.pop(0)
        if not np.isnan(r[1]):
            self.temp_setpoint_history.append(r[1])
            if len(self.temp_setpoint_history) > 10:
                self.temp_setpoint_history.pop(0)
        
        # Calculate adaptive gains
        avg_cb_sp = np.mean(self.cb_setpoint_history)
        avg_temp_sp = np.mean(self.temp_setpoint_history)
        kp_cb, ki_cb, kp_temp, ki_temp, kd_temp = self.calculate_gain_schedule(avg_cb_sp, avg_temp_sp)
        
        # Calculate errors with adaptive deadband
        error = np.zeros(2)
        adaptive_deadband = self.deadband_base.copy()
        
        # Increase deadband for noisy periods
        if self.prev_measurement is not None:
            measurement_variance = np.abs(y_filtered - self.prev_measurement)
            adaptive_deadband += measurement_variance * 0.5
        
        if not np.isnan(r[0]):
            raw_error = r[0] - y_filtered[0]
            if abs(raw_error) < adaptive_deadband[0]:
                error[0] = 0.0
            else:
                error[0] = raw_error
        if not np.isnan(r[1]):
            raw_error = r[1] - y_filtered[1]
            if abs(raw_error) < adaptive_deadband[1]:
                error[1] = 0.0
            else:
                error[1] = raw_error
        
        # Model-based feedforward
        if not np.isnan(r[0]) and not np.isnan(r[1]):
            feedforward = self.model_based_feedforward(r[0], r[1])
            # Update last setpoint
            self.last_setpoint = r.copy()
        else:
            feedforward = np.zeros(2)
        
        # Enhanced integral action with forgetting factor
        self.integral = 0.95 * self.integral + error * self.sample_time
        
        # Advanced anti-windup: track when control is saturated
        control_saturated = False
        for i in range(2):
            limit_name = 'F/V' if i == 0 else 'Q_K'
            if (u[i] >= self.output_limits[limit_name]['max'] - 0.01 and error[i] > 0) or \
               (u[i] <= self.output_limits[limit_name]['min'] + 0.01 and error[i] < 0):
                control_saturated = True
                # Back-calculate integral to prevent windup
                if i == 0:  # C_B control
                    desired_integral = (u[i] - self.u_init[0] - kp_cb * error[i] - feedforward[0]) / ki_cb
                    self.integral[i] = np.clip(desired_integral, self.integral_min[i], self.integral_max[i])
                else:  # Temperature control
                    desired_integral = (u[i] - self.u_init[1] - kp_temp * error[i] - feedforward[1]) / ki_temp
                    self.integral[i] = np.clip(desired_integral, self.integral_min[i], self.integral_max[i])
        
        # Clamp integral terms
        self.integral[0] = np.clip(self.integral[0], self.integral_min[0], self.integral_max[0])
        self.integral[1] = np.clip(self.integral[1], self.integral_min[1], self.integral_max[1])
        
        # Derivative term with noise rejection
        derivative = np.zeros(2)
        if self.prev_error is not None:
            raw_derivative = (error - self.prev_error) / self.sample_time
            # Reject noisy derivatives
            derivative_magnitude = np.abs(raw_derivative)
            if derivative_magnitude[0] > 0.1:  # Threshold for C_B
                raw_derivative[0] = 0.0
            if derivative_magnitude[1] > 1.0:  # Threshold for temperature
                raw_derivative[1] = 0.0
            
            derivative = self.derivative_filter * raw_derivative
        
        # Calculate control signals
        u_cb = kp_cb * error[0] + ki_cb * self.integral[0] + kd_temp * derivative[0] + feedforward[0]
        u_temp = kp_temp * error[1] + ki_temp * self.integral[1] + kd_temp * derivative[1] + feedforward[1]
        
        # Apply to actuators
        u[0] = self.u_init[0] + u_cb
        u[1] = self.u_init[1] + u_temp
        
        # Apply hard limits
        u[0] = np.clip(u[0], 3.0, 35.0)
        u[1] = np.clip(u[1], -9000.0, 0.0)
        
        # Smart rate limiting with prediction
        delta_u = u - self.prev_output
        delta_u_norm = np.sqrt(np.sum(delta_u**2))
        
        if delta_u_norm > self.max_rate:
            # Prioritize temperature control for safety
            if y_filtered[1] > 130.0:  # High temperature condition
                # Keep full temperature control action
                temp_scale = 1.0
                cb_scale = max(0.1, (self.max_rate - abs(delta_u[1])) / abs(delta_u[0]) if abs(delta_u[0]) > 0 else 0.1)
            else:
                # Scale both equally
                scale_factor = self.max_rate / delta_u_norm
                temp_scale = scale_factor
                cb_scale = scale_factor
            
            u[0] = self.prev_output[0] + delta_u[0] * cb_scale
            u[1] = self.prev_output[1] + delta_u[1] * temp_scale
        
        # Re-apply hard limits
        u[0] = np.clip(u[0], 3.0, 35.0)
        u[1] = np.clip(u[1], -9000.0, 0.0)
        
        # Proactive safety control
        temp_trend = 0.0
        if self.prev_measurement is not None:
            temp_trend = y_filtered[1] - self.prev_measurement[1]
        
        if y_filtered[1] > 130.0 or (y_filtered[1] > 120.0 and temp_trend > 0.1):
            # Aggressive cooling when temperature is high or rising fast
            cooling_needed = min(1.0, (y_filtered[1] - 120.0) / 30.0)
            u[1] = max(-9000.0, u[1] * (1.0 - 0.5 * cooling_needed))  # More negative = more cooling
            
            # Reduce feed to slow reaction
            feed_reduction = min(1.0, (y_filtered[1] - 120.0) / 30.0)
            u[0] = max(3.0, u[0] * (1.0 - 0.7 * feed_reduction))
        
        # Update state
        self.prev_error = error.copy()
        self.prev_measurement = y_filtered.copy()
        self.prev_output = u.copy()
        
        return u
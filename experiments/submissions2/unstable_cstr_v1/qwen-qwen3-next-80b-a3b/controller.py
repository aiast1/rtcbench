import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.reset()

    def reset(self):
        # Initialize controller states
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_output = 300.0
        self.prev_time = 0.0
        self.last_valid_output = 300.0
        self.last_temp = 350.0
        self.prev_filtered_derivative = 0.0
        
        # Conservative PID parameters - prioritizing stability and smoothness
        self.Kp = 0.6
        self.Ki = 0.008
        self.Kd = 0.02
        
        # Anti-windup and constraints
        self.sat_limit = 15.0  # Very conservative integral windup buffer
        self.slew_rate_limit = 1.5 * self.brief.sample_time  # 4.5 K per step (hard limit)
        self.stiction_hysteresis = 0.15
        
        # Derivative filtering
        self.d_filter_alpha = 0.02  # Very strong filtering for noisy measurement
        
        # Safety margins - wider than physical limits for buffer
        self.safe_lower = 315.0
        self.safe_upper = 440.0
        self.critical_lower = 310.0
        self.critical_upper = 450.0
        
        # Emergency damping parameters
        self.emergency_Kp = 0.2
        self.emergency_Ki = 0.001
        self.emergency_Kd = 0.05
        
        # Disturbance detection and compensation
        self.temp_rate_limit = 5.0  # Max safe dT/dt (K/s)
        self.last_temp_rate = 0.0
        self.disturbance_compensation = 0.0
        
        # Actuator duty limit protection
        self.max_change_per_step = 1.5 * self.brief.sample_time  # 4.5 K/step (hard physical limit)
        self.min_output_change = 0.05  # Minimum change to overcome stiction
        
        # State history for trend detection
        self.temp_history = [350.0] * 5  # Last 5 temperature readings
        self.output_history = [300.0] * 5  # Last 5 outputs

    def step(self, t, y, r, quality):
        # Extract measurements and setpoints
        reactor_temp = y[0]
        coolant_temp_readback = y[1]
        setpoint = r[0]
        
        # Safety check - if we're near critical limits, go into emergency mode
        if reactor_temp <= self.critical_lower or reactor_temp >= self.critical_upper:
            Kp = self.emergency_Kp
            Ki = self.emergency_Ki
            Kd = self.emergency_Kd
            # Force setpoint toward safe zone
            if reactor_temp < self.critical_lower:
                setpoint = max(setpoint, self.safe_lower)
            else:
                setpoint = min(setpoint, self.safe_upper)
        else:
            Kp = self.Kp
            Ki = self.Ki
            Kd = self.Kd
        
        # Handle bad quality readings
        if not quality[0]:
            current_temp = self.last_temp
        else:
            current_temp = reactor_temp
            self.last_temp = reactor_temp
            self.temp_history.pop(0)
            self.temp_history.append(current_temp)
        
        # Calculate error
        error = setpoint - current_temp
        
        # Time step
        dt = t - self.prev_time if self.prev_time > 0 else self.brief.sample_time
        
        # Rate of temperature change (for disturbance detection and damping)
        temp_rate = (current_temp - self.last_temp) / dt if dt > 0 else 0.0
        self.last_temp_rate = temp_rate
        
        # Prevent runaway: if temperature is rising too fast, reduce output aggressively
        if temp_rate > self.temp_rate_limit and current_temp > 360.0:
            # Strong damping - reduce output immediately
            output = self.prev_output - 2.0
            output = np.clip(output, 270.0, 340.0)
            self.prev_output = output
            self.prev_time = t
            return np.array([output])
        
        # Integral term with anti-windup
        if self.prev_output >= 340.0 and error > 0:
            integral_increment = 0.0
        elif self.prev_output <= 270.0 and error < 0:
            integral_increment = 0.0
        else:
            integral_increment = error * dt
        
        self.integral_error += integral_increment
        self.integral_error = np.clip(self.integral_error, -self.sat_limit, self.sat_limit)
        
        # Derivative term with very strong filtering
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        filtered_derivative = self.d_filter_alpha * derivative + (1 - self.d_filter_alpha) * self.prev_filtered_derivative
        self.prev_filtered_derivative = filtered_derivative
        self.prev_error = error
        
        # PID output
        p_term = Kp * error
        i_term = Ki * self.integral_error
        d_term = Kd * filtered_derivative
        
        output = self.prev_output + p_term + i_term + d_term
        
        # Apply slew rate limit (hard physical constraint)
        output = np.clip(output, self.prev_output - self.max_change_per_step, self.prev_output + self.max_change_per_step)
        
        # Apply actuator hard limits
        output = np.clip(output, 270.0, 340.0)
        
        # Apply stiction hysteresis - only change if necessary
        if abs(output - self.last_valid_output) < self.stiction_hysteresis:
            output = self.last_valid_output
        else:
            self.last_valid_output = output
        
        # Emergency safety override: if temperature is approaching 470K, shut down cooling aggressively
        if current_temp > 450.0:
            output = 270.0
        elif current_temp > 440.0:
            output = min(output, 280.0)
        elif current_temp > 430.0:
            output = min(output, 290.0)
        
        # Emergency safety override: if temperature is approaching 300K, heat aggressively
        if current_temp < 310.0:
            output = 340.0
        elif current_temp < 320.0:
            output = max(output, 330.0)
        elif current_temp < 330.0:
            output = max(output, 320.0)
        
        # Prevent chattering - ensure minimum output change to overcome stiction
        if abs(output - self.prev_output) < self.min_output_change:
            output = self.prev_output
        
        # Final safety check: if output is oscillating, dampen it
        if len(self.output_history) >= 5:
            recent_outputs = self.output_history[-3:]
            if len(set([np.round(x, 1) for x in recent_outputs])) >= 3:  # If changing every step
                output = self.prev_output  # Hold position
        
        # Update history
        self.output_history.pop(0)
        self.output_history.append(output)
        
        # Update states
        self.prev_output = output
        self.prev_time = t
        
        return np.array([output])
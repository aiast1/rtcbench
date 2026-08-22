import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.reset()
        
    def reset(self):
        # Initialize controller states
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_output = 45.0  # Start near operating point
        self.prev_level = 0.0
        self.prev_steam_flow = 0.0
        self.prev_feedwater_flow = 0.0
        self.prev_pressure = 85.0
        
        # PID parameters (ultra-conservative for robustness)
        self.Kp = 0.15
        self.Ki = 0.0015
        self.Kd = 0.001
        
        # Anti-windup and slew rate limits
        self.max_slew = 0.5    # % per sample (0.1% per second - extremely conservative)
        self.windup_limit = 10.0  # Integral windup limit
        
        # Feedforward compensation gains (physics-based)
        self.compensation_gain = 0.08  # Steam flow feedforward
        self.pressure_comp_gain = 0.005  # Pressure compensation
        
        # Safety margins (aggressive but safe)
        self.safety_margin_high = 75.0   # mm below 250mm trip
        self.safety_margin_low = 75.0    # mm above -250mm trip (inventory)
        
        # Filter for noisy measurements
        self.level_filter_alpha = 0.05
        self.steam_filter_alpha = 0.03
        self.pressure_filter_alpha = 0.03
        self.filtered_level = 0.0
        self.filtered_steam = 0.0
        self.filtered_pressure = 85.0
        
        # State for derivative filtering
        self.prev_derivative = 0.0
        self.derivative_filter_alpha = 0.1
        
        # Duty cycle monitoring (prevent chattering)
        self.last_output = 45.0
        self.duty_counter = 0
        self.max_duty_cycles = 5   # Max changes per 100s (0.00169 duty limit)
        
        # State for level prediction (simple model)
        self.level_prediction = 0.0
        self.prev_feedwater_flow = 0.0
        
    def step(self, t, y, r, quality):
        # Extract measurements
        level_ind = y[0]  # Indicated level (mm)
        pressure = y[1]   # Drum pressure (bar)
        steam_flow = y[2] # Steam flow (kg/s)
        feedwater_flow = y[3]  # Feedwater flow (kg/s)
        
        # Extract setpoint (only scored channel is level)
        setpoint = r[0] if not np.isnan(r[0]) else 0.0
        
        # Apply filtering to noisy measurements
        self.filtered_level = self.level_filter_alpha * level_ind + (1 - self.level_filter_alpha) * self.filtered_level
        self.filtered_steam = self.steam_filter_alpha * steam_flow + (1 - self.steam_filter_alpha) * self.filtered_steam
        self.filtered_pressure = self.pressure_filter_alpha * pressure + (1 - self.pressure_filter_alpha) * self.filtered_pressure
        
        # Check quality flags - if bad, use filtered value
        if not quality[0]:
            level_ind = self.filtered_level
        if not quality[2]:
            steam_flow = self.filtered_steam
        if not quality[3]:
            feedwater_flow = self.prev_feedwater_flow
        if not quality[1]:
            pressure = self.filtered_pressure
            
        # Update previous values
        self.prev_level = level_ind
        self.prev_steam_flow = steam_flow
        self.prev_feedwater_flow = feedwater_flow
        self.prev_pressure = pressure
        
        # Calculate error
        error = self.filtered_level - setpoint
        
        # PID terms
        proportional = self.Kp * error
        
        # Integral term with anti-windup (very conservative)
        if self.prev_output < 95.0 and self.prev_output > 5.0:
            self.integral_error += self.Ki * error * self.sample_time
        else:
            # Anti-windup: only integrate if error opposes saturation
            if self.prev_output >= 95.0 and error < 0:
                self.integral_error += self.Ki * error * self.sample_time
            elif self.prev_output <= 5.0 and error > 0:
                self.integral_error += self.Ki * error * self.sample_time
        
        # Clamp integral term tightly
        self.integral_error = np.clip(self.integral_error, -self.windup_limit, self.windup_limit)
        
        # Derivative term with aggressive filtering
        if t > 0:
            derivative_raw = (error - self.prev_error) / self.sample_time
            derivative_filtered = self.derivative_filter_alpha * derivative_raw + (1 - self.derivative_filter_alpha) * self.prev_derivative
        else:
            derivative_filtered = 0.0
            
        self.prev_error = error
        self.prev_derivative = derivative_filtered
        
        derivative = self.Kd * derivative_filtered
        
        # Feedforward compensation
        steam_delta = self.filtered_steam - self.prev_steam_flow
        feedforward = self.compensation_gain * steam_delta
        
        # Pressure compensation
        pressure_delta = self.filtered_pressure - self.prev_pressure
        pressure_comp = self.pressure_comp_gain * pressure_delta
        
        # Combine all terms
        output_raw = self.prev_output + proportional + self.integral_error + derivative + feedforward + pressure_comp
        
        # Apply slew rate limit (extremely conservative)
        slew_limit = self.max_slew
        output_delta = np.clip(output_raw - self.prev_output, -slew_limit, slew_limit)
        output = self.prev_output + output_delta
        
        # Hard actuator limits
        output = np.clip(output, 0.0, 100.0)
        
        # Safety override: prioritize inventory safety over setpoint
        if self.filtered_level >= 250.0 - self.safety_margin_high:
            # Approaching carryover - aggressively reduce feed
            output = min(output, 30.0)
        elif self.filtered_level <= -250.0 + self.safety_margin_low:
            # Approaching low inventory - aggressively increase feed
            output = max(output, 70.0)
            
        # Pressure safety override
        if self.filtered_pressure < 70.0 + 15.0:  # Near low pressure
            output = max(output, 65.0)
        elif self.filtered_pressure > 100.0 - 15.0:  # Near high pressure
            output = min(output, 35.0)
            
        # Duty cycle limit enforcement (prevent chattering)
        if abs(output - self.last_output) > 0.3:  # Significant change
            self.duty_counter += 1
        else:
            self.duty_counter = max(0, self.duty_counter - 1)
            
        # If too many changes, freeze output
        if self.duty_counter > self.max_duty_cycles:
            output = self.last_output
            
        self.last_output = output
        
        # Final safety: if we're in danger zone, lock output at safe value
        if self.filtered_level >= 200.0:
            output = 25.0
        elif self.filtered_level <= -200.0:
            output = 75.0
            
        # Update previous output
        self.prev_output = output
        
        return np.array([output])
import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.u_min = 0.0
        self.u_max = 100.0
        self.max_rate = 0.00169 * self.sample_time
        
        # Conservative PID parameters for robust control
        self.Kp = 0.08
        self.Ki = 0.005
        self.Kd = 0.01
        
        # State variables
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_derivative = 0.0
        self.prev_output = 45.0
        self.prev_level = 0.0
        self.prev_steam_flow = 0.0
        self.prev_feedwater_flow = 0.0
        self.prev_pressure = 85.0
        
        # Filter parameters
        self.derivative_filter = 0.1
        self.flow_filter = 0.5
        self.pressure_filter = 0.5
        self.level_filter = 0.7
        
        # Safety monitoring
        self.pressure_violation_count = 0
        self.level_violation_count = 0
        self.inventory_violation_count = 0
        
        # Anti-windup
        self.integral_max = 20.0
        
        # Three-element control weights
        self.flow_weight = 0.3
        self.level_weight = 0.7
        
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_derivative = 0.0
        self.prev_output = 45.0
        self.prev_level = 0.0
        self.prev_steam_flow = 0.0
        self.prev_feedwater_flow = 0.0
        self.prev_pressure = 85.0
        self.pressure_violation_count = 0
        self.level_violation_count = 0
        self.inventory_violation_count = 0
        
    def step(self, t, y, r, quality):
        # Extract measurements with filtering for missing data
        level_indicated_raw = y[0]
        level_indicated = level_indicated_raw if quality[0] else self.prev_level
        level_indicated = level_indicated * self.level_filter + self.prev_level * (1 - self.level_filter)
        
        pressure_raw = y[1]
        pressure = pressure_raw if quality[1] else self.prev_pressure
        pressure = pressure * self.pressure_filter + self.prev_pressure * (1 - self.pressure_filter)
        
        steam_flow_raw = y[2]
        steam_flow = steam_flow_raw if quality[2] else self.prev_steam_flow
        steam_flow = steam_flow * self.flow_filter + self.prev_steam_flow * (1 - self.flow_filter)
        
        feedwater_flow_raw = y[3]
        feedwater_flow = feedwater_flow_raw if quality[3] else self.prev_feedwater_flow
        feedwater_flow = feedwater_flow * self.flow_filter + self.prev_feedwater_flow * (1 - self.flow_filter)
        
        # Update stored values
        self.prev_level = level_indicated
        self.prev_pressure = pressure
        self.prev_steam_flow = steam_flow
        self.prev_feedwater_flow = feedwater_flow
        
        # Get setpoint
        setpoint = r[0] if not np.isnan(r[0]) else 0.0
        
        # Three-element control: balance feedwater with steam flow, adjust for level
        flow_error = steam_flow - feedwater_flow
        
        # Level error with deadband to reduce unnecessary movement
        level_error = setpoint - level_indicated
        if abs(level_error) < 5.0:
            level_error = 0.0
        
        # Combined error for three-element control
        combined_error = self.flow_weight * flow_error + self.level_weight * level_error
        
        # Safety monitoring and constraint enforcement
        safety_adjustment = 0.0
        
        # Pressure constraints (70-100 bar)
        if pressure < 72.0:  # Conservative margin
            safety_adjustment += 2.0 * (72.0 - pressure)  # Increase feedwater to raise pressure
            self.pressure_violation_count += 1
        elif pressure > 98.0:  # Conservative margin
            safety_adjustment -= 2.0 * (pressure - 98.0)  # Decrease feedwater to lower pressure
            self.pressure_violation_count += 1
        else:
            self.pressure_violation_count = max(0, self.pressure_violation_count - 1)
        
        # Indicated level upper constraint (250 mm)
        if level_indicated > 220.0:  # Conservative margin
            safety_adjustment -= 3.0 * (level_indicated - 220.0)  # Force level down
            self.level_violation_count += 1
        else:
            self.level_violation_count = max(0, self.level_violation_count - 1)
        
        # Water inventory lower constraint (-250 mm)
        # Estimate inventory: indicated level minus swell effect
        swell_estimate = max(0, (steam_flow - 50.0) * 0.8)  # Swell increases with steam flow
        inventory_estimate = level_indicated - swell_estimate
        
        if inventory_estimate < -220.0:  # Conservative margin
            safety_adjustment += 3.0 * (-220.0 - inventory_estimate)  # Force level up
            self.inventory_violation_count += 1
        else:
            self.inventory_violation_count = max(0, self.inventory_violation_count - 1)
        
        # Apply safety adjustment if any constraint is near violation
        if safety_adjustment != 0.0:
            combined_error = safety_adjustment
        
        # Integral term with conditional reset and anti-windup
        if abs(combined_error) < 1.0:  # Near setpoint
            self.integral += combined_error * self.sample_time
        else:
            # Reset integral during large errors to prevent windup
            self.integral *= 0.9
        
        self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)
        
        # Derivative term with heavy filtering
        derivative = (combined_error - self.prev_error) / self.sample_time
        filtered_derivative = derivative * self.derivative_filter + self.prev_derivative * (1 - self.derivative_filter)
        
        # PID calculation
        P = self.Kp * combined_error
        I = self.Ki * self.integral
        D = self.Kd * filtered_derivative
        
        # Pressure compensation for valve characteristic
        pressure_factor = 85.0 / max(70.0, pressure)
        pressure_factor = np.clip(pressure_factor, 0.7, 1.3)
        
        # Calculate output
        pid_output = (P + I + D) * pressure_factor
        
        # Rate limiting with additional constraints during safety violations
        current_max_rate = self.max_rate
        if (self.pressure_violation_count > 10 or self.level_violation_count > 10 or 
            self.inventory_violation_count > 10):
            current_max_rate *= 0.5  # Slower movements during persistent violations
        
        delta_u = pid_output - self.prev_output
        delta_u = np.clip(delta_u, -current_max_rate, current_max_rate)
        u_output = self.prev_output + delta_u
        
        # Output clamping
        u_output = np.clip(u_output, self.u_min, self.u_max)
        
        # Update state variables
        self.prev_error = combined_error
        self.prev_derivative = filtered_derivative
        self.prev_output = u_output
        
        return np.array([u_output])
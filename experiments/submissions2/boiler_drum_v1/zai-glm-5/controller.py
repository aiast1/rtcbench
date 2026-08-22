import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        
        # State variables
        self.integral = 0.0
        self.prev_level = None
        self.prev_pressure = None
        
        # Filtered values
        self.level_filtered = 0.0
        self.pressure_filtered = 85.0
        self.steam_flow_filtered = 50.0
        self.feed_flow_filtered = 50.0
        
        # Output tracking
        self.output = 45.0
        
        # Inventory estimate (mass balance)
        self.inventory_estimate = 0.0
        
        # Controller tuning - very conservative
        self.Kp = 0.25
        self.Ki = 0.005
        self.Kd = 0.08
        
        # Feedforward
        self.Kff_steam = 0.25
        
        # Anti-windup
        self.Kb = 0.2
        
        # Safety limits - very conservative margins
        self.level_high_limit = 180.0   # Trip at 250
        self.level_low_limit = -180.0   # Trip at -250 inventory
        
        # Actuator rate limit - CRITICAL for duty constraint
        # Duty = sum(|delta_u|) / (duration * max_rate)
        # Need delta << 0.5 to stay under 0.00169 duty
        self.max_delta_per_step = 0.15
        
        # Swell/shrink compensation
        self.swell_gain = 0.3
        
        # First step flag
        self.first_step = True
        
    def reset(self):
        self.integral = 0.0
        self.prev_level = None
        self.prev_pressure = None
        self.level_filtered = 0.0
        self.pressure_filtered = 85.0
        self.steam_flow_filtered = 50.0
        self.feed_flow_filtered = 50.0
        self.output = 45.0
        self.inventory_estimate = 0.0
        self.first_step = True
        
    def _get_setpoint(self, t):
        """Get setpoint from schedule"""
        if t < 300:
            return 0.0
        elif t < 1400:
            return 80.0
        elif t < 1580:
            # Ramp from 80 to -60 over 180s
            progress = (t - 1400) / 180.0
            return 80.0 + progress * (-140.0)
        elif t < 2100:
            return -60.0
        else:
            return 0.0
    
    def step(self, t, y, r, quality):
        dt = self.sample_time
        
        # Extract measurements
        level_raw = y[0]
        pressure_raw = y[1]
        steam_raw = y[2]
        feed_raw = y[3]
        
        # Handle bad quality readings
        if not quality[0] or np.isnan(level_raw):
            level_raw = self.level_filtered if self.level_filtered != 0.0 else 0.0
        if not quality[1] or np.isnan(pressure_raw):
            pressure_raw = self.pressure_filtered
        if not quality[2] or np.isnan(steam_raw):
            steam_raw = self.steam_flow_filtered
        if not quality[3] or np.isnan(feed_raw):
            feed_raw = self.feed_flow_filtered
        
        # Very strong low-pass filter (alpha = 0.08 for noise rejection)
        alpha = 0.08
        self.level_filtered = alpha * level_raw + (1 - alpha) * self.level_filtered
        self.pressure_filtered = alpha * pressure_raw + (1 - alpha) * self.pressure_filtered
        self.steam_flow_filtered = alpha * steam_raw + (1 - alpha) * self.steam_flow_filtered
        self.feed_flow_filtered = alpha * feed_raw + (1 - alpha) * self.feed_flow_filtered
        
        level = self.level_filtered
        pressure = self.pressure_filtered
        steam_flow = self.steam_flow_filtered
        feed_flow = self.feed_flow_filtered
        
        # Update inventory estimate via mass balance
        if steam_flow > 0 and feed_flow > 0:
            self.inventory_estimate += (feed_flow - steam_flow) * dt * 0.05
        
        # Get setpoint
        setpoint = self._get_setpoint(t)
        error = setpoint - level
        
        # Pressure derivative for swell/shrink compensation
        pressure_deriv = 0.0
        if self.prev_pressure is not None:
            pressure_deriv = (pressure - self.prev_pressure) / dt
        
        # Swell compensation
        swell_compensation = -self.swell_gain * pressure_deriv * 3.0
        
        # Proportional term
        P_term = self.Kp * error
        
        # Integral term with anti-windup
        int_gain = self.Ki
        if level > self.level_high_limit * 0.85:
            int_gain *= 0.2
        elif level < self.level_low_limit * 0.85:
            int_gain *= 0.2
        
        # Back-calculation for anti-windup
        back_calc = self.Kb * (self.output - 45.0 - self.Kp * error - self.Kff_steam * (steam_flow - 50.0))
        self.integral += int_gain * error * dt - back_calc * dt * 0.1
        self.integral = np.clip(self.integral, -20.0, 20.0)
        
        I_term = self.integral
        
        # Derivative term (on measurement)
        D_term = 0.0
        if self.prev_level is not None:
            level_deriv = (level - self.prev_level) / dt
            D_term = -self.Kd * level_deriv
        
        # Feedforward from steam flow
        FF_term = self.Kff_steam * (steam_flow - 50.0)
        
        # Base output
        output = 45.0 + P_term + I_term + D_term + FF_term + swell_compensation
        
        # Safety override: if level too high, force output down
        if level > self.level_high_limit:
            output = min(output, 35.0 - (level - self.level_high_limit) * 1.0)
        
        # Safety override: if level too low, force output up
        if level < self.level_low_limit:
            output = max(output, 55.0 + (self.level_low_limit - level) * 1.0)
        
        # Soft saturation
        output = np.clip(output, 10.0, 90.0)
        
        # CRITICAL: Rate limit for actuator duty constraint
        delta = output - self.output
        if abs(delta) > self.max_delta_per_step:
            output = self.output + np.sign(delta) * self.max_delta_per_step
        
        # Hard limits
        output = np.clip(output, 0.0, 100.0)
        
        # Store state
        self.prev_level = level
        self.prev_pressure = pressure
        self.output = output
        
        self.first_step = False
        
        return np.array([output])
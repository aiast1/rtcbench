import numpy as np

class Controller:
    def __init__(self, brief):
        self.dt = brief.sample_time
        
        # Measurement indices
        self.i_level = 0
        self.i_pressure = 1
        self.i_steam = 2
        self.i_feedwater = 3
        
        # Actuator indices
        self.i_valve = 0
        
        # Very conservative PID - prioritize smoothness over aggression
        self.Kp = 0.3
        self.Ki = 0.003
        self.Kd = 0.5
        
        # Derivative filter
        self.tau_d = 20.0
        
        # Output limits
        self.out_min = 0.0
        self.out_max = 100.0
        
        # State variables
        self.reset()
        
    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_filtered_deriv = 0.0
        self.last_output = 45.0
        self.last_setpoint = 0.0
        self.last_measurement = 0.0
        
        # History for estimation
        self.level_history = []
        self.pressure_history = []
        self.steam_history = []
        self.feedwater_history = []
        self.history_len = 40
        
        # Mass balance for inventory
        self.estimated_inventory = 0.0
        self.inventory_initialized = False
        
        # For duty calculation - track total movement
        self.total_movement = 0.0
        
    def step(self, t, y, r, quality):
        # Extract measurements with fallback
        level = y[self.i_level] if quality[self.i_level] else self.last_measurement
        pressure = y[self.i_pressure] if quality[self.i_pressure] else (self.pressure_history[-1] if self.pressure_history else 85.0)
        steam_flow = y[self.i_steam] if quality[self.i_steam] else (self.steam_history[-1] if self.steam_history else 50.0)
        feedwater_flow = y[self.i_feedwater] if quality[self.i_feedwater] else (self.feedwater_history[-1] if self.feedwater_history else 50.0)
        
        # Update histories
        self.level_history.append(level)
        self.pressure_history.append(pressure)
        self.steam_history.append(steam_flow)
        self.feedwater_history.append(feedwater_flow)
        
        if len(self.level_history) > self.history_len:
            self.level_history.pop(0)
            self.pressure_history.pop(0)
            self.steam_history.pop(0)
            self.feedwater_history.pop(0)
        
        # Get setpoint
        setpoint = r[self.i_level]
        if np.isnan(setpoint):
            setpoint = self.last_setpoint
        self.last_setpoint = setpoint
        
        # Initialize inventory estimate
        if not self.inventory_initialized:
            self.estimated_inventory = level
            self.inventory_initialized = True
        
        # Simple inventory observer: mass balance with level correction
        mass_imbalance = feedwater_flow - steam_flow
        self.estimated_inventory += mass_imbalance * self.dt * 0.3
        
        # Slowly correct to level when conditions are stable
        if len(self.pressure_history) >= 10:
            recent_pressure = self.pressure_history[-10:]
            pressure_var = np.std(recent_pressure)
            if pressure_var < 2.0:  # Stable pressure
                self.estimated_inventory = 0.95 * self.estimated_inventory + 0.05 * level
        
        # Compute trends
        if len(self.pressure_history) >= 5:
            dp_dt = (pressure - self.pressure_history[-5]) / (5 * self.dt)
        else:
            dp_dt = 0.0
        
        if len(self.steam_history) >= 5:
            dsteam_dt = (steam_flow - self.steam_history[-5]) / (5 * self.dt)
        else:
            dsteam_dt = 0.0
        
        # Safety margins
        inventory_margin = self.estimated_inventory - (-250.0)
        level_margin = 250.0 - level
        
        # Very conservative control - prioritize safety over tracking
        effective_setpoint = setpoint
        
        # Critical safety: inventory low
        if inventory_margin < 80.0:
            effective_setpoint = -300.0  # Emergency fill
        elif inventory_margin < 150.0:
            effective_setpoint = setpoint - 50.0
        
        # Level high
        if level > 180:
            effective_setpoint = max(effective_setpoint, 100.0)
        
        # Compute error
        error = effective_setpoint - level
        
        # Compute derivative (heavily filtered)
        d_meas = (level - self.last_measurement) / self.dt
        alpha = self.dt / (self.tau_d + self.dt)
        filtered_deriv = alpha * d_meas + (1 - alpha) * self.last_filtered_deriv
        self.last_filtered_deriv = filtered_deriv
        
        # PID
        P = self.Kp * error
        self.integral += self.Ki * error * self.dt
        self.integral = np.clip(self.integral, -30.0, 30.0)
        D = -self.Kd * filtered_deriv
        
        # Feedforward - smooth response to steam flow changes
        pressure_factor = np.sqrt(85.0 / max(pressure, 70.0))
        ff_pos = 45.0 * (steam_flow / 50.0) * pressure_factor
        
        # Combine with moderate feedforward
        output = P + self.integral + D + 0.5 * ff_pos
        
        # Hard safety limits
        if inventory_margin < 100.0:
            output = max(output, 55.0)
        if level > 200:
            output = min(output, 35.0)
        
        # Clamp
        output = np.clip(output, self.out_min, self.out_max)
        
        # Very aggressive rate limiting to prevent duty breach
        # Duty limit is 0.00169 per second average over scenario
        # With 2600s scenario, total movement must be < 4.4
        # Be very conservative
        max_rate = 0.3  # % per second - very slow
        max_change = max_rate * self.dt
        output = np.clip(output, self.last_output - max_change, self.last_output + max_change)
        
        # Update state
        self.last_error = error
        self.last_measurement = level
        self.last_output = output
        
        return np.array([output])
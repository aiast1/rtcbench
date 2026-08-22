import numpy as np

class Controller:
    def __init__(self, brief):
        self.Ts = brief.sample_time
        model = brief.model_hint
        
        # Correct dictionary access for model parameters
        T_supply = model['T_supply']
        K_h = model['K_h']
        
        # Conservative safety margin with floor for minimum duty
        self.max_duty = min((90.0 - T_supply)/(K_h * 3.0), 100.0)
        self.min_duty = max((0.0 - T_supply)/(K_h * 1.2), 0.0)  # Prevents T_out < 0
        
        # Robust PI tuning for variable delay
        self.Kp = 0.25
        self.Ti = 250.0
        self.Tf = 50.0  # Slower setpoint filtering
        
        self.reset()

    def reset(self):
        self.integral = 50.0  # Match initial actuator position
        self.last_valid_y = 55.0
        self.filtered_sp = 55.0
        self.safety_cooldown = 0

    def step(self, t, y, r, quality):
        # Handle measurement quality with persistence
        current_y = y[0] if quality[0] else self.last_valid_y
        self.last_valid_y = current_y
        
        # Conservative safety pre-check
        if current_y > 88.0:  # Stay well below 90°C limit
            return np.array([0.0])
        
        # Setpoint processing with ramp tracking
        sp = r[0] if not np.isnan(r[0]) else self.filtered_sp
        self.filtered_sp += (sp - self.filtered_sp) * (self.Ts/(self.Tf + self.Ts))
        
        # PI control with dynamic clamping
        error = self.filtered_sp - current_y
        P = self.Kp * error
        self.integral += (self.Kp * self.Ts / self.Ti) * error
        
        # Calculate raw output with safety bounds
        u = P + self.integral
        u_clamped = np.clip(u, self.min_duty, self.max_duty)
        
        # Anti-windup: only integrate when not saturated
        if abs(u - u_clamped) > 1e-6:
            self.integral = u_clamped - P
        
        # Final actuator limits
        u_clamped = np.clip(u_clamped, 0.0, 100.0)
        return np.array([u_clamped])